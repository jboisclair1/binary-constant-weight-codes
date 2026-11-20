#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <utility>
#include <vector>
#include "ATS_v4_5_cuda.h"
using namespace std;

#ifndef ATS48_CPU_SHADOW_INCIDENCE
#define ATS48_CPU_SHADOW_INCIDENCE 0
#endif

#ifndef ATS48_RANKED_REPAIR_QUEUE
#define ATS48_RANKED_REPAIR_QUEUE 0
#endif

struct Config {
    int n=0,d=0,w=0,lambda=0,t=0,base_m=0,target_m=0;
    uint64_t choose[65][65]{};
    uint64_t shadow_universe=0;
    vector<vector<int>> comb_w_t, comb_wm1_tm1;
};

static uint64_t outsider_samples = 5000000ULL;
static size_t candidate_pool = 2048;
static size_t gpu_batch_size = 262144;
static uint32_t gpu_walkers = 8192;
static uint32_t gpu_proposals = 16;
static uint32_t cuda_repair_capacity = 4U * 1024U * 1024U;
static bool legacy_regression_pool = false;
static bool exact_repair_enabled = true;
static bool reference_policy_audit = false;
static bool first_defect1_forensics = false;
static bool stop_on_first_defect1_mode = false;
enum class GpuMode { Off, Auto, On };
static GpuMode gpu_mode = GpuMode::Auto;

static const char* gpu_mode_name(GpuMode mode){return mode==GpuMode::Off?"off":mode==GpuMode::On?"on":"auto";}
static GpuMode parse_gpu_mode(string value){
    transform(value.begin(),value.end(),value.begin(),[](unsigned char ch){return static_cast<char>(tolower(ch));});
    if(value=="off"||value=="cpu"||value=="0")return GpuMode::Off;
    if(value=="auto")return GpuMode::Auto;
    if(value=="on"||value=="gpu"||value=="1")return GpuMode::On;
    throw runtime_error("cuda_mode must be off, auto, or on");
}

static uint32_t gpu_mix32(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
static uint32_t gpu_random_word(uint64_t seed,uint64_t index,uint32_t draw){
    uint32_t lo=static_cast<uint32_t>(index),hi=static_cast<uint32_t>(index>>32);
    uint32_t x=static_cast<uint32_t>(seed)^(lo*0x9e3779b9U)^(hi*0x85ebca6bU);
    x^=static_cast<uint32_t>(seed>>32)+draw*0xc2b2ae35U;
    return gpu_mix32(x^gpu_mix32(draw+0x27d4eb2dU));
}
static uint64_t gpu_generated_block(uint64_t seed,uint64_t index,int n,int w){
    uint64_t mask=0;uint32_t draw=0;
    for(uint32_t j=static_cast<uint32_t>(n-w);j<static_cast<uint32_t>(n);++j){uint32_t selected=gpu_random_word(seed,index,draw++)%(j+1);uint32_t bit=(mask>>selected&1ULL)?j:selected;mask|=1ULL<<bit;}
    return mask;
}

static void comb_rec(int n,int k,int start,vector<int>&cur,vector<vector<int>>&out){
    if((int)cur.size()==k){out.push_back(cur);return;}
    int need=k-(int)cur.size();
    for(int x=start;x<=n-need;++x){cur.push_back(x);comb_rec(n,k,x+1,cur,out);cur.pop_back();}
}
static vector<vector<int>> combinations(int n,int k){vector<vector<int>>o;vector<int>c;if(k>=0&&k<=n)comb_rec(n,k,0,c,o);return o;}

static Config make_config(int n,int d,int w,int m){
    Config c;c.n=n;c.d=d;c.w=w;c.base_m=m;c.target_m=m+1;
    if(n<1||n>64||w<1||w>n) throw runtime_error("require 1<=w<=n<=64");
    if(d%2) throw runtime_error("distance must be even");
    c.lambda=w-d/2;c.t=c.lambda+1;
    if(c.lambda<0||c.t<1||c.t>w) throw runtime_error("invalid parameters");
    for(int i=0;i<=64;++i){c.choose[i][0]=1;for(int j=1;j<=i;++j)c.choose[i][j]=(j==i)?1:c.choose[i-1][j-1]+c.choose[i-1][j];}
    c.shadow_universe=c.choose[n][c.t];
    c.comb_w_t=combinations(w,c.t);c.comb_wm1_tm1=combinations(w-1,c.t-1);
    return c;
}
static inline int pc(uint64_t x){return static_cast<int>(std::popcount(x));}
static vector<int> coords(uint64_t m){vector<int>v;while(m){int b=static_cast<int>(std::countr_zero(m));v.push_back(b);m&=m-1;}return v;}
static inline uint64_t rank_shadow(uint64_t m,const Config&c){
    if(c.t==1)return std::countr_zero(m);
    const uint64_t a=std::countr_zero(m);m&=m-1;
    const uint64_t b=std::countr_zero(m);m&=m-1;
    if(c.t==2)return a+b*(b-1)/2;
    const uint64_t d=std::countr_zero(m);
    if(c.t==3)return a+b*(b-1)/2+d*(d-1)*(d-2)/6;
    uint64_t r=a+c.choose[b][2]+c.choose[d][3];m&=m-1;int i=4;
    while(m){int x=static_cast<int>(std::countr_zero(m));r+=c.choose[x][i++];m&=m-1;}
    return r;
}
static vector<uint64_t> shadows(uint64_t block,const Config&c){
    auto p=coords(block);if((int)p.size()!=c.w)throw runtime_error("bad block weight");
    vector<uint64_t>o;o.reserve(c.comb_w_t.size());
    for(auto&ix:c.comb_w_t){uint64_t s=0;for(int q:ix)s|=1ULL<<p[q];o.push_back(s);}return o;
}
static vector<uint64_t> change_bases(uint64_t block,int oldp,const Config&c){
    auto p=coords(block&~(1ULL<<oldp));vector<uint64_t>o;o.reserve(c.comb_wm1_tm1.size());
    for(auto&ix:c.comb_wm1_tm1){uint64_t s=0;for(int q:ix)s|=1ULL<<p[q];o.push_back(s);}return o;
}

static vector<uint64_t> parse_code(const string&path,int n,int w,string*detected_format=nullptr){
    ifstream f(path);if(!f)throw runtime_error("cannot open "+path);
    vector<uint64_t>blocks;string line;int base=0;bool global_zero=false;string format="support sets";
    static const regex re("[0-9]+");
    while(getline(f,line)){
        if(line.find("zero-based")!=string::npos||line.find("ZERO-BASED")!=string::npos)global_zero=true;
        string s=line;auto h=s.find('#');if(h!=string::npos)s=s.substr(0,h);
        string compact;for(char ch:s)if(!isspace(static_cast<unsigned char>(ch))&&ch!='\\')compact.push_back(ch);
        if(compact.empty())continue;
        if(compact.rfind("$BASE=",0)==0){
            base=stoi(compact.substr(6));
            if(base!=2&&base!=16)throw runtime_error("unsupported $BASE="+to_string(base));
            format=base==16?"$BASE=16":"$BASE=2";
            continue;
        }
        uint64_t mask=0;
        if(base==16){
            if(compact.size()>16)throw runtime_error("hex codeword exceeds 64 bits: "+compact);
            size_t used=0;
            try{mask=stoull(compact,&used,16);}catch(const exception&){throw runtime_error("invalid hexadecimal codeword: "+compact);}
            if(used!=compact.size())throw runtime_error("invalid hexadecimal codeword: "+compact);
        }else if((base==2||base==0)&&(int)compact.size()==n&&all_of(compact.begin(),compact.end(),[](char ch){return ch=='0'||ch=='1';})){
            if(base==0)format="binary incidence";
            for(int i=0;i<n;++i)if(compact[i]=='1')mask|=1ULL<<i;
        }else{
            vector<int>a;for(sregex_iterator it(compact.begin(),compact.end(),re),e;it!=e;++it)a.push_back(stoi(it->str()));
            if(a.empty())continue;bool zero=global_zero||find(a.begin(),a.end(),0)!=a.end();
            for(int x:a){int p=zero?x:x-1;if(p<0||p>=n)throw runtime_error("coordinate out of range");mask|=1ULL<<p;}
        }
        if(n<64&&(mask>>n)!=0){ostringstream o;o<<"codeword has bits outside n="<<n<<": "<<compact;throw runtime_error(o.str());}
        if(pc(mask)!=w){ostringstream o;o<<"parsed weight "<<pc(mask)<<" not "<<w<<" line="<<line;throw runtime_error(o.str());}
        blocks.push_back(mask);
    }
    if(blocks.empty())throw runtime_error("no codewords found in "+path);
    if(detected_format)*detected_format=format;
    return blocks;
}
static bool verify(const vector<uint64_t>&b,const Config&c,string*why=nullptr){
    unordered_set<uint64_t>seen;seen.reserve(b.size()*2+1);
    for(int i=0;i<(int)b.size();++i){
        if(pc(b[i])!=c.w){if(why)*why="wrong weight";return false;}
        if(c.n<64&&(b[i]>>c.n)){if(why)*why="range";return false;}
        if(!seen.insert(b[i]).second){if(why)*why="duplicate";return false;}
        for(int j=0;j<i;++j){int x=pc(b[i]&b[j]);if(x>c.lambda){if(why){ostringstream o;o<<"bad pair "<<j<<","<<i<<" inter="<<x;*why=o.str();}return false;}}
    }return true;
}
static void write_code(const vector<uint64_t>&b,const Config&c,const string&path,const string&note){
    auto v=b;sort(v.begin(),v.end());ofstream f(path);if(!f)throw runtime_error("cannot write "+path);
    f<<"# "<<v.size()<<" constant-weight codewords; "<<note<<"\n# A("<<c.n<<","<<c.d<<","<<c.w<<") >= "<<v.size()<<"; one-based supports\n\n";
    for(uint64_t m:v){f<<"{";bool first=true;for(int x=0;x<c.n;++x)if(m>>x&1ULL){if(!first)f<<",";first=false;f<<x+1;}f<<"}\n";}
}

struct Base{vector<uint64_t>blocks;vector<uint16_t>cnt;vector<int>deg;unordered_set<uint64_t>set;};
static Base build_base(const vector<uint64_t>&b,const Config&c){
    Base x;x.blocks=b;x.cnt.assign(c.shadow_universe,0);x.deg.assign(c.n,0);x.set.reserve(b.size()*2+1);
    for(uint64_t m:b){x.set.insert(m);for(int p=0;p<c.n;++p)if(m>>p&1ULL)x.deg[p]++;for(uint64_t s:shadows(m,c))if(++x.cnt[rank_shadow(s,c)]!=1)throw runtime_error("base repeats critical shadow");}
    return x;
}
static long long degcost(const vector<int>&d,int total,int n){long long z=0;for(int x:d){long long q=1LL*x*n-total;z+=q*q;}return z;}
struct State{
    vector<uint64_t>blocks;
    vector<uint16_t>cnt;
    vector<int>deg;
    vector<uint64_t>bad;
    unordered_map<uint64_t,uint16_t>occupied_count;
#if ATS48_CPU_SHADOW_INCIDENCE
    vector<uint16_t>shadow_head;
    vector<uint16_t>shadow_next;
    uint32_t incidence_shadows_per_block=0;
#endif
    long long score=0,dcost=0;
    int total=0;
};
#if ATS48_CPU_SHADOW_INCIDENCE
static constexpr uint16_t CPU_NO_SHADOW_NODE=UINT16_MAX;
static bool update_state_incidence(
    State&s,int bi,uint64_t block,const Config&c,bool insert)
{
    if(s.shadow_head.empty())return true;
    const auto block_shadows=shadows(block,c);
    if(block_shadows.size()!=s.incidence_shadows_per_block)return false;
    for(uint32_t local=0;local<s.incidence_shadows_per_block;++local){
        const uint64_t rank=rank_shadow(block_shadows[local],c);
        const uint16_t node=static_cast<uint16_t>(
            static_cast<uint32_t>(bi)*s.incidence_shadows_per_block+local);
        if(insert){
            s.shadow_next[node]=s.shadow_head[rank];
            s.shadow_head[rank]=node;
            continue;
        }
        uint16_t previous=CPU_NO_SHADOW_NODE,current=s.shadow_head[rank];
        while(current!=CPU_NO_SHADOW_NODE&&current!=node){
            previous=current;current=s.shadow_next[current];
        }
        if(current==CPU_NO_SHADOW_NODE)return false;
        if(previous==CPU_NO_SHADOW_NODE)s.shadow_head[rank]=s.shadow_next[current];
        else s.shadow_next[previous]=s.shadow_next[current];
        s.shadow_next[current]=CPU_NO_SHADOW_NODE;
    }
    return true;
}
static void initialize_state_incidence(State&s,const Config&c){
    if(c.target_m<512)return;
    const uint64_t shadows_per_block=c.choose[c.w][c.t];
    const uint64_t nodes=static_cast<uint64_t>(c.target_m)*shadows_per_block;
    if(!shadows_per_block||nodes>=CPU_NO_SHADOW_NODE)return;
    s.incidence_shadows_per_block=static_cast<uint32_t>(shadows_per_block);
    s.shadow_head.assign(static_cast<size_t>(c.shadow_universe),CPU_NO_SHADOW_NODE);
    s.shadow_next.assign(static_cast<size_t>(nodes),CPU_NO_SHADOW_NODE);
    for(int bi=0;bi<c.target_m;++bi)
        if(!update_state_incidence(s,bi,s.blocks[bi],c,true))
            throw runtime_error("CPU shadow incidence initialization failed");
}
static vector<int> shadow_owners(const State&s,uint64_t shadow,const Config&c){
    vector<int>owners;
    if(!s.shadow_head.empty()){
        for(uint16_t node=s.shadow_head[rank_shadow(shadow,c)];
            node!=CPU_NO_SHADOW_NODE;node=s.shadow_next[node])
            owners.push_back(node/s.incidence_shadows_per_block);
        return owners;
    }
    for(int bi=0;bi<c.target_m;++bi)if((s.blocks[bi]&shadow)==shadow)owners.push_back(bi);
    return owners;
}
static bool audit_state_incidence(const State&s,const Config&c){
    if(s.shadow_head.empty())return true;
    vector<uint64_t>expected_rank(s.shadow_next.size(),UINT64_MAX);
    for(int bi=0;bi<c.target_m;++bi){
        const auto block_shadows=shadows(s.blocks[bi],c);
        for(uint32_t local=0;local<s.incidence_shadows_per_block;++local){
            const size_t node=static_cast<size_t>(bi)*s.incidence_shadows_per_block+local;
            expected_rank[node]=rank_shadow(block_shadows[local],c);
        }
    }
    vector<uint8_t>seen(s.shadow_next.size(),0);
    for(size_t rank=0;rank<s.shadow_head.size();++rank){
        uint32_t count=0,steps=0;
        for(uint16_t node=s.shadow_head[rank];node!=CPU_NO_SHADOW_NODE;
            node=s.shadow_next[node]){
            if(node>=s.shadow_next.size()||seen[node]||expected_rank[node]!=rank)return false;
            seen[node]=1;++count;
            if(++steps>s.shadow_next.size())return false;
        }
        if(count!=s.cnt[rank])return false;
    }
    return all_of(seen.begin(),seen.end(),[](uint8_t value){return value!=0;});
}
#endif
static void add_block(State&s,uint64_t m,const Config&c){
    ++s.occupied_count[m];
    for(int p=0;p<c.n;++p)if(m>>p&1ULL)s.deg[p]++;
    for(uint64_t sh:shadows(m,c)){auto r=rank_shadow(sh,c);uint16_t z=s.cnt[r];s.score+=z;s.cnt[r]=z+1;if(z==1)s.bad.push_back(sh);}s.blocks.push_back(m);s.total+=c.w;s.dcost=degcost(s.deg,s.total,c.n);
}
static void remove_at(State&s,int bi,const Config&c){
    uint64_t m=s.blocks[bi];
#if ATS48_CPU_SHADOW_INCIDENCE
    if(!update_state_incidence(s,bi,m,c,false))throw runtime_error("CPU shadow incidence removal failed");
#endif
    for(uint64_t sh:shadows(m,c)){auto r=rank_shadow(sh,c);uint16_t z=s.cnt[r];if(!z)throw runtime_error("underflow");s.score-=z-1;s.cnt[r]=z-1;}
    for(int p=0;p<c.n;++p)if(m>>p&1ULL)s.deg[p]--;if(--s.occupied_count[m]==0)s.occupied_count.erase(m);s.blocks[bi]=0;s.total-=c.w;s.dcost=degcost(s.deg,s.total,c.n);
}
static void install_at(State&s,int bi,uint64_t m,const Config&c){
    if(s.blocks[bi])throw runtime_error("occupied slot");++s.occupied_count[m];for(int p=0;p<c.n;++p)if(m>>p&1ULL)s.deg[p]++;
    for(uint64_t sh:shadows(m,c)){auto r=rank_shadow(sh,c);uint16_t z=s.cnt[r];s.score+=z;s.cnt[r]=z+1;if(z==1)s.bad.push_back(sh);}s.blocks[bi]=m;
#if ATS48_CPU_SHADOW_INCIDENCE
    if(!update_state_incidence(s,bi,m,c,true))throw runtime_error("CPU shadow incidence insertion failed");
#endif
    s.total+=c.w;s.dcost=degcost(s.deg,s.total,c.n);
}
static State make_state(const Base&b,uint64_t extra,const Config&c){State s;s.blocks=b.blocks;s.cnt=b.cnt;s.deg=b.deg;s.occupied_count.reserve(c.target_m*2+1);for(uint64_t block:b.blocks)++s.occupied_count[block];s.total=c.base_m*c.w;s.dcost=degcost(s.deg,s.total,c.n);add_block(s,extra,c);
#if ATS48_CPU_SHADOW_INCIDENCE
    initialize_state_incidence(s,c);
#endif
    return s;}
static State make_state(const vector<uint64_t>&blocks,const Config&c){State s;s.cnt.assign(c.shadow_universe,0);s.deg.assign(c.n,0);s.blocks.reserve(blocks.size());s.occupied_count.reserve(blocks.size()*2+1);for(uint64_t block:blocks)add_block(s,block,c);
#if ATS48_CPU_SHADOW_INCIDENCE
    initialize_state_incidence(s,c);
#endif
    return s;}
[[maybe_unused]] static long long contribution(uint64_t m,const vector<uint16_t>&cnt,const Config&c){long long v=0;for(uint64_t sh:shadows(m,c))v+=cnt[rank_shadow(sh,c)];return v;}
static long long contribution_direct(uint64_t m,const Base&b,const Config&c){long long v=0;for(uint64_t x:b.blocks){int z=pc(m&x);if(z>=c.t)v+=c.choose[z][c.t];}return v;}

static uint64_t random_block(mt19937_64&r,int n,int w){vector<int>p(n);iota(p.begin(),p.end(),0);for(int i=0;i<w;++i){int j=uniform_int_distribution<int>(i,n-1)(r);swap(p[i],p[j]);}uint64_t m=0;for(int i=0;i<w;++i)m|=1ULL<<p[i];return m;}
static uint64_t greedy_block(mt19937_64&r,const vector<uint16_t>&cnt,const Config&c){
    uint64_t m=0;vector<int>chosen;chosen.reserve(c.w);
    for(int pos=0;pos<c.w;++pos){
        vector<vector<int>> ix=((int)chosen.size()>=c.t-1)?combinations((int)chosen.size(),c.t-1):vector<vector<int>>{};
        vector<pair<long long,int>>vals;vals.reserve(c.n-pos);
        for(int x=0;x<c.n;++x)if(!(m>>x&1ULL)){long long inc=0;for(auto&a:ix){uint64_t sh=1ULL<<x;for(int q:a)sh|=1ULL<<chosen[q];inc+=cnt[rank_shadow(sh,c)];}vals.push_back({inc*32+(long long)(r()%13),x});}
        int keep=min(4,static_cast<int>(vals.size()));nth_element(vals.begin(),vals.begin()+keep-1,vals.end());sort(vals.begin(),vals.begin()+keep);int x=vals[uniform_int_distribution<int>(0,keep-1)(r)].second;chosen.push_back(x);m|=1ULL<<x;
    }return m;
}

struct Extra{long long score;uint64_t mask;};
static void keep_top(priority_queue<pair<long long,uint64_t>>&h,long long sc,uint64_t m,size_t cap){pair<long long,uint64_t>x{sc,m};if(h.size()<cap)h.push(x);else if(x<h.top()){h.pop();h.push(x);}}
static bool fill_extra_pool_gpu(const Base&b,const Config&c,uint64_t seed,priority_queue<pair<long long,uint64_t>>&h,uint64_t&tested,long long&best,ostream&log,string&error){
    ats45cuda::CandidateScorer scorer;
    if(!scorer.initialize(b.blocks,c.n,c.w,c.t,error))return false;
    const uint64_t sample_seed=seed^0x9128f3b71cULL;
    const size_t batch=max<size_t>(1,min(gpu_batch_size,scorer.maximum_batch_size()));
    const size_t retention_cap=candidate_pool>numeric_limits<size_t>::max()/2?candidate_pool:candidate_pool*2;
    vector<ats45cuda::CandidateResult>results;
    uint64_t verified=0;
    auto started=chrono::steady_clock::now();
    for(uint64_t first=0;first<outsider_samples;first+=batch){
        size_t count=static_cast<size_t>(min<uint64_t>(batch,outsider_samples-first));
        if(!scorer.score_generated(sample_seed,first,count,results,error))return false;
        for(size_t i=0;i<count;++i){
            const auto&r=results[i];
            if(verified<4096){
                uint64_t expected_mask=gpu_generated_block(sample_seed,first+i,c.n,c.w);
                long long expected_score=contribution_direct(r.mask,b,c);
                if(r.mask!=expected_mask||pc(r.mask)!=c.w||expected_score!=r.score){
                    ostringstream o;o<<"CUDA scorer validation failed at sample "<<(first+i)
                        <<" mask=0x"<<hex<<r.mask<<" expected_mask=0x"<<expected_mask<<dec
                        <<" weight="<<pc(r.mask)<<" score="<<r.score<<" expected_score="<<expected_score;
                    error=o.str();return false;
                }
                verified++;
            }
            uint64_t m=r.mask;if(b.set.count(m))continue;
            long long sc=r.score;keep_top(h,sc,m,retention_cap);best=min(best,sc);tested++;
        }
    }
    double elapsed=chrono::duration<double>(chrono::steady_clock::now()-started).count();
    ostringstream stats;stats<<fixed<<setprecision(3)
        <<"extra_pool_backend=CUDA device=\""<<scorer.device_name()<<"\""
        <<" elapsed_seconds="<<elapsed
        <<" generated_per_second="<<(outsider_samples/max(elapsed,1e-9))
        <<" pair_scores_per_second="<<(outsider_samples*static_cast<double>(b.blocks.size())/max(elapsed,1e-9))
        <<" cpu_validated="<<verified;
    log<<stats.str()<<"\n";
    return true;
}
static vector<Extra> extra_pool(const Base&b,const Config&c,uint64_t seed,ostream&log){
    priority_queue<pair<long long,uint64_t>>h;uint64_t universe=c.choose[c.n][c.w],tested=0,outsider_fingerprint=1469598103934665603ULL;long long best=LLONG_MAX;
    if(legacy_regression_pool){
        auto started=chrono::steady_clock::now();mt19937_64 r(seed^0x9128f3b71cULL);unordered_set<uint64_t>seen;seen.reserve(outsider_samples*2);
        for(uint64_t i=0;i<outsider_samples;++i){uint64_t m=(i%4==0)?greedy_block(r,b.cnt,c):random_block(r,c.n,c.w);if(b.set.count(m)||!seen.insert(m).second)continue;long long sc=contribution_direct(m,b,c);outsider_fingerprint^=m+0x9e3779b97f4a7c15ULL+(outsider_fingerprint<<6)+(outsider_fingerprint>>2);outsider_fingerprint*=1099511628211ULL;outsider_fingerprint^=static_cast<uint64_t>(sc);keep_top(h,sc,m,candidate_pool);best=min(best,sc);tested++;}
        log<<"extra_pool_profile=ATS_v4_3_deep_pool_legacy backend=CPU elapsed_seconds="<<chrono::duration<double>(chrono::steady_clock::now()-started).count()<<"\n";
        log<<"extra_pool_mode=SAMPLED tested="<<tested<<" requested="<<outsider_samples<<" universe="<<universe<<" best="<<best<<"\n";
        log<<"outsider_pool_fingerprint=0x"<<hex<<outsider_fingerprint<<dec<<"\n";
    }else if(universe<=outsider_samples){
        auto started=chrono::steady_clock::now();
        vector<int>a(c.w);iota(a.begin(),a.end(),0);while(true){uint64_t m=0;for(int x:a)m|=1ULL<<x;if(!b.set.count(m)){long long sc=contribution_direct(m,b,c);keep_top(h,sc,m,candidate_pool);best=min(best,sc);tested++;}int i=c.w-1;while(i>=0&&a[i]==c.n-c.w+i)--i;if(i<0)break;++a[i];for(int j=i+1;j<c.w;++j)a[j]=a[j-1]+1;}
        log<<"extra_pool_backend=CPU exhaustive_elapsed_seconds="<<chrono::duration<double>(chrono::steady_clock::now()-started).count()<<"\n";
        log<<"extra_pool_mode=EXHAUSTIVE tested="<<tested<<" universe="<<universe<<" best="<<best<<"\n";
    }else{
        bool used_gpu=false;
        if(gpu_mode!=GpuMode::Off){
            string error;used_gpu=fill_extra_pool_gpu(b,c,seed,h,tested,best,log,error);
            if(!used_gpu){
                if(gpu_mode==GpuMode::On)throw runtime_error("CUDA candidate scorer required but unavailable: "+error);
                log<<"extra_pool_cuda_fallback reason=\""<<error<<"\"\n";
                h={};tested=0;best=LLONG_MAX;
            }
        }
        if(!used_gpu){
            auto started=chrono::steady_clock::now();
            uint64_t samples=outsider_samples,sample_seed=seed^0x9128f3b71cULL;size_t retention_cap=candidate_pool>numeric_limits<size_t>::max()/2?candidate_pool:candidate_pool*2;
            for(uint64_t i=0;i<samples;++i){uint64_t m=gpu_generated_block(sample_seed,i,c.n,c.w);if(b.set.count(m))continue;long long sc=contribution_direct(m,b,c);keep_top(h,sc,m,retention_cap);best=min(best,sc);tested++;}
            double elapsed=chrono::duration<double>(chrono::steady_clock::now()-started).count();
            log<<"extra_pool_backend=CPU elapsed_seconds="<<elapsed
               <<" generated_per_second="<<(samples/max(elapsed,1e-9))
               <<" pair_scores_per_second="<<(samples*static_cast<double>(b.blocks.size())/max(elapsed,1e-9))<<"\n";
        }
        log<<"extra_pool_mode=SAMPLED tested="<<tested<<" requested="<<outsider_samples<<" universe="<<universe<<" best="<<best<<"\n";
    }
    vector<Extra>o;while(!h.empty()){o.push_back({h.top().first,h.top().second});h.pop();}sort(o.begin(),o.end(),[](auto&a,auto&b){return a.score!=b.score?a.score<b.score:a.mask<b.mask;});vector<Extra>u;unordered_set<uint64_t>seen;for(auto e:o)if(seen.insert(e.mask).second)u.push_back(e);if(u.size()>candidate_pool)u.resize(candidate_pool);if(u.empty())throw runtime_error("empty extra pool");uint64_t fingerprint=1469598103934665603ULL;for(const auto&e:u){fingerprint^=e.mask+0x9e3779b97f4a7c15ULL+(fingerprint<<6)+(fingerprint>>2);fingerprint*=1099511628211ULL;fingerprint^=static_cast<uint64_t>(e.score);}log<<"extra_pool_size="<<u.size()<<" min="<<u.front().score<<" max_retained="<<u.back().score<<" fingerprint=0x"<<hex<<fingerprint<<" candidate_pool_fingerprint=0x"<<fingerprint<<dec<<"\n";return u;
}
static uint64_t live_bad(State&s,const Config&c,mt19937_64&r){
    if(!s.score)return 0;
    if(s.bad.size()>100000){vector<uint64_t>good;good.reserve(4096);unordered_set<uint64_t>seen;seen.reserve(8192);for(uint64_t m:s.bad)if(s.cnt[rank_shadow(m,c)]>=2&&seen.insert(m).second)good.push_back(m);s.bad.swap(good);}
    for(int k=0;k<2000&&!s.bad.empty();++k){uint64_t m=s.bad[uniform_int_distribution<size_t>(0,s.bad.size()-1)(r)];if(s.cnt[rank_shadow(m,c)]>=2)return m;}
    vector<uint64_t>good;good.reserve(s.bad.size());unordered_set<uint64_t>seen;seen.reserve(s.bad.size()*2+1);for(uint64_t m:s.bad)if(s.cnt[rank_shadow(m,c)]>=2&&seen.insert(m).second)good.push_back(m);s.bad.swap(good);if(s.bad.empty())throw runtime_error("positive score without bad shadow");return s.bad[uniform_int_distribution<size_t>(0,s.bad.size()-1)(r)];
}
struct Move{int bi=-1,oldp=-1,newp=-1;long long ds=LLONG_MAX,dd=LLONG_MAX;};
static pair<long long,long long> eval_move_from_bases(
    const State&s,int bi,int oldp,int newp,const vector<uint64_t>&bases,
    long long removal_delta,const Config&c){
    uint64_t m=s.blocks[bi];if(!(m>>oldp&1ULL)||(m>>newp&1ULL))return{LLONG_MAX/4,LLONG_MAX/4};long long ds=removal_delta;
    for(uint64_t b:bases)ds+=s.cnt[rank_shadow(b|(1ULL<<newp),c)];
    int total=s.total;long long a=1LL*s.deg[oldp]*c.n-total,b=1LL*(s.deg[oldp]-1)*c.n-total,x=1LL*s.deg[newp]*c.n-total,y=1LL*(s.deg[newp]+1)*c.n-total;return{ds,b*b-a*a+y*y-x*x};
}
static void apply_move(State&s,const Move&m,const Config&c){
    uint64_t block=s.blocks[m.bi];
#if ATS48_CPU_SHADOW_INCIDENCE
    if(!update_state_incidence(s,m.bi,block,c,false))throw runtime_error("CPU shadow incidence move removal failed");
#endif
    auto bases=change_bases(block,m.oldp,c);for(uint64_t b:bases){auto r=rank_shadow(b|(1ULL<<m.oldp),c);uint16_t z=s.cnt[r];s.score-=z-1;s.cnt[r]=z-1;}
    for(uint64_t b:bases){uint64_t sh=b|(1ULL<<m.newp);auto r=rank_shadow(sh,c);uint16_t z=s.cnt[r];s.score+=z;s.cnt[r]=z+1;if(z==1)s.bad.push_back(sh);}if(--s.occupied_count[block]==0)s.occupied_count.erase(block);s.blocks[m.bi]&=~(1ULL<<m.oldp);s.blocks[m.bi]|=1ULL<<m.newp;++s.occupied_count[s.blocks[m.bi]];
#if ATS48_CPU_SHADOW_INCIDENCE
    if(!update_state_incidence(s,m.bi,s.blocks[m.bi],c,true))throw runtime_error("CPU shadow incidence move insertion failed");
#endif
    s.deg[m.oldp]--;s.deg[m.newp]++;s.dcost+=m.dd;
}
static long long pair_contribution(uint64_t cand,const State&s,int skip,const Config&c){
    long long v=0;const uint64_t removed=skip>=0?s.blocks[skip]:0;
    for(uint64_t shadow:shadows(cand,c))v+=s.cnt[rank_shadow(shadow,c)];
    v-=c.choose[pc(cand&removed)][c.t];
    return v;
}
static bool exists_except(const State&s,uint64_t m,int ex){auto it=s.occupied_count.find(m);return it!=s.occupied_count.end()&&it->second>(s.blocks[ex]==m?1:0);}
static bool replace_kick(State&s,int bi,const Config&c,mt19937_64&r,int samples,double temp,uint64_t*evaluated=nullptr){
    uint64_t old=s.blocks[bi],best=old;long long oldc=pair_contribution(old,s,bi,c),bestc=oldc;
    for(int z=0;z<samples;++z){uint64_t cand;if(z%4==0)cand=greedy_block(r,s.cnt,c);else if(z%4==1)cand=random_block(r,c.n,c.w);else{cand=old;int changes=(z%4==2)?1:2;for(int q=0;q<changes;++q){auto in=coords(cand);int op=in[uniform_int_distribution<int>(0,(int)in.size()-1)(r)],np;do np=uniform_int_distribution<int>(0,c.n-1)(r);while(cand>>np&1ULL);cand&=~(1ULL<<op);cand|=1ULL<<np;}}
        if(cand==old||exists_except(s,cand,bi))continue;if(evaluated)++*evaluated;long long v=pair_contribution(cand,s,bi,c);if(v<bestc||(v==bestc&&(r()&1))){best=cand;bestc=v;}}
    long long d=bestc-oldc;if(best==old)return false;bool accept=d<=0||uniform_real_distribution<double>(0,1)(r)<exp(-double(d)/max(0.05,temp));if(!accept)return false;remove_at(s,bi,c);install_at(s,bi,best,c);return true;
}
static bool two_kick(State&s,int bi,uint64_t bad,const Config&c,mt19937_64&r,int samples,double temp,uint64_t*evaluated=nullptr){
    uint64_t old=s.blocks[bi],best=old;auto oc=coords(old),bc=coords(bad);long long oldv=pair_contribution(old,s,bi,c),bestv=oldv;
    for(int z=0;z<samples;++z){int o1=bc[uniform_int_distribution<int>(0,(int)bc.size()-1)(r)],o2=oc[uniform_int_distribution<int>(0,(int)oc.size()-1)(r)];if(o1==o2)continue;uint64_t m=old&~(1ULL<<o1)&~(1ULL<<o2);int n1,n2;do n1=uniform_int_distribution<int>(0,c.n-1)(r);while(m>>n1&1ULL);m|=1ULL<<n1;do n2=uniform_int_distribution<int>(0,c.n-1)(r);while(m>>n2&1ULL);m|=1ULL<<n2;if(pc(m)!=c.w||exists_except(s,m,bi))continue;if(evaluated)++*evaluated;long long v=pair_contribution(m,s,bi,c);if(v<bestv||(v==bestv&&(r()&1))){best=m;bestv=v;}}
    long long d=bestv-oldv;if(best==old)return false;bool accept=d<=0||uniform_real_distribution<double>(0,1)(r)<exp(-double(d)/max(0.05,temp));if(!accept)return false;remove_at(s,bi,c);install_at(s,bi,best,c);return true;
}

static uint32_t telemetry_mix32(uint32_t value){value^=value>>16;value*=0x7feb352dU;value^=value>>15;value*=0x846ca68bU;value^=value>>16;return value;}
static uint64_t telemetry_fingerprint(const vector<uint64_t>&blocks){uint32_t h0=2166136261U,h1=0x9e3779b9U;for(size_t i=0;i<blocks.size();++i){uint64_t mask=blocks[i];uint32_t a=telemetry_mix32(static_cast<uint32_t>(mask)^(static_cast<uint32_t>(i)*0x85ebca6bU));uint32_t b=telemetry_mix32(static_cast<uint32_t>(mask>>32)^(static_cast<uint32_t>(i)*0xc2b2ae35U));h0=(h0^a)*16777619U;h1+=b+0x9e3779b9U+(h1<<6)+(h1>>2);}return h0|(static_cast<uint64_t>(h1)<<32);}

static uint64_t canonical_fingerprint(vector<uint64_t>blocks){sort(blocks.begin(),blocks.end());uint64_t h=1469598103934665603ULL;for(uint64_t x:blocks){h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);h*=1099511628211ULL;}return h;}
static uint64_t conflict_signature(const vector<uint64_t>&blocks,const Config&c){uint32_t h0=0x811c9dc5U,h1=0x9e3779b9U;for(size_t i=0;i<blocks.size();++i)for(size_t j=0;j<i;++j){int overlap=pc(blocks[i]&blocks[j]);if(overlap<c.t)continue;uint32_t edge0=telemetry_mix32(static_cast<uint32_t>(blocks[i]))^telemetry_mix32(static_cast<uint32_t>(blocks[j]));uint32_t edge1=telemetry_mix32(static_cast<uint32_t>(blocks[i]>>32))^telemetry_mix32(static_cast<uint32_t>(blocks[j]>>32));h0^=telemetry_mix32(edge0+0x85ebca6bU);h1+=telemetry_mix32(edge1+0xc2b2ae35U);}return h0|(static_cast<uint64_t>(h1)<<32);}

struct ForensicMove {
    int block = -1;
    int remove0 = -1;
    int remove1 = -1;
    int add0 = -1;
    int add1 = -1;
    uint64_t score = numeric_limits<uint64_t>::max();
};

struct ForensicMoveStats {
    uint64_t one_attempted = 0;
    uint64_t one_legal = 0;
    uint64_t one_duplicate = 0;
    uint64_t two_attempted = 0;
    uint64_t two_legal = 0;
    uint64_t two_duplicate = 0;
    uint64_t best_score = numeric_limits<uint64_t>::max();
    uint64_t best_count = 0;
    ForensicMove best_move;
    vector<ForensicMove> zero_moves;
    map<uint64_t,uint64_t> score_histogram;
};

static uint64_t exact_repeated_shadow_score(const vector<uint64_t>&blocks,const Config&c){
    uint64_t score=0;
    for(size_t i=0;i<blocks.size();++i)for(size_t j=0;j<i;++j){
        const int overlap=pc(blocks[i]&blocks[j]);
        if(overlap>=c.t)score+=c.choose[overlap][c.t];
    }
    return score;
}

static string forensic_move_text(const ForensicMove&move){
    ostringstream out;
    out<<"block="<<move.block<<" remove={"<<move.remove0+1;
    if(move.remove1>=0)out<<","<<move.remove1+1;
    out<<"} add={"<<move.add0+1;
    if(move.add1>=0)out<<","<<move.add1+1;
    out<<"} exact_score="<<move.score;
    return out.str();
}

static void audit_first_defect1_state(
    const ats45cuda::CapturedTerminalState&terminal,const Config&c,const string&out_prefix)
{
    if(terminal.blocks.size()!=static_cast<size_t>(c.target_m))
        throw runtime_error("first defect-1 forensic state has wrong block count");
    const uint64_t initial_score=exact_repeated_shadow_score(terminal.blocks,c);
    if(initial_score!=1)throw runtime_error("first defect-1 forensic state is not exact score 1");
    if(telemetry_fingerprint(terminal.blocks)!=terminal.state_hash)
        throw runtime_error("first defect-1 forensic state hash mismatch");

    const string state_path=out_prefix+".first_defect1_state.txt";
    const string report_path=out_prefix+".first_defect1_forensics.txt";
    write_code(terminal.blocks,c,state_path,"first CUDA defect-1 state; exact CPU score=1");

    unordered_set<uint64_t>occupied(terminal.blocks.begin(),terminal.blocks.end());
    if(occupied.size()!=terminal.blocks.size())
        throw runtime_error("first defect-1 forensic state contains duplicate blocks");

    const unsigned worker_count=max(1U,min<unsigned>(
        static_cast<unsigned>(terminal.blocks.size()),thread::hardware_concurrency()));
    atomic<size_t>next_block{0};
    atomic<size_t>completed_blocks{0};
    vector<ForensicMoveStats>locals(worker_count);
    vector<thread>workers;
    workers.reserve(worker_count);
    auto score_candidate=[&](size_t changed,uint64_t candidate,uint64_t old_contribution){
        uint64_t new_contribution=0;
        for(size_t other=0;other<terminal.blocks.size();++other){
            if(other==changed)continue;
            const int overlap=pc(candidate&terminal.blocks[other]);
            if(overlap>=c.t)new_contribution+=c.choose[overlap][c.t];
        }
        return initial_score-old_contribution+new_contribution;
    };
    auto record=[&](ForensicMoveStats&stats,const ForensicMove&move){
        ++stats.score_histogram[move.score];
        if(move.score<stats.best_score){
            stats.best_score=move.score;
            stats.best_count=1;
            stats.best_move=move;
        }else if(move.score==stats.best_score)++stats.best_count;
        if(move.score==0)stats.zero_moves.push_back(move);
    };
    for(unsigned worker=0;worker<worker_count;++worker)workers.emplace_back([&,worker](){
        ForensicMoveStats&stats=locals[worker];
        for(;;){
            const size_t changed=next_block.fetch_add(1,memory_order_relaxed);
            if(changed>=terminal.blocks.size())break;
            const uint64_t old_block=terminal.blocks[changed];
            vector<int>inside,outside;
            inside.reserve(c.w);outside.reserve(c.n-c.w);
            for(int bit=0;bit<c.n;++bit)((old_block>>bit)&1ULL?inside:outside).push_back(bit);
            uint64_t old_contribution=0;
            for(size_t other=0;other<terminal.blocks.size();++other){
                if(other==changed)continue;
                const int overlap=pc(old_block&terminal.blocks[other]);
                if(overlap>=c.t)old_contribution+=c.choose[overlap][c.t];
            }
            for(int removed:inside)for(int added:outside){
                ++stats.one_attempted;
                const uint64_t candidate=(old_block&~(1ULL<<removed))|(1ULL<<added);
                if(occupied.find(candidate)!=occupied.end()){
                    ++stats.one_duplicate;
                    continue;
                }
                ++stats.one_legal;
                ForensicMove move;
                move.block=static_cast<int>(changed);move.remove0=removed;move.add0=added;
                move.score=score_candidate(changed,candidate,old_contribution);
                record(stats,move);
            }
            for(size_t r0=0;r0<inside.size();++r0)for(size_t r1=r0+1;r1<inside.size();++r1)
                for(size_t a0=0;a0<outside.size();++a0)for(size_t a1=a0+1;a1<outside.size();++a1){
                    ++stats.two_attempted;
                    const uint64_t candidate=(old_block&~(1ULL<<inside[r0])&~(1ULL<<inside[r1]))
                        |(1ULL<<outside[a0])|(1ULL<<outside[a1]);
                    if(occupied.find(candidate)!=occupied.end()){
                        ++stats.two_duplicate;
                        continue;
                    }
                    ++stats.two_legal;
                    ForensicMove move;
                    move.block=static_cast<int>(changed);move.remove0=inside[r0];move.remove1=inside[r1];
                    move.add0=outside[a0];move.add1=outside[a1];
                    move.score=score_candidate(changed,candidate,old_contribution);
                    record(stats,move);
                }
            const size_t done=completed_blocks.fetch_add(1,memory_order_relaxed)+1;
            if(done%100==0||done==terminal.blocks.size())
                cerr<<"DEFECT1_FORENSICS_PROGRESS blocks="<<done<<"/"<<terminal.blocks.size()<<"\n";
        }
    });
    for(auto&worker:workers)worker.join();

    ForensicMoveStats total;
    for(auto&local:locals){
        total.one_attempted+=local.one_attempted;
        total.one_legal+=local.one_legal;
        total.one_duplicate+=local.one_duplicate;
        total.two_attempted+=local.two_attempted;
        total.two_legal+=local.two_legal;
        total.two_duplicate+=local.two_duplicate;
        for(const auto&entry:local.score_histogram)total.score_histogram[entry.first]+=entry.second;
        total.zero_moves.insert(total.zero_moves.end(),local.zero_moves.begin(),local.zero_moves.end());
        if(local.best_score<total.best_score){
            total.best_score=local.best_score;total.best_count=local.best_count;total.best_move=local.best_move;
        }else if(local.best_score==total.best_score)total.best_count+=local.best_count;
    }
    sort(total.zero_moves.begin(),total.zero_moves.end(),[](const ForensicMove&a,const ForensicMove&b){
        if(a.block!=b.block)return a.block<b.block;
        if(a.remove0!=b.remove0)return a.remove0<b.remove0;
        if(a.remove1!=b.remove1)return a.remove1<b.remove1;
        if(a.add0!=b.add0)return a.add0<b.add0;
        return a.add1<b.add1;
    });

    ofstream report(report_path);
    if(!report)throw runtime_error("cannot write "+report_path);
    report<<hex<<setfill('0');
    report<<"ordered_state_hash=0x"<<setw(16)<<terminal.state_hash<<"\n";
    report<<"canonical_state_hash=0x"<<setw(16)<<canonical_fingerprint(terminal.blocks)<<"\n";
    report<<dec<<setfill(' ');
    report<<"cuda_walker="<<terminal.walker_id<<"\n";
    report<<"cuda_iteration="<<terminal.iteration<<"\n";
    report<<"block_count="<<terminal.blocks.size()<<"\n";
    report<<"exact_cpu_defect="<<initial_score<<"\n";
    report<<"state_file="<<state_path<<"\n";
    report<<"cpu_enumerator_threads="<<worker_count<<"\n";
    report<<"one_bit_attempted="<<total.one_attempted<<"\n";
    report<<"one_bit_legal="<<total.one_legal<<"\n";
    report<<"one_bit_duplicate_rejected="<<total.one_duplicate<<"\n";
    report<<"two_bit_attempted="<<total.two_attempted<<"\n";
    report<<"two_bit_legal="<<total.two_legal<<"\n";
    report<<"two_bit_duplicate_rejected="<<total.two_duplicate<<"\n";
    report<<"exact_scored_total="<<total.one_legal+total.two_legal<<"\n";
    report<<"best_exact_score="<<total.best_score<<"\n";
    report<<"best_delta="<<static_cast<int64_t>(total.best_score)-static_cast<int64_t>(initial_score)<<"\n";
    report<<"best_move_count="<<total.best_count<<"\n";
    report<<"best_move="<<forensic_move_text(total.best_move)<<"\n";
    report<<"defect0_move_count="<<total.zero_moves.size()<<"\n";
    report<<"gpu_terminal_guard=score <= 1 (no proposals are generated at this state)\n";
    report<<"acceptance_if_scored=every defect-0 move is strictly improving and would be accepted\n";
    report<<"score_histogram_begin\n";
    for(const auto&entry:total.score_histogram)report<<entry.first<<"\t"<<entry.second<<"\n";
    report<<"score_histogram_end\n";
    for(size_t i=0;i<total.zero_moves.size();++i){
        report<<"defect0_move["<<i<<"]="<<forensic_move_text(total.zero_moves[i])
            <<" rejection_reason=not_evaluated_by_CUDA_terminal_guard_score_le_1"
            <<" acceptance_if_evaluated=accepted_strict_improvement\n";
    }
    report.close();
    cerr<<"DEFECT1_FORENSICS_COMPLETE ordered_state_hash=0x"<<hex<<terminal.state_hash
        <<" canonical_state_hash=0x"<<canonical_fingerprint(terminal.blocks)<<dec
        <<" best_delta="<<static_cast<int64_t>(total.best_score)-static_cast<int64_t>(initial_score)
        <<" defect0_moves="<<total.zero_moves.size()<<" report=\""<<report_path<<"\"\n";
}

enum class MoveClass:uint32_t{Initial=0,OneBitSwap=1,TwoBitEjection=2,Replacement=3};
static const char*move_class_name(MoveClass value){switch(value){case MoveClass::OneBitSwap:return"one_bit_swap";case MoveClass::TwoBitEjection:return"two_bit_ejection";case MoveClass::Replacement:return"replacement";default:return"initial";}}
enum:uint32_t{SOURCE_CPU=1,SOURCE_GPU=2};
struct LowDefectProvenance{uint64_t state_hash=0,conflict_signature=0,parent_hash=0,iteration=0;uint32_t defect=0,source_bits=0,actor_id=UINT32_MAX;MoveClass move_class=MoveClass::Initial;};
struct TerminalEvent{vector<uint64_t>blocks;uint64_t neighborhood_hash=0;LowDefectProvenance provenance;};
struct GpuHandoff{vector<uint64_t>blocks;uint64_t state_hash=0,iteration=0;uint32_t defect=0,walker_id=0,source_bits=SOURCE_GPU;bool concentrated=false;};

struct Shared{
    atomic<long long>best{LLONG_MAX/4};atomic<bool>found{false},gpu_done{false},stop_requested{false};mutex m;vector<uint64_t>blocks;string out;chrono::steady_clock::time_point started,deadline;
    atomic<uint64_t>cpu_iterations{0},cpu_move_evaluations{0},cpu_accepted_moves{0},cpu_restarts{0},cpu_search_starts{0},gpu_walker_iterations{0},gpu_proposals_scored{0},gpu_accepted_moves{0},verified_children{0};
    atomic<uint32_t>first_hit_mask{0};
    atomic<uint64_t>handoff_pending{0},gpu_handoffs_enqueued{0},gpu_handoffs_deduplicated{0},gpu_handoffs_dropped{0},cpu_handoffs_started{0};
    atomic<uint32_t>gpu_walkers_used{0};
    mutex handoff_m;condition_variable handoff_cv;deque<GpuHandoff>defect1_handoffs,defect2_handoffs;unordered_set<uint64_t>handoff_seen;
    uint64_t parent_hash=0;
    uint32_t first_zero_source_bits=0;
    double first_hit_seconds[5]={-1.0,-1.0,-1.0,-1.0,-1.0};
    unordered_set<uint64_t>defect2_hashes,defect1_hashes,basin_hashes;
    vector<LowDefectProvenance>low_defect_records;
    vector<TerminalEvent>terminal_events;
    unordered_map<uint64_t,size_t>low_defect_index,terminal_index;
    bool scout_stop_on_first_defect1=false,promoted_captured=false;
    vector<uint64_t>promoted_blocks;
    LowDefectProvenance promoted_provenance;
    double promoted_elapsed_seconds=-1.0;
    bool concurrent_promotion_repair_ran=false;
    uint64_t concurrent_promotion_repair_candidates=0;
    double concurrent_promotion_repair_seconds=0.0;
    unordered_set<uint64_t>completed_repair_neighborhoods;
};
struct GpuDoneGuard{Shared&g;~GpuDoneGuard(){g.gpu_done.store(true,memory_order_release);g.handoff_cv.notify_all();}};
static void enqueue_gpu_handoff(Shared&g,const ats45cuda::CapturedTerminalState&captured,const Config&c,uint32_t source_bits=SOURCE_GPU){
    constexpr size_t capacity=512;
    GpuHandoff handoff;handoff.blocks=captured.blocks;handoff.state_hash=captured.state_hash;handoff.iteration=captured.iteration;handoff.defect=captured.defect;handoff.walker_id=captured.walker_id;handoff.source_bits=source_bits;
    if(handoff.defect==2){vector<int>owners(handoff.blocks.size(),0);for(size_t i=0;i<handoff.blocks.size();++i)for(size_t j=0;j<i;++j)if(pc(handoff.blocks[i]&handoff.blocks[j])>=c.t){++owners[i];++owners[j];}handoff.concentrated=*max_element(owners.begin(),owners.end())>=2;}
    const uint64_t key=canonical_fingerprint(handoff.blocks)^(0x9e3779b97f4a7c15ULL*handoff.defect);
    lock_guard<mutex>lock(g.handoff_m);
    if(!g.handoff_seen.insert(key).second){g.gpu_handoffs_deduplicated.fetch_add(1,memory_order_relaxed);return;}
    if(g.defect1_handoffs.size()+g.defect2_handoffs.size()>=capacity){
        if(handoff.defect==1&&!g.defect2_handoffs.empty()){g.defect2_handoffs.pop_back();g.handoff_pending.fetch_sub(1,memory_order_relaxed);g.gpu_handoffs_dropped.fetch_add(1,memory_order_relaxed);}
        else{g.gpu_handoffs_dropped.fetch_add(1,memory_order_relaxed);return;}
    }
    if(handoff.defect==1)g.defect1_handoffs.push_back(move(handoff));else if(handoff.concentrated)g.defect2_handoffs.push_front(move(handoff));else g.defect2_handoffs.push_back(move(handoff));
    g.gpu_handoffs_enqueued.fetch_add(1,memory_order_relaxed);g.handoff_pending.fetch_add(1,memory_order_release);
    g.handoff_cv.notify_one();
}
static bool try_pop_gpu_handoff(Shared&g,GpuHandoff&handoff){
    if(g.handoff_pending.load(memory_order_acquire)==0)return false;
    lock_guard<mutex>lock(g.handoff_m);
    if(!g.defect1_handoffs.empty()){handoff=move(g.defect1_handoffs.front());g.defect1_handoffs.pop_front();}
    else if(!g.defect2_handoffs.empty()){handoff=move(g.defect2_handoffs.front());g.defect2_handoffs.pop_front();}
    else return false;
    g.handoff_pending.fetch_sub(1,memory_order_release);return true;
}
static bool wait_pop_gpu_handoff(Shared&g,GpuHandoff&handoff){
    unique_lock<mutex>lock(g.handoff_m);
    g.handoff_cv.wait_until(lock,g.deadline,[&](){return g.found.load(memory_order_relaxed)||g.stop_requested.load(memory_order_relaxed)||g.gpu_done.load(memory_order_relaxed)||!g.defect1_handoffs.empty()||!g.defect2_handoffs.empty();});
    if(g.found.load(memory_order_relaxed)||g.stop_requested.load(memory_order_relaxed)||((g.gpu_done.load(memory_order_relaxed)||chrono::steady_clock::now()>=g.deadline)&&g.defect1_handoffs.empty()&&g.defect2_handoffs.empty()))return false;
    if(!g.defect1_handoffs.empty()){handoff=move(g.defect1_handoffs.front());g.defect1_handoffs.pop_front();}
    else{handoff=move(g.defect2_handoffs.front());g.defect2_handoffs.pop_front();}
    g.handoff_pending.fetch_sub(1,memory_order_release);return true;
}
static void merge_low_defect_record(Shared&g,LowDefectProvenance record,const vector<uint64_t>*blocks=nullptr){uint64_t key=record.state_hash^(0x9e3779b97f4a7c15ULL*record.defect);auto found=g.low_defect_index.find(key);if(found==g.low_defect_index.end()){g.low_defect_index.emplace(key,g.low_defect_records.size());g.low_defect_records.push_back(record);}else g.low_defect_records[found->second].source_bits|=record.source_bits;if(record.defect!=1||!blocks)return;uint64_t neighborhood=canonical_fingerprint(*blocks);auto terminal_found=g.terminal_index.find(neighborhood);if(terminal_found==g.terminal_index.end()){TerminalEvent event;event.blocks=*blocks;event.neighborhood_hash=neighborhood;event.provenance=record;g.terminal_index.emplace(neighborhood,g.terminal_events.size());g.terminal_events.push_back(move(event));}else g.terminal_events[terminal_found->second].provenance.source_bits|=record.source_bits;}
static void record_state(const State&s,const Config&c,Shared&g,uint32_t source_bits,uint32_t actor_id,uint64_t iteration,MoveClass move_class){
    if(s.score>4)return;
    uint32_t threshold_mask=0;
    for(int threshold=max<long long>(0,s.score);threshold<=4;++threshold)
        threshold_mask|=1U<<threshold;
    const uint32_t previous_mask=g.first_hit_mask.fetch_or(
        threshold_mask,memory_order_relaxed);
    if(s.score>2&&(previous_mask&threshold_mask)==threshold_mask)return;
    double elapsed=chrono::duration<double>(chrono::steady_clock::now()-g.started).count();
    uint64_t hash=(s.score==1||s.score==2)?telemetry_fingerprint(s.blocks):0;
    LowDefectProvenance record;
    if(s.score==1||s.score==2){record.state_hash=hash;record.conflict_signature=conflict_signature(s.blocks,c);record.parent_hash=g.parent_hash;record.iteration=iteration;record.defect=static_cast<uint32_t>(s.score);record.source_bits=source_bits;record.actor_id=actor_id;record.move_class=move_class;}
    bool promoted=false;
    {
        lock_guard<mutex>lock(g.m);
        for(int threshold=max<long long>(0,s.score);threshold<=4;++threshold)if(g.first_hit_seconds[threshold]<0.0)g.first_hit_seconds[threshold]=elapsed;
        if(s.score==2)g.defect2_hashes.insert(hash);
        if(s.score==1)g.defect1_hashes.insert(hash);
        if(s.score==1||s.score==2)merge_low_defect_record(g,record,s.score==1?&s.blocks:nullptr);
        if(s.score==1&&g.scout_stop_on_first_defect1&&!g.promoted_captured){g.promoted_captured=true;g.promoted_blocks=s.blocks;g.promoted_provenance=record;g.promoted_elapsed_seconds=elapsed;promoted=true;}
    }
    if(promoted){
        write_code(s.blocks,c,g.out+".promoted_defect1.txt","ATS v4.8 exact CPU defect-1 scout promotion");
        cerr<<"V48_SCOUT_PROMOTED source=CPU actor="<<actor_id<<" iteration="<<iteration<<" elapsed_seconds="<<elapsed<<" state_hash=0x"<<hex<<hash<<dec<<"\n";
        g.stop_requested.store(true,memory_order_release);g.handoff_cv.notify_all();
    }
}
static void record_basin(const State&s,Shared&g){lock_guard<mutex>lock(g.m);g.basin_hashes.insert(telemetry_fingerprint(s.blocks));}
static void publish(const State&s,const Config&c,Shared&g,int tid,int restart,long long it,MoveClass move_class){record_state(s,c,g,SOURCE_CPU,static_cast<uint32_t>(tid),static_cast<uint64_t>(it),move_class);long long cur=s.score;if(cur>=g.best.load())return;lock_guard<mutex>lock(g.m);if(cur<g.best.load()){g.best.store(cur);g.blocks=s.blocks;write_code(s.blocks,c,g.out+".best","ATS v4.8 bounded near-code; repeated-shadow score="+to_string(cur));cerr<<"BEST score="<<cur<<" tid="<<tid<<" restart="<<restart<<" iter="<<it<<" elapsed_seconds="<<chrono::duration<double>(chrono::steady_clock::now()-g.started).count()<<"\n";}}
static const char*source_name(uint32_t bits){return bits==(SOURCE_CPU|SOURCE_GPU)?"CPU+GPU":bits==SOURCE_GPU?"GPU":"CPU";}
static void write_low_defect_provenance(const Shared&g){ofstream f(g.out+".low_defect.tsv");if(!f)throw runtime_error("cannot write low-defect provenance");f<<"state_hash\tdefect\tsource\tactor_id\titeration\tparent_hash\tconflict_signature\tmove_class\n";for(const auto&record:g.low_defect_records){f<<"0x"<<hex<<record.state_hash<<dec<<'\t'<<record.defect<<'\t'<<source_name(record.source_bits)<<'\t'<<record.actor_id<<'\t'<<record.iteration<<"\t0x"<<hex<<record.parent_hash<<"\t0x"<<record.conflict_signature<<dec<<'\t'<<move_class_name(record.move_class)<<'\n';}}
static void worker(int tid,uint64_t seed,const Base&base,const vector<Extra>&extra,const Config&c,Shared&g,bool handoff_only){
    uint64_t counted_iterations=0,counted_move_evaluations=0,counted_accepted_moves=0;auto flush_stats=[&](){g.cpu_iterations.fetch_add(counted_iterations,memory_order_relaxed);g.cpu_move_evaluations.fetch_add(counted_move_evaluations,memory_order_relaxed);g.cpu_accepted_moves.fetch_add(counted_accepted_moves,memory_order_relaxed);counted_iterations=0;counted_move_evaluations=0;counted_accepted_moves=0;};
    mt19937_64 r(seed);int restart=0;while(!g.found.load()&&!g.stop_requested.load()&&chrono::steady_clock::now()<g.deadline){
        State s;bool handoff_origin=false;GpuHandoff handoff;
        if(handoff_only){if(!wait_pop_gpu_handoff(g,handoff))break;s=make_state(handoff.blocks,c);if(s.score!=handoff.defect)throw runtime_error("GPU handoff reconstruction score mismatch");handoff_origin=true;g.cpu_handoffs_started.fetch_add(1,memory_order_relaxed);}
        else{const Extra&e=extra[(tid*31+restart*17+(r()%extra.size()))%extra.size()];s=make_state(base,e.mask,c);}
        vector<vector<int>>tabu(c.target_m,vector<int>(c.n));long long local=s.score,stagn=0,plateau_moves=0,ejection_kicks=0;MoveClass arrival_move=MoveClass::Initial;
        g.cpu_search_starts.fetch_add(1,memory_order_relaxed);if(!handoff_only&&restart>0)g.cpu_restarts.fetch_add(1,memory_order_relaxed);record_basin(s,g);
        if(handoff_only)cerr<<"GPU_CPU_HANDOFF defect="<<s.score<<" tid="<<tid<<" handoff="<<restart<<" gpu_walker="<<handoff.walker_id<<" gpu_iteration="<<handoff.iteration<<" state_hash=0x"<<hex<<handoff.state_hash<<dec<<" source="<<source_name(handoff.source_bits)<<" concentrated="<<handoff.concentrated<<"\n";
        else cerr<<"INITIAL defect="<<s.score<<" tid="<<tid<<" restart="<<restart<<"\n";
        const long long iteration_limit=handoff_only?20000:5000000;
        for(long long it=0;it<iteration_limit&&!g.found.load()&&!g.stop_requested.load()&&chrono::steady_clock::now()<g.deadline;++it){counted_iterations++;
            const bool local_improvement=s.score<local;
            if(local_improvement){local=s.score;stagn=0;}else stagn++;
            if(it==0||local_improvement||s.score<=2||s.score<g.best.load(memory_order_relaxed))
                publish(s,c,g,tid,restart,it,arrival_move);
            if(!s.score){string why;if(!verify(s.blocks,c,&why))throw runtime_error("zero-score verify failed: "+why);lock_guard<mutex>lock(g.m);if(!g.found.load()){g.blocks=s.blocks;g.best.store(0);g.first_zero_source_bits=handoff_origin?(SOURCE_CPU|handoff.source_bits):SOURCE_CPU;write_code(s.blocks,c,g.out,"ATS v4.8 cooperative CUDA exploration and CPU exploitation");g.found.store(true);g.stop_requested.store(true);g.handoff_cv.notify_all();g.verified_children.fetch_add(1,memory_order_relaxed);cerr<<"VERIFIED CHILD size="<<s.blocks.size()<<" tid="<<tid<<" restart="<<restart<<" iter="<<it<<" handoff_origin="<<handoff_origin<<" plateau_moves="<<plateau_moves<<" ejection_kicks="<<ejection_kicks<<" child_hash=0x"<<hex<<canonical_fingerprint(s.blocks)<<dec<<"\n";}flush_stats();return;}
            if(g.stop_requested.load(memory_order_acquire)){flush_stats();return;}
            uint64_t bad=live_bad(s,c,r);vector<int>owners;
#if ATS48_CPU_SHADOW_INCIDENCE
            owners=shadow_owners(s,bad,c);
#else
            for(int bi=0;bi<c.target_m;++bi)if((s.blocks[bi]&bad)==bad)owners.push_back(bi);
#endif
            if(owners.size()<2)throw runtime_error("bad shadow owner mismatch");shuffle(owners.begin(),owners.end(),r);
            Move best;int q=static_cast<int>(c.comb_wm1_tm1.size()),npn=c.n-c.w,olim=min(static_cast<int>(owners.size()),3),ob=max(1,60000/max(1,q*npn*olim));ob=min(ob,c.t);auto bc=coords(bad);
            for(int oi=0;oi<olim;++oi){
                int bi=owners[oi];vector<pair<long long,int>>oldrank;
                for(int op:bc){
                    long long rem=0;
                    for(uint64_t b:change_bases(s.blocks[bi],op,c))
                        rem+=s.cnt[rank_shadow(b|(1ULL<<op),c)]-1;
                    oldrank.push_back({-rem,op});
                }
                sort(oldrank.begin(),oldrank.end());
                if((int)oldrank.size()>ob){
                    vector<pair<long long,int>>z(oldrank.begin(),oldrank.begin()+ob);
                    if(ob>=2)z.back()=oldrank[uniform_int_distribution<int>(ob-1,(int)oldrank.size()-1)(r)];
                    oldrank.swap(z);
                }
                for(auto [nr,op]:oldrank){
                    const auto bases=change_bases(s.blocks[bi],op,c);
                    for(int np=0;np<c.n;++np)if(!(s.blocks[bi]>>np&1ULL)){
                        counted_move_evaluations++;
                        auto [ds,dd]=eval_move_from_bases(s,bi,op,np,bases,nr,c);
                        if(tabu[bi][np]>(int)it&&ds>=0)continue;
                        if(ds<best.ds||(ds==best.ds&&dd<best.dd)||
                            (ds==best.ds&&dd==best.dd&&(r()&1)))
                            best={bi,op,np,ds,dd};
                    }
                }
            }
            double temp=max(0.12,1.8*exp(-double(it%120000)/45000.0));bool kicked=false;if((stagn>180&&s.score<=8)||stagn>900){int bi=owners[uniform_int_distribution<int>(0,(int)owners.size()-1)(r)];if(s.score<=4&&(r()&1)){kicked=two_kick(s,bi,bad,c,r,180,temp,&counted_move_evaluations);if(kicked)arrival_move=MoveClass::TwoBitEjection;}if(!kicked){kicked=replace_kick(s,bi,c,r,160,temp,&counted_move_evaluations);if(kicked)arrival_move=MoveClass::Replacement;}if(kicked){counted_accepted_moves++;ejection_kicks++;stagn=0;continue;}}
            if(best.bi<0){bool replaced=replace_kick(s,owners.front(),c,r,300,temp+.5,&counted_move_evaluations);if(!replaced)break;arrival_move=MoveClass::Replacement;counted_accepted_moves++;continue;}bool accept=best.ds<0;if(best.ds==0)accept=best.dd<=0||uniform_real_distribution<double>(0,1)(r)<.55;if(best.ds>0)accept=uniform_real_distribution<double>(0,1)(r)<exp(-double(best.ds)/temp);if(!accept&&stagn>2000){if(replace_kick(s,owners.front(),c,r,500,temp+1,&counted_move_evaluations)){arrival_move=MoveClass::Replacement;counted_accepted_moves++;stagn=0;continue;}}
            if(accept){arrival_move=MoveClass::OneBitSwap;counted_accepted_moves++;if(best.ds==0)plateau_moves++;int reverse=best.oldp;apply_move(s,best,c);tabu[best.bi][reverse]=(int)it+7+(r()%11);}
            long long period=(c.target_m>1000)?65536:8192;
            if(it%period==0){
                long long exact=0;
                for(int i=0;i<c.target_m;++i)for(int j=0;j<i;++j){
                    int z=pc(s.blocks[i]&s.blocks[j]);if(z>=c.t)exact+=c.choose[z][c.t];
                }
                if(exact!=s.score){ostringstream o;o<<"score mismatch exact="<<exact<<" state="<<s.score;throw runtime_error(o.str());}
#if ATS48_CPU_SHADOW_INCIDENCE
                if(!audit_state_incidence(s,c))throw runtime_error("CPU shadow incidence audit failed");
#endif
            }
            if(stagn>(handoff_only?10000:(reference_policy_audit?50000:500000)))break;
        }restart++;
    }flush_stats();
}

struct ExactRepairMetrics {
    uint64_t terminal_events = 0;
    uint64_t endpoint_cores = 0;
    uint64_t unique_core_scans = 0;
    uint64_t core_cache_hits = 0;
    uint64_t gpu_candidates_examined = 0;
    uint64_t qualifying_candidates = 0;
    uint64_t compatibility_pairs_tested = 0;
    uint64_t compatibility_triples_tested = 0;
    uint32_t solved_radius = 0;
    double elapsed_seconds = 0;
};

struct RepairScanCacheEntry {
    vector<uint64_t> core;
    ats45cuda::CompatibilityScanResult scan;
};

struct RepairScanCache {
    unordered_map<uint64_t,vector<unique_ptr<RepairScanCacheEntry>>> buckets;
};

static pair<int,int> unique_defect1_pair(
    const vector<uint64_t>& blocks,
    const Config& c)
{
    uint64_t score = 0;
    int first = -1, second = -1, conflict_count = 0;
    for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
        for (int j = 0; j < i; ++j) {
            const int overlap = pc(blocks[i] & blocks[j]);
            if (overlap < c.t) continue;
            score += c.choose[overlap][c.t];
            first = j;
            second = i;
            ++conflict_count;
        }
    }
    if (score != 1 || conflict_count != 1) {
        throw runtime_error(
            "defect-1 terminal does not contain exactly one unit conflict");
    }
    return {first, second};
}

static bool repair_compatible(uint64_t first,uint64_t second,const Config&c){
    return pc(first&second)<c.t;
}

static const ats45cuda::CompatibilityScanResult& cached_repair_scan(
    vector<uint64_t> core,
    const Config& c,
    RepairScanCache& cache,
    ExactRepairMetrics& metrics)
{
    sort(core.begin(),core.end());
    const uint64_t hash=canonical_fingerprint(core);
    auto& bucket=cache.buckets[hash];
    for(const auto&entry:bucket)if(entry->core==core){
        ++metrics.core_cache_hits;
        return entry->scan;
    }
    auto entry=make_unique<RepairScanCacheEntry>();
    entry->core=move(core);
    entry->scan=ats45cuda::scan_repair_candidates(
        entry->core,c.n,c.w,c.t,2,cuda_repair_capacity);
    metrics.gpu_candidates_examined+=entry->scan.candidates_examined;
    if(!entry->scan.error.empty())
        throw runtime_error("CUDA exact-repair scan failed: "+entry->scan.error);
    if(!entry->scan.exact&&entry->scan.overflow_candidates!=0){
        if(entry->scan.qualifying_candidates>numeric_limits<uint32_t>::max())
            throw runtime_error("CUDA exact-repair candidate pool exceeds 32-bit capacity");
        const uint32_t exact_capacity=static_cast<uint32_t>(
            max<uint64_t>(1,entry->scan.qualifying_candidates));
        cerr<<"CUDA_REPAIR_RETRY core_hash=0x"<<hex<<hash<<dec
            <<" qualifying="<<entry->scan.qualifying_candidates
            <<" initial_capacity="<<cuda_repair_capacity
            <<" exact_capacity="<<exact_capacity<<"\n";
        entry->scan=ats45cuda::scan_repair_candidates(
            entry->core,c.n,c.w,c.t,2,exact_capacity);
        metrics.gpu_candidates_examined+=entry->scan.candidates_examined;
        if(!entry->scan.error.empty())
            throw runtime_error("CUDA exact-repair retry failed: "+entry->scan.error);
    }
    if(!entry->scan.exact)
        throw runtime_error("CUDA exact-repair scan did not exhaust the outsider universe");
    ++metrics.unique_core_scans;
    metrics.qualifying_candidates+=entry->scan.qualifying_candidates;
    cerr<<"CUDA_REPAIR_SCAN core_hash=0x"<<hex<<hash<<dec
        <<" device=\""<<entry->scan.device_name<<"\""
        <<" universe="<<entry->scan.universe_size
        <<" examined="<<entry->scan.candidates_examined
        <<" qualifying="<<entry->scan.qualifying_candidates
        <<" elapsed_seconds="<<entry->scan.elapsed_seconds
        <<" exact="<<entry->scan.exact<<"\n";
    bucket.push_back(move(entry));
    return bucket.back()->scan;
}

static bool try_exact_defect1_repair(
    const TerminalEvent& terminal,
    const Config& c,
    RepairScanCache& cache,
    ExactRepairMetrics& metrics,
    vector<uint64_t>& child)
{
    const auto [conflict_first,conflict_second]=unique_defect1_pair(terminal.blocks,c);
    const int endpoints[2]={conflict_first,conflict_second};
    for(int side=0;side<2;++side){
        vector<uint64_t>core;
        core.reserve(terminal.blocks.size()-1);
        for(int i=0;i<static_cast<int>(terminal.blocks.size());++i)
            if(i!=endpoints[side])core.push_back(terminal.blocks[i]);
        sort(core.begin(),core.end());
        string core_why;
        if(!verify(core,c,&core_why))
            throw runtime_error("defect-1 endpoint removal did not produce a valid core: "+core_why);
        ++metrics.endpoint_cores;
        const auto&scan=cached_repair_scan(core,c,cache,metrics);

        auto accept_child=[&](vector<uint64_t>candidate,uint32_t radius){
            string why;
            if(candidate.size()!=terminal.blocks.size()||!verify(candidate,c,&why))
                throw runtime_error("exact radius-"+to_string(radius)
                    +" repair failed full CPU verification: "+why);
            child=move(candidate);
            metrics.solved_radius=radius;
            return true;
        };

        for(const auto&candidate:scan.candidates){
            if(candidate.blocker_count!=0)break;
            vector<uint64_t>rebuilt=core;
            rebuilt.push_back(candidate.mask);
            return accept_child(move(rebuilt),1);
        }

        vector<vector<size_t>>single_blocker(core.size());
        map<pair<uint32_t,uint32_t>,vector<size_t>>double_blocker;
        for(size_t candidate_id=0;candidate_id<scan.candidates.size();++candidate_id){
            const auto&candidate=scan.candidates[candidate_id];
            if(candidate.blocker_count==1){
                if(candidate.blocker0>=core.size())
                    throw runtime_error("CUDA repair candidate has invalid blocker ID");
                single_blocker[candidate.blocker0].push_back(candidate_id);
            }else if(candidate.blocker_count==2){
                if(candidate.blocker0>=candidate.blocker1||candidate.blocker1>=core.size())
                    throw runtime_error("CUDA repair candidate has invalid blocker pair");
                double_blocker[{candidate.blocker0,candidate.blocker1}].push_back(candidate_id);
            }
        }

        for(uint32_t removed=0;removed<single_blocker.size();++removed){
            const auto&group=single_blocker[removed];
            for(size_t i=0;i<group.size();++i){
                for(size_t j=i+1;j<group.size();++j){
                    ++metrics.compatibility_pairs_tested;
                    const uint64_t first=scan.candidates[group[i]].mask;
                    const uint64_t second=scan.candidates[group[j]].mask;
                    if(!repair_compatible(first,second,c))continue;
                    vector<uint64_t>rebuilt;
                    rebuilt.reserve(terminal.blocks.size());
                    for(uint32_t core_id=0;core_id<core.size();++core_id)
                        if(core_id!=removed)rebuilt.push_back(core[core_id]);
                    rebuilt.push_back(first);
                    rebuilt.push_back(second);
                    return accept_child(move(rebuilt),2);
                }
            }
        }

        for(const auto&[removed_pair,pair_candidates]:double_blocker){
            const uint32_t removed_first=removed_pair.first;
            const uint32_t removed_second=removed_pair.second;
            vector<size_t>eligible;
            eligible.reserve(single_blocker[removed_first].size()
                +single_blocker[removed_second].size()+pair_candidates.size());
            eligible.insert(eligible.end(),single_blocker[removed_first].begin(),
                single_blocker[removed_first].end());
            eligible.insert(eligible.end(),single_blocker[removed_second].begin(),
                single_blocker[removed_second].end());
            eligible.insert(eligible.end(),pair_candidates.begin(),pair_candidates.end());
            for(size_t i=0;i<eligible.size();++i){
                const uint64_t first=scan.candidates[eligible[i]].mask;
                for(size_t j=i+1;j<eligible.size();++j){
                    const uint64_t second=scan.candidates[eligible[j]].mask;
                    ++metrics.compatibility_pairs_tested;
                    if(!repair_compatible(first,second,c))continue;
                    for(size_t k=j+1;k<eligible.size();++k){
                        ++metrics.compatibility_triples_tested;
                        const uint64_t third=scan.candidates[eligible[k]].mask;
                        if(!repair_compatible(first,third,c)
                            ||!repair_compatible(second,third,c))continue;
                        vector<uint64_t>rebuilt;
                        rebuilt.reserve(terminal.blocks.size());
                        for(uint32_t core_id=0;core_id<core.size();++core_id)
                            if(core_id!=removed_first&&core_id!=removed_second)
                                rebuilt.push_back(core[core_id]);
                        rebuilt.push_back(first);
                        rebuilt.push_back(second);
                        rebuilt.push_back(third);
                        return accept_child(move(rebuilt),3);
                    }
                }
            }
        }
    }
    return false;
}

static bool run_exact_repair_queue(
    const vector<TerminalEvent>& terminals,
    const Config& c,
    ExactRepairMetrics& metrics,
    vector<uint64_t>& child)
{
    const auto started=chrono::steady_clock::now();
    RepairScanCache cache;
    for(const auto&terminal:terminals){
        ++metrics.terminal_events;
        if(try_exact_defect1_repair(terminal,c,cache,metrics,child)){
            metrics.elapsed_seconds=chrono::duration<double>(
                chrono::steady_clock::now()-started).count();
            return true;
        }
    }
    metrics.elapsed_seconds=chrono::duration<double>(
        chrono::steady_clock::now()-started).count();
    return false;
}

#if ATS48_RANKED_REPAIR_QUEUE
static vector<TerminalEvent> ranked_repair_terminals(
    const vector<TerminalEvent>&terminals,const Config&c,size_t limit)
{
    struct Ranked{size_t index=0;uint64_t residual_pressure=0,total_pressure=0;};
    vector<Ranked>ranked;ranked.reserve(terminals.size());
    for(size_t index=0;index<terminals.size();++index){
        const auto&terminal=terminals[index];
        const auto conflict=unique_defect1_pair(terminal.blocks,c);
        uint64_t total=0,endpoint_pressure[2]={0,0};
        for(size_t first=0;first<terminal.blocks.size();++first){
            for(size_t second=0;second<first;++second){
                if(pc(terminal.blocks[first]&terminal.blocks[second])!=c.t-1)continue;
                ++total;
                if(static_cast<int>(first)==conflict.first||static_cast<int>(second)==conflict.first)
                    ++endpoint_pressure[0];
                if(static_cast<int>(first)==conflict.second||static_cast<int>(second)==conflict.second)
                    ++endpoint_pressure[1];
            }
        }
        const uint64_t best_residual=total-max(endpoint_pressure[0],endpoint_pressure[1]);
        ranked.push_back({index,best_residual,total});
    }
    sort(ranked.begin(),ranked.end(),[&](const Ranked&first,const Ranked&second){
        if(first.residual_pressure!=second.residual_pressure)
            return first.residual_pressure<second.residual_pressure;
        if(first.total_pressure!=second.total_pressure)
            return first.total_pressure<second.total_pressure;
        const auto&first_event=terminals[first.index];const auto&second_event=terminals[second.index];
        if(first_event.provenance.iteration!=second_event.provenance.iteration)
            return first_event.provenance.iteration<second_event.provenance.iteration;
        return first_event.neighborhood_hash<second_event.neighborhood_hash;
    });
    vector<TerminalEvent>selected;selected.reserve(min(limit,ranked.size()));
    unordered_set<uint64_t>signatures;
    vector<bool>used(ranked.size(),false);
    for(size_t position=0;position<ranked.size()&&selected.size()<limit;++position){
        const auto&event=terminals[ranked[position].index];
        if(!signatures.insert(event.provenance.conflict_signature).second)continue;
        selected.push_back(event);used[position]=true;
    }
    for(size_t position=0;position<ranked.size()&&selected.size()<limit;++position){
        if(used[position])continue;selected.push_back(terminals[ranked[position].index]);
    }
    cerr<<"CUDA_REPAIR_QUEUE_RANKED input="<<terminals.size()
        <<" selected="<<selected.size()<<" limit="<<limit;
    if(!ranked.empty())cerr<<" best_residual_pressure="<<ranked.front().residual_pressure
        <<" best_total_pressure="<<ranked.front().total_pressure;
    cerr<<"\n";
    return selected;
}
#endif


// ============================================================================
// ATS v4.4 — Branch-Aware Look-Back Chaining
//
// v4.3 kernel retained above:
//   * critical t-shadow score
//   * deep plateau traversal
//   * tabu bit-swaps
//   * one/two-point plateau ejection
//   * independently verified score-0 acceptance
//
// v4.4 controller added below:
//   * multiple stochastic trajectories per valid parent
//   * retention/deduplication of multiple siblings at each cardinality
//   * light future-augmentability probe used to rank siblings
//   * automatic M -> M+1 -> M+2 ... chaining
//   * ancestor look-back epochs when the newest frontier stalls
//   * hardware-aware thread count and checkpoint manifest
//
// The mathematical acceptance criterion is unchanged: a child is kept only
// after verify() confirms exact constant weight and pairwise intersection <= lambda.
// ============================================================================

namespace fs = std::filesystem;

struct ProbeScore {
    long long min_extra_score = LLONG_MAX / 4;
    int near_count = 0;
    long long degree_cost = LLONG_MAX / 4;
};

struct BranchNode {
    vector<uint64_t> blocks;
    ProbeScore probe;
    uint64_t fingerprint = 0;
    string lineage;
    string file;
};

struct RunResult {
    bool found = false;
    bool captured_defect1 = false;
    long long best_score = LLONG_MAX / 4;
    vector<uint64_t> child;
    vector<uint64_t> promoted_state;
    LowDefectProvenance promoted_provenance;
    double promoted_elapsed_seconds = -1.0;
    string best_file;
    string out_file;
};

struct TrajectoryOptions {
    bool stop_on_first_defect1 = false;
    const vector<uint64_t>* promoted_state = nullptr;
    LowDefectProvenance promoted_provenance;
    bool immediate_promoted_repair = true;
    bool concurrent_promoted_repair = false;
};

static vector<uint64_t> sorted_blocks(vector<uint64_t> b) {
    sort(b.begin(), b.end());
    return b;
}

static uint64_t fingerprint_blocks(const vector<uint64_t>& in) {
    auto b = sorted_blocks(in);
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t x : b) {
        h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(b.size()) * 0x517cc1b727220a95ULL;
    return h;
}

static ProbeScore probe_augmentability(
    const vector<uint64_t>& blocks,
    int n,
    int d,
    int w,
    uint64_t seed,
    int samples = 25000)
{
    Config c = make_config(n, d, w, static_cast<int>(blocks.size()));
    Base base = build_base(blocks, c);
    mt19937_64 rng(seed ^ 0x6a09e667f3bcc909ULL);
    unordered_set<uint64_t> sampled;
    sampled.reserve(samples * 2 + 1);

    long long best = LLONG_MAX / 4;
    int near = 0;
    for (int i = 0; i < samples; ++i) {
        uint64_t m = random_block(rng, n, w);
        if (base.set.count(m) || !sampled.insert(m).second) continue;
        long long s = contribution_direct(m, base, c);
        if (s < best) {
            best = s;
            near = 1;
        } else if (s <= best + 1) {
            ++near;
        }
    }

    ProbeScore p;
    p.min_extra_score = best;
    p.near_count = near;
    p.degree_cost = degcost(base.deg, static_cast<int>(blocks.size()) * w, n);
    return p;
}

static bool better_probe(const BranchNode& a, const BranchNode& b) {
    if (a.probe.min_extra_score != b.probe.min_extra_score)
        return a.probe.min_extra_score < b.probe.min_extra_score;
    if (a.probe.near_count != b.probe.near_count)
        return a.probe.near_count > b.probe.near_count;
    if (a.probe.degree_cost != b.probe.degree_cost)
        return a.probe.degree_cost < b.probe.degree_cost;
    return a.fingerprint < b.fingerprint;
}

static RunResult run_one_trajectory(
    const vector<uint64_t>& blocks,
    int n,
    int d,
    int w,
    double seconds,
    int threads,
    const string& out,
    uint64_t seed,
    const TrajectoryOptions& options = {})
{
    RunResult rr;
    rr.out_file = out;

    Config c = make_config(n, d, w, static_cast<int>(blocks.size()));
    string why;
    if (!verify(blocks, c, &why))
        throw runtime_error("v4.8 CUDA parent invalid: " + why);

    Base base = build_base(blocks, c);
    cerr << "V48_CUDA_TRAJECTORY A(" << n << "," << d << "," << w << ") base="
         << blocks.size() << " target=" << blocks.size() + 1
         << " seconds=" << seconds << " threads=" << threads
         << " seed=" << seed << " parent_hash=0x" << hex
         << fingerprint_blocks(blocks) << dec << "\n";

    auto extras = extra_pool(base, c, seed, cerr);
    if (extras.front().score == 0) {
        auto child = blocks;
        child.push_back(extras.front().mask);
        if (!verify(child, c, &why))
            throw runtime_error("v4.8 CUDA direct child verify failed: " + why);
        write_code(child, c, out, "ATS v4.8 CUDA direct critical-shadow extension");
        rr.found = true;
        rr.best_score = 0;
        rr.child = std::move(child);
        return rr;
    }

    Shared g;
    g.out = out;
    g.parent_hash = fingerprint_blocks(blocks);
    g.scout_stop_on_first_defect1 = options.stop_on_first_defect1;
    g.started = chrono::steady_clock::now();
    g.deadline = g.started +
        chrono::duration_cast<chrono::steady_clock::duration>(chrono::duration<double>(seconds));

    if(options.promoted_state){
        if(options.promoted_state->size()!=blocks.size()+1)
            throw runtime_error("ATS v4.8 promoted state has wrong cardinality");
        const uint64_t promoted_score=exact_repeated_shadow_score(*options.promoted_state,c);
        if(promoted_score!=1)
            throw runtime_error("ATS v4.8 promoted state is not exact defect 1");
        LowDefectProvenance record=options.promoted_provenance;
        record.state_hash=telemetry_fingerprint(*options.promoted_state);
        record.conflict_signature=conflict_signature(*options.promoted_state,c);
        record.parent_hash=g.parent_hash;
        record.defect=1;
        if(record.source_bits==0)record.source_bits=SOURCE_CPU;
        g.promoted_captured=true;g.promoted_blocks=*options.promoted_state;g.promoted_provenance=record;g.promoted_elapsed_seconds=0.0;
        g.best.store(1);g.blocks=*options.promoted_state;g.first_hit_seconds[1]=0.0;g.first_hit_seconds[2]=0.0;g.first_hit_seconds[3]=0.0;g.first_hit_seconds[4]=0.0;
        g.defect1_hashes.insert(record.state_hash);merge_low_defect_record(g,record,&g.promoted_blocks);
        write_code(g.promoted_blocks,c,g.out+".promoted_defect1.txt","ATS v4.8 promoted exact defect-1 state");
        write_code(g.promoted_blocks,c,g.out+".best","ATS v4.8 promoted exact defect-1 state");
        cerr<<"V48_PROMOTION_ACCEPTED source="<<source_name(record.source_bits)<<" state_hash=0x"<<hex<<record.state_hash<<dec<<" immediate_repair="<<options.immediate_promoted_repair<<" concurrent_repair="<<options.concurrent_promoted_repair<<"\n";

        if(options.immediate_promoted_repair&&!options.concurrent_promoted_repair&&exact_repair_enabled&&gpu_mode!=GpuMode::Off){
            ExactRepairMetrics promotion_repair_metrics;vector<uint64_t>repaired_child;
            if(run_exact_repair_queue(g.terminal_events,c,promotion_repair_metrics,repaired_child)){
                if(!verify(repaired_child,c,&why))throw runtime_error("ATS v4.8 promoted repair child failed verification: "+why);
                write_code(repaired_child,c,g.out,"ATS v4.8 immediate promoted defect-1 repair");
                cerr<<"V48_PROMOTION_REPAIR_VERIFIED_CHILD size="<<repaired_child.size()<<" radius="<<promotion_repair_metrics.solved_radius<<" elapsed_seconds="<<promotion_repair_metrics.elapsed_seconds<<"\n";
                rr.found=true;rr.best_score=0;rr.child=move(repaired_child);return rr;
            }
            cerr<<"V48_PROMOTION_REPAIR_COMPLETE solved=0 terminal_events="<<promotion_repair_metrics.terminal_events<<" candidates_examined="<<promotion_repair_metrics.gpu_candidates_examined<<" elapsed_seconds="<<promotion_repair_metrics.elapsed_seconds<<"\n";
        }

        ats45cuda::CapturedTerminalState promoted;
        promoted.blocks=g.promoted_blocks;promoted.state_hash=record.state_hash;promoted.iteration=record.iteration;promoted.defect=1;promoted.walker_id=record.actor_id;
        enqueue_gpu_handoff(g,promoted,c,record.source_bits);
        g.started=chrono::steady_clock::now();
        g.deadline=g.started+chrono::duration_cast<chrono::steady_clock::duration>(chrono::duration<double>(seconds));
    }

    exception_ptr gpu_exception;thread gpu_thread;
    if(gpu_mode!=GpuMode::Off){
        vector<uint64_t>gpu_extras;gpu_extras.reserve(extras.size());for(const auto&e:extras)gpu_extras.push_back(e.mask);
        gpu_thread=thread([&,gpu_extras=move(gpu_extras)](){
            GpuDoneGuard done_guard{g};
            try{
                if(options.concurrent_promoted_repair&&options.promoted_state&&exact_repair_enabled){
                    vector<TerminalEvent>promotion_terminals;
                    {lock_guard<mutex>lock(g.m);for(const auto&terminal:g.terminal_events)if(terminal.neighborhood_hash==canonical_fingerprint(*options.promoted_state)){promotion_terminals.push_back(terminal);break;}}
                    if(promotion_terminals.empty())throw runtime_error("ATS v4.8 concurrent promoted repair lost the promoted terminal state");
                    cerr<<"V48_CONCURRENT_REPAIR_START state_hash=0x"<<hex<<options.promoted_provenance.state_hash<<dec<<" cpu_ats=running gpu_walkers=waiting\n";
                    ExactRepairMetrics concurrent_metrics;vector<uint64_t>repaired_child;
                    const bool repaired=run_exact_repair_queue(promotion_terminals,c,concurrent_metrics,repaired_child);
                    {lock_guard<mutex>lock(g.m);g.concurrent_promotion_repair_ran=true;g.concurrent_promotion_repair_candidates=concurrent_metrics.gpu_candidates_examined;g.concurrent_promotion_repair_seconds=concurrent_metrics.elapsed_seconds;g.completed_repair_neighborhoods.insert(promotion_terminals.front().neighborhood_hash);}
                    if(repaired){
                        string repair_why;if(!verify(repaired_child,c,&repair_why))throw runtime_error("ATS v4.8 concurrent promoted repair child failed CPU verification: "+repair_why);
                        bool published=false;
                        {lock_guard<mutex>lock(g.m);if(!g.found.load()){g.blocks=move(repaired_child);g.best.store(0);g.found.store(true);g.stop_requested.store(true);g.verified_children.fetch_add(1,memory_order_relaxed);g.first_zero_source_bits=SOURCE_CPU|SOURCE_GPU;g.first_hit_seconds[0]=chrono::duration<double>(chrono::steady_clock::now()-g.started).count();write_code(g.blocks,c,g.out,"ATS v4.8 concurrent GPU repair with CPU exact verification");published=true;}}
                        g.handoff_cv.notify_all();
                        cerr<<"V48_CONCURRENT_REPAIR_VERIFIED_CHILD published="<<published<<" radius="<<concurrent_metrics.solved_radius<<" elapsed_seconds="<<concurrent_metrics.elapsed_seconds<<" candidates_examined="<<concurrent_metrics.gpu_candidates_examined<<"\n";
                        return;
                    }
                    cerr<<"V48_CONCURRENT_REPAIR_COMPLETE solved=0 elapsed_seconds="<<concurrent_metrics.elapsed_seconds<<" candidates_examined="<<concurrent_metrics.gpu_candidates_examined<<" cpu_ats_continues=1 gpu_walkers=resuming\n";
                    if(g.found.load())return;
                }
                double gpu_start_offset=chrono::duration<double>(chrono::steady_clock::now()-g.started).count();
                function<void(const ats45cuda::CapturedTerminalState&)>handoff_callback;
                if(!first_defect1_forensics)handoff_callback=[&](const ats45cuda::CapturedTerminalState&captured){
                    if(options.stop_on_first_defect1&&captured.defect==1){
                        const uint64_t exact=exact_repeated_shadow_score(captured.blocks,c);
                        if(exact!=1)throw runtime_error("ATS v4.8 CUDA scout promotion failed exact CPU score check");
                        LowDefectProvenance record;record.state_hash=telemetry_fingerprint(captured.blocks);record.conflict_signature=conflict_signature(captured.blocks,c);record.parent_hash=g.parent_hash;record.iteration=captured.iteration;record.defect=1;record.source_bits=SOURCE_GPU;record.actor_id=captured.walker_id;record.move_class=MoveClass::OneBitSwap;
                        bool promoted=false;double elapsed=chrono::duration<double>(chrono::steady_clock::now()-g.started).count();
                        {lock_guard<mutex>lock(g.m);if(!g.promoted_captured){g.promoted_captured=true;g.promoted_blocks=captured.blocks;g.promoted_provenance=record;g.promoted_elapsed_seconds=elapsed;g.defect1_hashes.insert(record.state_hash);merge_low_defect_record(g,record,&g.promoted_blocks);promoted=true;}}
                        if(promoted){write_code(captured.blocks,c,g.out+".promoted_defect1.txt","ATS v4.8 exact CUDA defect-1 scout promotion");cerr<<"V48_SCOUT_PROMOTED source=GPU actor="<<captured.walker_id<<" iteration="<<captured.iteration<<" elapsed_seconds="<<elapsed<<" state_hash=0x"<<hex<<record.state_hash<<dec<<"\n";g.stop_requested.store(true,memory_order_release);g.handoff_cv.notify_all();}
                        return;
                    }
                    if(threads>0)enqueue_gpu_handoff(g,captured,c);
                };
                auto result=ats45cuda::run_swap_search(base.blocks,gpu_extras,c.n,c.w,c.t,gpu_walkers,gpu_proposals,seed^0xd1b54a32d192ed03ULL,g.deadline,&g.stop_requested,options.stop_on_first_defect1||first_defect1_forensics,handoff_callback);
                g.gpu_walkers_used.store(result.walkers,memory_order_relaxed);
                g.gpu_walker_iterations.store(result.walker_iterations,memory_order_relaxed);
                g.gpu_proposals_scored.store(result.proposals_scored,memory_order_relaxed);
                g.gpu_accepted_moves.store(result.accepted_moves,memory_order_relaxed);
                {lock_guard<mutex>lock(g.m);for(int threshold=0;threshold<=4;++threshold)if(result.first_hit_seconds[threshold]>=0.0){double hit=gpu_start_offset+result.first_hit_seconds[threshold];if(g.first_hit_seconds[threshold]<0.0||hit<g.first_hit_seconds[threshold])g.first_hit_seconds[threshold]=hit;}g.defect2_hashes.insert(result.defect2_state_hashes.begin(),result.defect2_state_hashes.end());g.defect1_hashes.insert(result.defect1_state_hashes.begin(),result.defect1_state_hashes.end());g.basin_hashes.insert(result.basin_hashes.begin(),result.basin_hashes.end());for(const auto&gpu_record:result.low_defect_records){LowDefectProvenance record;record.state_hash=gpu_record.state_hash;record.conflict_signature=gpu_record.conflict_signature;record.parent_hash=g.parent_hash;record.iteration=gpu_record.iteration;record.defect=gpu_record.defect;record.source_bits=SOURCE_GPU;record.actor_id=gpu_record.walker_id;record.move_class=gpu_record.move_class==2?MoveClass::TwoBitEjection:MoveClass::OneBitSwap;merge_low_defect_record(g,record);}for(const auto&terminal:result.terminal_states){LowDefectProvenance record;record.state_hash=telemetry_fingerprint(terminal.blocks);record.conflict_signature=conflict_signature(terminal.blocks,c);record.parent_hash=g.parent_hash;record.iteration=terminal.iteration;record.defect=1;record.source_bits=SOURCE_GPU;record.actor_id=terminal.walker_id;record.move_class=MoveClass::OneBitSwap;merge_low_defect_record(g,record,&terminal.blocks);}}
                cerr<<"CUDA_SEARCH_STATS ran="<<result.ran<<" device=\""<<result.device_name<<"\" walkers="<<result.walkers
                    <<" elapsed_seconds="<<result.elapsed_seconds<<" walker_iterations="<<result.walker_iterations
                    <<" proposals_scored="<<result.proposals_scored<<" accepted_moves="<<result.accepted_moves
                    <<" best_score="<<result.best_score<<" low_defect_events_dropped="<<result.low_defect_events_dropped
                    <<" low_defect_states_emitted="<<result.low_defect_states_emitted
                    <<" low_defect_walkers_screened="<<result.low_defect_walkers_screened
                    <<" low_defect_state_copies="<<result.low_defect_state_copies
                    <<" incremental_shadow_scoring="<<result.incremental_shadow_scoring
                    <<" incremental_shadow_bytes="<<result.incremental_shadow_bytes
                    <<" incremental_shadow_audits="<<result.incremental_shadow_audits
                    <<" inverted_shadow_incidence="<<result.inverted_shadow_incidence
                    <<" inverted_shadow_incidence_bytes="<<result.inverted_shadow_incidence_bytes;
                if(!result.error.empty())cerr<<" error=\""<<result.error<<"\"";cerr<<"\n";
                if(!result.error.empty()||!result.ran){
                    if(gpu_mode==GpuMode::On)throw runtime_error("CUDA swap search required but failed: "+result.error);
                    return;
                }
                if(result.has_first_terminal_state&&first_defect1_forensics){
                    audit_first_defect1_state(result.first_terminal_state,c,g.out);
                }else if(first_defect1_forensics){
                    throw runtime_error("CUDA forensic run ended without a captured first defect-1 state");
                }
                if(result.best_blocks.empty()||result.best_score>=g.best.load())return;
                long long exact=0;for(int i=0;i<(int)result.best_blocks.size();++i)for(int j=0;j<i;++j){int z=pc(result.best_blocks[i]&result.best_blocks[j]);if(z>=c.t)exact+=c.choose[z][c.t];}
                if(exact!=result.best_score)throw runtime_error("CUDA published state score mismatch");
                if(exact==0){string gpu_why;if(!verify(result.best_blocks,c,&gpu_why))throw runtime_error("CUDA score-0 state failed independent verify: "+gpu_why);}
                lock_guard<mutex>lock(g.m);
                if(exact<g.best.load()){
                    g.best.store(exact);g.blocks=result.best_blocks;
                    if(exact==0){g.first_zero_source_bits=SOURCE_GPU;write_code(g.blocks,c,g.out,"ATS v4.8 CUDA verified exploration search");g.found.store(true);g.stop_requested.store(true);g.handoff_cv.notify_all();g.verified_children.fetch_add(1,memory_order_relaxed);cerr<<"CUDA_VERIFIED_CHILD size="<<g.blocks.size()<<" child_hash=0x"<<hex<<canonical_fingerprint(g.blocks)<<dec<<"\n";}
                    else{write_code(g.blocks,c,g.out+".best","ATS v4.8 CUDA bounded near-code; repeated-shadow score="+to_string(exact));cerr<<"CUDA_BEST score="<<exact<<"\n";}
                }
            }catch(...){gpu_exception=current_exception();}
        });
    }

    vector<thread> ts;
    ts.reserve(threads);
    const int handoff_threads=((gpu_mode!=GpuMode::Off||options.promoted_state)&&threads>=2)?min(4,max(1,threads/6)):0;
    const int ordinary_threads=threads-handoff_threads;
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back(
            worker,
            t,
            seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(t + 1),
            cref(base),
            cref(extras),
            cref(c),
            ref(g),
            t>=ordinary_threads);
    }
    for (auto& th : ts) th.join();
    if(gpu_thread.joinable())gpu_thread.join();
    if(gpu_exception)rethrow_exception(gpu_exception);
    const double search_elapsed=chrono::duration<double>(
        chrono::steady_clock::now()-g.started).count();
    ExactRepairMetrics repair_metrics;
    vector<TerminalEvent>pending_repair_terminals;
    for(const auto&terminal:g.terminal_events)if(!g.completed_repair_neighborhoods.count(terminal.neighborhood_hash))pending_repair_terminals.push_back(terminal);
#if ATS48_RANKED_REPAIR_QUEUE
    pending_repair_terminals=ranked_repair_terminals(pending_repair_terminals,c,32);
#endif
    if(exact_repair_enabled&&!options.stop_on_first_defect1&&!g.found.load()&&!pending_repair_terminals.empty()&&gpu_mode!=GpuMode::Off){
        string repair_device,repair_error;
        if(!ats45cuda::runtime_available(repair_device,repair_error)){
            if(gpu_mode==GpuMode::On)
                throw runtime_error("CUDA exact repair required but unavailable: "+repair_error);
            cerr<<"CUDA_REPAIR_SKIPPED reason=\""<<repair_error<<"\"\n";
        }else{
            vector<uint64_t>repaired_child;
            if(run_exact_repair_queue(pending_repair_terminals,c,repair_metrics,repaired_child)){
                string repair_why;
                if(!verify(repaired_child,c,&repair_why))
                    throw runtime_error("CUDA-assisted exact repair child failed final verification: "+repair_why);
                g.blocks=move(repaired_child);
                g.best.store(0);
                g.found.store(true);
                g.verified_children.fetch_add(1,memory_order_relaxed);
                g.first_hit_seconds[0]=chrono::duration<double>(
                    chrono::steady_clock::now()-g.started).count();
                write_code(g.blocks,c,g.out,
                    "ATS v4.8 CUDA exact defect-1 radius-"+to_string(repair_metrics.solved_radius)+" repair");
                cerr<<"CUDA_REPAIR_VERIFIED_CHILD size="<<g.blocks.size()
                    <<" radius="<<repair_metrics.solved_radius<<"\n";
            }
        }
    }
    cerr<<"CUDA_REPAIR_STATS terminal_events="<<repair_metrics.terminal_events
        <<" endpoint_cores="<<repair_metrics.endpoint_cores
        <<" unique_core_scans="<<repair_metrics.unique_core_scans
        <<" core_cache_hits="<<repair_metrics.core_cache_hits
        <<" candidates_examined="<<repair_metrics.gpu_candidates_examined
        <<" qualifying_candidates="<<repair_metrics.qualifying_candidates
        <<" compatibility_pairs_tested="<<repair_metrics.compatibility_pairs_tested
        <<" compatibility_triples_tested="<<repair_metrics.compatibility_triples_tested
        <<" solved_radius="<<repair_metrics.solved_radius
        <<" elapsed_seconds="<<repair_metrics.elapsed_seconds<<"\n";
    cerr<<"V48_CONCURRENT_REPAIR_STATS ran="<<g.concurrent_promotion_repair_ran
        <<" candidates_examined="<<g.concurrent_promotion_repair_candidates
        <<" elapsed_seconds="<<g.concurrent_promotion_repair_seconds
        <<" completed_neighborhoods="<<g.completed_repair_neighborhoods.size()<<"\n";
    write_low_defect_provenance(g);
    size_t defect1_cpu_only=0,defect1_gpu_only=0,defect1_both=0;
    for(const auto&record:g.low_defect_records)if(record.defect==1){if(record.source_bits==(SOURCE_CPU|SOURCE_GPU))++defect1_both;else if(record.source_bits==SOURCE_GPU)++defect1_gpu_only;else ++defect1_cpu_only;}
    cerr<<"LOW_DEFECT_PROVENANCE records="<<g.low_defect_records.size()
        <<" terminal_neighborhoods="<<g.terminal_events.size()
        <<" defect1_cpu_only="<<defect1_cpu_only
        <<" defect1_gpu_only="<<defect1_gpu_only
        <<" defect1_both="<<defect1_both
        <<" file=\""<<g.out<<".low_defect.tsv\"\n";
    cerr<<"SEARCH_BACKEND_STATS cpu_iterations="<<g.cpu_iterations.load()
        <<" cpu_move_evaluations="<<g.cpu_move_evaluations.load()
        <<" cpu_accepted_moves="<<g.cpu_accepted_moves.load()
        <<" cpu_restarts="<<g.cpu_restarts.load()
        <<" cpu_search_starts="<<g.cpu_search_starts.load()
        <<" gpu_walkers="<<g.gpu_walkers_used.load()
        <<" gpu_walker_iterations="<<g.gpu_walker_iterations.load()
        <<" gpu_proposals_scored="<<g.gpu_proposals_scored.load()
        <<" gpu_accepted_moves="<<g.gpu_accepted_moves.load()
        <<" ordinary_cpu_workers="<<ordinary_threads
        <<" handoff_cpu_workers="<<handoff_threads
        <<" gpu_handoffs_enqueued="<<g.gpu_handoffs_enqueued.load()
        <<" gpu_handoffs_deduplicated="<<g.gpu_handoffs_deduplicated.load()
        <<" gpu_handoffs_dropped="<<g.gpu_handoffs_dropped.load()
        <<" cpu_handoffs_started="<<g.cpu_handoffs_started.load()
        <<" handoff_queue_remaining="<<g.handoff_pending.load()<<"\n";
    cerr<<"BENCHMARK_TELEMETRY search_elapsed_seconds="<<search_elapsed
        <<" best_defect="<<g.best.load()
        <<" first_le4="<<g.first_hit_seconds[4]
        <<" first_le3="<<g.first_hit_seconds[3]
        <<" first_le2="<<g.first_hit_seconds[2]
        <<" first_le1="<<g.first_hit_seconds[1]
        <<" first_le0="<<g.first_hit_seconds[0]
        <<" unique_defect2="<<g.defect2_hashes.size()
        <<" unique_defect1="<<g.defect1_hashes.size()
        <<" proposed_swaps="<<(g.cpu_move_evaluations.load()+g.gpu_proposals_scored.load())
        <<" accepted_moves="<<(g.cpu_accepted_moves.load()+g.gpu_accepted_moves.load())
        <<" restarts="<<g.cpu_restarts.load()
        <<" distinct_basin_hashes="<<g.basin_hashes.size()
        <<" verified_children="<<g.verified_children.load()
        <<" first_zero_source="<<(g.first_zero_source_bits?source_name(g.first_zero_source_bits):"none")
        <<" exact_repair_seconds="<<repair_metrics.elapsed_seconds
        <<" exact_repair_radius="<<repair_metrics.solved_radius
        <<" concurrent_repair_seconds="<<g.concurrent_promotion_repair_seconds
        <<" concurrent_repair_candidates="<<g.concurrent_promotion_repair_candidates<<"\n";

    rr.best_score = g.best.load();
    rr.best_file = out + ".best";
    rr.captured_defect1 = g.promoted_captured;
    rr.promoted_state = g.promoted_blocks;
    rr.promoted_provenance = g.promoted_provenance;
    rr.promoted_elapsed_seconds = g.promoted_elapsed_seconds;
    if (!g.found.load()) return rr;

    if (!verify(g.blocks, c, &why))
        throw runtime_error("v4.8 CUDA score-0 child failed independent verify: " + why);

    rr.found = true;
    rr.best_score = 0;
    rr.child = g.blocks;
    return rr;
}

static void write_manifest(
    const string& path,
    int n,
    int d,
    int w,
    const vector<vector<BranchNode>>& levels,
    const string& status)
{
    ofstream f(path);
    if (!f) throw runtime_error("cannot write manifest " + path);
    f << "# ATS v4.8 CUDA Branch-Aware Look-Back Chaining Checkpoint\n";
    f << "status=" << status << "\n";
    f << "parameters=A(" << n << "," << d << "," << w << ")\n";
    f << "levels=" << levels.size() << "\n\n";
    for (size_t li = 0; li < levels.size(); ++li) {
        f << "## level " << li << " size="
          << (levels[li].empty() ? 0 : levels[li][0].blocks.size()) << "\n";
        for (size_t j = 0; j < levels[li].size(); ++j) {
            const auto& b = levels[li][j];
            f << j
              << " fingerprint=0x" << hex << b.fingerprint << dec
              << " probe_min=" << b.probe.min_extra_score
              << " probe_near=" << b.probe.near_count
              << " degree_cost=" << b.probe.degree_cost
              << " lineage=" << b.lineage
              << " file=" << b.file << "\n";
        }
        f << "\n";
    }
}

static vector<BranchNode> expand_parents(
    const vector<BranchNode>& parents,
    int n,
    int d,
    int w,
    double seconds_per_trajectory,
    int threads,
    int rounds_per_parent,
    int beam_width,
    uint64_t master_seed,
    uint64_t epoch,
    const string& output_dir,
    int target_level,
    const vector<BranchNode>& already = {})
{
    vector<BranchNode> children = already;
    unordered_set<uint64_t> seen;
    for (const auto& c : children) seen.insert(c.fingerprint);

    for (size_t pi = 0; pi < parents.size(); ++pi) {
        for (int round = 0; round < rounds_per_parent; ++round) {
            uint64_t seed = master_seed
                ^ (0x94d049bb133111ebULL * (epoch + 1))
                ^ (0xd6e8feb86659fd93ULL * (parents[pi].blocks.size() + 1))
                ^ (0xbf58476d1ce4e5b9ULL * (pi + 1))
                ^ (0x9e3779b97f4a7c15ULL * (round + 1));

            int target_m = static_cast<int>(parents[pi].blocks.size()) + 1;
            ostringstream name;
            name << output_dir << "/A_" << n << "_" << d << "_" << w << "_"
                 << target_m << "_L" << target_level << "_P" << pi
                 << "_E" << epoch << "_R" << round << ".txt";

            cerr << "[level " << parents[pi].blocks.size() << "]"
                 << "[parent " << pi << "]"
                 << "[trajectory " << round << "]\n";

            TrajectoryOptions trajectory_options;
            trajectory_options.stop_on_first_defect1=stop_on_first_defect1_mode;
            RunResult rr = run_one_trajectory(
                parents[pi].blocks,
                n, d, w,
                seconds_per_trajectory,
                threads,
                name.str(),
                seed,
                trajectory_options);

            cerr << "V48_CUDA_RESULT parent=" << pi << " level=" << target_level
                 << " epoch=" << epoch << " round=" << round
                 << " found=" << rr.found << " best=" << rr.best_score << "\n";

            if (!rr.found) continue;
            uint64_t fp = fingerprint_blocks(rr.child);
            if (!seen.insert(fp).second) continue;

            Config cc = make_config(n, d, w, static_cast<int>(rr.child.size()));
            string why;
            if (!verify(rr.child, cc, &why))
                throw runtime_error("v4.8 CUDA retained child invalid: " + why);

            BranchNode child;
            child.blocks = rr.child;
            child.fingerprint = fp;
            child.lineage = parents[pi].lineage + "->" + to_string(target_m)
                + "/E" + to_string(epoch) + "R" + to_string(round);
            child.file = rr.out_file;
            child.probe = probe_augmentability(
                child.blocks, n, d, w,
                seed ^ 0x3c6ef372fe94f82bULL);
            children.push_back(std::move(child));
        }
    }

    sort(children.begin(), children.end(), better_probe);
    if ((int)children.size() > beam_width) children.resize(beam_width);
    return children;
}

static int run_staged_frontier(
    const string& input,
    int n,
    int d,
    int w,
    int threads,
    const string& output_dir,
    uint64_t base_seed,
    double full_seconds,
    int scouts_per_stage,
    double interval_seconds)
{
    string input_format;vector<uint64_t>blocks=parse_code(input,n,w,&input_format);
    Config c=make_config(n,d,w,static_cast<int>(blocks.size()));string why;
    if(!verify(blocks,c,&why))throw runtime_error("ATS v4.8 staged frontier parent invalid: "+why);
    if(full_seconds<=0||scouts_per_stage<=0||interval_seconds<=0)
        throw runtime_error("ATS v4.8 staged frontier times and scout count must be positive");
    fs::create_directories(output_dir);
    const string status_path=output_dir+"/ATS_v4_8_staged_status.txt";
    uint64_t attempt=0;
    cerr<<"V48_STAGED_START input=\""<<input<<"\" format="<<input_format<<" base="<<blocks.size()<<" target="<<blocks.size()+1<<" threads="<<threads<<" scouts_per_stage="<<scouts_per_stage<<" interval_seconds="<<interval_seconds<<" full_seconds="<<full_seconds<<" base_seed="<<base_seed<<"\n";

    for(uint64_t stage=1;;++stage){
        const double scout_seconds=interval_seconds*static_cast<double>(stage);
        for(int slot=0;slot<scouts_per_stage;++slot,++attempt){
            const uint64_t scout_seed=base_seed+attempt;
            ostringstream prefix;prefix<<output_dir<<"/scout_stage_"<<stage<<"_slot_"<<(slot+1)<<"_seconds_"<<static_cast<uint64_t>(scout_seconds)<<"_seed_"<<scout_seed;
            {ofstream status(status_path);status<<"status=scouting\nstage="<<stage<<"\nslot="<<(slot+1)<<"\nscout_seconds="<<scout_seconds<<"\nscout_seed="<<scout_seed<<"\nattempt="<<(attempt+1)<<"\n";}
            cerr<<"V48_SCOUT_START stage="<<stage<<" slot="<<(slot+1)<<" attempt="<<(attempt+1)<<" seconds="<<scout_seconds<<" seed="<<scout_seed<<"\n";
            TrajectoryOptions scout_options;scout_options.stop_on_first_defect1=true;
            RunResult scout=run_one_trajectory(blocks,n,d,w,scout_seconds,threads,prefix.str(),scout_seed,scout_options);
            cerr<<"V48_SCOUT_RESULT stage="<<stage<<" slot="<<(slot+1)<<" seed="<<scout_seed<<" found="<<scout.found<<" captured_defect1="<<scout.captured_defect1<<" best="<<scout.best_score<<" promoted_elapsed_seconds="<<scout.promoted_elapsed_seconds<<"\n";
            if(scout.found){
                {ofstream status(status_path);status<<"status=verified_child_during_scout\nseed="<<scout_seed<<"\noutput="<<prefix.str()<<"\n";}
                return 0;
            }
            if(!scout.captured_defect1)continue;

            const uint64_t exploitation_seed=scout_seed^0xa0761d6478bd642fULL;
            ostringstream full_prefix;full_prefix<<output_dir<<"/promoted_full_"<<static_cast<uint64_t>(full_seconds)<<"s_scout_seed_"<<scout_seed<<"_exploit_seed_"<<exploitation_seed;
            {ofstream status(status_path);status<<"status=promoted_exploitation\nstage="<<stage<<"\nslot="<<(slot+1)<<"\nscout_seed="<<scout_seed<<"\nexploitation_seed="<<exploitation_seed<<"\npromoted_state_hash=0x"<<hex<<scout.promoted_provenance.state_hash<<dec<<"\nfull_seconds="<<full_seconds<<"\n";}
            cerr<<"V48_PROMOTED_RUN_START scout_seed="<<scout_seed<<" exploitation_seed="<<exploitation_seed<<" source="<<source_name(scout.promoted_provenance.source_bits)<<" state_hash=0x"<<hex<<scout.promoted_provenance.state_hash<<dec<<" full_seconds="<<full_seconds<<" gpu_role=concurrent_exact_repair_then_basin_discovery cpu_role=simultaneous_promoted_and_ordinary_ats\n";
            TrajectoryOptions exploit_options;exploit_options.promoted_state=&scout.promoted_state;exploit_options.promoted_provenance=scout.promoted_provenance;exploit_options.immediate_promoted_repair=false;exploit_options.concurrent_promoted_repair=true;
            RunResult full=run_one_trajectory(blocks,n,d,w,full_seconds,threads,full_prefix.str(),exploitation_seed,exploit_options);
            {ofstream status(status_path);status<<"status=complete\nscout_seed="<<scout_seed<<"\nexploitation_seed="<<exploitation_seed<<"\nfound="<<full.found<<"\nbest_defect="<<full.best_score<<"\noutput="<<full_prefix.str()<<"\n";}
            cerr<<"V48_PROMOTED_RUN_RESULT found="<<full.found<<" best="<<full.best_score<<" scout_seed="<<scout_seed<<" exploitation_seed="<<exploitation_seed<<"\n";
            return full.found?0:2;
        }
    }
}

static int benchmark_candidate_scorer(const vector<uint64_t>&blocks,const Config&c,double seconds,size_t batch,uint64_t seed){
    if(seconds<=0)throw runtime_error("benchmark seconds must be positive");
    batch=max<size_t>(1,batch);
    ats45cuda::CandidateScorer gpu;string error;
    if(!gpu.initialize(blocks,c.n,c.w,c.t,error))throw runtime_error("CUDA benchmark initialization failed: "+error);
    batch=min(batch,gpu.maximum_batch_size());

    Base base=build_base(blocks,c);
    size_t validation_count=min<size_t>(batch,65536);vector<ats45cuda::CandidateResult>validation;
    if(!gpu.score_generated(seed,0,validation_count,validation,error))throw runtime_error("CUDA benchmark validation dispatch failed: "+error);
    for(size_t i=0;i<validation.size();++i){
        uint64_t expected_mask=gpu_generated_block(seed,i,c.n,c.w);
        long long expected_score=contribution_direct(expected_mask,base,c);
        if(validation[i].mask!=expected_mask||validation[i].score!=expected_score||pc(validation[i].mask)!=c.w){
            ostringstream o;o<<"CUDA benchmark mismatch at candidate "<<i<<" mask=0x"<<hex<<validation[i].mask
                <<" expected_mask=0x"<<expected_mask<<dec<<" score="<<validation[i].score
                <<" expected_score="<<expected_score;throw runtime_error(o.str());
        }
    }

    uint64_t cpu_candidates=0,cpu_checksum=0,cpu_first=0;
    auto cpu_start=chrono::steady_clock::now(),cpu_deadline=cpu_start+chrono::duration_cast<chrono::steady_clock::duration>(chrono::duration<double>(seconds));
    do{
        for(size_t i=0;i<batch;++i){uint64_t m=gpu_generated_block(seed,cpu_first+i,c.n,c.w);long long score=contribution_direct(m,base,c);cpu_checksum^=gpu_mix32(static_cast<uint32_t>(m)^static_cast<uint32_t>(m>>32)^static_cast<uint32_t>(score)+static_cast<uint32_t>(cpu_first+i));}
        cpu_first+=batch;cpu_candidates+=batch;
    }while(chrono::steady_clock::now()<cpu_deadline);
    double cpu_elapsed=chrono::duration<double>(chrono::steady_clock::now()-cpu_start).count();

    uint64_t gpu_candidates=0,gpu_checksum=0,gpu_first=0;vector<ats45cuda::CandidateResult>results;
    auto gpu_start=chrono::steady_clock::now(),gpu_deadline=gpu_start+chrono::duration_cast<chrono::steady_clock::duration>(chrono::duration<double>(seconds));
    do{
        if(!gpu.score_generated(seed,gpu_first,batch,results,error))throw runtime_error("CUDA benchmark dispatch failed: "+error);
        for(size_t i=0;i<results.size();++i){const auto&r=results[i];gpu_checksum^=gpu_mix32(static_cast<uint32_t>(r.mask)^static_cast<uint32_t>(r.mask>>32)^r.score+static_cast<uint32_t>(gpu_first+i));}
        gpu_first+=batch;gpu_candidates+=batch;
    }while(chrono::steady_clock::now()<gpu_deadline);
    double gpu_elapsed=chrono::duration<double>(chrono::steady_clock::now()-gpu_start).count();

    double cpu_rate=cpu_candidates/max(cpu_elapsed,1e-9),gpu_rate=gpu_candidates/max(gpu_elapsed,1e-9);
    cout<<fixed<<setprecision(3)
        <<"ATS_V4_8_CUDA_BENCHMARK device=\""<<gpu.device_name()<<"\" base_blocks="<<blocks.size()
        <<" n="<<c.n<<" d="<<c.d<<" w="<<c.w<<" t="<<c.t<<" batch="<<batch
        <<" validation_candidates="<<validation_count<<" validation=PASS\n"
        <<"CPU seconds="<<cpu_elapsed<<" candidates="<<cpu_candidates
        <<" candidates_per_second="<<cpu_rate
        <<" pair_scores_per_second="<<(cpu_rate*blocks.size())<<" checksum="<<cpu_checksum<<"\n"
        <<"CPU_GPU seconds="<<gpu_elapsed<<" candidates="<<gpu_candidates
        <<" candidates_per_second="<<gpu_rate
        <<" pair_scores_per_second="<<(gpu_rate*blocks.size())<<" checksum="<<gpu_checksum<<"\n"
        <<"SPEEDUP cpu_gpu_over_cpu="<<(gpu_rate/max(cpu_rate,1e-9))<<"x\n";
    return 0;
}

static void usage() {
    cerr
      << "ATS v4.8 CUDA — Staged Defect-1 Scouting / Exact-State Promotion\n\n"
      << "usage:\n"
      << "  ATS_v4_8_cuda seed.txt n d w seconds threads output_dir rng_seed"
      << " [rounds_per_parent] [beam_width] [lookback_depth] [max_gain] [lookback_epochs]"
      << " [outsider_samples] [candidate_pool] [cuda_mode] [cuda_batch] [cuda_walkers] [cuda_proposals]"
      << " [cuda_repair_capacity]\n"
      << "  ATS_v4_8_cuda --staged-frontier seed.txt n d w threads output_dir rng_seed"
      << " [full_seconds] [scouts_per_stage] [interval_seconds] [outsider_samples] [candidate_pool]"
      << " [cuda_mode] [cuda_batch] [cuda_walkers] [cuda_proposals] [cuda_repair_capacity]\n"
      << "  ATS_v4_8_cuda --benchmark-cuda seed.txt n d w seconds [batch] [rng_seed]\n"
      << "  ATS_v4_8_cuda --known-zero-regression seed.txt n d w seconds cpu_threads output rng_seed cuda_mode"
      << " [outsider_samples] [candidate_pool] [cuda_batch] [cuda_walkers] [cuda_proposals] [audit_mode: reference|forensics]\n"
      << "  ATS_v4_8_cuda --repair-defect1 state.txt n d w output.txt [capacity]\n\n"
      << "defaults:\n"
      << "  staged frontier: full_seconds=600, scouts_per_stage=6, interval_seconds=10\n"
      << "  rounds_per_parent=2\n"
      << "  beam_width=4\n"
      << "  lookback_depth=2\n"
      << "  max_gain=16\n"
      << "  lookback_epochs=2\n"
      << "  outsider_samples=5000000\n"
      << "  candidate_pool=2048\n"
      << "  cuda_mode=auto (off, auto, or on)\n"
      << "  cuda_batch=262144\n"
      << "  cuda_walkers=8192\n"
      << "  cuda_proposals=16\n"
      << "  cuda_repair_capacity=4194304\n\n"
      << "threads=0 uses std::thread::hardware_concurrency().\n"
      << "Each accepted child is independently verified before retention.\n";
}

int main(int argc, char** argv) {
    try {
        if(argc>=2&&string(argv[1])=="--staged-frontier"){
            if(argc<9){usage();return 64;}
            const string input=argv[2];const int n=stoi(argv[3]),d=stoi(argv[4]),w=stoi(argv[5]);int threads=stoi(argv[6]);const string output_dir=argv[7];const uint64_t seed=stoull(argv[8]);
            const double full_seconds=argc>9?stod(argv[9]):600.0;const int scouts_per_stage=argc>10?max(1,stoi(argv[10])):6;const double interval_seconds=argc>11?stod(argv[11]):10.0;
            outsider_samples=argc>12?max<uint64_t>(1,stoull(argv[12])):5000000ULL;candidate_pool=argc>13?max<size_t>(1,stoull(argv[13])):2048;gpu_mode=argc>14?parse_gpu_mode(argv[14]):GpuMode::On;gpu_batch_size=argc>15?max<size_t>(1,stoull(argv[15])):262144;gpu_walkers=argc>16?max<uint32_t>(1,stoul(argv[16])):8192;gpu_proposals=argc>17?max<uint32_t>(1,stoul(argv[17])):16;cuda_repair_capacity=argc>18?max<uint32_t>(1,stoul(argv[18])):4U*1024U*1024U;
            unsigned hc=thread::hardware_concurrency();if(threads==0)threads=hc?static_cast<int>(hc):1;if(threads<1)throw runtime_error("threads must be nonnegative");
            return run_staged_frontier(input,n,d,w,threads,output_dir,seed,full_seconds,scouts_per_stage,interval_seconds);
        }
        if(argc>=2&&string(argv[1])=="--known-zero-regression"){
            if(argc<11){usage();return 64;}
            const string input=argv[2];const int n=stoi(argv[3]),d=stoi(argv[4]),w=stoi(argv[5]);
            const double seconds=stod(argv[6]);const int cpu_threads=stoi(argv[7]);const string output=argv[8];
            const uint64_t seed=stoull(argv[9]);gpu_mode=parse_gpu_mode(argv[10]);
            outsider_samples=argc>11?max<uint64_t>(1,stoull(argv[11])):450000ULL;
            candidate_pool=argc>12?max<size_t>(1,stoull(argv[12])):2048;
            gpu_batch_size=argc>13?max<size_t>(1,stoull(argv[13])):262144;
            gpu_walkers=argc>14?max<uint32_t>(1,stoul(argv[14])):8192;
            gpu_proposals=argc>15?max<uint32_t>(1,stoul(argv[15])):16;
            const string audit_mode=argc>16?string(argv[16]):string();
            reference_policy_audit=audit_mode=="reference";
            first_defect1_forensics=audit_mode=="forensics";
            if(!audit_mode.empty()&&!reference_policy_audit&&!first_defect1_forensics)
                throw runtime_error("known-zero regression audit mode must be reference or forensics");
            if(first_defect1_forensics&&(cpu_threads!=0||gpu_mode==GpuMode::Off))
                throw runtime_error("first-defect1 forensics requires CUDA-only (cpu_threads=0, cuda_mode=on/auto)");
            if(cpu_threads<0)throw runtime_error("known-zero regression cpu_threads must be >= 0");
            if(cpu_threads==0&&gpu_mode==GpuMode::Off)throw runtime_error("known-zero regression requires at least one search backend");
            legacy_regression_pool=true;exact_repair_enabled=false;
            string input_format;auto blocks=parse_code(input,n,w,&input_format);Config c=make_config(n,d,w,static_cast<int>(blocks.size()));string why;
            if(!verify(blocks,c,&why))throw runtime_error("known-zero regression parent invalid: "+why);
            cerr<<"KNOWN_ZERO_REGRESSION input=\""<<input<<"\" format="<<input_format
                <<" cpu_threads="<<cpu_threads<<" cuda_mode="<<gpu_mode_name(gpu_mode)
                <<" exact_repair=off legacy_pool=on reference_policy_audit="<<reference_policy_audit
                <<" first_defect1_forensics="<<first_defect1_forensics<<"\n";
            RunResult result=run_one_trajectory(blocks,n,d,w,seconds,cpu_threads,output,seed);
            cout<<"KNOWN_ZERO_RESULT found="<<result.found<<" best_defect="<<result.best_score
                <<" source_mode="<<(cpu_threads==0?"CUDA-only":gpu_mode==GpuMode::Off?"CPU-only":"CPU+CUDA")<<"\n";
            return result.found?0:2;
        }
        if(argc>=2&&(string(argv[1])=="--benchmark-cuda"||string(argv[1])=="--benchmark-gpu")){
            if(argc<7){usage();return 64;}
            string input=argv[2];int n=stoi(argv[3]),d=stoi(argv[4]),w=stoi(argv[5]);double seconds=stod(argv[6]);
            size_t batch=argc>7?max<size_t>(1,stoull(argv[7])):gpu_batch_size;uint64_t seed=argc>8?stoull(argv[8]):53042820260824ULL;
            string input_format;auto blocks=parse_code(input,n,w,&input_format);Config c=make_config(n,d,w,static_cast<int>(blocks.size()));string why;
            if(!verify(blocks,c,&why))throw runtime_error("benchmark seed invalid: "+why);
            cerr<<"ATS v4.8 CUDA benchmark input="<<input<<" format="<<input_format<<" blocks="<<blocks.size()<<" seconds_per_backend="<<seconds<<"\n";
            return benchmark_candidate_scorer(blocks,c,seconds,batch,seed);
        }
        if(argc>=2&&string(argv[1])=="--repair-defect1"){
            if(argc<7){usage();return 64;}
            const string input=argv[2];
            const int n=stoi(argv[3]),d=stoi(argv[4]),w=stoi(argv[5]);
            const string output=argv[6];
            cuda_repair_capacity=argc>7?max<uint32_t>(1,stoul(argv[7]))
                :4U*1024U*1024U;
            string input_format;
            auto blocks=parse_code(input,n,w,&input_format);
            Config c=make_config(n,d,w,static_cast<int>(blocks.size())-1);
            const auto conflict=unique_defect1_pair(blocks,c);
            TerminalEvent terminal;
            terminal.blocks=blocks;
            terminal.neighborhood_hash=canonical_fingerprint(blocks);
            ExactRepairMetrics metrics;
            vector<uint64_t>child;
            cerr<<"ATS v4.8 CUDA exact defect-1 repair input="<<input
                <<" format="<<input_format<<" blocks="<<blocks.size()
                <<" conflict_pair="<<conflict.first<<","<<conflict.second
                <<" capacity="<<cuda_repair_capacity<<"\n";
            const bool found=run_exact_repair_queue({terminal},c,metrics,child);
            cout<<"ATS_V4_8_CUDA_REPAIR found="<<found
                <<" radius="<<metrics.solved_radius
                <<" terminal_events="<<metrics.terminal_events
                <<" core_scans="<<metrics.unique_core_scans
                <<" candidates_examined="<<metrics.gpu_candidates_examined
                <<" qualifying_candidates="<<metrics.qualifying_candidates
                <<" elapsed_seconds="<<metrics.elapsed_seconds<<"\n";
            if(!found)return 2;
            string why;
            if(!verify(child,c,&why))
                throw runtime_error("standalone repaired child failed verification: "+why);
            write_code(child,c,output,
                "ATS v4.8 CUDA exact defect-1 radius-"+to_string(metrics.solved_radius)+" repair");
            return 0;
        }
        if (argc < 9) {
            usage();
            return 64;
        }

        for(int argument=1;argument<argc;++argument)
            if(string(argv[argument])=="--stop-on-first-defect1")
                stop_on_first_defect1_mode=true;
        const string input = argv[1];
        const int n = stoi(argv[2]);
        const int d = stoi(argv[3]);
        const int w = stoi(argv[4]);
        const double seconds = stod(argv[5]);
        int threads = stoi(argv[6]);
        const string output_dir = argv[7];
        const uint64_t master_seed = stoull(argv[8]);
        const int rounds_per_parent = argc > 9 ? max(1, stoi(argv[9])) : 2;
        const int beam_width = argc > 10 ? max(1, stoi(argv[10])) : 4;
        const int lookback_depth = argc > 11 ? max(0, stoi(argv[11])) : 2;
        const int max_gain = argc > 12 ? max(1, stoi(argv[12])) : 16;
        const int lookback_epochs = argc > 13 ? max(0, stoi(argv[13])) : 2;
        outsider_samples = argc > 14 ? max<uint64_t>(1, stoull(argv[14])) : 5000000ULL;
        candidate_pool = argc > 15 ? max<size_t>(1, stoull(argv[15])) : 2048;
        gpu_mode = argc > 16 ? parse_gpu_mode(argv[16]) : GpuMode::Auto;
        gpu_batch_size = argc > 17 ? max<size_t>(1, stoull(argv[17])) : 262144;
        gpu_walkers = argc > 18 ? max<uint32_t>(1, stoul(argv[18])) : 8192;
        gpu_proposals = argc > 19 ? max<uint32_t>(1, stoul(argv[19])) : 16;
        cuda_repair_capacity = argc > 20 ? max<uint32_t>(1, stoul(argv[20]))
            : 4U * 1024U * 1024U;

        unsigned hc = thread::hardware_concurrency();
        if (threads <= 0) threads = hc ? static_cast<int>(hc) : 1;
        if (hc) threads = min(threads, static_cast<int>(hc));
        threads = max(1, threads);

        cerr << "ATS v4.8 CUDA Staged Defect-1 Scouting / Exact-State Promotion\n"
             << "Kernel: ATS v4.3 critical-shadow plateau/ejection\n"
             << "Input: " << input << "\n"
             << "Threads: " << threads << "\n"
             << "Seconds per trajectory: " << seconds << "\n"
             << "Outsider sample limit: " << outsider_samples << "\n"
             << "Candidate pool: " << candidate_pool << "\n"
             << "CUDA mode: " << gpu_mode_name(gpu_mode) << "\n"
             << "CUDA batch: " << gpu_batch_size << "\n"
             << "CUDA walkers: " << gpu_walkers << "\n"
             << "CUDA proposals per walker iteration: " << gpu_proposals << "\n"
             << "CUDA exact-repair output capacity: " << cuda_repair_capacity << "\n"
             << "Stop on first defect-1: " << stop_on_first_defect1_mode << "\n"
             << "Trajectories per parent: " << rounds_per_parent << "\n"
             << "Sibling beam width: " << beam_width << "\n"
             << "Look-back depth: " << lookback_depth << "\n"
             << "Maximum chain gain: " << max_gain << "\n"
             << "Look-back epochs: " << lookback_epochs << "\n"
             << "RNG base seed: " << master_seed << "\n";

        fs::create_directories(output_dir);

        string input_format;
        vector<uint64_t> initial = parse_code(input, n, w, &input_format);
        Config initial_cfg = make_config(n, d, w, static_cast<int>(initial.size()));
        string why;
        if (!verify(initial, initial_cfg, &why))
            throw runtime_error("initial seed invalid: " + why);
        cerr << "Input format: " << input_format << "\n"
             << "Input size: " << initial.size() << "\n"
             << "Target size: " << initial.size() + max_gain << "\n";

        BranchNode root;
        root.blocks = initial;
        root.fingerprint = fingerprint_blocks(initial);
        root.lineage = to_string(initial.size());
        root.file = input;
        root.probe = probe_augmentability(initial, n, d, w, master_seed);

        vector<vector<BranchNode>> levels;
        levels.push_back({root});
        vector<uint64_t> epochs(max_gain + lookback_depth + 4, 0);

        const string manifest = output_dir + "/ATS_v4_8_CUDA_checkpoint.txt";
        write_manifest(manifest, n, d, w, levels, "initialized");

        int gain = 0;
        bool stopped = false;

        while (gain < max_gain && !stopped) {
            int li = static_cast<int>(levels.size()) - 1;
            auto children = expand_parents(
                levels[li], n, d, w,
                seconds, threads,
                rounds_per_parent, beam_width,
                master_seed, epochs[li]++, output_dir,
                li + 1);

            if (!children.empty()) {
                levels.push_back(std::move(children));
                ++gain;
                cerr << "V48_CUDA_FRONTIER_GAIN size=" << levels.back()[0].blocks.size()
                     << " siblings=" << levels.back().size() << "\n";
                write_manifest(manifest, n, d, w, levels, "forward gain");
                continue;
            }

            // No child from the newest frontier. Look back to older ancestors,
            // generate fresh siblings there, and propagate those branches forward.
            bool escaped = false;
            for (int lb_epoch = 0; lb_epoch < lookback_epochs && !escaped; ++lb_epoch) {
                int newest = static_cast<int>(levels.size()) - 1;
                int min_level = max(0, newest - lookback_depth);

                for (int back = newest - 1; back >= min_level && !escaped; --back) {
                    cerr << "V48_CUDA_LOOKBACK epoch=" << lb_epoch
                         << " ancestor_level=" << back
                         << " ancestor_size=" << levels[back][0].blocks.size() << "\n";

                    // Refresh the child level from this ancestor with a new epoch.
                    vector<BranchNode> existing = levels[back + 1];
                    auto refreshed = expand_parents(
                        levels[back], n, d, w,
                        seconds, threads,
                        rounds_per_parent, beam_width,
                        master_seed, epochs[back]++, output_dir,
                        back + 1, existing);
                    levels[back + 1] = std::move(refreshed);

                    // Propagate the refreshed siblings back up to the current frontier.
                    for (int k = back + 1; k < newest; ++k) {
                        vector<BranchNode> existing_next = levels[k + 1];
                        auto propagated = expand_parents(
                            levels[k], n, d, w,
                            seconds, threads,
                            rounds_per_parent, beam_width,
                            master_seed, epochs[k]++, output_dir,
                            k + 1, existing_next);
                        levels[k + 1] = std::move(propagated);
                    }

                    // Re-attack the stalled frontier from the enriched newest level.
                    auto escaped_children = expand_parents(
                        levels[newest], n, d, w,
                        seconds, threads,
                        rounds_per_parent, beam_width,
                        master_seed, epochs[newest]++, output_dir,
                        newest + 1);
                    if (!escaped_children.empty()) {
                        levels.push_back(std::move(escaped_children));
                        ++gain;
                        escaped = true;
                        cerr << "V48_CUDA_LOOKBACK_ESCAPE size=" << levels.back()[0].blocks.size()
                             << " siblings=" << levels.back().size() << "\n";
                        write_manifest(manifest, n, d, w, levels, "look-back escape");
                    }
                }
            }

            if (!escaped) {
                stopped = true;
                write_manifest(manifest, n, d, w, levels, "bounded stall");
            }
        }

        const auto& best_level = levels.back();
        int best_m = static_cast<int>(best_level[0].blocks.size());
        string final_path = output_dir + "/A_" + to_string(n) + "_" + to_string(d)
            + "_" + to_string(w) + "_" + to_string(best_m) + "_ATS_v4_8_CUDA_FINAL.txt";
        Config final_cfg = make_config(n, d, w, best_m);
        write_code(best_level[0].blocks, final_cfg, final_path,
                   "ATS v4.8 CUDA staged scouting/exact-state promotion; selected retained sibling");

        cout << "ATS_V4_8_CUDA_DONE base=" << initial.size()
             << " best=" << best_m
             << " gain=" << (best_m - static_cast<int>(initial.size()))
             << " retained_siblings=" << best_level.size()
             << " final=" << final_path
             << " manifest=" << manifest << "\n";
        return 0;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << "\n";
        return 70;
    }
}
