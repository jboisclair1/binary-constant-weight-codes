// ATS 4.9 "Shadow Pressure": target-size constant-weight-code search with
// learned conflict weights (breakout-style landscape reshaping).
//
// ATS 4.8 minimises the static score S(C) = number of repeated critical
// shadows and escapes stalls with annealing/kicks. Its documented failure mode
// is the defect-1 terminal basin: the last conflict cannot be told apart from
// a repairable one, so the walker keeps returning to it.
//
// Here every conflict is identified by the intersection I = B_i & B_j that
// causes it (|I| >= t). Whenever the search sits in a local minimum, the
// intersections of the conflicts it is stuck on are charged one unit of
// "pressure". Moves are scored by  sum (|I|-t+1) + P[I], so a conflict that
// keeps reappearing in the same place becomes progressively more expensive
// and the walker is pushed to trade it for conflicts in fresh places. The
// pressure table is learned per restart and slowly decays.
//
// Build:  cl /nologo /std:c++20 /O2 /EHsc ats49_pressure.cpp
// Run:    ats49_pressure seed.txt n d w seconds threads outdir rngseed [opts]
//   --no-weights        static score + ATS-style annealing (ablation baseline)
//   --decay K           decrement all pressures once every K charge events; 0 = portfolio
//                       (default: per-thread portfolio 4/16/64/256/1024 + 1 static in 6)
//   --noise P           probability of taking a worsening move at a minimum
//   --restart R         restart after R iterations without a new run best
//   --plus K            add K codewords per level instead of 1
//   --max-gain G        stop after G codewords gained (0 = unlimited)
//   --rng-swap-seconds S  retry with the next RNG seed after S seconds without a gain
//                         (0 = off; when enabled, seconds is the total search budget)
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
using namespace std;
typedef uint64_t u64;
static inline int pc(u64 x) { return (int)std::popcount(x); }

struct Params {
    int n = 0, d = 0, w = 0, t = 0;
    double seconds = 60;
    double rngSwapSeconds = 0;
    int threads = 1;
    u64 seed = 1;
    bool weights = true;
    int decayEvery = 64;
    bool portfolio = true;
    double noise = 0.02;
    long long restart = 3000000;
    int plus = 1;
    int maxGain = 0;
    string outdir = ".";
};

static vector<u64> parse_code(const string& path, int n, int w) {
    ifstream f(path);
    if (!f) throw runtime_error("cannot open " + path);
    vector<u64> blocks;
    string line;
    int base = 0;
    bool global_zero = false;
    static const regex re("[0-9]+");
    while (getline(f, line)) {
        if (line.find("zero-based") != string::npos || line.find("ZERO-BASED") != string::npos) global_zero = true;
        string s = line;
        auto h = s.find('#');
        if (h != string::npos) s = s.substr(0, h);
        string c;
        for (char ch : s)
            if (!isspace((unsigned char)ch) && ch != '\\') c.push_back(ch);
        if (c.empty()) continue;
        if (c.rfind("$BASE=", 0) == 0) { base = stoi(c.substr(6)); continue; }
        u64 mask = 0;
        if (base == 16) {
            mask = stoull(c, nullptr, 16);
        } else if ((int)c.size() == n && all_of(c.begin(), c.end(), [](char ch) { return ch == '0' || ch == '1'; })) {
            for (int i = 0; i < n; ++i) if (c[i] == '1') mask |= 1ULL << i;
        } else {
            vector<int> a;
            for (sregex_iterator it(c.begin(), c.end(), re), e; it != e; ++it) a.push_back(stoi(it->str()));
            if (a.empty()) continue;
            bool zero = global_zero || find(a.begin(), a.end(), 0) != a.end();
            for (int x : a) {
                int p = zero ? x : x - 1;
                if (p < 0 || p >= n) throw runtime_error("coordinate out of range in " + path);
                mask |= 1ULL << p;
            }
        }
        if (pc(mask) != w) throw runtime_error("wrong weight in " + path + " line: " + line);
        blocks.push_back(mask);
    }
    return blocks;
}

static bool verify(const vector<u64>& b, const Params& p, string* why = nullptr) {
    unordered_set<u64> seen;
    for (size_t i = 0; i < b.size(); ++i) {
        if (pc(b[i]) != p.w) { if (why) *why = "weight"; return false; }
        if (p.n < 64 && (b[i] >> p.n)) { if (why) *why = "range"; return false; }
        if (!seen.insert(b[i]).second) { if (why) *why = "duplicate"; return false; }
        for (size_t j = 0; j < i; ++j)
            if (pc(b[i] & b[j]) >= p.t) { if (why) *why = "pair " + to_string(j) + "," + to_string(i); return false; }
    }
    return true;
}

static void write_code(const vector<u64>& b, const Params& p, const string& path, const string& note) {
    auto v = b;
    sort(v.begin(), v.end());
    ofstream f(path);
    if (!f) throw runtime_error("cannot write " + path);
    f << "# " << v.size() << " constant-weight codewords; " << note << "\n# A(" << p.n << "," << p.d << "," << p.w
      << ") >= " << v.size() << "; one-based supports\n\n";
    for (u64 m : v) {
        f << "{";
        bool first = true;
        for (int x = 0; x < p.n; ++x)
            if (m >> x & 1ULL) { if (!first) f << ","; first = false; f << x + 1; }
        f << "}\n";
    }
}

struct Shared {
    static constexpr int unobservedDefect = 1 << 30;
    atomic<bool> found{false};
    atomic<int> best{unobservedDefect};
    atomic<long long> iters{0}, bumps{0}, restarts{0}, defect1{0};
    mutex m;
    vector<u64> solution;
    chrono::steady_clock::time_point deadline, started;

    void observe_defect(int defect) {
        int previous = best.load();
        while (defect < previous && !best.compare_exchange_weak(previous, defect)) {}
    }
};

static const int HB = 20;
static inline uint32_t hkey(u64 m) { return (uint32_t)((m * 0x9E3779B97F4A7C15ULL) >> (64 - HB)); }

struct Walker {
    Params p;
    Shared& g;
    mt19937_64 r;
    int M = 0;
    vector<u64> B;
    vector<int> cc, confList, confPos;
    long long nconf = 0;
    vector<uint16_t> W;
    vector<uint32_t> touched;
    vector<long long> tabu;
    int delta[64][64];
    long long bumpsSinceDecay = 0;

    Walker(const Params& p_, Shared& g_, u64 seed) : p(p_), g(g_), r(seed), W(1u << HB, 0) {}

    u64 random_block() {
        int a[64];
        for (int i = 0; i < p.n; ++i) a[i] = i;
        u64 m = 0;
        for (int i = 0; i < p.w; ++i) {
            int j = i + (int)(r() % (u64)(p.n - i));
            swap(a[i], a[j]);
            m |= 1ULL << a[i];
        }
        return m;
    }
    void conf_inc(int i) { if (cc[i]++ == 0) { confPos[i] = (int)confList.size(); confList.push_back(i); } }
    void conf_dec(int i) {
        if (--cc[i] == 0) {
            int pos = confPos[i], last = confList.back();
            confList[pos] = last; confPos[last] = pos; confList.pop_back(); confPos[i] = -1;
        }
    }
    bool swap_budget_spent() const {
        return p.rngSwapSeconds > 0 && (g.found.load() || chrono::steady_clock::now() >= g.deadline);
    }
    bool init(const vector<u64>& base) {
        B = base;
        for (int k = 0; k < p.plus; ++k) {
            u64 bestm = 0; int bestc = 1 << 30, ties = 0;
            for (int s = 0; s < 4000; ++s) {
                if ((s & 63) == 0 && swap_budget_spent()) return false;
                u64 m = random_block();
                int c = 0;
                for (u64 x : B) { int z = pc(m & x); if (z >= p.t) c += z - p.t + 1; if (c > bestc) break; }
                if (c < bestc) { bestc = c; bestm = m; ties = 1; }
                else if (c == bestc && r() % ++ties == 0) bestm = m;
            }
            B.push_back(bestm);
        }
        M = (int)B.size();
        cc.assign(M, 0); confPos.assign(M, -1); confList.clear(); nconf = 0;
        for (int i = 0; i < M; ++i) {
            if ((i & 63) == 0 && swap_budget_spent()) return false;
            for (int j = 0; j < i; ++j)
                if (pc(B[i] & B[j]) >= p.t) { conf_inc(i); conf_inc(j); ++nconf; }
        }
        for (uint32_t k : touched) W[k] = 0;
        touched.clear();
        tabu.assign((size_t)M * p.n, 0);
        bumpsSinceDecay = 0;
        return !swap_budget_spent();
    }
    inline int cost(int z, u64 I) const { return (z - p.t + 1) + (p.weights ? W[hkey(I)] : 0); }

    // Fill delta[a][b] for every swap a->b on block bi; returns a random conflict partner.
    int evaluate(int bi) {
        const u64 Bm = B[bi];
        const int t = p.t, n = p.n;
        for (u64 x = Bm; x; x &= x - 1) memset(delta[countr_zero(x)], 0, sizeof(int) * n);
        const u64 full = n == 64 ? ~0ULL : ((1ULL << n) - 1);
        const u64 outside = full & ~Bm;
        int partner = -1, seen = 0;
        for (int j = 0; j < M; ++j) {
            if (j == bi) continue;
            const u64 X = B[j], I = Bm & X;
            const int z = pc(I);
            if (z < t - 1) continue;
            if (z == t - 1) {
                const u64 Bo = Bm & ~X;
                for (u64 xb = X & ~Bm; xb; xb &= xb - 1) {
                    const int b = countr_zero(xb);
                    const int c = cost(t, I | (1ULL << b));
                    for (u64 xa = Bo; xa; xa &= xa - 1) delta[countr_zero(xa)][b] += c;
                }
            } else {
                if (r() % ++seen == 0) partner = j;
                const int cur = cost(z, I);
                for (u64 xa = Bm; xa; xa &= xa - 1) {
                    const int a = countr_zero(xa);
                    const int ain = (int)(X >> a & 1ULL);
                    for (u64 xb = outside; xb; xb &= xb - 1) {
                        const int b = countr_zero(xb);
                        const int bin = (int)(X >> b & 1ULL);
                        if (!ain && !bin) continue;
                        const int z2 = z - ain + bin;
                        int nc = 0;
                        if (z2 >= t) nc = cost(z2, (I & ~(1ULL << a)) | ((1ULL << b) & X));
                        delta[a][b] += nc - cur;
                    }
                }
            }
        }
        return partner;
    }
    struct Mv { int bi = -1, a = -1, b = -1, d = INT32_MAX; };
    void pick(int bi, long long it, Mv& best, int& ties) {
        const u64 Bm = B[bi];
        const u64 full = p.n == 64 ? ~0ULL : ((1ULL << p.n) - 1);
        const long long* tb = &tabu[(size_t)bi * p.n];
        for (u64 xa = Bm; xa; xa &= xa - 1) {
            const int a = countr_zero(xa);
            for (u64 xb = full & ~Bm; xb; xb &= xb - 1) {
                const int b = countr_zero(xb);
                const int dv = delta[a][b];
                if (dv >= 0 && tb[b] > it) continue;
                if (dv < best.d) { best = {bi, a, b, dv}; ties = 1; }
                else if (dv == best.d && r() % ++ties == 0) best = {bi, a, b, dv};
            }
        }
    }
    void apply(const Mv& m, long long it) {
        const u64 old = B[m.bi], nw = (old & ~(1ULL << m.a)) | (1ULL << m.b);
        for (int j = 0; j < M; ++j) {
            if (j == m.bi) continue;
            const bool zo = pc(old & B[j]) >= p.t, zn = pc(nw & B[j]) >= p.t;
            if (zo && !zn) { conf_dec(m.bi); conf_dec(j); --nconf; }
            else if (!zo && zn) { conf_inc(m.bi); conf_inc(j); ++nconf; }
        }
        B[m.bi] = nw;
        tabu[(size_t)m.bi * p.n + m.a] = it + 7 + (long long)(r() % 11);
    }
    void bump(int bi) {
        const u64 Bm = B[bi];
        for (int j = 0; j < M; ++j) {
            if (j == bi) continue;
            const u64 I = Bm & B[j];
            if (pc(I) < p.t) continue;
            const uint32_t k = hkey(I);
            if (W[k] == 0) touched.push_back(k);
            if (W[k] < 60000) ++W[k];
        }
        if (++bumpsSinceDecay >= p.decayEvery) {
            bumpsSinceDecay = 0;
            size_t o = 0;
            for (uint32_t k : touched) if (--W[k]) touched[o++] = k;
            touched.resize(o);
        }
    }
    void run(const vector<u64>& base) {
        uniform_real_distribution<double> U(0, 1);
        long long counted = 0, cbumps = 0;
        while (!g.found.load() && chrono::steady_clock::now() < g.deadline) {
            if (!init(base)) break;
            long long runBest = nconf, lastImp = 0;
            if (runBest > 0) g.observe_defect((int)runBest);
            bool wasD1 = false;
            for (long long it = 1;; ++it) {
                if ((it & 63) == 0 && swap_budget_spent()) break;
                if (nconf == 0) {
                    if (swap_budget_spent()) break;
                    string why;
                    if (!verify(B, p, &why)) throw runtime_error("zero-conflict state failed verify: " + why);
                    lock_guard<mutex> lock(g.m);
                    if (!g.found.load() && !swap_budget_spent()) {
                        g.solution = B; g.best.store(0); g.found.store(true);
                    }
                    break;
                }
                if (nconf < runBest) {
                    runBest = nconf; lastImp = it;
                    g.observe_defect((int)runBest);
                }
                if (nconf == 1) { if (!wasD1) { g.defect1.fetch_add(1, memory_order_relaxed); wasD1 = true; } }
                else if (nconf > 2) wasD1 = false;
                if ((it & 1023) == 0) {
                    g.iters.fetch_add(1024, memory_order_relaxed);
                    g.bumps.fetch_add(cbumps, memory_order_relaxed); cbumps = 0;
                    if (g.found.load() || chrono::steady_clock::now() >= g.deadline) break;
                    if (it - lastImp > p.restart) { g.restarts.fetch_add(1, memory_order_relaxed); break; }
                }
                const int bi = confList[r() % confList.size()];
                Mv best; int ties = 0;
                const int bj = evaluate(bi);
                pick(bi, it, best, ties);
                if (bj >= 0) { evaluate(bj); pick(bj, it, best, ties); }
                if (best.bi < 0) continue;
                bool accept;
                if (best.d < 0) accept = true;
                else if (p.weights) {
                    if (best.d == 0) accept = U(r) < 0.6;
                    else { bump(bi); if (bj >= 0) bump(bj); ++cbumps; accept = U(r) < p.noise; }
                } else {
                    const double temp = max(0.12, 1.8 * exp(-double(it % 120000) / 45000.0));
                    accept = best.d == 0 ? U(r) < 0.6 : U(r) < exp(-double(best.d) / temp);
                }
                if (accept) apply(best, it);
            }
            (void)counted;
        }
    }
};

int main(int argc, char** argv) {
    try {
        if (argc < 9) {
            cerr << "usage: ats49_pressure seed n d w seconds threads outdir rngseed [--no-weights] [--decay K] "
                    "[--noise P] [--restart R] [--plus K] [--max-gain G] [--rng-swap-seconds S]\n";
            return 64;
        }
        Params p;
        const string seedPath = argv[1];
        p.n = stoi(argv[2]); p.d = stoi(argv[3]); p.w = stoi(argv[4]);
        p.seconds = stod(argv[5]); p.threads = stoi(argv[6]);
        if (p.threads <= 0) p.threads = max(1u, thread::hardware_concurrency()); p.outdir = argv[7]; p.seed = stoull(argv[8]);
        for (int i = 9; i < argc; ++i) {
            string a = argv[i];
            if (a == "--no-weights") { p.weights = false; p.portfolio = false; }
            else if (a == "--decay") { int k = stoi(argv[++i]); if (k > 0) { p.decayEvery = k; p.portfolio = false; } }
            else if (a == "--noise") p.noise = stod(argv[++i]);
            else if (a == "--restart") p.restart = stoll(argv[++i]);
            else if (a == "--plus") p.plus = max(1, stoi(argv[++i]));
            else if (a == "--max-gain") p.maxGain = stoi(argv[++i]);
            else if (a == "--rng-swap-seconds") {
                if (i + 1 == argc) throw runtime_error("--rng-swap-seconds requires a value");
                size_t parsed = 0;
                const string interval = argv[++i];
                p.rngSwapSeconds = stod(interval, &parsed);
                if (parsed != interval.size() || !isfinite(p.rngSwapSeconds) || p.rngSwapSeconds < 0)
                    throw runtime_error("--rng-swap-seconds must be a finite nonnegative number");
            }
            else throw runtime_error("unknown option " + a);
        }
        if (p.n < 2 || p.n > 64 || p.d % 2 || p.d < 2 || p.d > 2 * p.w) throw runtime_error("unsupported family");
        p.t = p.w - p.d / 2 + 1;
        using Clock = chrono::steady_clock;
        const bool swapRng = p.rngSwapSeconds > 0;
        const auto runStarted = Clock::now();
        if (swapRng && (!isfinite(p.seconds) || p.seconds <= 0 ||
            p.seconds >= chrono::duration<double>(Clock::time_point::max() - runStarted).count()))
            throw runtime_error("RNG swapping requires a finite positive total runtime in range");
        const auto runtime = chrono::duration_cast<Clock::duration>(chrono::duration<double>(p.seconds));
        const auto attemptBudget = swapRng
            ? chrono::duration_cast<Clock::duration>(chrono::duration<double>(min(p.seconds, p.rngSwapSeconds)))
            : runtime;
        const auto runDeadline = swapRng ? runStarted + runtime : Clock::time_point::max();
        vector<u64> base = parse_code(seedPath, p.n, p.w);
        string why;
        if (!verify(base, p, &why)) throw runtime_error("seed is not a valid code: " + why);
        const int initial = (int)base.size();
        cout << "PRESSURE_CONFIG family=A(" << p.n << "," << p.d << "," << p.w << ") base=" << initial << " t=" << p.t
             << " weights=" << p.weights << " decay=" << (p.portfolio ? string("portfolio") : to_string(p.decayEvery)) << " noise=" << p.noise << " restart=" << p.restart
             << " plus=" << p.plus << " threads=" << p.threads << " seconds=" << p.seconds << " seed=" << p.seed
             << " rng_swap_seconds=" << p.rngSwapSeconds << endl;
        int level = 0;
        u64 activeSeed = p.seed;
        int bestAtTarget = Shared::unobservedDefect;
        while ((p.maxGain == 0 || (int)base.size() - initial < p.maxGain) && Clock::now() < runDeadline) {
            Params levelParams = p;
            if (p.maxGain > 0) {
                const int remainingGain = p.maxGain - ((int)base.size() - initial);
                levelParams.plus = min(p.plus, remainingGain);
            }
            const size_t target = base.size() + levelParams.plus;
            Shared g;
            g.best.store(bestAtTarget);
            g.started = Clock::now();
            g.deadline = g.started + min(attemptBudget, runDeadline - g.started);
            // A swap attempt starts the same RNG stream as a fresh invocation.
            const u64 attemptSeed = activeSeed * 1000003ULL + (swapRng ? 0 : (u64)level * 7919ULL);
            vector<thread> th;
            vector<string> errors(p.threads);
            for (int i = 0; i < p.threads; ++i)
                th.emplace_back([&, i] {
                    try {
                        Walker wk(levelParams, g, attemptSeed + (u64)i * 0x9E3779B97F4A7C15ULL + 1);
                        if (p.portfolio) {
                            // Mixed strategies: pressure walkers at several decay rates plus one static walker in six.
                            static const int rates[5] = {4, 16, 64, 256, 1024};
                            if (i % 6 == 5) wk.p.weights = false;
                            else wk.p.decayEvery = rates[i % 6];
                        }
                        wk.run(base);
                    } catch (const exception& e) { errors[i] = e.what(); g.found.store(true); }
                });
            // Heartbeat for front ends: one line every few seconds while walkers run.
            for (auto next = g.started + chrono::seconds(5); !g.found.load() && chrono::steady_clock::now() < g.deadline;) {
                if (swapRng) this_thread::sleep_until(min(g.deadline, Clock::now() + chrono::milliseconds(5)));
                else this_thread::sleep_for(chrono::milliseconds(50));
                if (Clock::now() < next) continue;
                next += chrono::seconds(5);
                cout << "PRESSURE_PROGRESS target=" << target << " best_defect=" << g.best.load()
                     << " iterations=" << g.iters.load() << " charges=" << g.bumps.load() << " defect1_visits="
                     << g.defect1.load() << " seconds="
                     << chrono::duration<double>(chrono::steady_clock::now() - g.started).count() << endl;
            }
            for (auto& x : th) x.join();
            for (auto& e : errors) if (!e.empty()) throw runtime_error(e);
            const double el = chrono::duration<double>(chrono::steady_clock::now() - g.started).count();
            const bool ok = !g.solution.empty();
            if (!ok && swapRng && Clock::now() < runDeadline) {
                bestAtTarget = g.best.load();
                ++activeSeed;
                cout << "PRESSURE_RNG_SWAP target=" << target << " best_defect=" << bestAtTarget
                     << " rng_seed=" << activeSeed << " seconds="
                     << chrono::duration<double>(Clock::now() - runStarted).count() << endl;
                continue;
            }
            cout << "PRESSURE_LEVEL base=" << base.size() << " target=" << target << " success=" << ok
                 << " best_defect=" << g.best.load() << " seconds=" << el << " iterations=" << g.iters.load()
                 << " charges=" << g.bumps.load() << " restarts=" << g.restarts.load()
                 << " defect1_visits=" << g.defect1.load() << " rng_seed=" << activeSeed << endl;
            if (!ok) break;
            if (!verify(g.solution, p, &why)) throw runtime_error("solution failed verification: " + why);
            base = g.solution;
            bestAtTarget = Shared::unobservedDefect;
            if (swapRng) sort(base.begin(), base.end()); // Match the order in a saved child used as the next input.
            ostringstream name;
            name << p.outdir << "/A_" << p.n << "_" << p.d << "_" << p.w << "_" << base.size() << "_ATS_4_9_PRESSURE.txt";
            write_code(base, p, name.str(), p.weights ? "ATS 4.9 shadow-pressure search" : "ATS 4.9 static-score ablation");
            cout << "PRESSURE_VERIFIED_CHILD size=" << base.size() << " A(" << p.n << "," << p.d << "," << p.w << ") >= "
                 << base.size() << " file=" << name.str() << endl;
            ++level;
        }
        cout << "PRESSURE_DONE base=" << initial << " best=" << base.size() << " gain=" << base.size() - initial << endl;
        return base.size() > (size_t)initial ? 0 : 2;
    } catch (const exception& e) {
        cerr << "PRESSURE_ERROR " << e.what() << endl;
        return 70;
    }
}
