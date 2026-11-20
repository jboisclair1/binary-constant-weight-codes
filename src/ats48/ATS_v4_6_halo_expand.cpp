#include <bits/stdc++.h>

using namespace std;

#ifndef ATS_HALO_VERSION_LABEL
#define ATS_HALO_VERSION_LABEL "ATS 4.6"
#endif

static int pc(uint64_t value) { return __builtin_popcountll(value); }

struct Geometry {
    int n;
    int d;
    int w;
    int overlap_limit;
    int conflict_subset;
};

static vector<int> coordinates(uint64_t mask, int n) {
    vector<int> result;
    for (int coordinate = 0; coordinate < n; ++coordinate)
        if ((mask >> coordinate) & 1ULL) result.push_back(coordinate);
    return result;
}

static void combinations_recursive(
    const vector<int>& values, int needed, int at, uint64_t mask,
    vector<uint64_t>& output) {
    if (!needed) { output.push_back(mask); return; }
    for (int i = at; i <= static_cast<int>(values.size()) - needed; ++i)
        combinations_recursive(
            values, needed - 1, i + 1, mask | (1ULL << values[i]), output);
}

static vector<uint64_t> combinations(const vector<int>& values, int needed) {
    vector<uint64_t> output;
    combinations_recursive(values, needed, 0, 0, output);
    return output;
}

static vector<uint64_t> conflict_subsets(uint64_t mask, const Geometry& geometry) {
    return combinations(coordinates(mask, geometry.n), geometry.conflict_subset);
}

static vector<uint64_t> parse_code(const string& path, const Geometry& geometry) {
    ifstream input(path);
    if (!input) throw runtime_error("cannot open code: " + path);
    vector<uint64_t> blocks;
    string line;
    int base = 0;
    bool global_zero_based = false;
    static const regex number("[0-9]+");
    while (getline(input, line)) {
        if (line.find("zero-based") != string::npos
            || line.find("ZERO-BASED") != string::npos) global_zero_based = true;
        string content = line;
        const auto comment = content.find('#');
        if (comment != string::npos) content = content.substr(0, comment);
        string compact;
        for (char ch : content)
            if (!isspace(static_cast<unsigned char>(ch)) && ch != '\\') compact.push_back(ch);
        if (compact.empty()) continue;
        if (compact.rfind("$BASE=", 0) == 0) {
            base = stoi(compact.substr(6));
            if (base != 2 && base != 16) throw runtime_error("unsupported $BASE");
            continue;
        }
        uint64_t mask = 0;
        if (base == 16) {
            size_t used = 0;
            mask = stoull(compact, &used, 16);
            if (used != compact.size()) throw runtime_error("invalid hexadecimal codeword");
        } else if ((base == 2 || base == 0)
            && static_cast<int>(compact.size()) == geometry.n
            && all_of(compact.begin(), compact.end(), [](char ch) { return ch == '0' || ch == '1'; })) {
            for (int i = 0; i < geometry.n; ++i)
                if (compact[i] == '1') mask |= 1ULL << i;
        } else {
            vector<int> values;
            for (sregex_iterator it(compact.begin(), compact.end(), number), end;
                 it != end; ++it) values.push_back(stoi(it->str()));
            if (values.empty()) continue;
            const bool zero_based = global_zero_based
                || find(values.begin(), values.end(), 0) != values.end();
            for (int value : values) {
                const int coordinate = zero_based ? value : value - 1;
                if (coordinate < 0 || coordinate >= geometry.n)
                    throw runtime_error("coordinate outside code geometry");
                mask |= 1ULL << coordinate;
            }
        }
        if (pc(mask) != geometry.w) throw runtime_error("codeword has wrong weight");
        blocks.push_back(mask);
    }
    return blocks;
}

static bool verify(const vector<uint64_t>& blocks, const Geometry& geometry) {
    unordered_set<uint64_t> distinct;
    for (uint64_t block : blocks)
        if (pc(block) != geometry.w || !distinct.insert(block).second) return false;
    for (size_t i = 0; i < blocks.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (pc(blocks[i] & blocks[j]) > geometry.overlap_limit) return false;
    return true;
}

static int repeated_shadow_score_capped_at_two(
    const vector<uint64_t>& blocks,
    const Geometry& geometry) {
    int score = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            const int overlap = pc(blocks[i] & blocks[j]);
            if (overlap < geometry.conflict_subset) continue;
            if (overlap > geometry.conflict_subset || ++score > 1) return 2;
        }
    }
    return score;
}

static void write_code(
    const vector<uint64_t>& blocks, const string& path, const string& header,
    const Geometry& geometry) {
    ofstream output(path);
    if (!output) throw runtime_error("cannot write output: " + path);
    output << "# " << header << '\n';
    for (uint64_t mask : blocks) {
        const auto values = coordinates(mask, geometry.n);
        output << '{';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) output << ',';
            output << values[i] + 1;
        }
        output << "}\n";
    }
}

struct OwnerIndex {
    const Geometry& geometry;
    unordered_map<uint64_t, int> owner;

    explicit OwnerIndex(const Geometry& value) : geometry(value) {}

    void build(const vector<uint64_t>& blocks) {
        owner.clear();
        for (int id = 0; id < static_cast<int>(blocks.size()); ++id)
            for (uint64_t subset : conflict_subsets(blocks[id], geometry)) {
                const auto [it, inserted] = owner.emplace(subset, id);
                if (!inserted) throw runtime_error("invalid overlap in endpoint core");
            }
    }

    int blocker_count(uint64_t candidate, int& one) const {
        vector<int> blockers;
        for (uint64_t subset : conflict_subsets(candidate, geometry)) {
            const auto it = owner.find(subset);
            if (it == owner.end()) continue;
            if (find(blockers.begin(), blockers.end(), it->second) == blockers.end()) {
                blockers.push_back(it->second);
                if (blockers.size() > 1) return static_cast<int>(blockers.size());
            }
        }
        one = blockers.empty() ? -1 : blockers.front();
        return static_cast<int>(blockers.size());
    }
};

static uint64_t fingerprint(vector<uint64_t> blocks) {
    sort(blocks.begin(), blocks.end());
    uint64_t hash = 1469598103934665603ULL;
    for (uint64_t block : blocks) {
        hash ^= block + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        hash *= 1099511628211ULL;
    }
    return hash;
}

int main(int argc, char** argv) {
    try {
        if (argc < 8) {
            cerr << "usage: near outdir maxswap root n d w [direct_only]\n";
            return 2;
        }
        const string near_path = argv[1];
        const string output_dir = argv[2];
        const int max_swap = stoi(argv[3]);
        const string root_path = argv[4];
        const Geometry geometry{
            stoi(argv[5]), stoi(argv[6]), stoi(argv[7]),
            stoi(argv[7]) - (stoi(argv[6]) + 1) / 2,
            stoi(argv[7]) - (stoi(argv[6]) + 1) / 2 + 1};
        const bool direct_only = argc >= 9 && stoi(argv[8]) != 0;
        if (geometry.n < 1 || geometry.n > 64 || geometry.w < 1
            || geometry.w >= geometry.n || geometry.d < 1
            || geometry.conflict_subset < 1 || geometry.conflict_subset > geometry.w)
            throw runtime_error("unsupported geometry");

        filesystem::create_directories(output_dir);
        const auto near_code = parse_code(near_path, geometry);
        const auto root = parse_code(root_path, geometry);
        const size_t target_size = near_code.size();
        if (target_size < 2 || root.size() + 1 != target_size) return 3;
        const unordered_set<uint64_t> root_set(root.begin(), root.end());

        vector<pair<int, int>> conflicts;
        for (int i = 0; i < static_cast<int>(near_code.size()); ++i)
            for (int j = 0; j < i; ++j)
                if (pc(near_code[i] & near_code[j]) > geometry.overlap_limit)
                    conflicts.push_back({j, i});
        if (conflicts.size() != 1
            || repeated_shadow_score_capped_at_two(near_code, geometry) != 1)
            return 3;

        unordered_set<uint64_t> seen;
        int emitted = 0;
        const auto [first, second] = conflicts.front();
        for (int orientation = 0; orientation < 2; ++orientation) {
            const int drop = orientation ? second : first;
            const int retained = orientation ? first : second;
            const uint64_t endpoint = near_code[drop];
            const uint64_t repair_base = near_code[retained];
            vector<uint64_t> core;
            for (int i = 0; i < static_cast<int>(near_code.size()); ++i)
                if (i != drop && i != retained) core.push_back(near_code[i]);
            if (!verify(core, geometry)) throw runtime_error("endpoint core is invalid");

            OwnerIndex index(geometry);
            index.build(core);
            const unordered_set<uint64_t> core_set(core.begin(), core.end());
            const auto inside = coordinates(repair_base, geometry.n);
            vector<int> outside;
            for (int coordinate = 0; coordinate < geometry.n; ++coordinate)
                if (!((repair_base >> coordinate) & 1ULL)) outside.push_back(coordinate);

            const int effective_max_swap = min({max_swap, geometry.w, geometry.n - geometry.w});
            for (int swap = 1; swap <= effective_max_swap; ++swap) {
                const auto removals = combinations(inside, swap);
                const auto additions = combinations(outside, swap);
                long long tested = 0;
                long long one_blocker = 0;
                for (uint64_t removal : removals) for (uint64_t addition : additions) {
                    ++tested;
                    const uint64_t candidate = (repair_base & ~removal) | addition;
                    if (candidate == endpoint || core_set.count(candidate)
                        || pc(candidate & endpoint) > geometry.overlap_limit) continue;
                    int blocker = -1;
                    const int count = index.blocker_count(candidate, blocker);
                    if (count == 0) {
                        vector<uint64_t> child = core;
                        child.push_back(endpoint);
                        child.push_back(candidate);
                        if (child.size() == target_size && verify(child, geometry)) {
                            const string path = output_dir + "/A_" + to_string(geometry.n)
                                + "_" + to_string(geometry.d) + "_" + to_string(geometry.w)
                                + "_" + to_string(target_size) + "_DIRECT.txt";
                            write_code(child, path, string(ATS_HALO_VERSION_LABEL) + " generic multiswap HALO direct child", geometry);
                            cerr << "FOUND orient=" << orientation << " sw=" << swap
                                 << " tested=" << tested << "\n";
                            return 0;
                        }
                    } else if (count == 1 && !direct_only) {
                        ++one_blocker;
                        vector<uint64_t> sibling;
                        sibling.reserve(target_size - 1);
                        for (int i = 0; i < static_cast<int>(core.size()); ++i)
                            if (i != blocker) sibling.push_back(core[i]);
                        sibling.push_back(endpoint);
                        sibling.push_back(candidate);
                        if (!verify(sibling, geometry) || sibling.size() + 1 != target_size)
                            throw runtime_error("neutral sibling is invalid");
                        const uint64_t hash = fingerprint(sibling);
                        if (!seen.insert(hash).second) continue;
                        int novelty = 0;
                        for (uint64_t block : sibling)
                            if (!root_set.count(block)) ++novelty;
                        const string path = output_dir + "/halo_o" + to_string(orientation)
                            + "_s" + to_string(swap) + "_n" + to_string(novelty)
                            + "_" + to_string(emitted++) + ".txt";
                        write_code(sibling, path, string(ATS_HALO_VERSION_LABEL) + " generic multiswap HALO sibling", geometry);
                    }
                }
                cerr << "orient=" << orientation << " sw=" << swap << " tested=" << tested
                     << " e1raw=" << one_blocker << " emitted_total=" << emitted << "\n";
            }
        }
        cerr << "DONE emitted=" << emitted << "\n";
        return 1;
    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << "\n";
        return 70;
    }
}
