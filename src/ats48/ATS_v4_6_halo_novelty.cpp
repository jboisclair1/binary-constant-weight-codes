#include <bits/stdc++.h>

using namespace std;

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

static void subset_masks_recursive(
    const vector<int>& values, int needed, int at, uint64_t mask,
    vector<uint64_t>& output) {
    if (!needed) { output.push_back(mask); return; }
    for (int i = at; i <= static_cast<int>(values.size()) - needed; ++i)
        subset_masks_recursive(
            values, needed - 1, i + 1, mask | (1ULL << values[i]), output);
}

static vector<uint64_t> conflict_subsets(uint64_t mask, const Geometry& geometry) {
    const auto values = coordinates(mask, geometry.n);
    vector<uint64_t> output;
    subset_masks_recursive(values, geometry.conflict_subset, 0, 0, output);
    return output;
}

static vector<uint64_t> parse_code(const string& path, const Geometry& geometry) {
    ifstream input(path);
    if (!input) throw runtime_error("cannot open seed: " + path);
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
                    throw runtime_error("coordinate outside seed geometry");
                mask |= 1ULL << coordinate;
            }
        }
        if (pc(mask) != geometry.w) throw runtime_error("seed codeword has wrong weight");
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
                if (!inserted) throw runtime_error("seed contains an illegal overlap");
            }
    }

    void remove(uint64_t block, int id) {
        for (uint64_t subset : conflict_subsets(block, geometry)) {
            const auto it = owner.find(subset);
            if (it == owner.end() || it->second != id)
                throw runtime_error("owner index removal mismatch");
            owner.erase(it);
        }
    }

    void add(uint64_t block, int id) {
        for (uint64_t subset : conflict_subsets(block, geometry)) {
            const auto [it, inserted] = owner.emplace(subset, id);
            if (!inserted) throw runtime_error("owner index insertion conflict");
        }
    }

    vector<int> blockers(uint64_t candidate) const {
        vector<int> result;
        for (uint64_t subset : conflict_subsets(candidate, geometry)) {
            const auto it = owner.find(subset);
            if (it == owner.end()) continue;
            if (find(result.begin(), result.end(), it->second) == result.end())
                result.push_back(it->second);
        }
        return result;
    }
};

static uint64_t mutate(
    uint64_t old, mt19937_64& rng, const Geometry& geometry, int changes) {
    uint64_t result = old;
    for (int step = 0; step < changes; ++step) {
        const auto inside = coordinates(result, geometry.n);
        const int removed = inside[uniform_int_distribution<int>(
            0, static_cast<int>(inside.size()) - 1)(rng)];
        result &= ~(1ULL << removed);
        int added;
        do {
            added = uniform_int_distribution<int>(0, geometry.n - 1)(rng);
        } while ((result >> added) & 1ULL);
        result |= 1ULL << added;
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc < 9) {
            cerr << "usage: seed outdir rng steps archive_every n d w\n";
            return 2;
        }
        const string seed_path = argv[1];
        const string output_dir = argv[2];
        const uint64_t rng_seed = stoull(argv[3]);
        const long long steps = stoll(argv[4]);
        const int archive_every = max(1, stoi(argv[5]));
        const Geometry geometry{
            stoi(argv[6]), stoi(argv[7]), stoi(argv[8]),
            stoi(argv[8]) - (stoi(argv[7]) + 1) / 2,
            stoi(argv[8]) - (stoi(argv[7]) + 1) / 2 + 1};
        if (geometry.n < 1 || geometry.n > 64 || geometry.w < 1
            || geometry.w >= geometry.n || geometry.d < 1
            || geometry.conflict_subset < 1 || geometry.conflict_subset > geometry.w)
            throw runtime_error("unsupported geometry; require 1<=n<=64, 1<=w<n, and a positive overlap bound");

        filesystem::create_directories(output_dir);
        auto current = parse_code(seed_path, geometry);
        if (current.empty() || !verify(current, geometry))
            throw runtime_error("invalid seed code");
        const auto root = current;
        const unordered_set<uint64_t> root_set(root.begin(), root.end());
        unordered_set<uint64_t> current_set(current.begin(), current.end());
        OwnerIndex index(geometry);
        index.build(current);
        mt19937_64 rng(rng_seed);
        int novelty = 0;
        int best_novelty = 0;
        int archive_id = 0;
        long long accepted = 0;
        const int max_changes = min({3, geometry.w, geometry.n - geometry.w});

        for (long long iteration = 1; iteration <= steps; ++iteration) {
            const int block_id = uniform_int_distribution<int>(
                0, static_cast<int>(current.size()) - 1)(rng);
            const uint64_t old = current[block_id];
            uint64_t candidate = 0;
            bool got = false;
            for (int attempt = 0; attempt < 80; ++attempt) {
                const int requested = attempt < 25 ? 1 : attempt < 60 ? 2 : 3;
                const int changes = min(requested, max_changes);
                const uint64_t proposal = mutate(old, rng, geometry, changes);
                if (proposal == old || current_set.count(proposal)) continue;
                const auto blockers = index.blockers(proposal);
                if (blockers.empty()) {
                    auto child = current;
                    child.push_back(proposal);
                    if (verify(child, geometry)) {
                        const string path = output_dir + "/A_" + to_string(geometry.n)
                            + "_" + to_string(geometry.d) + "_" + to_string(geometry.w)
                            + "_" + to_string(child.size()) + "_DIRECT.txt";
                        write_code(child, path, "ATS 4.6 generic novelty direct child", geometry);
                        cerr << "DIRECT size=" << child.size() << " iter=" << iteration
                             << " file=" << path << "\n";
                        return 0;
                    }
                }
                if (blockers.size() == 1 && blockers[0] == block_id) {
                    candidate = proposal;
                    got = true;
                    break;
                }
            }
            if (!got) continue;

            const int delta = (root_set.count(candidate) ? 0 : 1)
                - (root_set.count(old) ? 0 : 1);
            const bool accept = delta > 0 || (delta == 0 && rng() % 100 < 60)
                || (delta < 0 && rng() % 10000 < 3);
            if (!accept) continue;
            index.remove(old, block_id);
            current_set.erase(old);
            current[block_id] = candidate;
            current_set.insert(candidate);
            index.add(candidate, block_id);
            novelty += delta;
            ++accepted;

            if (novelty > best_novelty) {
                best_novelty = novelty;
                cerr << "BESTNOV " << best_novelty << " iter=" << iteration
                     << " accepted=" << accepted << "\n";
                if (best_novelty % archive_every == 0 || best_novelty < archive_every) {
                    const string path = output_dir + "/sibling_nov_"
                        + to_string(best_novelty) + "_" + to_string(archive_id++) + ".txt";
                    write_code(current, path, "ATS 4.6 generic novelty-polished sibling", geometry);
                }
            }
            if (accepted % 2000 == 0 && !verify(current, geometry))
                throw runtime_error("novelty walk verification failed");
        }

        const string path = output_dir + "/best_sibling.txt";
        write_code(current, path, "ATS 4.6 generic novelty-polished final sibling", geometry);
        cerr << "DONE bestnov=" << best_novelty << " finalnov=" << novelty
             << " accepted=" << accepted << "\n";
        return 1;
    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << "\n";
        return 70;
    }
}
