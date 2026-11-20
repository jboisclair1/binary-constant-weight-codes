#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ats45cuda {

struct CandidateResult {
    std::uint64_t mask = 0;
    std::uint32_t score = 0;
    std::uint32_t reserved = 0;
};

class CandidateScorer {
public:
    CandidateScorer();
    ~CandidateScorer();
    CandidateScorer(CandidateScorer&&) noexcept;
    CandidateScorer& operator=(CandidateScorer&&) noexcept;
    CandidateScorer(const CandidateScorer&) = delete;
    CandidateScorer& operator=(const CandidateScorer&) = delete;

    bool initialize(
        const std::vector<std::uint64_t>& base_blocks,
        int n,
        int w,
        int critical_shadow_size,
        std::string& error);

    bool score_generated(
        std::uint64_t seed,
        std::uint64_t first_sample,
        std::size_t count,
        std::vector<CandidateResult>& results,
        std::string& error);

    bool score_masks(
        const std::vector<std::uint64_t>& masks,
        std::vector<CandidateResult>& results,
        std::string& error);

    const std::string& device_name() const;
    std::size_t maximum_batch_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct CapturedTerminalState {
    std::vector<std::uint64_t> blocks;
    std::uint64_t state_hash = 0;
    std::uint64_t iteration = 0;
    std::uint32_t defect = 1;
    std::uint32_t walker_id = UINT32_MAX;
};

struct CapturedLowDefectRecord {
    std::uint64_t state_hash = 0;
    std::uint64_t conflict_signature = 0;
    std::uint64_t iteration = 0;
    std::uint32_t defect = UINT32_MAX;
    std::uint32_t walker_id = UINT32_MAX;
    std::uint32_t move_class = 0;
};

struct RepairCandidate {
    std::uint64_t mask = 0;
    std::uint32_t blocker_count = 0;
    std::uint32_t blocker0 = UINT32_MAX;
    std::uint32_t blocker1 = UINT32_MAX;
};

struct CompatibilityScanResult {
    bool ran = false;
    bool exact = false;
    std::uint64_t universe_size = 0;
    std::uint64_t candidates_examined = 0;
    std::uint64_t qualifying_candidates = 0;
    std::uint64_t overflow_candidates = 0;
    double elapsed_seconds = 0;
    std::vector<RepairCandidate> candidates;
    std::string device_name;
    std::string error;
};

struct SearchResult {
    bool ran = false;
    std::uint32_t best_score = UINT32_MAX;
    std::vector<std::uint64_t> best_blocks;
    std::uint64_t walker_iterations = 0;
    std::uint64_t proposals_scored = 0;
    std::uint64_t accepted_moves = 0;
    std::uint64_t low_defect_events_dropped = 0;
    std::uint64_t low_defect_states_emitted = 0;
    std::uint64_t low_defect_walkers_screened = 0;
    std::uint64_t low_defect_state_copies = 0;
    bool incremental_shadow_scoring = false;
    std::uint64_t incremental_shadow_bytes = 0;
    std::uint32_t incremental_shadow_audits = 0;
    bool inverted_shadow_incidence = false;
    std::uint64_t inverted_shadow_incidence_bytes = 0;
    std::uint32_t walkers = 0;
    double elapsed_seconds = 0;
    double first_hit_seconds[5] = {-1.0, -1.0, -1.0, -1.0, -1.0};
    std::vector<std::uint64_t> defect2_state_hashes;
    std::vector<std::uint64_t> defect1_state_hashes;
    std::vector<std::uint64_t> basin_hashes;
    bool has_first_terminal_state = false;
    CapturedTerminalState first_terminal_state;
    std::vector<CapturedTerminalState> terminal_states;
    std::vector<CapturedLowDefectRecord> low_defect_records;
    std::string device_name;
    std::string error;
};

SearchResult run_swap_search(
    const std::vector<std::uint64_t>& base_blocks,
    const std::vector<std::uint64_t>& extra_blocks,
    int n,
    int w,
    int critical_shadow_size,
    std::uint32_t walkers,
    std::uint32_t proposals_per_iteration,
    std::uint64_t seed,
    std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool>* stop,
    bool stop_on_first_defect1,
    const std::function<void(const CapturedTerminalState&)>& low_defect_callback);

CompatibilityScanResult scan_repair_candidates(
    const std::vector<std::uint64_t>& valid_core,
    int n,
    int w,
    int critical_shadow_size,
    std::uint32_t max_blockers,
    std::uint32_t output_capacity);

bool runtime_available(std::string& device_name, std::string& error);

} // namespace ats45cuda
