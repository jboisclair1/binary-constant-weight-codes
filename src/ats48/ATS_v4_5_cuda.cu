#include "ATS_v4_5_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <vector>

#ifndef ATS48_DEVICE_DEDUP_HANDOFF
#define ATS48_DEVICE_DEDUP_HANDOFF 0
#endif

#ifndef ATS48_DEVICE_DEFECT2_INTENSIFICATION
#define ATS48_DEVICE_DEFECT2_INTENSIFICATION 0
#endif

#ifndef ATS48_INCREMENTAL_SHADOW_SCORING
#define ATS48_INCREMENTAL_SHADOW_SCORING 0
#endif

#ifndef ATS48_INVERTED_SHADOW_INCIDENCE
#define ATS48_INVERTED_SHADOW_INCIDENCE 0
#endif

// Compile-time isolated experiment for the A(24,6,12) record search.  Ordinary
// ATS builds leave this disabled, preserving the established move kernel.
#ifndef ATS_A24_RICH_MOVES
#define ATS_A24_RICH_MOVES 0
#endif

#ifndef ATS_A24_RICH_PERIOD
#define ATS_A24_RICH_PERIOD 4
#endif

namespace {

constexpr std::uint32_t kCandidateThreads = 256;
constexpr std::uint32_t kWalkThreads = 256;
constexpr std::uint32_t kRepairThreads = 256;
constexpr std::uint32_t kBadCapacity = 256;
constexpr std::uint16_t kNoShadowNode = UINT16_MAX;
constexpr std::uint32_t kIncidenceMinimumTargetM = 512;
#ifndef ATS48_ITERATIONS_PER_LAUNCH
#define ATS48_ITERATIONS_PER_LAUNCH 8
#endif
constexpr std::uint32_t kIterationsPerLaunch = ATS48_ITERATIONS_PER_LAUNCH;
constexpr std::uint32_t kLowDefectCapacity = 1U << 20;
constexpr std::uint32_t kRepairRanksPerLaunch = 8U * 1024U * 1024U;
// A shadow with k owners induces C(k,2) distinct bad pairs.  The maintained
// 256-edge cap therefore implies k <= 23, so an 8-bit counter is exact.
using ShadowCount = std::uint8_t;

// CUDA 13 can target the installed VS 2019 host compiler, but that pairing
// cannot enable the C++20 <bit> operations in a .cu translation unit. These
// small host-side equivalents keep the CUDA backend C++17-compatible.
std::uint32_t host_popcount64(std::uint64_t value) {
    std::uint32_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

std::uint32_t host_countr_zero64(std::uint64_t value) {
    std::uint32_t count = 0;
    while ((value & 1ULL) == 0) {
        value >>= 1;
        ++count;
    }
    return count;
}

std::string cuda_error(cudaError_t error) {
    return std::string(cudaGetErrorName(error)) + ": " + cudaGetErrorString(error);
}

bool check_cuda(cudaError_t status, const char* operation, std::string& error) {
    if (status == cudaSuccess) return true;
    error = std::string(operation) + " failed: " + cuda_error(status);
    return false;
}

std::string cuda_device_name(int device) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return "CUDA device";
    return properties.name;
}

__host__ __device__ std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__host__ __device__ std::uint32_t random_word(
    std::uint64_t seed,
    std::uint64_t index,
    std::uint32_t draw)
{
    const std::uint32_t lo = static_cast<std::uint32_t>(index);
    const std::uint32_t hi = static_cast<std::uint32_t>(index >> 32);
    std::uint32_t value = static_cast<std::uint32_t>(seed)
        ^ (lo * 0x9e3779b9U) ^ (hi * 0x85ebca6bU);
    value ^= static_cast<std::uint32_t>(seed >> 32) + draw * 0xc2b2ae35U;
    return mix32(value ^ mix32(draw + 0x27d4eb2dU));
}

__host__ __device__ std::uint64_t generated_block(
    std::uint64_t seed,
    std::uint64_t index,
    std::uint32_t n,
    std::uint32_t w)
{
    std::uint64_t mask = 0;
    std::uint32_t draw = 0;
    for (std::uint32_t j = n - w; j < n; ++j) {
        const std::uint32_t selected = random_word(seed, index, draw++) % (j + 1);
        const std::uint32_t bit = ((mask >> selected) & 1ULL) != 0 ? j : selected;
        mask |= 1ULL << bit;
    }
    return mask;
}

__host__ __device__ std::uint32_t choose_small(
    std::uint32_t value,
    std::uint32_t choose)
{
    if (value < choose) return 0;
    std::uint32_t result = 1;
    for (std::uint32_t i = 1; i <= choose; ++i) {
        const std::uint64_t next = static_cast<std::uint64_t>(result)
            * (value - choose + i) / i;
        if (next > UINT32_MAX) return UINT32_MAX;
        result = static_cast<std::uint32_t>(next);
    }
    return result;
}

__host__ __device__ std::uint32_t saturating_add(
    std::uint32_t first,
    std::uint32_t second)
{
    return first > UINT32_MAX - second ? UINT32_MAX : first + second;
}

std::uint32_t exact_score(
    const std::vector<std::uint64_t>& blocks,
    std::uint32_t critical_shadow_size)
{
    std::uint64_t score = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            score += choose_small(
                host_popcount64(blocks[i] & blocks[j]),
                critical_shadow_size);
        }
    }
    return score > std::numeric_limits<std::uint32_t>::max()
        ? std::numeric_limits<std::uint32_t>::max()
        : static_cast<std::uint32_t>(score);
}

std::uint64_t telemetry_hash(const std::vector<std::uint64_t>& blocks) {
    std::uint32_t h0 = 2166136261U;
    std::uint32_t h1 = 0x9e3779b9U;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const std::uint64_t mask = blocks[i];
        const std::uint32_t a = mix32(
            static_cast<std::uint32_t>(mask)
            ^ (static_cast<std::uint32_t>(i) * 0x85ebca6bU));
        const std::uint32_t b = mix32(
            static_cast<std::uint32_t>(mask >> 32)
            ^ (static_cast<std::uint32_t>(i) * 0xc2b2ae35U));
        h0 = (h0 ^ a) * 16777619U;
        h1 += b + 0x9e3779b9U + (h1 << 6) + (h1 >> 2);
    }
    return h0 | (static_cast<std::uint64_t>(h1) << 32);
}

std::uint64_t telemetry_hash(
    const std::vector<std::uint64_t>& base,
    std::uint64_t extra)
{
    std::vector<std::uint64_t> blocks = base;
    blocks.push_back(extra);
    return telemetry_hash(blocks);
}

std::vector<ShadowCount> base_shadow_counts(
    const std::vector<std::uint64_t>& blocks,
    std::uint32_t n,
    std::uint32_t critical_shadow_size)
{
    const std::uint32_t universe = choose_small(n, critical_shadow_size);
    std::vector<ShadowCount> counts(universe, 0);
    for (const std::uint64_t block : blocks) {
        std::uint32_t coordinates[64];
        std::uint32_t coordinate_count = 0;
        std::uint64_t remaining = block;
        while (remaining != 0) {
            coordinates[coordinate_count++] = static_cast<std::uint32_t>(
                host_countr_zero64(remaining));
            remaining &= remaining - 1;
        }
        std::uint32_t selected[3] = {0, 1, 2};
        while (true) {
            std::uint32_t rank = 0;
            for (std::uint32_t i = 0; i < critical_shadow_size; ++i)
                rank += choose_small(coordinates[selected[i]], i + 1);
            if (counts[rank] == UINT8_MAX)
                throw std::runtime_error("incremental CUDA shadow count overflow");
            ++counts[rank];

            int position = static_cast<int>(critical_shadow_size) - 1;
            while (position >= 0 && selected[position]
                    == coordinate_count - critical_shadow_size
                        + static_cast<std::uint32_t>(position))
                --position;
            if (position < 0) break;
            ++selected[position];
            for (std::uint32_t i = static_cast<std::uint32_t>(position) + 1;
                 i < critical_shadow_size; ++i)
                selected[i] = selected[i - 1] + 1;
        }
    }
    return counts;
}

__global__ void score_candidates_kernel(
    const std::uint64_t* base_blocks,
    std::uint32_t base_count,
    std::uint32_t n,
    std::uint32_t w,
    std::uint32_t critical_shadow_size,
    std::uint64_t seed,
    std::uint64_t first_sample,
    std::uint32_t count,
    ats45cuda::CandidateResult* results)
{
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp_in_block = threadIdx.x >> 5;
    const std::uint32_t warps_per_block = blockDim.x >> 5;
    const std::uint32_t candidate_id = blockIdx.x * warps_per_block + warp_in_block;
    if (candidate_id >= count) return;
    std::uint64_t candidate = 0;
    if (lane == 0) {
        candidate = generated_block(seed, first_sample + candidate_id, n, w);
    }
    std::uint32_t low = __shfl_sync(0xffffffffU, static_cast<std::uint32_t>(candidate), 0);
    std::uint32_t high = __shfl_sync(
        0xffffffffU, static_cast<std::uint32_t>(candidate >> 32), 0);
    candidate = low | (static_cast<std::uint64_t>(high) << 32);
    std::uint64_t local_score = 0;
    for (std::uint32_t i = lane; i < base_count; i += 32) {
        const std::uint32_t overlap = static_cast<std::uint32_t>(
            __popcll(candidate & base_blocks[i]));
        local_score += choose_small(overlap, critical_shadow_size);
    }
    for (int offset = 16; offset != 0; offset >>= 1)
        local_score += __shfl_down_sync(0xffffffffU, local_score, offset);
    if (lane == 0) {
        ats45cuda::CandidateResult result;
        result.mask = candidate;
        result.score = local_score > UINT32_MAX
            ? UINT32_MAX : static_cast<std::uint32_t>(local_score);
        result.reserved = 0;
        results[candidate_id] = result;
    }
}

__global__ void score_explicit_candidates_kernel(
    const std::uint64_t* base_blocks,
    std::uint32_t base_count,
    std::uint32_t critical_shadow_size,
    const std::uint64_t* candidate_masks,
    std::uint32_t count,
    ats45cuda::CandidateResult* results)
{
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp_in_block = threadIdx.x >> 5;
    const std::uint32_t warps_per_block = blockDim.x >> 5;
    const std::uint32_t candidate_id = blockIdx.x * warps_per_block + warp_in_block;
    if (candidate_id >= count) return;
    std::uint64_t candidate = 0;
    if (lane == 0) candidate = candidate_masks[candidate_id];
    const std::uint32_t low = __shfl_sync(
        0xffffffffU, static_cast<std::uint32_t>(candidate), 0);
    const std::uint32_t high = __shfl_sync(
        0xffffffffU, static_cast<std::uint32_t>(candidate >> 32), 0);
    candidate = low | (static_cast<std::uint64_t>(high) << 32);
    std::uint64_t local_score = 0;
    for (std::uint32_t i = lane; i < base_count; i += 32) {
        const std::uint32_t overlap = static_cast<std::uint32_t>(
            __popcll(candidate & base_blocks[i]));
        local_score += choose_small(overlap, critical_shadow_size);
    }
    for (int offset = 16; offset != 0; offset >>= 1)
        local_score += __shfl_down_sync(0xffffffffU, local_score, offset);
    if (lane == 0) {
        ats45cuda::CandidateResult result;
        result.mask = candidate;
        result.score = local_score > UINT32_MAX
            ? UINT32_MAX : static_cast<std::uint32_t>(local_score);
        result.reserved = 0;
        results[candidate_id] = result;
    }
}

template <typename T>
void cuda_free(T*& pointer) {
    if (pointer) cudaFree(pointer);
    pointer = nullptr;
}

} // namespace

namespace ats45cuda {

struct CandidateScorer::Impl {
    int device = 0;
    std::uint64_t* base_blocks = nullptr;
    CandidateResult* results = nullptr;
    std::uint64_t* explicit_masks = nullptr;
    std::size_t result_capacity = 0;
    std::size_t explicit_capacity = 0;
    std::uint32_t base_count = 0;
    std::uint32_t n = 0;
    std::uint32_t w = 0;
    std::uint32_t critical_shadow_size = 0;
    cudaStream_t stream = nullptr;
    std::string name;

    ~Impl() {
        cuda_free(results);
        cuda_free(explicit_masks);
        cuda_free(base_blocks);
        if (stream) cudaStreamDestroy(stream);
    }
};

CandidateScorer::CandidateScorer() : impl_(std::make_unique<Impl>()) {}
CandidateScorer::~CandidateScorer() = default;
CandidateScorer::CandidateScorer(CandidateScorer&&) noexcept = default;
CandidateScorer& CandidateScorer::operator=(CandidateScorer&&) noexcept = default;

bool CandidateScorer::initialize(
    const std::vector<std::uint64_t>& base_blocks,
    int n,
    int w,
    int critical_shadow_size,
    std::string& error)
{
    if (base_blocks.empty() || n < 1 || n > 64 || w < 1 || w > n ||
        critical_shadow_size < 1 || critical_shadow_size > w) {
        error = "invalid CUDA candidate scorer parameters";
        return false;
    }
    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error)) return false;
    if (device_count <= 0) {
        error = "no CUDA-capable device found";
        return false;
    }
    impl_->device = 0;
    if (!check_cuda(cudaSetDevice(impl_->device), "cudaSetDevice", error)) return false;
    if (!impl_->stream && !check_cuda(
            cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags", error)) return false;
    cuda_free(impl_->base_blocks);
    impl_->base_count = static_cast<std::uint32_t>(base_blocks.size());
    impl_->n = static_cast<std::uint32_t>(n);
    impl_->w = static_cast<std::uint32_t>(w);
    impl_->critical_shadow_size = static_cast<std::uint32_t>(critical_shadow_size);
    impl_->name = cuda_device_name(impl_->device);
    const std::size_t bytes = base_blocks.size() * sizeof(std::uint64_t);
    if (!check_cuda(cudaMalloc(&impl_->base_blocks, bytes), "cudaMalloc(base blocks)", error) ||
        !check_cuda(cudaMemcpyAsync(
            impl_->base_blocks, base_blocks.data(), bytes, cudaMemcpyHostToDevice,
            impl_->stream), "cudaMemcpyAsync(base blocks)", error) ||
        !check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize", error)) {
        cuda_free(impl_->base_blocks);
        return false;
    }
    return true;
}

bool CandidateScorer::score_generated(
    std::uint64_t seed,
    std::uint64_t first_sample,
    std::size_t count,
    std::vector<CandidateResult>& results,
    std::string& error)
{
    if (!impl_->base_blocks || !impl_->stream) {
        error = "CUDA candidate scorer is not initialized";
        return false;
    }
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
        error = "invalid CUDA candidate batch size";
        return false;
    }
    if (count > impl_->result_capacity) {
        cuda_free(impl_->results);
        if (!check_cuda(cudaMalloc(
                &impl_->results, count * sizeof(CandidateResult)),
                "cudaMalloc(candidate results)", error)) return false;
        impl_->result_capacity = count;
    }
    const std::uint32_t warps_per_block = kCandidateThreads / 32;
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        (count + warps_per_block - 1) / warps_per_block);
    score_candidates_kernel<<<blocks, kCandidateThreads, 0, impl_->stream>>>(
        impl_->base_blocks,
        impl_->base_count,
        impl_->n,
        impl_->w,
        impl_->critical_shadow_size,
        seed,
        first_sample,
        static_cast<std::uint32_t>(count),
        impl_->results);
    if (!check_cuda(cudaGetLastError(), "score_candidates_kernel launch", error)) return false;
    results.resize(count);
    if (!check_cuda(cudaMemcpyAsync(
            results.data(), impl_->results, count * sizeof(CandidateResult),
            cudaMemcpyDeviceToHost, impl_->stream),
            "cudaMemcpyAsync(candidate results)", error) ||
        !check_cuda(cudaStreamSynchronize(impl_->stream),
            "cudaStreamSynchronize(candidate results)", error)) return false;
    return true;
}

bool CandidateScorer::score_masks(
    const std::vector<std::uint64_t>& masks,
    std::vector<CandidateResult>& results,
    std::string& error)
{
    if (!impl_->base_blocks || !impl_->stream) {
        error = "CUDA candidate scorer is not initialized";
        return false;
    }
    const std::size_t count = masks.size();
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
        error = "invalid explicit CUDA candidate batch size";
        return false;
    }
    if (count > impl_->result_capacity) {
        cuda_free(impl_->results);
        if (!check_cuda(cudaMalloc(
                &impl_->results, count * sizeof(CandidateResult)),
                "cudaMalloc(explicit candidate results)", error)) return false;
        impl_->result_capacity = count;
    }
    if (count > impl_->explicit_capacity) {
        cuda_free(impl_->explicit_masks);
        if (!check_cuda(cudaMalloc(
                &impl_->explicit_masks, count * sizeof(std::uint64_t)),
                "cudaMalloc(explicit candidate masks)", error)) return false;
        impl_->explicit_capacity = count;
    }
    if (!check_cuda(cudaMemcpyAsync(
            impl_->explicit_masks, masks.data(), count * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice, impl_->stream),
            "cudaMemcpyAsync(explicit candidate masks)", error)) return false;
    const std::uint32_t warps_per_block = kCandidateThreads / 32;
    const std::uint32_t blocks = static_cast<std::uint32_t>(
        (count + warps_per_block - 1) / warps_per_block);
    score_explicit_candidates_kernel<<<blocks, kCandidateThreads, 0, impl_->stream>>>(
        impl_->base_blocks,
        impl_->base_count,
        impl_->critical_shadow_size,
        impl_->explicit_masks,
        static_cast<std::uint32_t>(count),
        impl_->results);
    if (!check_cuda(cudaGetLastError(),
            "score_explicit_candidates_kernel launch", error)) return false;
    results.resize(count);
    if (!check_cuda(cudaMemcpyAsync(
            results.data(), impl_->results, count * sizeof(CandidateResult),
            cudaMemcpyDeviceToHost, impl_->stream),
            "cudaMemcpyAsync(explicit candidate results)", error) ||
        !check_cuda(cudaStreamSynchronize(impl_->stream),
            "cudaStreamSynchronize(explicit candidate results)", error)) return false;
    return true;
}

const std::string& CandidateScorer::device_name() const { return impl_->name; }
std::size_t CandidateScorer::maximum_batch_size() const {
    return static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
}

bool runtime_available(std::string& device_name, std::string& error) {
    int count = 0;
    if (!check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount", error)) return false;
    if (count <= 0) {
        error = "no CUDA-capable device found";
        return false;
    }
    device_name = cuda_device_name(0);
    return true;
}

namespace {

struct DeviceMoveCounters {
    std::uint64_t accepted = 0;
    std::uint64_t proposals = 0;
    std::uint64_t iterations = 0;
    std::uint64_t incidence_errors = 0;
};

struct DeviceLowDefectEvent {
    std::uint64_t state_hash = 0;
    std::uint64_t conflict_signature = 0;
    std::uint64_t iteration = 0;
    std::uint32_t score = 0;
    std::uint32_t walker = 0;
    std::uint32_t move_class = 0;
    std::uint32_t reserved = 0;
};

template <typename T>
struct CudaBuffer {
    T* data = nullptr;
    std::size_t count = 0;
    ~CudaBuffer() { cuda_free(data); }
    bool allocate(std::size_t requested, const char* label, std::string& error) {
        cuda_free(data);
        count = 0;
        if (requested == 0) {
            error = std::string(label) + " requested an empty CUDA buffer";
            return false;
        }
        if (!check_cuda(cudaMalloc(&data, requested * sizeof(T)), label, error)) return false;
        count = requested;
        return true;
    }
};

__device__ std::uint32_t next_random(std::uint32_t& state) {
    if (state == 0) state = 0xa341316cU;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ std::uint32_t nth_set_bit(std::uint64_t mask, std::uint32_t ordinal) {
    while (ordinal-- != 0) mask &= mask - 1;
    return static_cast<std::uint32_t>(__ffsll(static_cast<long long>(mask)) - 1);
}

__device__ std::uint64_t swap_bit(
    std::uint64_t mask,
    std::uint32_t old_bit,
    std::uint32_t new_bit)
{
    mask &= ~(1ULL << old_bit);
    mask |= 1ULL << new_bit;
    return mask;
}

__device__ std::uint32_t shadow_rank_1(std::uint32_t a) {
    return a;
}

__device__ std::uint32_t shadow_rank_2(std::uint32_t a, std::uint32_t b) {
    return a + b * (b - 1U) / 2U;
}

__device__ std::uint32_t shadow_rank_3(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t c)
{
    return shadow_rank_2(a, b) + c * (c - 1U) * (c - 2U) / 6U;
}

__device__ std::uint32_t removal_shadow_contribution(
    const ShadowCount* counts,
    std::uint64_t block,
    std::uint32_t critical_shadow_size)
{
    std::uint32_t contribution = 0;
    std::uint64_t first_remaining = block;
    while (first_remaining != 0) {
        const std::uint32_t a = static_cast<std::uint32_t>(__ffsll(
            static_cast<long long>(first_remaining)) - 1);
        first_remaining &= first_remaining - 1;
        if (critical_shadow_size == 1) {
            contribution = saturating_add(
                contribution, static_cast<std::uint32_t>(counts[shadow_rank_1(a)]));
            continue;
        }
        std::uint64_t second_remaining = first_remaining;
        while (second_remaining != 0) {
            const std::uint32_t b = static_cast<std::uint32_t>(__ffsll(
                static_cast<long long>(second_remaining)) - 1);
            second_remaining &= second_remaining - 1;
            if (critical_shadow_size == 2) {
                contribution = saturating_add(
                    contribution, static_cast<std::uint32_t>(
                        counts[shadow_rank_2(a, b)]));
                continue;
            }
            std::uint64_t third_remaining = second_remaining;
            while (third_remaining != 0) {
                const std::uint32_t c = static_cast<std::uint32_t>(__ffsll(
                    static_cast<long long>(third_remaining)) - 1);
                third_remaining &= third_remaining - 1;
                contribution = saturating_add(
                    contribution, static_cast<std::uint32_t>(
                        counts[shadow_rank_3(a, b, c)]));
            }
        }
    }
    return contribution - choose_small(
        static_cast<std::uint32_t>(__popcll(block)), critical_shadow_size);
}

__device__ std::uint32_t addition_shadow_contribution(
    const ShadowCount* counts,
    std::uint64_t candidate,
    std::uint64_t removed_block,
    std::uint32_t critical_shadow_size)
{
    std::uint32_t contribution = 0;
    std::uint64_t first_remaining = candidate;
    while (first_remaining != 0) {
        const std::uint32_t a = static_cast<std::uint32_t>(__ffsll(
            static_cast<long long>(first_remaining)) - 1);
        first_remaining &= first_remaining - 1;
        if (critical_shadow_size == 1) {
            contribution = saturating_add(
                contribution, static_cast<std::uint32_t>(counts[shadow_rank_1(a)]));
            continue;
        }
        std::uint64_t second_remaining = first_remaining;
        while (second_remaining != 0) {
            const std::uint32_t b = static_cast<std::uint32_t>(__ffsll(
                static_cast<long long>(second_remaining)) - 1);
            second_remaining &= second_remaining - 1;
            if (critical_shadow_size == 2) {
                contribution = saturating_add(
                    contribution, static_cast<std::uint32_t>(counts[shadow_rank_2(a, b)]));
                continue;
            }
            std::uint64_t third_remaining = second_remaining;
            while (third_remaining != 0) {
                const std::uint32_t c = static_cast<std::uint32_t>(__ffsll(
                    static_cast<long long>(third_remaining)) - 1);
                third_remaining &= third_remaining - 1;
                contribution = saturating_add(
                    contribution, static_cast<std::uint32_t>(
                        counts[shadow_rank_3(a, b, c)]));
            }
        }
    }
    return contribution - choose_small(
        static_cast<std::uint32_t>(__popcll(candidate & removed_block)),
        critical_shadow_size);
}

__device__ void adjust_shadow_counts(
    ShadowCount* counts,
    std::uint64_t block,
    std::uint32_t critical_shadow_size,
    int adjustment)
{
    std::uint64_t first_remaining = block;
    while (first_remaining != 0) {
        const std::uint32_t a = static_cast<std::uint32_t>(__ffsll(
            static_cast<long long>(first_remaining)) - 1);
        first_remaining &= first_remaining - 1;
        if (critical_shadow_size == 1) {
            counts[shadow_rank_1(a)] = static_cast<ShadowCount>(
                static_cast<int>(counts[shadow_rank_1(a)]) + adjustment);
            continue;
        }
        std::uint64_t second_remaining = first_remaining;
        while (second_remaining != 0) {
            const std::uint32_t b = static_cast<std::uint32_t>(__ffsll(
                static_cast<long long>(second_remaining)) - 1);
            second_remaining &= second_remaining - 1;
            if (critical_shadow_size == 2) {
                const std::uint32_t rank = shadow_rank_2(a, b);
                counts[rank] = static_cast<ShadowCount>(
                    static_cast<int>(counts[rank]) + adjustment);
                continue;
            }
            std::uint64_t third_remaining = second_remaining;
            while (third_remaining != 0) {
                const std::uint32_t c = static_cast<std::uint32_t>(__ffsll(
                    static_cast<long long>(third_remaining)) - 1);
                third_remaining &= third_remaining - 1;
                const std::uint32_t rank = shadow_rank_3(a, b, c);
                counts[rank] = static_cast<ShadowCount>(
                    static_cast<int>(counts[rank]) + adjustment);
            }
        }
    }
}

__global__ void initialize_shadow_counts_kernel(
    ShadowCount* all_counts,
    const ShadowCount* base_counts,
    const std::uint64_t* all_blocks,
    std::uint32_t shadow_universe,
    std::uint32_t target_m,
    std::uint32_t critical_shadow_size,
    std::uint32_t walker_count)
{
    const std::uint32_t walker = blockIdx.x;
    if (walker >= walker_count) return;
    ShadowCount* counts = all_counts
        + static_cast<std::size_t>(walker) * shadow_universe;
    for (std::uint32_t rank = threadIdx.x; rank < shadow_universe;
         rank += blockDim.x)
        counts[rank] = base_counts[rank];
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint64_t extra = all_blocks[
            static_cast<std::size_t>(walker) * target_m + target_m - 1];
        adjust_shadow_counts(counts, extra, critical_shadow_size, 1);
    }
}

#if ATS48_INVERTED_SHADOW_INCIDENCE
__device__ std::uint32_t incidence_rank_contribution(
    const std::uint16_t* heads,
    const std::uint16_t* next,
    const std::uint16_t* node_owners,
    std::uint32_t rank,
    std::uint32_t changed)
{
    std::uint32_t contribution = 0;
    for (std::uint16_t node = heads[rank]; node != kNoShadowNode;
         node = next[node]) {
        if (node_owners[node] != changed) ++contribution;
    }
    return contribution;
}

__device__ std::uint32_t incidence_block_contribution(
    const std::uint16_t* heads,
    const std::uint16_t* next,
    const std::uint16_t* node_owners,
    std::uint64_t block,
    std::uint32_t changed,
    std::uint32_t critical_shadow_size)
{
    std::uint32_t contribution = 0;
    std::uint64_t first_remaining = block;
    while (first_remaining != 0) {
        const std::uint32_t a = static_cast<std::uint32_t>(__ffsll(
            static_cast<long long>(first_remaining)) - 1);
        first_remaining &= first_remaining - 1;
        if (critical_shadow_size == 1) {
            contribution = saturating_add(contribution, incidence_rank_contribution(
                heads, next, node_owners, shadow_rank_1(a), changed));
            continue;
        }
        std::uint64_t second_remaining = first_remaining;
        while (second_remaining != 0) {
            const std::uint32_t b = static_cast<std::uint32_t>(__ffsll(
                static_cast<long long>(second_remaining)) - 1);
            second_remaining &= second_remaining - 1;
            if (critical_shadow_size == 2) {
                contribution = saturating_add(contribution, incidence_rank_contribution(
                    heads, next, node_owners, shadow_rank_2(a, b), changed));
                continue;
            }
            std::uint64_t third_remaining = second_remaining;
            while (third_remaining != 0) {
                const std::uint32_t c = static_cast<std::uint32_t>(__ffsll(
                    static_cast<long long>(third_remaining)) - 1);
                third_remaining &= third_remaining - 1;
                contribution = saturating_add(contribution, incidence_rank_contribution(
                    heads, next, node_owners, shadow_rank_3(a, b, c), changed));
            }
        }
    }
    return contribution;
}

__device__ bool gather_incidence_rank_owners(
    const std::uint16_t* heads,
    const std::uint16_t* next,
    const std::uint16_t* node_owners,
    std::uint32_t rank,
    std::uint32_t changed,
    std::uint16_t* scratch,
    std::uint32_t limit,
    std::uint32_t& count)
{
    for (std::uint16_t node = heads[rank]; node != kNoShadowNode;
         node = next[node]) {
        const std::uint16_t owner = node_owners[node];
        if (owner == changed) continue;
        bool present = false;
        for (std::uint32_t i = 0; i < count; ++i) {
            if (scratch[i] == owner) {
                present = true;
                break;
            }
        }
        if (present) continue;
        if (count >= limit) return false;
        scratch[count++] = owner;
    }
    return true;
}

__device__ std::uint32_t gather_incident_owners(
    const std::uint16_t* heads,
    const std::uint16_t* next,
    const std::uint16_t* node_owners,
    std::uint64_t block,
    std::uint32_t changed,
    std::uint32_t critical_shadow_size,
    std::uint16_t* scratch,
    std::uint32_t limit)
{
    std::uint32_t count = 0;
    std::uint64_t first_remaining = block;
    while (first_remaining != 0) {
        const std::uint32_t a = static_cast<std::uint32_t>(__ffsll(
            static_cast<long long>(first_remaining)) - 1);
        first_remaining &= first_remaining - 1;
        if (critical_shadow_size == 1) {
            if (!gather_incidence_rank_owners(
                    heads, next, node_owners, shadow_rank_1(a), changed,
                    scratch, limit, count))
                return limit + 1;
            continue;
        }
        std::uint64_t second_remaining = first_remaining;
        while (second_remaining != 0) {
            const std::uint32_t b = static_cast<std::uint32_t>(__ffsll(
                static_cast<long long>(second_remaining)) - 1);
            second_remaining &= second_remaining - 1;
            if (critical_shadow_size == 2) {
                if (!gather_incidence_rank_owners(
                        heads, next, node_owners, shadow_rank_2(a, b), changed,
                        scratch, limit, count))
                    return limit + 1;
                continue;
            }
            std::uint64_t third_remaining = second_remaining;
            while (third_remaining != 0) {
                const std::uint32_t c = static_cast<std::uint32_t>(__ffsll(
                    static_cast<long long>(third_remaining)) - 1);
                third_remaining &= third_remaining - 1;
                if (!gather_incidence_rank_owners(
                        heads, next, node_owners, shadow_rank_3(a, b, c), changed,
                        scratch, limit, count))
                    return limit + 1;
            }
        }
    }
    return count;
}

__device__ void insert_incidence_rank(
    std::uint16_t* heads,
    std::uint16_t* next,
    std::uint32_t rank,
    std::uint16_t node)
{
    next[node] = heads[rank];
    heads[rank] = node;
}

__device__ bool remove_incidence_rank(
    std::uint16_t* heads,
    std::uint16_t* next,
    std::uint32_t rank,
    std::uint16_t node)
{
    std::uint16_t previous = kNoShadowNode;
    std::uint16_t current = heads[rank];
    while (current != kNoShadowNode && current != node) {
        previous = current;
        current = next[current];
    }
    if (current == kNoShadowNode) return false;
    if (previous == kNoShadowNode) heads[rank] = next[current];
    else next[previous] = next[current];
    next[current] = kNoShadowNode;
    return true;
}

__device__ bool update_incidence_block(
    std::uint16_t* heads,
    std::uint16_t* next,
    std::uint64_t block,
    std::uint32_t block_index,
    std::uint32_t shadows_per_block,
    std::uint32_t critical_shadow_size,
    bool insert)
{
    std::uint32_t local_shadow = 0;
    std::uint64_t first_remaining = block;
    while (first_remaining != 0) {
        const std::uint32_t a = static_cast<std::uint32_t>(__ffsll(
            static_cast<long long>(first_remaining)) - 1);
        first_remaining &= first_remaining - 1;
        if (critical_shadow_size == 1) {
            const std::uint16_t node = static_cast<std::uint16_t>(
                block_index * shadows_per_block + local_shadow++);
            if (insert) insert_incidence_rank(heads, next, shadow_rank_1(a), node);
            else if (!remove_incidence_rank(heads, next, shadow_rank_1(a), node))
                return false;
            continue;
        }
        std::uint64_t second_remaining = first_remaining;
        while (second_remaining != 0) {
            const std::uint32_t b = static_cast<std::uint32_t>(__ffsll(
                static_cast<long long>(second_remaining)) - 1);
            second_remaining &= second_remaining - 1;
            if (critical_shadow_size == 2) {
                const std::uint16_t node = static_cast<std::uint16_t>(
                    block_index * shadows_per_block + local_shadow++);
                if (insert) insert_incidence_rank(heads, next, shadow_rank_2(a, b), node);
                else if (!remove_incidence_rank(
                        heads, next, shadow_rank_2(a, b), node))
                    return false;
                continue;
            }
            std::uint64_t third_remaining = second_remaining;
            while (third_remaining != 0) {
                const std::uint32_t c = static_cast<std::uint32_t>(__ffsll(
                    static_cast<long long>(third_remaining)) - 1);
                third_remaining &= third_remaining - 1;
                const std::uint16_t node = static_cast<std::uint16_t>(
                    block_index * shadows_per_block + local_shadow++);
                if (insert) insert_incidence_rank(
                    heads, next, shadow_rank_3(a, b, c), node);
                else if (!remove_incidence_rank(
                        heads, next, shadow_rank_3(a, b, c), node))
                    return false;
            }
        }
    }
    return local_shadow == shadows_per_block;
}

__global__ void initialize_shadow_incidence_kernel(
    std::uint16_t* all_heads,
    std::uint16_t* all_next,
    const std::uint64_t* all_blocks,
    std::uint32_t shadow_universe,
    std::uint32_t nodes_per_walker,
    std::uint32_t shadows_per_block,
    std::uint32_t target_m,
    std::uint32_t critical_shadow_size,
    std::uint32_t walker_count)
{
    const std::uint32_t walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= walker_count) return;
    std::uint16_t* heads = all_heads
        + static_cast<std::size_t>(walker) * shadow_universe;
    std::uint16_t* next = all_next
        + static_cast<std::size_t>(walker) * nodes_per_walker;
    const std::uint64_t* blocks = all_blocks
        + static_cast<std::size_t>(walker) * target_m;
    for (std::uint32_t block = 0; block < target_m; ++block) {
        update_incidence_block(
            heads, next, blocks[block], block, shadows_per_block,
            critical_shadow_size, true);
    }
}
#endif

__device__ std::uint64_t device_state_hash(
    const std::uint64_t* blocks,
    std::uint32_t block_count)
{
    std::uint32_t h0 = 2166136261U;
    std::uint32_t h1 = 0x9e3779b9U;
    for (std::uint32_t i = 0; i < block_count; ++i) {
        const std::uint64_t mask = blocks[i];
        const std::uint32_t a = mix32(
            static_cast<std::uint32_t>(mask) ^ (i * 0x85ebca6bU));
        const std::uint32_t b = mix32(
            static_cast<std::uint32_t>(mask >> 32) ^ (i * 0xc2b2ae35U));
        h0 = (h0 ^ a) * 16777619U;
        h1 += b + 0x9e3779b9U + (h1 << 6) + (h1 >> 2);
    }
    return h0 | (static_cast<std::uint64_t>(h1) << 32);
}

__device__ std::uint64_t device_conflict_signature(
    const std::uint64_t* blocks,
    const std::uint32_t* bad_pairs,
    std::uint32_t bad_count)
{
    std::uint32_t h0 = 0x811c9dc5U;
    std::uint32_t h1 = 0x9e3779b9U;
    for (std::uint32_t q = 0; q < bad_count; ++q) {
        const std::uint32_t pair = bad_pairs[q];
        const std::uint32_t a = pair & 0xffffU;
        const std::uint32_t b = pair >> 16;
        const std::uint64_t first = blocks[a];
        const std::uint64_t second = blocks[b];
        const std::uint32_t edge0 = mix32(static_cast<std::uint32_t>(first))
            ^ mix32(static_cast<std::uint32_t>(second));
        const std::uint32_t edge1 = mix32(static_cast<std::uint32_t>(first >> 32))
            ^ mix32(static_cast<std::uint32_t>(second >> 32));
        h0 ^= mix32(edge0 + 0x85ebca6bU);
        h1 += mix32(edge1 + 0xc2b2ae35U);
    }
    return h0 | (static_cast<std::uint64_t>(h1) << 32);
}

__device__ std::uint32_t defect2_geometry(
    const std::uint64_t* blocks,
    std::uint32_t target_m,
    const std::uint32_t* bad_pairs,
    std::uint32_t bad_count,
    std::uint32_t changed,
    std::uint64_t candidate,
    std::uint32_t critical_shadow_size)
{
    std::uint32_t edge_a[2] = {UINT32_MAX, UINT32_MAX};
    std::uint32_t edge_b[2] = {UINT32_MAX, UINT32_MAX};
    std::uint32_t edge_count = 0;
    for (std::uint32_t q = 0; q < bad_count; ++q) {
        const std::uint32_t pair = bad_pairs[q];
        const std::uint32_t a = pair & 0xffffU;
        const std::uint32_t b = pair >> 16;
        if (a == changed || b == changed) continue;
        if (edge_count >= 2) return UINT32_MAX;
        edge_a[edge_count] = a;
        edge_b[edge_count] = b;
        ++edge_count;
    }
    for (std::uint32_t other = 0; other < target_m; ++other) {
        if (other == changed) continue;
        if (__popcll(candidate & blocks[other]) < critical_shadow_size) continue;
        if (edge_count >= 2) return UINT32_MAX;
        edge_a[edge_count] = min(changed, other);
        edge_b[edge_count] = max(changed, other);
        ++edge_count;
    }
    if (edge_count != 2) return UINT32_MAX;
    const bool shared = edge_a[0] == edge_a[1] || edge_a[0] == edge_b[1]
        || edge_b[0] == edge_a[1] || edge_b[0] == edge_b[1];
    std::uint64_t support_union = 0;
    const std::uint32_t endpoints[4] = {
        edge_a[0], edge_b[0], edge_a[1], edge_b[1]};
    for (std::uint32_t i = 0; i < 4; ++i) {
        support_union |= endpoints[i] == changed ? candidate : blocks[endpoints[i]];
    }
    return (shared ? 0U : 1024U)
        + static_cast<std::uint32_t>(__popcll(support_union));
}

#if ATS48_INVERTED_SHADOW_INCIDENCE
__device__ std::uint32_t defect2_geometry_from_incidence(
    const std::uint64_t* blocks,
    const std::uint32_t* bad_pairs,
    std::uint32_t bad_count,
    std::uint32_t changed,
    std::uint64_t candidate,
    std::uint32_t critical_shadow_size,
    const std::uint16_t* shadow_heads,
    const std::uint16_t* shadow_next,
    const std::uint16_t* node_owners,
    std::uint16_t* incident_scratch)
{
    std::uint32_t edge_a[2] = {UINT32_MAX, UINT32_MAX};
    std::uint32_t edge_b[2] = {UINT32_MAX, UINT32_MAX};
    std::uint32_t edge_count = 0;
    for (std::uint32_t q = 0; q < bad_count; ++q) {
        const std::uint32_t pair = bad_pairs[q];
        const std::uint32_t a = pair & 0xffffU;
        const std::uint32_t b = pair >> 16;
        if (a == changed || b == changed) continue;
        if (edge_count >= 2) return UINT32_MAX;
        edge_a[edge_count] = a;
        edge_b[edge_count] = b;
        ++edge_count;
    }
    const std::uint32_t available = 2U - edge_count;
    const std::uint32_t incident_count = gather_incident_owners(
        shadow_heads, shadow_next, node_owners, candidate, changed,
        critical_shadow_size, incident_scratch, available);
    if (incident_count > available) return UINT32_MAX;
    for (std::uint32_t i = 0; i < incident_count; ++i) {
        const std::uint32_t other = incident_scratch[i];
        edge_a[edge_count] = min(changed, other);
        edge_b[edge_count] = max(changed, other);
        ++edge_count;
    }
    if (edge_count != 2) return UINT32_MAX;
    const bool shared = edge_a[0] == edge_a[1] || edge_a[0] == edge_b[1]
        || edge_b[0] == edge_a[1] || edge_b[0] == edge_b[1];
    std::uint64_t support_union = 0;
    const std::uint32_t endpoints[4] = {
        edge_a[0], edge_b[0], edge_a[1], edge_b[1]};
    for (std::uint32_t i = 0; i < 4; ++i) {
        support_union |= endpoints[i] == changed ? candidate : blocks[endpoints[i]];
    }
    return (shared ? 0U : 1024U)
        + static_cast<std::uint32_t>(__popcll(support_union));
}
#endif

__global__ void swap_walkers_kernel(
    std::uint64_t* all_blocks,
    std::uint32_t* scores,
    std::uint32_t* all_bad_pairs,
    std::uint32_t* bad_counts,
    std::uint32_t* rng_states,
    DeviceMoveCounters* move_counts,
    DeviceLowDefectEvent* low_defect_events,
    std::uint32_t* low_defect_event_count,
    std::uint32_t low_defect_capacity,
    std::uint64_t* first_defect1_blocks,
    DeviceLowDefectEvent* first_defect1_event,
    std::uint32_t* first_defect1_flag,
#if ATS48_DEVICE_DEDUP_HANDOFF
    std::uint64_t* low_defect_state_hashes,
#endif
#if ATS48_INCREMENTAL_SHADOW_SCORING
    ShadowCount* all_shadow_counts,
    std::uint32_t shadow_universe,
#endif
#if ATS48_INVERTED_SHADOW_INCIDENCE
    std::uint16_t* all_shadow_heads,
    std::uint16_t* all_shadow_next,
    const std::uint16_t* shadow_node_owners,
    std::uint16_t* all_incident_scratch,
    std::uint32_t shadow_nodes_per_walker,
    std::uint32_t shadows_per_block,
#endif
    std::uint32_t target_m,
    std::uint32_t n,
    std::uint32_t w,
    std::uint32_t critical_shadow_size,
    std::uint32_t walker_count,
    std::uint32_t proposals_per_iteration,
    float temperature,
    bool freeze_at_defect1)
{
    const std::uint32_t walker = blockIdx.x * blockDim.x + threadIdx.x;
    if (walker >= walker_count) return;
    std::uint64_t* blocks = all_blocks + static_cast<std::size_t>(walker) * target_m;
    std::uint32_t* bad_pairs = all_bad_pairs
        + static_cast<std::size_t>(walker) * kBadCapacity;
#if ATS48_INCREMENTAL_SHADOW_SCORING
    ShadowCount* shadow_counts = all_shadow_counts == nullptr ? nullptr
        : all_shadow_counts + static_cast<std::size_t>(walker) * shadow_universe;
#endif
#if ATS48_INVERTED_SHADOW_INCIDENCE
    std::uint16_t* shadow_heads = all_shadow_heads == nullptr ? nullptr
        : all_shadow_heads + static_cast<std::size_t>(walker) * shadow_universe;
    std::uint16_t* shadow_next = all_shadow_next == nullptr ? nullptr
        : all_shadow_next + static_cast<std::size_t>(walker) * shadow_nodes_per_walker;
    std::uint16_t* incident_scratch = all_incident_scratch == nullptr ? nullptr
        : all_incident_scratch + static_cast<std::size_t>(walker) * kBadCapacity;
#endif
    std::uint32_t score = scores[walker];
    std::uint32_t bad_count = bad_counts[walker];
    std::uint32_t rng = rng_states[walker];
    DeviceMoveCounters counts = move_counts[walker];
    for (std::uint32_t iteration = 0; iteration < kIterationsPerLaunch; ++iteration) {
        if (score == 0 || (freeze_at_defect1 && score == 1) || bad_count == 0) break;
        ++counts.iterations;
        const std::uint32_t pair_slot = next_random(rng) % bad_count;
        const std::uint32_t pair = bad_pairs[pair_slot];
        const std::uint32_t first = pair & 0xffffU;
        const std::uint32_t second = pair >> 16;
        const std::uint64_t common = blocks[first] & blocks[second];
        const std::uint32_t common_count = static_cast<std::uint32_t>(__popcll(common));
        if (common_count < critical_shadow_size) {
            bad_pairs[pair_slot] = bad_pairs[bad_count - 1];
            --bad_count;
            continue;
        }
        std::uint32_t changed = (next_random(rng) & 1U) != 0 ? first : second;
#if ATS_A24_RICH_MOVES
        bool rich_global = false;
        const bool rich_a24_state = n == 24U && w == 12U
            && critical_shadow_size == 10U && score <= 4U;
        if (rich_a24_state
                && counts.iterations % static_cast<std::uint32_t>(ATS_A24_RICH_PERIOD) == 0U) {
            // Conflict-endpoint selection freezes almost the entire incumbent
            // code at low defect.  Select a genuinely non-incident block so a
            // bounded uphill structural kick can alter that frozen background.
            for (std::uint32_t attempt = 0; attempt < 16U; ++attempt) {
                const std::uint32_t candidate_changed = next_random(rng) % target_m;
                bool incident = false;
                for (std::uint32_t q = 0; q < bad_count; ++q) {
                    const std::uint32_t existing = bad_pairs[q];
                    const std::uint32_t a = existing & 0xffffU;
                    const std::uint32_t b = existing >> 16;
                    if (a == candidate_changed || b == candidate_changed) {
                        incident = true;
                        break;
                    }
                }
                if (!incident) {
                    changed = candidate_changed;
                    rich_global = true;
                    break;
                }
            }
        }
#endif
        const std::uint64_t old_block = blocks[changed];
        const std::uint32_t old_bit_offset = next_random(rng) % common_count;
        std::uint32_t old_contribution = 0;
        std::uint32_t unaffected_bad_count = 0;
        for (std::uint32_t q = 0; q < bad_count; ++q) {
            const std::uint32_t existing = bad_pairs[q];
            const std::uint32_t a = existing & 0xffffU;
            const std::uint32_t b = existing >> 16;
            if (a != changed && b != changed) ++unaffected_bad_count;
        }
#if ATS48_INCREMENTAL_SHADOW_SCORING
        if (shadow_counts != nullptr) {
            old_contribution = removal_shadow_contribution(
                shadow_counts, old_block, critical_shadow_size);
        } else
#endif
#if ATS48_INVERTED_SHADOW_INCIDENCE
        if (shadow_heads != nullptr) {
            old_contribution = incidence_block_contribution(
                shadow_heads, shadow_next, shadow_node_owners,
                old_block, changed, critical_shadow_size);
        } else
#endif
        {
            for (std::uint32_t other = 0; other < target_m; ++other) {
                if (other == changed) continue;
                old_contribution = saturating_add(
                    old_contribution,
                    choose_small(
                        static_cast<std::uint32_t>(__popcll(old_block & blocks[other])),
                        critical_shadow_size));
            }
        }
        std::uint32_t best_total = UINT32_MAX;
        std::uint32_t best_geometry = UINT32_MAX;
        std::uint32_t best_move_class = 0;
        std::uint64_t best_block = old_block;
        bool best_available = false;
        for (std::uint32_t proposal = 0; proposal < proposals_per_iteration; ++proposal) {
            ++counts.proposals;
            std::uint64_t candidate = old_block;
            std::uint32_t proposal_move_class = 1;
#if ATS_A24_RICH_MOVES
            if (rich_global) {
                // Alternate three- and four-coordinate exchanges.  Every new
                // coordinate comes from outside the original word, so the
                // result always retains weight 12 and changes exactly 3 or 4
                // positions on each side of the exchange.
                const std::uint32_t arity = 3U + ((proposal + counts.iterations) & 1U);
                proposal_move_class = arity == 3U ? 4U : 5U;
                std::uint64_t remaining_old = old_block;
                std::uint64_t selected_new = 0;
                for (std::uint32_t step = 0; step < arity; ++step) {
                    const std::uint32_t old_bit = nth_set_bit(
                        remaining_old, next_random(rng) % (w - step));
                    remaining_old &= ~(1ULL << old_bit);
                    std::uint32_t new_bit = 0;
                    do { new_bit = next_random(rng) % n; }
                    while (((old_block >> new_bit) & 1ULL) != 0
                        || ((selected_new >> new_bit) & 1ULL) != 0);
                    selected_new |= 1ULL << new_bit;
                    candidate &= ~(1ULL << old_bit);
                    candidate |= 1ULL << new_bit;
                }
            } else {
#endif
            const std::uint32_t old_bit = nth_set_bit(
                common, (old_bit_offset + proposal) % common_count);
            std::uint32_t new_bit = 0;
            do { new_bit = next_random(rng) % n; }
            while (((old_block >> new_bit) & 1ULL) != 0);
            candidate = swap_bit(old_block, old_bit, new_bit);
            if (score <= 4 && w > 1 && proposal >= proposals_per_iteration / 2) {
                proposal_move_class = 2;
                const std::uint64_t remaining = old_block & ~(1ULL << old_bit);
                const std::uint32_t second_old = nth_set_bit(
                    remaining, next_random(rng) % (w - 1));
                std::uint32_t second_new = 0;
                do { second_new = next_random(rng) % n; }
                while (((old_block >> second_new) & 1ULL) != 0 || second_new == new_bit);
                candidate = swap_bit(candidate, second_old, second_new);
            }
#if ATS_A24_RICH_MOVES
            }
#endif
            std::uint32_t new_contribution = 0;
            std::uint32_t incident_bad_count = 0;
#if ATS48_INCREMENTAL_SHADOW_SCORING
            if (shadow_counts != nullptr) {
                new_contribution = addition_shadow_contribution(
                    shadow_counts, candidate, old_block, critical_shadow_size);
                if (new_contribution > kBadCapacity - unaffected_bad_count) {
#if ATS48_INVERTED_SHADOW_INCIDENCE
                    if (shadow_heads != nullptr) {
                        incident_bad_count = gather_incident_owners(
                            shadow_heads, shadow_next, shadow_node_owners,
                            candidate, changed, critical_shadow_size,
                            incident_scratch, kBadCapacity - unaffected_bad_count);
                    } else
#endif
                    {
                        for (std::uint32_t other = 0; other < target_m; ++other) {
                            if (other == changed) continue;
                            if (__popcll(candidate & blocks[other])
                                    >= critical_shadow_size)
                                ++incident_bad_count;
                        }
                    }
                    if (incident_bad_count > kBadCapacity - unaffected_bad_count)
                        continue;
                }
            } else
#endif
#if ATS48_INVERTED_SHADOW_INCIDENCE
            if (shadow_heads != nullptr) {
                new_contribution = incidence_block_contribution(
                    shadow_heads, shadow_next, shadow_node_owners,
                    candidate, changed, critical_shadow_size);
                if (new_contribution > kBadCapacity - unaffected_bad_count) {
                    incident_bad_count = gather_incident_owners(
                        shadow_heads, shadow_next, shadow_node_owners,
                        candidate, changed, critical_shadow_size,
                        incident_scratch, kBadCapacity - unaffected_bad_count);
                    if (incident_bad_count > kBadCapacity - unaffected_bad_count)
                        continue;
                }
            } else
#endif
            {
                for (std::uint32_t other = 0; other < target_m; ++other) {
                    if (other == changed) continue;
                    const std::uint32_t overlap = static_cast<std::uint32_t>(
                        __popcll(candidate & blocks[other]));
                    new_contribution = saturating_add(
                        new_contribution, choose_small(overlap, critical_shadow_size));
                    if (overlap >= critical_shadow_size) ++incident_bad_count;
                }
                if (incident_bad_count > kBadCapacity - unaffected_bad_count) continue;
            }
            const std::uint32_t candidate_total = saturating_add(
                score - old_contribution, new_contribution);
            std::uint32_t candidate_geometry = 0;
            if (candidate_total == 2) {
#if ATS48_INVERTED_SHADOW_INCIDENCE
                if (shadow_heads != nullptr) {
                    candidate_geometry = defect2_geometry_from_incidence(
                        blocks, bad_pairs, bad_count, changed, candidate,
                        critical_shadow_size, shadow_heads, shadow_next,
                        shadow_node_owners, incident_scratch);
                } else
#endif
                {
                    candidate_geometry = defect2_geometry(
                        blocks, target_m, bad_pairs, bad_count, changed, candidate,
                        critical_shadow_size);
                }
            }
            if (candidate_total < best_total ||
                (candidate_total == best_total && candidate_geometry < best_geometry) ||
                (candidate_total == best_total && candidate_geometry == best_geometry
                    && (next_random(rng) & 1U) != 0)) {
                best_total = candidate_total;
                best_geometry = candidate_geometry;
                best_move_class = proposal_move_class;
                best_block = candidate;
                best_available = true;
            }
        }
        if (!best_available) continue;
        bool accept = false;
        if (best_total < score) accept = true;
        else if (best_total == score) {
#if ATS_A24_RICH_MOVES
            accept = next_random(rng) % 10000 < (rich_global ? 8500U : 5500U);
#else
            accept = next_random(rng) % 10000 < 5500U;
#endif
        }
        else {
#if ATS_A24_RICH_MOVES
            const float effective_temperature = rich_global
                ? fmaxf(temperature, 1.25f) : fmaxf(temperature, 0.05f);
#else
            const float effective_temperature = fmaxf(temperature, 0.05f);
#endif
            const float probability = expf(
                -static_cast<float>(best_total - score) / effective_temperature);
#if ATS_A24_RICH_MOVES
            // A24 calibration found that exact 3/4-bit mutations of a frozen
            // incumbent typically add 6-7 conflicts even after selecting the
            // best of 16 proposals.  Let 30% of bounded (+8) kicks cross that
            // barrier; ordinary endpoint repair takes over immediately because
            // rich_global is restricted to score <= 4.
            const float kick_probability = rich_global && best_total - score <= 8U
                ? fmaxf(probability, 0.30f) : probability;
#else
            const float kick_probability = probability;
#endif
            accept = static_cast<float>(next_random(rng)) / 4294967296.0f
                < kick_probability;
        }
        if (!accept) continue;
        std::uint32_t write_count = 0;
        for (std::uint32_t q = 0; q < bad_count; ++q) {
            const std::uint32_t existing = bad_pairs[q];
            const std::uint32_t a = existing & 0xffffU;
            const std::uint32_t b = existing >> 16;
            if (a != changed && b != changed) bad_pairs[write_count++] = existing;
        }
#if ATS48_INVERTED_SHADOW_INCIDENCE
        if (shadow_heads != nullptr) {
            const std::uint32_t available = kBadCapacity - write_count;
            const std::uint32_t incident_count = gather_incident_owners(
                shadow_heads, shadow_next, shadow_node_owners,
                best_block, changed, critical_shadow_size,
                incident_scratch, available);
            if (incident_count > available) {
                ++counts.incidence_errors;
                bad_count = 0;
                break;
            }
            const bool removed = update_incidence_block(
                shadow_heads, shadow_next, old_block, changed, shadows_per_block,
                critical_shadow_size, false);
            const bool inserted = removed && update_incidence_block(
                shadow_heads, shadow_next, best_block, changed, shadows_per_block,
                critical_shadow_size, true);
            if (!inserted) {
                ++counts.incidence_errors;
                bad_count = 0;
                break;
            }
#if ATS48_INCREMENTAL_SHADOW_SCORING
            if (shadow_counts != nullptr) {
                adjust_shadow_counts(shadow_counts, old_block, critical_shadow_size, -1);
                adjust_shadow_counts(shadow_counts, best_block, critical_shadow_size, 1);
            }
#endif
            for (std::uint32_t i = 0; i < incident_count; ++i) {
                const std::uint32_t other = incident_scratch[i];
                const std::uint32_t a = min(changed, other);
                const std::uint32_t b = max(changed, other);
                bad_pairs[write_count++] = a | (b << 16);
            }
        } else
#endif
        {
#if ATS48_INCREMENTAL_SHADOW_SCORING
            if (shadow_counts != nullptr) {
                adjust_shadow_counts(shadow_counts, old_block, critical_shadow_size, -1);
                adjust_shadow_counts(shadow_counts, best_block, critical_shadow_size, 1);
            }
#endif
            for (std::uint32_t other = 0; other < target_m; ++other) {
                if (other == changed) continue;
                if (__popcll(best_block & blocks[other]) >= critical_shadow_size) {
                    const std::uint32_t a = min(changed, other);
                    const std::uint32_t b = max(changed, other);
                    bad_pairs[write_count++] = a | (b << 16);
                }
            }
        }
        blocks[changed] = best_block;
        score = best_total;
        ++counts.accepted;
        bad_count = write_count;
        if (score == 1 || score == 2) {
#if ATS48_DEVICE_DEDUP_HANDOFF
            const std::uint64_t state_hash = device_state_hash(blocks, target_m);
            low_defect_state_hashes[walker] = state_hash;
#endif
            const std::uint32_t event_index = atomicAdd(low_defect_event_count, 1U);
            if (event_index < low_defect_capacity) {
                DeviceLowDefectEvent event;
#if ATS48_DEVICE_DEDUP_HANDOFF
                event.state_hash = state_hash;
#else
                event.state_hash = device_state_hash(blocks, target_m);
#endif
                event.conflict_signature = device_conflict_signature(
                    blocks, bad_pairs, bad_count);
                event.iteration = counts.iterations;
                event.score = score;
                event.walker = walker;
                event.move_class = best_move_class;
                low_defect_events[event_index] = event;
                if (score == 1 && atomicCAS(first_defect1_flag, 0U, 1U) == 0U) {
                    for (std::uint32_t block = 0; block < target_m; ++block)
                        first_defect1_blocks[block] = blocks[block];
                    first_defect1_event[0] = event;
                    __threadfence_system();
                    atomicExch(first_defect1_flag, 2U);
                }
            }
        }
#if ATS48_DEVICE_DEFECT2_INTENSIFICATION
        if (score == 1) break;
#else
        if (score <= 2) break;
#endif
    }
    scores[walker] = score;
    bad_counts[walker] = bad_count;
    rng_states[walker] = rng;
    move_counts[walker] = counts;
}

__device__ std::uint64_t device_choose(
    const std::uint64_t* choose_table,
    std::uint32_t value,
    std::uint32_t choose)
{
    return choose_table[value * 65U + choose];
}

__device__ std::int32_t find_shadow_owner(
    const std::uint64_t* shadow_ranks,
    const std::int32_t* shadow_owners,
    std::uint64_t shadow_count,
    std::uint64_t rank)
{
    std::uint64_t first = 0;
    std::uint64_t last = shadow_count;
    while (first < last) {
        const std::uint64_t middle = first + (last - first) / 2;
        if (shadow_ranks[middle] < rank) first = middle + 1;
        else last = middle;
    }
    return first < shadow_count && shadow_ranks[first] == rank
        ? shadow_owners[first] : -1;
}

__device__ std::uint64_t unrank_lexicographic_combination(
    std::uint64_t rank,
    std::uint32_t n,
    std::uint32_t w,
    const std::uint64_t* choose_table,
    std::uint32_t* coordinates)
{
    std::uint64_t mask = 0;
    std::uint32_t first = 0;
    for (std::uint32_t position = 0; position < w; ++position) {
        const std::uint32_t remaining = w - position - 1;
        const std::uint32_t last = n - (w - position);
        for (std::uint32_t coordinate = first; coordinate <= last; ++coordinate) {
            const std::uint64_t suffixes = device_choose(
                choose_table, n - coordinate - 1, remaining);
            if (rank < suffixes) {
                coordinates[position] = coordinate;
                mask |= 1ULL << coordinate;
                first = coordinate + 1;
                break;
            }
            rank -= suffixes;
        }
    }
    return mask;
}

__global__ void scan_repair_candidates_kernel(
    const std::uint64_t* valid_core,
    const std::uint64_t* shadow_ranks,
    const std::int32_t* shadow_owners,
    std::uint64_t shadow_count,
    const std::uint64_t* choose_table,
    std::uint32_t n,
    std::uint32_t w,
    std::uint32_t critical_shadow_size,
    std::uint32_t max_blockers,
    std::uint64_t first_rank,
    std::uint32_t rank_count,
    RepairCandidate* output,
    std::uint32_t output_capacity,
    unsigned long long* qualifying_count)
{
    const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= rank_count) return;

    std::uint32_t coordinates[64];
    const std::uint64_t mask = unrank_lexicographic_combination(
        first_rank + thread, n, w, choose_table, coordinates);
    std::uint32_t blockers[2] = {UINT32_MAX, UINT32_MAX};
    std::uint32_t blocker_count = 0;
    std::uint32_t selected[64];
    for (std::uint32_t i = 0; i < critical_shadow_size; ++i) selected[i] = i;
    while (true) {
        std::uint64_t shadow_rank = 0;
        for (std::uint32_t i = 0; i < critical_shadow_size; ++i)
            shadow_rank += device_choose(
                choose_table, coordinates[selected[i]], i + 1);
        const std::int32_t owner = find_shadow_owner(
            shadow_ranks, shadow_owners, shadow_count, shadow_rank);
        if (owner >= 0) {
            const std::uint32_t owner_id = static_cast<std::uint32_t>(owner);
            if (valid_core[owner_id] == mask) return;
            bool duplicate = false;
            for (std::uint32_t q = 0; q < blocker_count; ++q)
                duplicate |= blockers[q] == owner_id;
            if (!duplicate &&
                __popcll(mask & valid_core[owner_id]) >= critical_shadow_size) {
                if (blocker_count == max_blockers) return;
                blockers[blocker_count++] = owner_id;
            }
        }

        int position = static_cast<int>(critical_shadow_size) - 1;
        while (position >= 0 && selected[position]
            == w - critical_shadow_size + static_cast<std::uint32_t>(position))
            --position;
        if (position < 0) break;
        ++selected[position];
        for (std::uint32_t i = static_cast<std::uint32_t>(position) + 1;
             i < critical_shadow_size; ++i)
            selected[i] = selected[i - 1] + 1;
    }
    if (blocker_count == 2 && blockers[1] < blockers[0]) {
        const std::uint32_t temporary = blockers[0];
        blockers[0] = blockers[1];
        blockers[1] = temporary;
    }
    const unsigned long long slot = atomicAdd(qualifying_count, 1ULL);
    if (slot >= output_capacity) return;
    RepairCandidate candidate;
    candidate.mask = mask;
    candidate.blocker_count = blocker_count;
    candidate.blocker0 = blockers[0];
    candidate.blocker1 = blockers[1];
    output[slot] = candidate;
}

} // namespace

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
    const std::function<void(const CapturedTerminalState&)>& low_defect_callback)
{
    SearchResult result;
    const auto started = std::chrono::steady_clock::now();
    auto done = [&]() {
        result.elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    };
    try {
        walkers = std::max<std::uint32_t>(1, walkers);
        proposals_per_iteration = std::max<std::uint32_t>(1, proposals_per_iteration);
        result.walkers = walkers;
        if (base_blocks.empty() || extra_blocks.empty()) {
            result.error = "CUDA walker search requires base and outsider blocks";
            return done();
        }
        if (n < 1 || n > 64 || w < 1 || w > n ||
            critical_shadow_size < 1 || critical_shadow_size > w) {
            result.error = "invalid CUDA walker parameters";
            return done();
        }
        const std::uint32_t target_m =
            static_cast<std::uint32_t>(base_blocks.size() + 1);
        if (target_m >= 65536) {
            result.error = "CUDA walker pair encoding supports fewer than 65536 blocks";
            return done();
        }
        struct SeedTemplate {
            std::uint64_t mask = 0;
            std::uint32_t score = 0;
            std::vector<std::uint32_t> bad_pairs;
        };
        std::vector<SeedTemplate> templates;
        templates.reserve(extra_blocks.size());
        std::size_t score_overflow_templates = 0;
        std::size_t edge_overflow_templates = 0;
        for (std::uint64_t extra : extra_blocks) {
            SeedTemplate item;
            item.mask = extra;
            for (std::uint32_t j = 0; j < base_blocks.size(); ++j) {
                const std::uint32_t overlap = static_cast<std::uint32_t>(
                    host_popcount64(extra & base_blocks[j]));
                const std::uint32_t contribution = choose_small(
                    overlap, static_cast<std::uint32_t>(critical_shadow_size));
                item.score = saturating_add(item.score, contribution);
                if (contribution != 0)
                    item.bad_pairs.push_back(j | ((target_m - 1) << 16));
            }
            if (item.score == UINT32_MAX) {
                ++score_overflow_templates;
            } else if (item.bad_pairs.size() > kBadCapacity) {
                ++edge_overflow_templates;
            } else {
                templates.push_back(std::move(item));
            }
        }
        if (templates.empty()) {
            result.error = "CUDA walker search has no exactly representable seed template"
                "; score_overflow=" + std::to_string(score_overflow_templates)
                + "; edge_capacity_overflow=" + std::to_string(edge_overflow_templates)
                + "; edge_capacity=" + std::to_string(kBadCapacity);
            return done();
        }

        const std::size_t total_blocks = static_cast<std::size_t>(walkers) * target_m;
        const std::size_t total_bad_pairs =
            static_cast<std::size_t>(walkers) * kBadCapacity;
        std::vector<std::uint64_t> host_blocks(total_blocks);
        std::vector<std::uint32_t> host_scores(walkers);
        std::vector<std::uint32_t> host_bad_pairs(total_bad_pairs, 0);
        std::vector<std::uint32_t> host_bad_counts(walkers);
        std::vector<std::uint32_t> host_rng(walkers);
        std::vector<DeviceMoveCounters> host_move_counts(walkers);
#if ATS48_DEVICE_DEDUP_HANDOFF
        std::vector<std::uint64_t> host_low_defect_hashes(walkers);
#endif
        result.best_score = UINT32_MAX;
        for (std::uint32_t walker = 0; walker < walkers; ++walker) {
            const SeedTemplate& item = templates[walker % templates.size()];
            const std::size_t block_offset = static_cast<std::size_t>(walker) * target_m;
            std::copy(base_blocks.begin(), base_blocks.end(),
                host_blocks.begin() + block_offset);
            host_blocks[block_offset + target_m - 1] = item.mask;
            host_scores[walker] = item.score;
            host_bad_counts[walker] = static_cast<std::uint32_t>(item.bad_pairs.size());
            std::copy(item.bad_pairs.begin(), item.bad_pairs.end(),
                host_bad_pairs.begin() + static_cast<std::size_t>(walker) * kBadCapacity);
            std::uint32_t state = static_cast<std::uint32_t>(seed)
                ^ static_cast<std::uint32_t>(seed >> 32)
                ^ (walker * 0x9e3779b9U) ^ 0xa341316cU;
            state = mix32(state);
            host_rng[walker] = state ? state : walker + 1;
            result.basin_hashes.push_back(telemetry_hash(base_blocks, item.mask));
#if ATS48_DEVICE_DEDUP_HANDOFF
            host_low_defect_hashes[walker] = result.basin_hashes.back();
#endif
            if (item.score == 2) result.defect2_state_hashes.push_back(
                result.basin_hashes.back());
            if (item.score == 1) result.defect1_state_hashes.push_back(
                result.basin_hashes.back());
            if (item.score < result.best_score) {
                result.best_score = item.score;
                result.best_blocks = base_blocks;
                result.best_blocks.push_back(item.mask);
            }
        }
        if (result.best_score <= 4) {
            for (std::uint32_t threshold = result.best_score; threshold <= 4; ++threshold)
                result.first_hit_seconds[threshold] = 0.0;
        }

        std::string error;
        int device_count = 0;
        if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error) ||
            device_count <= 0) {
            result.error = device_count <= 0 && error.empty()
                ? "no CUDA-capable device found" : error;
            return done();
        }
        if (!check_cuda(cudaSetDevice(0), "cudaSetDevice", error)) {
            result.error = error;
            return done();
        }
        result.device_name = cuda_device_name(0);

        CudaBuffer<std::uint64_t> device_blocks;
        CudaBuffer<std::uint32_t> device_scores;
        CudaBuffer<std::uint32_t> device_bad_pairs;
        CudaBuffer<std::uint32_t> device_bad_counts;
        CudaBuffer<std::uint32_t> device_rng;
        CudaBuffer<DeviceMoveCounters> device_move_counts;
        CudaBuffer<DeviceLowDefectEvent> device_events;
        CudaBuffer<std::uint32_t> device_event_count;
        CudaBuffer<std::uint64_t> device_first_defect1_blocks;
        CudaBuffer<DeviceLowDefectEvent> device_first_defect1_event;
        CudaBuffer<std::uint32_t> device_first_defect1_flag;
#if ATS48_DEVICE_DEDUP_HANDOFF
        CudaBuffer<std::uint64_t> device_low_defect_hashes;
#endif
        std::uint32_t shadow_universe = 0;
#if ATS48_INCREMENTAL_SHADOW_SCORING
        CudaBuffer<ShadowCount> device_base_shadow_counts;
        CudaBuffer<ShadowCount> device_shadow_counts;
#endif
#if ATS48_INVERTED_SHADOW_INCIDENCE
        CudaBuffer<std::uint16_t> device_shadow_heads;
        CudaBuffer<std::uint16_t> device_shadow_next;
        CudaBuffer<std::uint16_t> device_shadow_node_owners;
        CudaBuffer<std::uint16_t> device_incident_scratch;
        std::uint32_t shadow_nodes_per_walker = 0;
        std::uint32_t shadows_per_block = 0;
#endif
        if (!device_blocks.allocate(total_blocks, "cudaMalloc(walker blocks)", error) ||
            !device_scores.allocate(walkers, "cudaMalloc(walker scores)", error) ||
            !device_bad_pairs.allocate(total_bad_pairs, "cudaMalloc(walker bad pairs)", error) ||
            !device_bad_counts.allocate(walkers, "cudaMalloc(walker bad counts)", error) ||
            !device_rng.allocate(walkers, "cudaMalloc(walker RNG)", error) ||
            !device_move_counts.allocate(walkers, "cudaMalloc(walker counters)", error) ||
            !device_events.allocate(kLowDefectCapacity, "cudaMalloc(low-defect events)", error) ||
            !device_event_count.allocate(1, "cudaMalloc(low-defect event count)", error) ||
            !device_first_defect1_blocks.allocate(target_m, "cudaMalloc(first defect-1 blocks)", error) ||
            !device_first_defect1_event.allocate(1, "cudaMalloc(first defect-1 event)", error) ||
            !device_first_defect1_flag.allocate(1, "cudaMalloc(first defect-1 flag)", error)) {
            result.error = error;
            return done();
        }
#if ATS48_DEVICE_DEDUP_HANDOFF
        if (!device_low_defect_hashes.allocate(
                walkers, "cudaMalloc(low-defect state hashes)", error)) {
            result.error = error;
            return done();
        }
#endif
#if ATS48_INCREMENTAL_SHADOW_SCORING
        if (critical_shadow_size <= 3) {
            shadow_universe = choose_small(
                static_cast<std::uint32_t>(n),
                static_cast<std::uint32_t>(critical_shadow_size));
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                    "cudaMemGetInfo(incremental shadows)", error)) {
                result.error = error;
                return done();
            }
#if ATS48_INVERTED_SHADOW_INCIDENCE
            shadows_per_block = choose_small(
                static_cast<std::uint32_t>(w),
                static_cast<std::uint32_t>(critical_shadow_size));
            const std::uint64_t incidence_nodes = static_cast<std::uint64_t>(target_m)
                * shadows_per_block;
            const std::size_t head_elements = static_cast<std::size_t>(walkers)
                * shadow_universe;
            const std::size_t next_elements = static_cast<std::size_t>(walkers)
                * incidence_nodes;
            const std::size_t scratch_elements = static_cast<std::size_t>(walkers)
                * kBadCapacity;
            const std::size_t incidence_bytes = (
                head_elements + next_elements + scratch_elements + incidence_nodes)
                * sizeof(std::uint16_t);
            if (target_m >= kIncidenceMinimumTargetM
                && shadow_universe != 0 && shadows_per_block != 0
                && incidence_nodes < kNoShadowNode
                && incidence_bytes <= free_bytes / 2) {
                shadow_nodes_per_walker = static_cast<std::uint32_t>(incidence_nodes);
                std::vector<std::uint16_t> host_node_owners(shadow_nodes_per_walker);
                for (std::uint32_t block = 0; block < target_m; ++block) {
                    const std::uint32_t begin = block * shadows_per_block;
                    std::fill(
                        host_node_owners.begin() + begin,
                        host_node_owners.begin() + begin + shadows_per_block,
                        static_cast<std::uint16_t>(block));
                }
                if (!device_shadow_heads.allocate(
                        head_elements, "cudaMalloc(walker shadow heads)", error) ||
                    !device_shadow_next.allocate(
                        next_elements, "cudaMalloc(walker shadow next)", error) ||
                    !device_shadow_node_owners.allocate(
                        shadow_nodes_per_walker, "cudaMalloc(shadow node owners)", error) ||
                    !device_incident_scratch.allocate(
                        scratch_elements, "cudaMalloc(incident owner scratch)", error) ||
                    !check_cuda(cudaMemcpy(
                        device_shadow_node_owners.data, host_node_owners.data(),
                        shadow_nodes_per_walker * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice), "cudaMemcpy(shadow node owners)", error)) {
                    result.error = error;
                    return done();
                }
                result.incremental_shadow_scoring = true;
                result.incremental_shadow_bytes = incidence_bytes;
                result.inverted_shadow_incidence = true;
                result.inverted_shadow_incidence_bytes = incidence_bytes;
            }
#endif
            const std::size_t shadow_elements = static_cast<std::size_t>(walkers)
                * shadow_universe;
            const std::size_t shadow_bytes = shadow_elements * sizeof(ShadowCount);
            if (shadow_universe != 0
                && shadow_bytes + result.inverted_shadow_incidence_bytes
                    <= free_bytes / 2) {
                const std::vector<ShadowCount> host_base_shadow_counts =
                    base_shadow_counts(
                        base_blocks, static_cast<std::uint32_t>(n),
                        static_cast<std::uint32_t>(critical_shadow_size));
                if (!device_base_shadow_counts.allocate(
                        shadow_universe, "cudaMalloc(base shadow counts)", error) ||
                    !device_shadow_counts.allocate(
                        shadow_elements, "cudaMalloc(walker shadow counts)", error) ||
                    !check_cuda(cudaMemcpy(
                        device_base_shadow_counts.data, host_base_shadow_counts.data(),
                        shadow_universe * sizeof(ShadowCount), cudaMemcpyHostToDevice),
                        "cudaMemcpy(base shadow counts)", error)) {
                    result.error = error;
                    return done();
                }
                result.incremental_shadow_scoring = true;
                result.incremental_shadow_bytes = shadow_bytes
                    + result.inverted_shadow_incidence_bytes;
            }
        }
#endif
        auto copy_to_device = [&](auto& buffer, const auto& host, const char* label) {
            return check_cuda(cudaMemcpy(
                buffer.data, host.data(), host.size() * sizeof(host[0]),
                cudaMemcpyHostToDevice), label, error);
        };
        if (!copy_to_device(device_blocks, host_blocks, "cudaMemcpy(walker blocks)") ||
            !copy_to_device(device_scores, host_scores, "cudaMemcpy(walker scores)") ||
            !copy_to_device(device_bad_pairs, host_bad_pairs, "cudaMemcpy(walker bad pairs)") ||
            !copy_to_device(device_bad_counts, host_bad_counts, "cudaMemcpy(walker bad counts)") ||
            !copy_to_device(device_rng, host_rng, "cudaMemcpy(walker RNG)") ||
            !copy_to_device(device_move_counts, host_move_counts, "cudaMemcpy(walker counters)") ||
#if ATS48_DEVICE_DEDUP_HANDOFF
            !copy_to_device(
                device_low_defect_hashes, host_low_defect_hashes,
                "cudaMemcpy(low-defect state hashes)") ||
#endif
            !check_cuda(cudaMemset(device_event_count.data, 0, sizeof(std::uint32_t)),
                "cudaMemset(low-defect event count)", error) ||
            !check_cuda(cudaMemset(device_first_defect1_flag.data, 0, sizeof(std::uint32_t)),
                "cudaMemset(first defect-1 flag)", error)) {
            result.error = error;
            return done();
        }
#if ATS48_INVERTED_SHADOW_INCIDENCE
        if (result.inverted_shadow_incidence) {
            if (!check_cuda(cudaMemset(
                    device_shadow_heads.data, 0xff,
                    device_shadow_heads.count * sizeof(std::uint16_t)),
                    "cudaMemset(walker shadow heads)", error) ||
                !check_cuda(cudaMemset(
                    device_shadow_next.data, 0xff,
                    device_shadow_next.count * sizeof(std::uint16_t)),
                    "cudaMemset(walker shadow next)", error)) {
                result.error = error;
                return done();
            }
            const std::uint32_t incidence_grid =
                (walkers + kWalkThreads - 1) / kWalkThreads;
            initialize_shadow_incidence_kernel<<<incidence_grid, kWalkThreads>>>(
                device_shadow_heads.data,
                device_shadow_next.data,
                device_blocks.data,
                shadow_universe,
                shadow_nodes_per_walker,
                shadows_per_block,
                target_m,
                static_cast<std::uint32_t>(critical_shadow_size),
                walkers);
            if (!check_cuda(cudaGetLastError(),
                    "initialize_shadow_incidence_kernel launch", error)) {
                result.error = error;
                return done();
            }
        }
#endif
#if ATS48_INCREMENTAL_SHADOW_SCORING
        if (device_shadow_counts.data != nullptr) {
            initialize_shadow_counts_kernel<<<walkers, kWalkThreads>>>(
                device_shadow_counts.data,
                device_base_shadow_counts.data,
                device_blocks.data,
                shadow_universe,
                target_m,
                static_cast<std::uint32_t>(critical_shadow_size),
                walkers);
            if (!check_cuda(cudaGetLastError(),
                    "initialize_shadow_counts_kernel launch", error)) {
                result.error = error;
                return done();
            }
        }
#endif
        result.ran = true;
        const std::uint32_t grid = (walkers + kWalkThreads - 1) / kWalkThreads;
        std::unordered_set<std::uint64_t> emitted_low_defect_hashes;
        while (std::chrono::steady_clock::now() < deadline &&
            !(stop && stop->load(std::memory_order_relaxed))) {
            swap_walkers_kernel<<<grid, kWalkThreads>>>(
                device_blocks.data,
                device_scores.data,
                device_bad_pairs.data,
                device_bad_counts.data,
                device_rng.data,
                device_move_counts.data,
                device_events.data,
                device_event_count.data,
                kLowDefectCapacity,
                device_first_defect1_blocks.data,
                device_first_defect1_event.data,
                device_first_defect1_flag.data,
#if ATS48_DEVICE_DEDUP_HANDOFF
                device_low_defect_hashes.data,
#endif
#if ATS48_INCREMENTAL_SHADOW_SCORING
                result.incremental_shadow_scoring ? device_shadow_counts.data : nullptr,
                shadow_universe,
#endif
#if ATS48_INVERTED_SHADOW_INCIDENCE
                result.inverted_shadow_incidence ? device_shadow_heads.data : nullptr,
                result.inverted_shadow_incidence ? device_shadow_next.data : nullptr,
                result.inverted_shadow_incidence ? device_shadow_node_owners.data : nullptr,
                result.inverted_shadow_incidence ? device_incident_scratch.data : nullptr,
                shadow_nodes_per_walker,
                shadows_per_block,
#endif
                target_m,
                static_cast<std::uint32_t>(n),
                static_cast<std::uint32_t>(w),
                static_cast<std::uint32_t>(critical_shadow_size),
                walkers,
                proposals_per_iteration,
                0.75f,
                stop_on_first_defect1);
            if (!check_cuda(cudaGetLastError(), "swap_walkers_kernel launch", error) ||
                !check_cuda(cudaMemcpy(
                    host_scores.data(), device_scores.data,
                    walkers * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(walker scores)", error)) {
                result.error = error;
                break;
            }
            const auto minimum = std::min_element(host_scores.begin(), host_scores.end());
            const std::uint32_t minimum_score = *minimum;
            const std::uint32_t minimum_walker = static_cast<std::uint32_t>(
                std::distance(host_scores.begin(), minimum));
            const double hit_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            if (minimum_score <= 4) {
                for (std::uint32_t threshold = minimum_score; threshold <= 4; ++threshold) {
                    if (result.first_hit_seconds[threshold] < 0.0)
                        result.first_hit_seconds[threshold] = hit_seconds;
                }
            }
            if (minimum_score < result.best_score) {
                std::vector<std::uint64_t> candidate(target_m);
                if (!check_cuda(cudaMemcpy(
                        candidate.data(),
                        device_blocks.data + static_cast<std::size_t>(minimum_walker) * target_m,
                        target_m * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy(best walker state)", error)) {
                    result.error = error;
                    break;
                }
                const std::uint32_t checked = exact_score(
                    candidate, static_cast<std::uint32_t>(critical_shadow_size));
                if (checked != minimum_score) {
                    result.error = "CUDA walker score cross-check failed: device="
                        + std::to_string(minimum_score) + " CPU=" + std::to_string(checked);
                    break;
                }
                result.best_score = minimum_score;
                result.best_blocks = std::move(candidate);
            }
            if (low_defect_callback) {
#if ATS48_DEVICE_DEDUP_HANDOFF
                const bool has_low_defect_walker = std::any_of(
                    host_scores.begin(), host_scores.end(),
                    [](std::uint32_t score) { return score == 1 || score == 2; });
                if (has_low_defect_walker && !check_cuda(cudaMemcpy(
                        host_low_defect_hashes.data(), device_low_defect_hashes.data,
                        walkers * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy(low-defect state hashes)", error)) {
                    result.error = error;
                    break;
                }
#endif
                for (std::uint32_t walker = 0; walker < walkers; ++walker) {
                    if (host_scores[walker] != 1 && host_scores[walker] != 2) continue;
                    ++result.low_defect_walkers_screened;
#if ATS48_DEVICE_DEDUP_HANDOFF
                    if (emitted_low_defect_hashes.contains(
                            host_low_defect_hashes[walker])) continue;
#endif
                    CapturedTerminalState captured;
                    captured.blocks.resize(target_m);
                    DeviceMoveCounters counts;
                    if (!check_cuda(cudaMemcpy(
                            captured.blocks.data(),
                            device_blocks.data + static_cast<std::size_t>(walker) * target_m,
                            target_m * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy(low-defect handoff blocks)", error) ||
                        !check_cuda(cudaMemcpy(
                            &counts, device_move_counts.data + walker, sizeof(counts),
                            cudaMemcpyDeviceToHost),
                            "cudaMemcpy(low-defect handoff counters)", error)) {
                        result.error = error;
                        break;
                    }
                    ++result.low_defect_state_copies;
                    const std::uint32_t checked = exact_score(
                        captured.blocks, static_cast<std::uint32_t>(critical_shadow_size));
                    if (checked != host_scores[walker]) {
                        result.error = "CUDA low-defect handoff score cross-check failed";
                        break;
                    }
                    captured.state_hash = telemetry_hash(captured.blocks);
#if ATS48_DEVICE_DEDUP_HANDOFF
                    if (captured.state_hash != host_low_defect_hashes[walker]) {
                        result.error = "CUDA low-defect handoff hash cross-check failed";
                        break;
                    }
#endif
                    if (!emitted_low_defect_hashes.insert(captured.state_hash).second) continue;
                    captured.iteration = counts.iterations;
                    captured.defect = checked;
                    captured.walker_id = walker;
                    low_defect_callback(captured);
                    ++result.low_defect_states_emitted;
                }
                if (!result.error.empty()) break;
            }
            if (result.best_score == 0 || (stop_on_first_defect1 && minimum_score == 1)) break;
        }

        if (!check_cuda(cudaMemcpy(
                host_move_counts.data(), device_move_counts.data,
                walkers * sizeof(DeviceMoveCounters), cudaMemcpyDeviceToHost),
                "cudaMemcpy(final walker counters)", error) ||
            !check_cuda(cudaMemcpy(
                host_scores.data(), device_scores.data,
                walkers * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                "cudaMemcpy(final walker scores)", error) ||
            !check_cuda(cudaMemcpy(
                host_bad_counts.data(), device_bad_counts.data,
                walkers * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                "cudaMemcpy(final walker bad counts)", error) ||
            !check_cuda(cudaMemcpy(
                host_blocks.data(), device_blocks.data,
                total_blocks * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                "cudaMemcpy(final walker blocks)", error)) {
            if (result.error.empty()) result.error = error;
            return done();
        }
        for (const DeviceMoveCounters& counts : host_move_counts) {
            if (counts.incidence_errors != 0) {
                result.error = "inverted CUDA shadow incidence update failed";
                return done();
            }
        }
#if ATS48_INCREMENTAL_SHADOW_SCORING
        if (result.incremental_shadow_scoring) {
            const auto most_active = std::max_element(
                host_move_counts.begin(), host_move_counts.end(),
                [](const DeviceMoveCounters& first, const DeviceMoveCounters& second) {
                    return first.accepted < second.accepted;
                });
            const std::uint32_t audit_walker = static_cast<std::uint32_t>(
                std::distance(host_move_counts.begin(), most_active));
            const auto begin = host_blocks.begin()
                + static_cast<std::size_t>(audit_walker) * target_m;
            const std::vector<std::uint64_t> audit_blocks(begin, begin + target_m);
            const std::uint32_t checked = exact_score(
                audit_blocks, static_cast<std::uint32_t>(critical_shadow_size));
            if (checked != host_scores[audit_walker]) {
                result.error = "incremental CUDA shadow score final audit failed: device="
                    + std::to_string(host_scores[audit_walker])
                    + " CPU=" + std::to_string(checked);
                return done();
            }
            if (result.inverted_shadow_incidence) {
                std::vector<std::uint32_t> device_pairs(kBadCapacity);
                if (!check_cuda(cudaMemcpy(
                        device_pairs.data(),
                        device_bad_pairs.data
                            + static_cast<std::size_t>(audit_walker) * kBadCapacity,
                        kBadCapacity * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy(audit walker bad pairs)", error)) {
                    result.error = error;
                    return done();
                }
                device_pairs.resize(host_bad_counts[audit_walker]);
                std::vector<std::uint32_t> expected_pairs;
                for (std::uint32_t i = 0; i < target_m; ++i) {
                    for (std::uint32_t j = 0; j < i; ++j) {
                        if (host_popcount64(audit_blocks[i] & audit_blocks[j])
                                >= critical_shadow_size)
                            expected_pairs.push_back(j | (i << 16));
                    }
                }
                std::sort(device_pairs.begin(), device_pairs.end());
                std::sort(expected_pairs.begin(), expected_pairs.end());
                if (device_pairs != expected_pairs) {
                    result.error = "inverted CUDA shadow incidence bad-pair audit failed";
                    return done();
                }
            }
            result.incremental_shadow_audits = 1;
        }
#endif
        for (const DeviceMoveCounters& counts : host_move_counts) {
            result.accepted_moves += counts.accepted;
            result.proposals_scored += counts.proposals;
            result.walker_iterations += counts.iterations;
        }
        for (std::uint32_t walker = 0; walker < walkers; ++walker) {
            if (host_scores[walker] != 1) continue;
            CapturedTerminalState terminal;
            const auto begin = host_blocks.begin()
                + static_cast<std::size_t>(walker) * target_m;
            terminal.blocks.assign(begin, begin + target_m);
            const std::uint32_t checked = exact_score(
                terminal.blocks, static_cast<std::uint32_t>(critical_shadow_size));
            if (checked != 1) {
                result.error = "CUDA terminal state score cross-check failed";
                return done();
            }
            terminal.state_hash = telemetry_hash(terminal.blocks);
            terminal.iteration = host_move_counts[walker].iterations;
            terminal.walker_id = walker;
            result.terminal_states.push_back(std::move(terminal));
        }

        std::uint32_t first_defect1_flag = 0;
        if (!check_cuda(cudaMemcpy(
                &first_defect1_flag, device_first_defect1_flag.data,
                sizeof(first_defect1_flag), cudaMemcpyDeviceToHost),
                "cudaMemcpy(first defect-1 flag)", error)) {
            result.error = error;
            return done();
        }
        if (first_defect1_flag == 1) {
            result.error = "first CUDA defect-1 snapshot was not committed";
            return done();
        }
        if (first_defect1_flag == 2) {
            DeviceLowDefectEvent event;
            CapturedTerminalState terminal;
            terminal.blocks.resize(target_m);
            if (!check_cuda(cudaMemcpy(
                    &event, device_first_defect1_event.data, sizeof(event),
                    cudaMemcpyDeviceToHost), "cudaMemcpy(first defect-1 event)", error) ||
                !check_cuda(cudaMemcpy(
                    terminal.blocks.data(), device_first_defect1_blocks.data,
                    target_m * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(first defect-1 blocks)", error)) {
                result.error = error;
                return done();
            }
            const std::uint32_t checked = exact_score(
                terminal.blocks, static_cast<std::uint32_t>(critical_shadow_size));
            if (checked != 1 || telemetry_hash(terminal.blocks) != event.state_hash) {
                result.error = "first CUDA defect-1 snapshot failed CPU state/hash cross-check";
                return done();
            }
            terminal.state_hash = event.state_hash;
            terminal.iteration = event.iteration;
            terminal.walker_id = event.walker;
            result.first_terminal_state = std::move(terminal);
            result.has_first_terminal_state = true;
        }

        std::uint32_t event_count = 0;
        if (!check_cuda(cudaMemcpy(
                &event_count, device_event_count.data, sizeof(event_count),
                cudaMemcpyDeviceToHost), "cudaMemcpy(low-defect event count)", error)) {
            result.error = error;
            return done();
        }
        const std::uint32_t stored_events = std::min(event_count, kLowDefectCapacity);
        result.low_defect_events_dropped = event_count - stored_events;
        if (stored_events != 0) {
            std::vector<DeviceLowDefectEvent> events(stored_events);
            if (!check_cuda(cudaMemcpy(
                    events.data(), device_events.data,
                    stored_events * sizeof(DeviceLowDefectEvent), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(low-defect events)", error)) {
                result.error = error;
                return done();
            }
            result.low_defect_records.reserve(stored_events);
            for (const DeviceLowDefectEvent& event : events) {
                if (event.score == 2) result.defect2_state_hashes.push_back(event.state_hash);
                if (event.score == 1) result.defect1_state_hashes.push_back(event.state_hash);
                CapturedLowDefectRecord record;
                record.state_hash = event.state_hash;
                record.conflict_signature = event.conflict_signature;
                record.iteration = event.iteration;
                record.defect = event.score;
                record.walker_id = event.walker;
                record.move_class = event.move_class;
                result.low_defect_records.push_back(record);
            }
        }
    } catch (const std::exception& exception) {
        result.error = exception.what();
    }
    return done();
}

CompatibilityScanResult scan_repair_candidates(
    const std::vector<std::uint64_t>& valid_core,
    int n,
    int w,
    int critical_shadow_size,
    std::uint32_t max_blockers,
    std::uint32_t output_capacity)
{
    CompatibilityScanResult result;
    const auto started = std::chrono::steady_clock::now();
    auto done = [&]() {
        result.elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    };
    try {
        if (valid_core.empty() || n < 1 || n > 64 || w < 1 || w > n ||
            critical_shadow_size < 1 || critical_shadow_size > w) {
            result.error = "invalid CUDA repair scanner parameters";
            return done();
        }
        if (max_blockers > 2 || output_capacity == 0) {
            result.error = "CUDA repair scanner supports max_blockers 0..2 and nonzero output capacity";
            return done();
        }

        std::vector<std::uint64_t> choose_table(65U * 65U, 0);
        for (std::uint32_t value = 0; value <= 64; ++value) {
            choose_table[value * 65U] = 1;
            for (std::uint32_t choose = 1; choose <= value; ++choose) {
                choose_table[value * 65U + choose] = choose == value
                    ? 1
                    : choose_table[(value - 1) * 65U + choose - 1]
                        + choose_table[(value - 1) * 65U + choose];
            }
        }
        result.universe_size = choose_table[static_cast<std::size_t>(n) * 65U + w];
        const std::uint64_t shadows_per_block =
            choose_table[static_cast<std::size_t>(w) * 65U + critical_shadow_size];
        if (result.universe_size == 0 || shadows_per_block == 0) {
            result.error = "CUDA repair scanner combination table overflow or empty universe";
            return done();
        }
        if (valid_core.size() > static_cast<std::size_t>(INT32_MAX)) {
            result.error = "CUDA repair core exceeds blocker owner index range";
            return done();
        }

        if (shadows_per_block > std::numeric_limits<std::size_t>::max()
                / valid_core.size()) {
            result.error = "CUDA repair shadow index size overflow";
            return done();
        }
        std::vector<std::pair<std::uint64_t, std::int32_t>> shadow_index;
        shadow_index.reserve(
            static_cast<std::size_t>(shadows_per_block) * valid_core.size());
        std::vector<std::uint64_t> sorted_core = valid_core;
        std::sort(sorted_core.begin(), sorted_core.end());
        if (std::adjacent_find(sorted_core.begin(), sorted_core.end()) != sorted_core.end()) {
            result.error = "CUDA repair core contains duplicate blocks";
            return done();
        }
        for (std::size_t block_id = 0; block_id < valid_core.size(); ++block_id) {
            const std::uint64_t block = valid_core[block_id];
            if (static_cast<int>(host_popcount64(block)) != w ||
                (n < 64 && (block >> n) != 0)) {
                result.error = "CUDA repair core contains an invalid block";
                return done();
            }
            for (std::size_t earlier = 0; earlier < block_id; ++earlier) {
                if (host_popcount64(block & valid_core[earlier]) >= critical_shadow_size) {
                    result.error = "CUDA repair core is not valid at its critical shadow size";
                    return done();
                }
            }
            std::uint32_t coordinates[64];
            std::uint32_t coordinate_count = 0;
            std::uint64_t remaining = block;
            while (remaining != 0) {
                coordinates[coordinate_count++] = static_cast<std::uint32_t>(
                    host_countr_zero64(remaining));
                remaining &= remaining - 1;
            }
            std::uint32_t selected[64];
            for (int i = 0; i < critical_shadow_size; ++i) selected[i] = i;
            while (true) {
                std::uint64_t shadow_rank = 0;
                for (int i = 0; i < critical_shadow_size; ++i)
                    shadow_rank += choose_table[
                        coordinates[selected[i]] * 65U + static_cast<std::uint32_t>(i + 1)];
                shadow_index.emplace_back(
                    shadow_rank, static_cast<std::int32_t>(block_id));

                int position = critical_shadow_size - 1;
                while (position >= 0 && selected[position]
                    == coordinate_count - critical_shadow_size
                        + static_cast<std::uint32_t>(position))
                    --position;
                if (position < 0) break;
                ++selected[position];
                for (int i = position + 1; i < critical_shadow_size; ++i)
                    selected[i] = selected[i - 1] + 1;
            }
        }
        std::sort(shadow_index.begin(), shadow_index.end());
        for (std::size_t i = 1; i < shadow_index.size(); ++i) {
            if (shadow_index[i - 1].first == shadow_index[i].first) {
                result.error = "CUDA repair core repeats a critical shadow owner";
                return done();
            }
        }
        std::vector<std::uint64_t> shadow_ranks;
        std::vector<std::int32_t> shadow_owners;
        shadow_ranks.reserve(shadow_index.size());
        shadow_owners.reserve(shadow_index.size());
        for (const auto& [rank, owner] : shadow_index) {
            shadow_ranks.push_back(rank);
            shadow_owners.push_back(owner);
        }

        std::string error;
        int device_count = 0;
        if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", error) ||
            device_count <= 0) {
            result.error = device_count <= 0 && error.empty()
                ? "no CUDA-capable device found" : error;
            return done();
        }
        if (!check_cuda(cudaSetDevice(0), "cudaSetDevice", error)) {
            result.error = error;
            return done();
        }
        result.device_name = cuda_device_name(0);

        CudaBuffer<std::uint64_t> device_core;
        CudaBuffer<std::uint64_t> device_shadow_ranks;
        CudaBuffer<std::int32_t> device_shadow_owners;
        CudaBuffer<std::uint64_t> device_choose_table;
        CudaBuffer<RepairCandidate> device_output;
        CudaBuffer<unsigned long long> device_qualifying_count;
        if (!device_core.allocate(valid_core.size(), "cudaMalloc(repair core)", error) ||
            !device_shadow_ranks.allocate(
                shadow_ranks.size(), "cudaMalloc(shadow rank index)", error) ||
            !device_shadow_owners.allocate(
                shadow_owners.size(), "cudaMalloc(shadow owner index)", error) ||
            !device_choose_table.allocate(
                choose_table.size(), "cudaMalloc(combination table)", error) ||
            !device_output.allocate(output_capacity, "cudaMalloc(repair output)", error) ||
            !device_qualifying_count.allocate(
                1, "cudaMalloc(repair qualifying count)", error)) {
            result.error = error;
            return done();
        }
        if (!check_cuda(cudaMemcpy(
                device_core.data, valid_core.data(),
                valid_core.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                "cudaMemcpy(repair core)", error) ||
            !check_cuda(cudaMemcpy(
                device_shadow_ranks.data, shadow_ranks.data(),
                shadow_ranks.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                "cudaMemcpy(shadow rank index)", error) ||
            !check_cuda(cudaMemcpy(
                device_shadow_owners.data, shadow_owners.data(),
                shadow_owners.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                "cudaMemcpy(shadow owner index)", error) ||
            !check_cuda(cudaMemcpy(
                device_choose_table.data, choose_table.data(),
                choose_table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                "cudaMemcpy(combination table)", error) ||
            !check_cuda(cudaMemset(
                device_qualifying_count.data, 0, sizeof(unsigned long long)),
                "cudaMemset(repair qualifying count)", error)) {
            result.error = error;
            return done();
        }

        result.ran = true;
        for (std::uint64_t first_rank = 0;
             first_rank < result.universe_size;
             first_rank += kRepairRanksPerLaunch) {
            const std::uint32_t rank_count = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    kRepairRanksPerLaunch, result.universe_size - first_rank));
            const std::uint32_t blocks =
                (rank_count + kRepairThreads - 1) / kRepairThreads;
            scan_repair_candidates_kernel<<<blocks, kRepairThreads>>>(
                device_core.data,
                device_shadow_ranks.data,
                device_shadow_owners.data,
                static_cast<std::uint64_t>(shadow_ranks.size()),
                device_choose_table.data,
                static_cast<std::uint32_t>(n),
                static_cast<std::uint32_t>(w),
                static_cast<std::uint32_t>(critical_shadow_size),
                max_blockers,
                first_rank,
                rank_count,
                device_output.data,
                output_capacity,
                device_qualifying_count.data);
            if (!check_cuda(
                    cudaGetLastError(), "scan_repair_candidates_kernel launch", error) ||
                !check_cuda(
                    cudaDeviceSynchronize(), "scan_repair_candidates_kernel", error)) {
                result.error = error;
                return done();
            }
            result.candidates_examined += rank_count;
        }

        unsigned long long qualifying = 0;
        if (!check_cuda(cudaMemcpy(
                &qualifying, device_qualifying_count.data, sizeof(qualifying),
                cudaMemcpyDeviceToHost), "cudaMemcpy(repair qualifying count)", error)) {
            result.error = error;
            return done();
        }
        result.qualifying_candidates = qualifying;
        const std::size_t stored = static_cast<std::size_t>(
            std::min<unsigned long long>(qualifying, output_capacity));
        result.overflow_candidates = qualifying - stored;
        result.candidates.resize(stored);
        if (stored != 0 && !check_cuda(cudaMemcpy(
                result.candidates.data(), device_output.data,
                stored * sizeof(RepairCandidate), cudaMemcpyDeviceToHost),
                "cudaMemcpy(repair candidates)", error)) {
            result.error = error;
            return done();
        }
        std::sort(result.candidates.begin(), result.candidates.end(),
            [](const RepairCandidate& first, const RepairCandidate& second) {
                if (first.blocker_count != second.blocker_count)
                    return first.blocker_count < second.blocker_count;
                if (first.blocker0 != second.blocker0)
                    return first.blocker0 < second.blocker0;
                if (first.blocker1 != second.blocker1)
                    return first.blocker1 < second.blocker1;
                return first.mask < second.mask;
            });
        for (std::size_t candidate_id = 0;
             candidate_id < result.candidates.size(); ++candidate_id) {
            const RepairCandidate& candidate = result.candidates[candidate_id];
            if (static_cast<int>(host_popcount64(candidate.mask)) != w ||
                (n < 64 && (candidate.mask >> n) != 0) ||
                (candidate_id != 0 &&
                    candidate.mask == result.candidates[candidate_id - 1].mask)) {
                result.error = "CUDA repair candidate generation validation failed";
                return done();
            }
            std::uint32_t blockers[2] = {UINT32_MAX, UINT32_MAX};
            std::uint32_t blocker_count = 0;
            for (std::uint32_t core_id = 0; core_id < valid_core.size(); ++core_id) {
                if (candidate.mask == valid_core[core_id]) {
                    result.error = "CUDA repair scanner returned an incumbent block";
                    return done();
                }
                if (host_popcount64(candidate.mask & valid_core[core_id]) <
                    critical_shadow_size) continue;
                if (blocker_count < 2) blockers[blocker_count] = core_id;
                ++blocker_count;
            }
            if (blocker_count != candidate.blocker_count || blocker_count > 2 ||
                blockers[0] != candidate.blocker0 || blockers[1] != candidate.blocker1) {
                std::ostringstream message;
                message << "CUDA repair blocker validation failed for candidate 0x"
                        << std::hex << candidate.mask << std::dec;
                result.error = message.str();
                return done();
            }
        }
        result.exact = result.candidates_examined == result.universe_size &&
            result.overflow_candidates == 0;
    } catch (const std::exception& exception) {
        result.error = exception.what();
    }
    return done();
}

} // namespace ats45cuda
