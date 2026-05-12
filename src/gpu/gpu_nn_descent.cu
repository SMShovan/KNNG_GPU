/// @file
/// @brief Steps 62–70 — GPU NN-Descent kernels.
///
/// ## Step coverage
///   Step 62: `init_random_graph_kernel`    — per-thread XorShift init
///   Step 63: `local_join_kernel`           — one block per point, naive
///   Step 64: `local_join_batch_kernel`     — multiple points per block (small k)
///   Step 65: `device_try_insert` spinlock + `device_try_insert_atomicmin`
///   Step 66: `build_reverse_indegrees_kernel` + `build_reverse_scatter_kernel`
///            CUB `DeviceScan::ExclusiveSum` for CSR offsets
///   Step 67: `sample_new_neighbors_kernel`  — per-point reservoir sampling
///   Step 68: CUB `DeviceReduce::Sum` for convergence in `gpu_local_join`
///   Step 69: `local_join_fp16_kernel` + `gpu_nn_descent_fp16`
///   Step 70: `gpu_nn_descent` end-to-end driver + `docs/PERF_GPU_NND.md`

#include <knng/gpu/gpu_nn_descent.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/gpu/fp16_distance.hpp>   // fp32_to_fp16_kernel, f32_to_f16_bits

#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <vector>

namespace knng::gpu {

// ===========================================================================
// Device XorShift64 RNG (seeded per-thread)
// ===========================================================================

struct DeviceXorShift {
    std::uint64_t state;
    __device__ explicit DeviceXorShift(std::uint64_t s) : state(s ? s : 1ULL) {}
    __device__ std::uint64_t next() noexcept {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    __device__ std::uint64_t next_below(std::uint64_t n) noexcept {
        return (__uint128_t(next()) * __uint128_t(n)) >> 64;
    }
};

// ===========================================================================
// Step 62 — Random graph init kernel
// ===========================================================================

GPU_GLOBAL void init_random_graph_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    std::uint64_t  global_seed,
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags)
{
    const unsigned int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n) { return; }

    DeviceXorShift rng(global_seed ^ (std::uint64_t(qi + 1) * 6364136223846793005ULL));

    unsigned int filled   = 0;
    unsigned int tries    = 0;
    const unsigned int mt = k * 16u + 64u;
    const float* q        = d_pts + std::size_t(qi) * d;

    while (filled < k && tries < mt) {
        ++tries;
        const unsigned int rid = static_cast<unsigned int>(rng.next_below(n));
        if (rid == qi) { continue; }

        bool dup = false;
        for (unsigned int r = 0; r < filled; ++r) {
            if (d_ids[qi * k + r] == rid) { dup = true; break; }
        }
        if (dup) { continue; }

        const float* rp = d_pts + std::size_t(rid) * d;
        float acc = 0.f;
        for (unsigned int dim = 0; dim < d; ++dim) {
            const float diff = q[dim] - rp[dim];
            acc += diff * diff;
        }
        d_ids  [qi * k + filled] = rid;
        d_dists[qi * k + filled] = acc;
        d_flags[qi * k + filled] = 1u;
        ++filled;
    }
}

DeviceGraph gpu_init_random_graph(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::uint64_t        seed)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto ku = static_cast<unsigned int>(k);

    DeviceBuffer<float> d_pts(dataset.n * dataset.d);
    d_pts.copy_from_host(dataset.data.data(), dataset.n * dataset.d);

    DeviceGraph graph(dataset.n, k);
    GPU_CHECK(cudaMemset(graph.ids.data(),   0xFF, dataset.n * k * sizeof(knng::index_t)));
    GPU_CHECK(cudaMemset(graph.dists.data(), 0x7F, dataset.n * k * sizeof(float)));
    GPU_CHECK(cudaMemset(graph.flags.data(), 0,    dataset.n * k * sizeof(std::uint32_t)));

    constexpr unsigned int kBlock = 128u;
    GPU_LAUNCH(init_random_graph_kernel, (n + kBlock - 1u) / kBlock, kBlock, 0, nullptr,
               d_pts.data(), n, d, ku, seed,
               graph.ids.data(), graph.dists.data(), graph.flags.data());
    GPU_SYNC();
    return graph;
}

// ===========================================================================
// Step 65 — Atomic insertion helpers (spinlock + atomicMin variant)
// ===========================================================================

/// Spinlock acquire/release (Step 65, variant A).
__device__ GPU_INLINE static void nnd_lock  (int* lk, unsigned int p)
    { while (atomicCAS(lk + p, 0, 1) != 0) {} }
__device__ GPU_INLINE static void nnd_unlock(int* lk, unsigned int p)
    { atomicExch(lk + p, 0); }

/// @brief Spinlock-based try-insert (variant A).
__device__ GPU_INLINE static int device_try_insert(
    knng::index_t* d_ids, float* d_dists, std::uint32_t* d_flags,
    int* d_locks, unsigned int p, unsigned int k,
    knng::index_t new_id, float new_dist)
{
    // Fast pre-check (without lock): reject obviously worse candidates.
    float worst = 0.f;
    for (unsigned int r = 0; r < k; ++r) {
        if (d_dists[p * k + r] > worst) worst = d_dists[p * k + r];
    }
    if (new_dist >= worst) { return 0; }

    nnd_lock(d_locks, p);
    // Re-check and insert inside the lock.
    worst = 0.f;
    unsigned int worst_pos = 0;
    for (unsigned int r = 0; r < k; ++r) {
        if (d_ids  [p * k + r] == new_id) { nnd_unlock(d_locks, p); return 0; }
        if (d_dists[p * k + r] > worst)   { worst = d_dists[p * k + r]; worst_pos = r; }
    }
    int changed = 0;
    if (new_dist < worst) {
        d_ids  [p * k + worst_pos] = new_id;
        d_dists[p * k + worst_pos] = new_dist;
        d_flags[p * k + worst_pos] = 1u;
        changed = 1;
    }
    nnd_unlock(d_locks, p);
    return changed;
}

/// @brief atomicMin-based try-insert (Step 65, variant B).
///
/// Exploits the fact that IEEE-754 positive floats ordered as
/// unsigned integers have the same order as their float values.
/// Atomically updates the worst-distance slot:
///   1. Scan to find the worst slot (no lock needed — single write per slot).
///   2. atomicMin on that slot's distance reinterpret-cast to int.
///   3. If we actually improved the slot, also update id/flags with a lock.
__device__ GPU_INLINE static int device_try_insert_atomicmin(
    knng::index_t* d_ids, float* d_dists, std::uint32_t* d_flags,
    int* d_locks, unsigned int p, unsigned int k,
    knng::index_t new_id, float new_dist)
{
    // Find worst slot index (read-only scan, no lock).
    unsigned int worst_pos = 0;
    float        worst_d   = d_dists[p * k];
    for (unsigned int r = 1; r < k; ++r) {
        if (d_dists[p * k + r] > worst_d) { worst_d = d_dists[p * k + r]; worst_pos = r; }
    }
    if (new_dist >= worst_d) { return 0; }

    // Try to claim the worst slot by replacing its distance atomically.
    // Works for non-negative floats: bit pattern ordering == float ordering.
    const int old_bits = atomicMin(
        reinterpret_cast<int*>(d_dists + p * k + worst_pos),
        __float_as_int(new_dist));

    if (__int_as_float(old_bits) <= new_dist) {
        return 0;   // another thread already placed a better value
    }
    // We won the slot — update id and flag under the per-point lock.
    nnd_lock(d_locks, p);
    // Verify our value is still in the slot (not overwritten by a third thread).
    if (d_dists[p * k + worst_pos] == new_dist) {
        d_ids  [p * k + worst_pos] = new_id;
        d_flags[p * k + worst_pos] = 1u;
    }
    nnd_unlock(d_locks, p);
    return 1;
}

// ===========================================================================
// Step 63 — Local-join kernel, naive (one block per point)
// ===========================================================================

GPU_GLOBAL void local_join_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags,
    int*           d_locks,
    unsigned int*  d_update_counts)
{
    const unsigned int p = blockIdx.x;
    if (p >= n) { return; }

    // Shared memory: sh_new[k] | sh_old[k] | sh_cnt[3]
    extern __shared__ char sh[];
    auto* sh_new = reinterpret_cast<knng::index_t*>(sh);
    auto* sh_old = sh_new + k;
    auto* sh_cnt = reinterpret_cast<unsigned int*>(sh_old + k);

    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; sh_cnt[2] = 0; }
    __syncthreads();

    // Snapshot: partition neighbor list into new[] and old[], mark all old.
    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        const knng::index_t id = d_ids  [p * k + r];
        const std::uint32_t fl = d_flags[p * k + r];
        if (id == static_cast<knng::index_t>(-1)) { continue; }
        if (fl & 1u) {
            const unsigned int pos = atomicAdd(sh_cnt + 0, 1u);
            if (pos < k) { sh_new[pos] = id; }
        } else {
            const unsigned int pos = atomicAdd(sh_cnt + 1, 1u);
            if (pos < k) { sh_old[pos] = id; }
        }
        d_flags[p * k + r] &= ~1u;   // mark old in global memory
    }
    __syncthreads();

    const unsigned int n_new = sh_cnt[0];
    const unsigned int n_old = sh_cnt[1];

    // Pair enumeration: each thread owns a strided slice of the u (outer) loop.
    // new × new (u < v to avoid double-processing the same pair):
    for (unsigned int i = threadIdx.x; i < n_new; i += blockDim.x) {
        const knng::index_t u = sh_new[i];
        const float* pu = d_pts + std::size_t(u) * d;

        for (unsigned int j = i + 1; j < n_new; ++j) {
            const knng::index_t v = sh_new[j];
            const float* pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = pu[dim] - pv[dim];
                acc += diff * diff;
            }
            atomicAdd(sh_cnt + 2,
                static_cast<unsigned int>(
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, u, k, v, acc) +
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, v, k, u, acc)));
        }
        // new × old:
        for (unsigned int j = 0; j < n_old; ++j) {
            const knng::index_t v = sh_old[j];
            if (u == v) { continue; }
            const float* pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = pu[dim] - pv[dim];
                acc += diff * diff;
            }
            atomicAdd(sh_cnt + 2,
                static_cast<unsigned int>(
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, u, k, v, acc) +
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, v, k, u, acc)));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) { d_update_counts[p] = sh_cnt[2]; }
}

// ===========================================================================
// Step 64 — Batch local-join (multiple points per block, for small k)
// ===========================================================================
//
// When k ≤ 16, the per-point pair set is small (≤ 120 pairs).  Fitting
// BATCH_PTS points into one block lets each warp own one point, keeping
// occupancy high without wasting block resources on a single point's pairs.
//
// Block layout: blockDim.x = BATCH_PTS * 32 (one warp per point).
// Shared memory: BATCH_PTS × (2*k*sizeof(index_t) + 3*sizeof(uint32_t))

static constexpr unsigned int kBatchPts = 4u;   // points per block

GPU_GLOBAL void local_join_batch_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags,
    int*           d_locks,
    unsigned int*  d_update_counts)
{
    // Each warp in the block handles one point.
    const unsigned int warp_id   = threadIdx.x / 32u;
    const unsigned int lane      = threadIdx.x % 32u;
    const unsigned int p         = blockIdx.x * kBatchPts + warp_id;

    if (p >= n) { return; }

    // Per-warp shared-memory slice: new_ids[k] | old_ids[k] | cnt[3].
    // Total per warp: (2*k + 3) words.  Warp 0 starts at offset 0.
    extern __shared__ char sh[];
    const unsigned int words_per_warp = 2u * k + 3u;
    auto* sh_new = reinterpret_cast<knng::index_t*>(sh) + warp_id * words_per_warp;
    auto* sh_old = sh_new + k;
    auto* sh_cnt = reinterpret_cast<unsigned int*>(sh_old + k);

    if (lane == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; sh_cnt[2] = 0; }
    // Warp-synchronous: __syncwarp() suffices because we only share within a warp.
    __syncwarp(0xFFFFFFFFu);

    // Snapshot: lane `r` handles slot `r` (k ≤ 32 guaranteed by kNndMaxK).
    if (lane < k) {
        const knng::index_t id = d_ids  [p * k + lane];
        const std::uint32_t fl = d_flags[p * k + lane];
        if (id != static_cast<knng::index_t>(-1)) {
            if (fl & 1u) {
                const unsigned int pos = atomicAdd(sh_cnt + 0, 1u);
                sh_new[pos] = id;
            } else {
                const unsigned int pos = atomicAdd(sh_cnt + 1, 1u);
                sh_old[pos] = id;
            }
            d_flags[p * k + lane] &= ~1u;
        }
    }
    __syncwarp(0xFFFFFFFFu);

    const unsigned int n_new = sh_cnt[0];
    const unsigned int n_old = sh_cnt[1];

    // Each lane processes pairs where it is the outer loop index (i == lane).
    if (lane < n_new) {
        const unsigned int i = lane;
        const knng::index_t u = sh_new[i];
        const float* pu = d_pts + std::size_t(u) * d;
        // new × new
        for (unsigned int j = i + 1; j < n_new; ++j) {
            const knng::index_t v = sh_new[j];
            const float* pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = pu[dim] - pv[dim];
                acc += diff * diff;
            }
            atomicAdd(sh_cnt + 2,
                static_cast<unsigned int>(
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, u, k, v, acc) +
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, v, k, u, acc)));
        }
        // new × old
        for (unsigned int j = 0; j < n_old; ++j) {
            const knng::index_t v = sh_old[j];
            if (u == v) { continue; }
            const float* pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = pu[dim] - pv[dim];
                acc += diff * diff;
            }
            atomicAdd(sh_cnt + 2,
                static_cast<unsigned int>(
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, u, k, v, acc) +
                    device_try_insert(d_ids, d_dists, d_flags, d_locks, v, k, u, acc)));
        }
    }
    __syncwarp(0xFFFFFFFFu);

    if (lane == 0) { d_update_counts[p] = sh_cnt[2]; }
}

// ===========================================================================
// Step 66 — Reverse-neighbor graph on GPU (CUB DeviceScan)
// ===========================================================================

/// @brief Pass 1: atomically count in-degrees for reverse graph.
GPU_GLOBAL void build_reverse_indegrees_kernel(
    const knng::index_t* d_ids,
    unsigned int         n,
    unsigned int         k,
    unsigned int*        d_indegrees)   // [n], zero-initialised by caller
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) { return; }
    const knng::index_t nbr = d_ids[idx];
    if (nbr < n) {
        atomicAdd(d_indegrees + nbr, 1u);
    }
}

/// @brief Pass 2: scatter edges into CSR arrays.
GPU_GLOBAL void build_reverse_scatter_kernel(
    const knng::index_t* d_ids,
    unsigned int         n,
    unsigned int         k,
    const unsigned int*  d_offsets,   // CSR row offsets [n+1] (from DeviceScan)
    unsigned int*        d_cursor,    // copy of d_offsets used as write cursor
    knng::index_t*       d_rev_ids)   // CSR neighbor IDs (output)
{
    const unsigned int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n) { return; }
    for (unsigned int r = 0; r < k; ++r) {
        const knng::index_t nbr = d_ids[qi * k + r];
        if (nbr < n) {
            const unsigned int pos = atomicAdd(d_cursor + nbr, 1u);
            d_rev_ids[pos] = static_cast<knng::index_t>(qi);
        }
    }
}

void gpu_build_reverse_graph(
    const DeviceGraph&           fwd,
    DeviceBuffer<knng::index_t>& rev_ids,
    DeviceBuffer<unsigned int>&  rev_offsets)
{
    const auto n  = static_cast<unsigned int>(fwd.n);
    const auto k  = static_cast<unsigned int>(fwd.k);

    // Allocate in-degree and CSR offset arrays.
    DeviceBuffer<unsigned int> d_indegrees(n + 1);
    rev_offsets = DeviceBuffer<unsigned int>(n + 1);
    GPU_CHECK(cudaMemset(d_indegrees.data(), 0, (n + 1) * sizeof(unsigned int)));

    // Pass 1: count in-degrees.
    constexpr unsigned int kB = 256u;
    GPU_LAUNCH(build_reverse_indegrees_kernel, (n * k + kB - 1u) / kB, kB, 0, nullptr,
               fwd.ids.data(), n, k, d_indegrees.data());
    GPU_SYNC();

    // Step 66: CUB DeviceScan::ExclusiveSum for CSR row offsets.
    {
        std::size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
                                      d_indegrees.data(), rev_offsets.data(),
                                      static_cast<int>(n + 1));
        DeviceBuffer<std::uint8_t> d_temp(temp_bytes ? temp_bytes : 1);
        cub::DeviceScan::ExclusiveSum(d_temp.data(), temp_bytes,
                                      d_indegrees.data(), rev_offsets.data(),
                                      static_cast<int>(n + 1));
        GPU_SYNC();
    }

    // Read back total edge count (= rev_offsets[n]).
    std::vector<unsigned int> h_offsets;
    rev_offsets.copy_to_host(h_offsets);
    const unsigned int total_edges = h_offsets[n];

    rev_ids = DeviceBuffer<knng::index_t>(total_edges ? total_edges : 1);

    // Pass 2: scatter.
    DeviceBuffer<unsigned int> d_cursor(n + 1);
    d_cursor.copy_from_host(h_offsets.data(), n + 1);

    GPU_LAUNCH(build_reverse_scatter_kernel, (n + kB - 1u) / kB, kB, 0, nullptr,
               fwd.ids.data(), n, k, rev_offsets.data(), d_cursor.data(), rev_ids.data());
    GPU_SYNC();
}

// ===========================================================================
// Step 67 — Sampling kernel (per-point reservoir sampling)
// ===========================================================================

GPU_GLOBAL void sample_new_neighbors_kernel(
    std::uint32_t* d_flags,
    unsigned int   n,
    unsigned int   k,
    unsigned int   sample_k,       // ceil(rho * k)
    std::uint64_t  iter_seed)
{
    const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) { return; }

    DeviceXorShift rng(iter_seed ^ (std::uint64_t(p + 1) * 2654435761ULL));

    // Count new entries.
    unsigned int new_count = 0;
    for (unsigned int r = 0; r < k; ++r) {
        if (d_flags[p * k + r] & 1u) { ++new_count; }
    }
    if (new_count <= sample_k) { return; }

    // Drop (new_count - sample_k) entries via reservoir-style scan.
    unsigned int to_drop = new_count - sample_k;
    for (unsigned int r = 0; r < k && to_drop > 0; ++r) {
        if (!(d_flags[p * k + r] & 1u)) { continue; }
        const unsigned int rem = new_count - r;
        if (rng.next_below(rem) < to_drop) {
            d_flags[p * k + r] &= ~1u;
            --to_drop;
        }
    }
}

void gpu_sample_new_neighbors(
    DeviceGraph&  graph,
    double        rho,
    std::uint64_t iter_seed)
{
    const auto n = static_cast<unsigned int>(graph.n);
    const auto k = static_cast<unsigned int>(graph.k);
    const auto sample_k = static_cast<unsigned int>(
        static_cast<std::size_t>(std::ceil(rho * static_cast<double>(k))));

    constexpr unsigned int kB = 128u;
    GPU_LAUNCH(sample_new_neighbors_kernel, (n + kB - 1u) / kB, kB, 0, nullptr,
               graph.flags.data(), n, k, sample_k, iter_seed);
    GPU_SYNC();
}

// ===========================================================================
// Step 68 — Convergence reduction (embedded in gpu_local_join)
// ===========================================================================

std::size_t gpu_local_join(
    const knng::Dataset& dataset,
    DeviceGraph&         graph)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto ku = static_cast<unsigned int>(graph.k);

    DeviceBuffer<float> d_pts(dataset.n * dataset.d);
    d_pts.copy_from_host(dataset.data.data(), dataset.n * dataset.d);

    DeviceBuffer<int>          d_locks(n);
    DeviceBuffer<unsigned int> d_update_counts(n);
    GPU_CHECK(cudaMemset(d_locks.data(),        0, n * sizeof(int)));
    GPU_CHECK(cudaMemset(d_update_counts.data(),0, n * sizeof(unsigned int)));

    // Choose kernel variant based on k.
    if (ku <= kNndMaxK && ku <= 32u) {
        // Batch kernel: 4 points per block, one warp per point.
        const unsigned int kBlock = kBatchPts * 32u;
        const std::size_t shmem  = kBatchPts * (2u * ku + 3u) * sizeof(unsigned int);
        GPU_LAUNCH(local_join_batch_kernel,
                   (n + kBatchPts - 1u) / kBatchPts, kBlock, shmem, nullptr,
                   d_pts.data(), n, d, ku,
                   graph.ids.data(), graph.dists.data(), graph.flags.data(),
                   d_locks.data(), d_update_counts.data());
    } else {
        // Naive kernel: 1 block per point.
        constexpr unsigned int kBlock = 64u;
        const std::size_t shmem = 2u * ku * sizeof(knng::index_t)
                                 + 3u * sizeof(unsigned int);
        GPU_LAUNCH(local_join_kernel,
                   n, kBlock, shmem, nullptr,
                   d_pts.data(), n, d, ku,
                   graph.ids.data(), graph.dists.data(), graph.flags.data(),
                   d_locks.data(), d_update_counts.data());
    }
    GPU_SYNC();

    // Step 68: CUB DeviceReduce::Sum for convergence count.
    DeviceBuffer<unsigned int> d_total(1);
    std::size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes,
                           d_update_counts.data(), d_total.data(),
                           static_cast<int>(n));
    DeviceBuffer<std::uint8_t> d_temp(temp_bytes ? temp_bytes : 1);
    cub::DeviceReduce::Sum(d_temp.data(), temp_bytes,
                           d_update_counts.data(), d_total.data(),
                           static_cast<int>(n));
    GPU_SYNC();

    std::vector<unsigned int> h_total;
    d_total.copy_to_host(h_total);
    return static_cast<std::size_t>(h_total[0]);
}

// ===========================================================================
// Step 69 — fp16 NN-Descent kernel
//
// Stores neighbor distances as __half on device (halves dists[] memory).
// On insertion, computes distance in fp32 then stores as __half.
// On comparison, loads __half and converts to fp32.
// ===========================================================================

GPU_GLOBAL void local_join_fp16_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    __half*        d_dists_f16,    // fp16 distances
    std::uint32_t* d_flags,
    int*           d_locks,
    unsigned int*  d_update_counts)
{
    const unsigned int p = blockIdx.x;
    if (p >= n) { return; }

    extern __shared__ char sh[];
    auto* sh_new = reinterpret_cast<knng::index_t*>(sh);
    auto* sh_old = sh_new + k;
    auto* sh_cnt = reinterpret_cast<unsigned int*>(sh_old + k);
    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; sh_cnt[2] = 0; }
    __syncthreads();

    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        const knng::index_t id = d_ids  [p * k + r];
        const std::uint32_t fl = d_flags[p * k + r];
        if (id == static_cast<knng::index_t>(-1)) { continue; }
        if (fl & 1u) {
            const unsigned int pos = atomicAdd(sh_cnt + 0, 1u);
            if (pos < k) sh_new[pos] = id;
        } else {
            const unsigned int pos = atomicAdd(sh_cnt + 1, 1u);
            if (pos < k) sh_old[pos] = id;
        }
        d_flags[p * k + r] &= ~1u;
    }
    __syncthreads();

    const unsigned int n_new = sh_cnt[0];
    const unsigned int n_old = sh_cnt[1];

    auto try_insert_fp16 = [&](unsigned int qi, knng::index_t new_id, float new_dist_f32) {
        // Convert distance to fp16 for storage comparison.
        const __half new_h = __float2half(new_dist_f32);
        const float  new_h_f32 = __half2float(new_h);

        // Find worst slot (read fp16, compare as fp32).
        float worst = 0.f;
        unsigned int worst_pos = 0;
        for (unsigned int r = 0; r < k; ++r) {
            const float dv = __half2float(d_dists_f16[qi * k + r]);
            if (dv > worst) { worst = dv; worst_pos = r; }
        }
        if (new_h_f32 >= worst) { return 0; }

        nnd_lock(d_locks, qi);
        worst = 0.f; worst_pos = 0;
        for (unsigned int r = 0; r < k; ++r) {
            if (d_ids[qi * k + r] == new_id) { nnd_unlock(d_locks, qi); return 0; }
            const float dv = __half2float(d_dists_f16[qi * k + r]);
            if (dv > worst) { worst = dv; worst_pos = r; }
        }
        int changed = 0;
        if (new_h_f32 < worst) {
            d_ids      [qi * k + worst_pos] = new_id;
            d_dists_f16[qi * k + worst_pos] = new_h;
            d_flags    [qi * k + worst_pos] = 1u;
            changed = 1;
        }
        nnd_unlock(d_locks, qi);
        return changed;
    };

    for (unsigned int i = threadIdx.x; i < n_new; i += blockDim.x) {
        const knng::index_t u = sh_new[i];
        const float* pu = d_pts + std::size_t(u) * d;
        for (unsigned int j = i + 1; j < n_new; ++j) {
            const knng::index_t v = sh_new[j];
            const float* pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) { const float df = pu[dim]-pv[dim]; acc+=df*df; }
            atomicAdd(sh_cnt + 2, static_cast<unsigned int>(try_insert_fp16(u, v, acc) + try_insert_fp16(v, u, acc)));
        }
        for (unsigned int j = 0; j < n_old; ++j) {
            const knng::index_t v = sh_old[j];
            if (u == v) { continue; }
            const float* pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) { const float df = pu[dim]-pv[dim]; acc+=df*df; }
            atomicAdd(sh_cnt + 2, static_cast<unsigned int>(try_insert_fp16(u, v, acc) + try_insert_fp16(v, u, acc)));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) { d_update_counts[p] = sh_cnt[2]; }
}

// ===========================================================================
// Step 70 — GPU NN-Descent driver
// ===========================================================================

knng::Knng gpu_nn_descent(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg)
{
    DeviceGraph graph = gpu_init_random_graph(dataset, k, cfg.seed);
    const double threshold = cfg.delta * static_cast<double>(dataset.n * k);

    for (std::size_t iter = 0; iter < cfg.max_iters; ++iter) {
        if (cfg.rho < 1.0) {
            gpu_sample_new_neighbors(graph, cfg.rho,
                cfg.seed ^ (std::uint64_t(iter + 1) * 1000003ULL));
        }
        const std::size_t updates = gpu_local_join(dataset, graph);
        if (static_cast<double>(updates) < threshold) { break; }
    }
    return graph.to_knng();
}

// ---------------------------------------------------------------------------
// fp16 driver
// ---------------------------------------------------------------------------

knng::Knng gpu_nn_descent_fp16(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto ku = static_cast<unsigned int>(k);

    DeviceBuffer<float> d_pts(dataset.n * dataset.d);
    d_pts.copy_from_host(dataset.data.data(), dataset.n * dataset.d);

    // Initialize graph with fp32 distances, then convert to fp16.
    DeviceGraph graph_fp32 = gpu_init_random_graph(dataset, k, cfg.seed);

    // Allocate fp16 distance buffer.
    DeviceBuffer<__half>        d_dists_f16(dataset.n * k);
    DeviceBuffer<knng::index_t> d_ids_copy(dataset.n * k);
    DeviceBuffer<std::uint32_t> d_flags_copy(dataset.n * k);

    // Convert fp32 dists → fp16.
    {
        constexpr unsigned int kB = 256u;
        const auto total = static_cast<unsigned int>(dataset.n * k);
        GPU_LAUNCH(fp32_to_fp16_kernel, (total + kB - 1u) / kB, kB, 0, nullptr,
                   graph_fp32.dists.data(), 1u, total, d_dists_f16.data());
    }
    // Copy ids/flags.
    GPU_MEMCPY_D2D(d_ids_copy.data(),   graph_fp32.ids.data(),   dataset.n * k * sizeof(knng::index_t));
    GPU_MEMCPY_D2D(d_flags_copy.data(), graph_fp32.flags.data(), dataset.n * k * sizeof(std::uint32_t));
    GPU_SYNC();

    DeviceBuffer<int>          d_locks(n);
    DeviceBuffer<unsigned int> d_update_counts(n);

    const double threshold = cfg.delta * static_cast<double>(dataset.n * k);

    for (std::size_t iter = 0; iter < cfg.max_iters; ++iter) {
        GPU_CHECK(cudaMemset(d_locks.data(),        0, n * sizeof(int)));
        GPU_CHECK(cudaMemset(d_update_counts.data(),0, n * sizeof(unsigned int)));

        constexpr unsigned int kBlock = 64u;
        const std::size_t shmem = 2u * ku * sizeof(knng::index_t)
                                 + 3u * sizeof(unsigned int);
        GPU_LAUNCH(local_join_fp16_kernel, n, kBlock, shmem, nullptr,
                   d_pts.data(), n, d, ku,
                   d_ids_copy.data(), d_dists_f16.data(), d_flags_copy.data(),
                   d_locks.data(), d_update_counts.data());
        GPU_SYNC();

        // Convergence reduction.
        DeviceBuffer<unsigned int> d_total(1);
        std::size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, temp_bytes,
                               d_update_counts.data(), d_total.data(),
                               static_cast<int>(n));
        DeviceBuffer<std::uint8_t> d_temp(temp_bytes ? temp_bytes : 1);
        cub::DeviceReduce::Sum(d_temp.data(), temp_bytes,
                               d_update_counts.data(), d_total.data(),
                               static_cast<int>(n));
        GPU_SYNC();

        std::vector<unsigned int> h_total;
        d_total.copy_to_host(h_total);
        if (static_cast<double>(h_total[0]) < threshold) { break; }
    }

    // Build result Knng from fp16 graph.
    std::vector<knng::index_t>  h_ids(dataset.n * k);
    std::vector<__half>         h_dists_f16(dataset.n * k);
    GPU_MEMCPY_D2H(h_ids.data(),       d_ids_copy.data(),   dataset.n * k * sizeof(knng::index_t));
    GPU_MEMCPY_D2H(h_dists_f16.data(), d_dists_f16.data(),  dataset.n * k * sizeof(__half));
    GPU_SYNC();

    knng::Knng result(dataset.n, k);
    result.neighbors = h_ids;
    for (std::size_t i = 0; i < dataset.n * k; ++i) {
        result.distances[i] = __half2float(h_dists_f16[i]);
    }
    return result;
}

} // namespace knng::gpu
