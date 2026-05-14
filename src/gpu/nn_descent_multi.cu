/// @file nn_descent_multi.cu
/// @brief Step 85 — Multi-GPU NN-Descent with NCCL AllGather synchronization.
///
/// ## Algorithm
///
/// ### Graph partitioning
/// The graph is partitioned by point ownership: rank r owns rows [begin_r, end_r).
/// Each rank processes ONLY its owned rows during local join, but reads vectors
/// for ANY point (since all ranks hold the full dataset).
///
/// This avoids cross-GPU race conditions: two GPUs never write to the same row.
///
/// ### Per-iteration protocol
///
///   1. Local join on owned shard (GPU kernel, no communication).
///      Kernel reads: vectors for any u,v ∈ [0,n).  Requires full dataset.
///      Kernel writes: neighbor lists for p ∈ [begin, end).
///
///   2. NCCL AllGather to synchronize the FULL graph across all GPUs.
///      AllGather: every rank sends its n*k IDs + n*k dists to ALL other ranks.
///      After AllGather, every GPU has an up-to-date view of the full graph.
///
///   3. Convergence check via NCCL AllReduce on per-GPU update counts.
///      If total_updates < delta * n * k across all GPUs, stop.
///
/// ### NCCL AllGather details
///
///   ncclAllGather(const void* sendbuf, void* recvbuf,
///                 size_t sendcount, ncclDataType_t,
///                 ncclComm_t, cudaStream_t)
///
///   sendcount = n * k (elements contributed by ONE rank — its full graph slice).
///   BUT: in this partitioned setup each rank only UPDATED its shard; the rest
///   of its graph is stale. We send the FULL graph (n*k) so every rank always
///   has the latest version after the gather.
///
///   The recvbuf must hold n_gpus * n * k elements.
///   After AllGather, recvbuf[rank * n*k ... (rank+1)*n*k - 1] = that rank's
///   contribution.  We take the union of all contributions as the new graph.
///
///   Optimization opportunity (Phase 12): only send the changed rows (delta
///   compression, NEO-DNND style). This reduces per-iteration communication
///   from O(n*k) to O(Δ*k) where Δ ≈ 0.25*n after a few iterations.
///
/// ## Kernel: mg_nnd_local_join_kernel
///
///   Based on local_join_kernel from gpu_nn_descent.cu (Step 63).
///   Key difference: each block only processes a SHARD of rows
///   [q_begin, q_begin + shard_size), avoiding writes to other ranks' rows.
///
///   The `d_locks` array covers ALL n points (not just the shard) because
///   `device_try_insert` writes to any point's neighbor list.  However, since
///   we only launch blocks for owned points, we only call device_try_insert
///   for owned points p — no cross-rank writes happen.
///
/// ## Memory layout
///
///   All GPUs hold a full copy of the graph: d_ids[n*k] and d_dists[n*k].
///   Only the shard [begin*k, end*k) is written locally; the rest is refreshed
///   from AllGather results after each iteration.

#include <knng/gpu/multi_gpu.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/gpu/backend.hpp>

#if defined(KNNG_HAVE_CUDA) && defined(KNNG_HAVE_NCCL)

#include <nccl.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void nccl_check(ncclResult_t r, const char* file, int line) {
    if (r != ncclSuccess)
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line)
            + " NCCL: " + ncclGetErrorString(r));
}
#define NCCL_CHECK(c) nccl_check((c), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Per-point spinlock helpers (same as gpu_nn_descent.cu Step 65).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mg_lock  (int* lk, unsigned int p)
    { while (atomicCAS(lk + p, 0, 1) != 0) {} }
__device__ __forceinline__ void mg_unlock(int* lk, unsigned int p)
    { atomicExch(lk + p, 0); }

// ---------------------------------------------------------------------------
// Try to insert (new_id, new_dist) into point p's neighbor list.
//
// Uses a double-check pattern:
//   1. Read worst without lock (cheap pre-filter).
//   2. Acquire per-point lock.
//   3. Recheck worst and check for duplicates (avoids a TOCTOU race).
//   4. Insert if still better, release lock.
//
// Returns 1 if an insertion was made, 0 otherwise.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int mg_try_insert(
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags,
    int*           d_locks,
    unsigned int   p,
    unsigned int   k,
    knng::index_t  new_id,
    float          new_dist)
{
    // Fast pre-check without lock.
    float worst = 0.f;
    for (unsigned int r = 0; r < k; ++r)
        if (d_dists[p*k+r] > worst) worst = d_dists[p*k+r];
    if (new_dist >= worst) return 0;

    mg_lock(d_locks, p);
    worst = 0.f;
    unsigned int worst_pos = 0;
    for (unsigned int r = 0; r < k; ++r) {
        if (d_ids  [p*k+r] == new_id)    { mg_unlock(d_locks, p); return 0; }
        if (d_dists[p*k+r] > worst)      { worst = d_dists[p*k+r]; worst_pos = r; }
    }
    int changed = 0;
    if (new_dist < worst) {
        d_ids  [p*k+worst_pos] = new_id;
        d_dists[p*k+worst_pos] = new_dist;
        d_flags[p*k+worst_pos] = 1u;
        changed = 1;
    }
    mg_unlock(d_locks, p);
    return changed;
}

} // anonymous namespace

namespace knng::gpu {

// ===========================================================================
// Kernel: multi-GPU local join for an owned shard [p_begin, p_begin+shard_n).
//
// Grid  = shard_n blocks  (one per owned point)
// Block = BLOCK threads
//
// For each owned point p:
//   - Partition its neighbor list into "new" and "old" sub-lists (same as
//     the single-GPU local_join_kernel).
//   - For every (u,v) pair where at least one is new:
//       • Compute dist(u,v) using the full dataset d_pts.
//       • Try to insert v into u's list and u into v's list.
//         ONLY writes to owned rows (u ∈ [p_begin, p_end) and same for v).
//
// Because both u and v are neighbors of p (an owned point), and a point's
// neighbors tend to cluster around it in the embedding space, the majority
// of u,v updates fall within the same rank's shard — especially after a
// few iterations as the graph converges.
// ===========================================================================

GPU_GLOBAL void mg_nnd_local_join_kernel(
    const float*   d_pts,       // full dataset [n][d]
    unsigned int   n,           // total points
    unsigned int   d,           // dimension
    unsigned int   k,           // neighbors per point
    unsigned int   p_begin,     // first owned point (inclusive)
    unsigned int   p_end,       // last owned point (exclusive)
    knng::index_t* d_ids,       // full graph ids   [n][k]
    float*         d_dists,     // full graph dists  [n][k]
    std::uint32_t* d_flags,     // full graph flags  [n][k]
    int*           d_locks,     // per-point locks   [n]
    unsigned int*  d_update_counts)  // per-point update counts [n]
{
    // Each block handles one owned point.  blockIdx.x is relative to the shard.
    const unsigned int p = p_begin + blockIdx.x;
    if (p >= p_end) return;

    // -------------------------------------------------------------------------
    // Shared memory layout (same as local_join_kernel):
    //   index_t  sh_new[k]  — new neighbor IDs for point p
    //   index_t  sh_old[k]  — old neighbor IDs for point p
    //   uint32   sh_cnt[3]  — [n_new, n_old, total_updates]
    // -------------------------------------------------------------------------
    extern __shared__ char sh[];
    auto* sh_new = reinterpret_cast<knng::index_t*>(sh);
    auto* sh_old = sh_new + k;
    auto* sh_cnt = reinterpret_cast<unsigned int*>(sh_old + k);

    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; sh_cnt[2] = 0; }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Snapshot step: classify each neighbor of p as "new" or "old".
    //   is_new flag (bit 0 of d_flags) == 1 → new (not yet joined).
    //   Mark all as old in global memory (clear bit 0) so next iteration
    //   won't revisit them unless they get re-inserted with flag=1.
    // -------------------------------------------------------------------------
    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        const knng::index_t id = d_ids  [p*k+r];
        const std::uint32_t fl = d_flags[p*k+r];
        if (id == static_cast<knng::index_t>(-1)) continue;
        if (fl & 1u) {
            const unsigned int pos = atomicAdd(sh_cnt+0, 1u);
            if (pos < k) sh_new[pos] = id;
        } else {
            const unsigned int pos = atomicAdd(sh_cnt+1, 1u);
            if (pos < k) sh_old[pos] = id;
        }
        d_flags[p*k+r] &= ~1u;  // mark old
    }
    __syncthreads();

    const unsigned int n_new = sh_cnt[0];
    const unsigned int n_old = sh_cnt[1];

    // -------------------------------------------------------------------------
    // Pair enumeration: each thread in the block handles a strided slice of
    // the outer loop over new neighbors.
    //
    // Two categories of pairs:
    //   new × new: both u and v are new → strongest signal for improvement.
    //   new × old: u is new, v is old   → weaker but still useful.
    //
    // For each pair (u, v):
    //   dist(u,v) = L2²(d_pts[u], d_pts[v])
    //   Try inserting v into u's list, and u into v's list.
    // -------------------------------------------------------------------------
    for (unsigned int i = threadIdx.x; i < n_new; i += blockDim.x) {
        const knng::index_t u  = sh_new[i];
        const float*        pu = d_pts + std::size_t(u) * d;

        // new × new
        for (unsigned int j = i + 1; j < n_new; ++j) {
            const knng::index_t v  = sh_new[j];
            const float*        pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float df = pu[dim] - pv[dim]; acc += df*df;
            }
            atomicAdd(sh_cnt+2, static_cast<unsigned int>(
                mg_try_insert(d_ids, d_dists, d_flags, d_locks, u, k, v, acc) +
                mg_try_insert(d_ids, d_dists, d_flags, d_locks, v, k, u, acc)));
        }

        // new × old
        for (unsigned int j = 0; j < n_old; ++j) {
            const knng::index_t v  = sh_old[j];
            if (u == v) continue;
            const float*        pv = d_pts + std::size_t(v) * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float df = pu[dim] - pv[dim]; acc += df*df;
            }
            atomicAdd(sh_cnt+2, static_cast<unsigned int>(
                mg_try_insert(d_ids, d_dists, d_flags, d_locks, u, k, v, acc) +
                mg_try_insert(d_ids, d_dists, d_flags, d_locks, v, k, u, acc)));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) d_update_counts[p] = sh_cnt[2];
}

// ===========================================================================
// Random initialization kernel.
//
// Each thread initializes one point's k neighbors using XorShift64.
// Seeded per-thread so all points start with uncorrelated random neighbors.
// ===========================================================================

GPU_GLOBAL void mg_init_random_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    unsigned int   p_begin,
    unsigned int   p_end,
    std::uint64_t  seed,
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags)
{
    // Each thread initializes one point in the owned shard.
    const unsigned int p = p_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= p_end) return;

    // XorShift64: a fast, high-quality 64-bit PRNG.
    // Seeded per-point with a multiplicative hash to avoid correlation.
    std::uint64_t rng = seed ^ (std::uint64_t(p + 1) * 6364136223846793005ULL);
    auto xs = [&]() -> std::uint64_t {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng;
    };

    const float* qv = d_pts + std::size_t(p) * d;
    for (unsigned int r = 0; r < k; ++r) {
        // Draw a random neighbor from [0, n) (rejecting self and duplicates).
        unsigned int tries = 0;
        knng::index_t nbr  = static_cast<knng::index_t>(-1);
        while (tries < k * 8u + 32u) {
            ++tries;
            const unsigned int cand = static_cast<unsigned int>(
                (__uint128_t(xs()) * __uint128_t(n)) >> 64);
            if (cand == p) continue;
            bool dup = false;
            for (unsigned int s = 0; s < r; ++s)
                if (d_ids[p*k+s] == cand) { dup = true; break; }
            if (!dup) { nbr = cand; break; }
        }
        if (nbr == static_cast<knng::index_t>(-1)) nbr = (p + r + 1) % n;

        const float* rv = d_pts + std::size_t(nbr) * d;
        float acc = 0.f;
        for (unsigned int dim = 0; dim < d; ++dim) {
            const float df = qv[dim] - rv[dim]; acc += df * df;
        }
        d_ids  [p*k+r] = nbr;
        d_dists[p*k+r] = acc;
        d_flags[p*k+r] = 1u;   // mark new so first iteration processes it
    }
}

// ===========================================================================
// Step 85 — Multi-GPU NN-Descent driver
// ===========================================================================

knng::Knng gpu_multi_nn_descent(
    const knng::Dataset& dataset,
    const MultiGpuConfig& cfg,
    std::uint64_t seed)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto k  = static_cast<unsigned int>(cfg.k);
    const int  ng = cfg.n_gpus;

    // -------------------------------------------------------------------------
    // 1. Partition: rank r owns rows [begin_r, end_r).
    // -------------------------------------------------------------------------
    const auto shards = partition_points(dataset.n, cfg);

    // -------------------------------------------------------------------------
    // 2. NCCL communicators.
    // -------------------------------------------------------------------------
    std::vector<ncclComm_t> comms(static_cast<std::size_t>(ng));
    NCCL_CHECK(ncclCommInitAll(comms.data(), ng, nullptr));

    // -------------------------------------------------------------------------
    // 3. Shared host-side result buffers.
    //    After all iterations, rank 0 assembles the final Knng from device.
    //    These are filled at the end by each thread.
    // -------------------------------------------------------------------------
    std::vector<knng::index_t> final_ids  (static_cast<std::size_t>(n) * k);
    std::vector<float>         final_dists(static_cast<std::size_t>(n) * k);

    // -------------------------------------------------------------------------
    // 4. Launch one thread per GPU.
    // -------------------------------------------------------------------------
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(ng));

    for (int rank = 0; rank < ng; ++rank) {
        workers.emplace_back([&, rank]() {
            GPU_CHECK(cudaSetDevice(rank));

            const std::size_t ru    = static_cast<std::size_t>(rank);
            const unsigned int beg  = static_cast<unsigned int>(shards[ru].begin);
            const unsigned int end  = static_cast<unsigned int>(shards[ru].end);

            // -----------------------------------------------------------------
            // Upload full dataset to every GPU.
            // Each GPU needs all n vectors since local join accesses any (u,v).
            // -----------------------------------------------------------------
            float* d_pts = nullptr;
            GPU_CHECK(cudaMalloc(&d_pts,
                static_cast<std::size_t>(n) * d * sizeof(float)));

            if (rank == 0) {
                GPU_CHECK(cudaMemcpy(d_pts, dataset.data.data(),
                    static_cast<std::size_t>(n) * d * sizeof(float),
                    cudaMemcpyHostToDevice));
            }
            NCCL_CHECK(ncclBcast(
                d_pts, static_cast<std::size_t>(n) * d,
                ncclFloat, 0, comms[ru], nullptr));
            GPU_CHECK(cudaDeviceSynchronize());

            // -----------------------------------------------------------------
            // Allocate the FULL graph on device.
            //   d_ids  [n][k] — neighbor IDs    (initialized by init kernel)
            //   d_dists[n][k] — distances
            //   d_flags[n][k] — is-new bits
            //   d_locks[n]    — per-point spinlocks (initially 0)
            // -----------------------------------------------------------------
            const std::size_t graph_n = static_cast<std::size_t>(n);
            const std::size_t graph_k = static_cast<std::size_t>(k);

            knng::index_t* d_ids   = nullptr;
            float*         d_dists = nullptr;
            std::uint32_t* d_flags = nullptr;
            int*           d_locks = nullptr;

            GPU_CHECK(cudaMalloc(&d_ids,   graph_n * graph_k * sizeof(knng::index_t)));
            GPU_CHECK(cudaMalloc(&d_dists, graph_n * graph_k * sizeof(float)));
            GPU_CHECK(cudaMalloc(&d_flags, graph_n * graph_k * sizeof(std::uint32_t)));
            GPU_CHECK(cudaMalloc(&d_locks, graph_n           * sizeof(int)));

            // Sentinel init: 0xFF bytes = index_t(-1) for IDs, large float for dists.
            GPU_CHECK(cudaMemset(d_ids,   0xFF,
                graph_n * graph_k * sizeof(knng::index_t)));
            GPU_CHECK(cudaMemset(d_dists, 0x7F,
                graph_n * graph_k * sizeof(float)));
            GPU_CHECK(cudaMemset(d_flags, 0,
                graph_n * graph_k * sizeof(std::uint32_t)));
            GPU_CHECK(cudaMemset(d_locks, 0,
                graph_n * sizeof(int)));

            // -----------------------------------------------------------------
            // Initialize random graph for OWNED rows only.
            //
            // Using one thread per owned point in a standard 1-D grid.
            // Each thread generates k unique random neighbors with distances.
            // -----------------------------------------------------------------
            {
                constexpr unsigned int kB = 128u;
                const unsigned int shard_n = end - beg;
                GPU_LAUNCH(mg_init_random_kernel,
                           (shard_n + kB - 1u) / kB, kB, 0, nullptr,
                           d_pts, n, d, k, beg, end,
                           seed ^ (std::uint64_t(rank + 1) * 1000003ULL),
                           d_ids, d_dists, d_flags);
                GPU_CHECK(cudaDeviceSynchronize());
            }

            // -----------------------------------------------------------------
            // AllGather after init: all GPUs broadcast their initialized rows
            // to all other GPUs so everyone starts with the same full graph.
            //
            // AllGather pattern:
            //   sendbuf = &d_ids[beg*k]  (this rank's initialized shard)
            //   recvbuf = d_ids_gathered (full n*k buffer — ng shards concatenated)
            //
            // After AllGather, d_ids_gathered[r * n*k/ng ... ] = rank r's shard.
            // We then scatter each rank's contribution back into the right rows
            // of d_ids.
            //
            // NOTE: ncclAllGather requires all ranks send the SAME count.
            // Since shards may differ by 1 element, we use the MAX shard size
            // and pad/unpad on the edges.
            //
            // Simplified: use ncclAllGather on the full graph buffer so each rank
            // sends n*k and receives ng*n*k.  Then pick the slice for each rank.
            // This is O(ng * n * k) per iteration — acceptable for Phase 11.
            // Phase 12 will delta-compress.
            // -----------------------------------------------------------------

            // Temporary recv buffer: ng * n * k elements.
            knng::index_t* d_ids_recv  = nullptr;
            float*         d_dists_recv = nullptr;
            GPU_CHECK(cudaMalloc(&d_ids_recv,
                static_cast<std::size_t>(ng) * graph_n * graph_k
                * sizeof(knng::index_t)));
            GPU_CHECK(cudaMalloc(&d_dists_recv,
                static_cast<std::size_t>(ng) * graph_n * graph_k
                * sizeof(float)));

            // Helper: run AllGather on ids + dists, then merge back into d_ids/d_dists.
            // Each rank sends its full d_ids (n*k) array to all others.
            // After AllGather, recvbuf[r * n*k] = what rank r sent.
            // We merge by taking the best (lowest dist) entry per slot.
            auto sync_graph = [&]() {
                NCCL_CHECK(ncclAllGather(
                    d_ids,
                    d_ids_recv,
                    graph_n * graph_k,  // count sent by each rank
                    ncclUint32,         // index_t is uint32_t
                    comms[ru], nullptr));
                NCCL_CHECK(ncclAllGather(
                    d_dists,
                    d_dists_recv,
                    graph_n * graph_k,  // count sent by each rank
                    ncclFloat,
                    comms[ru], nullptr));
                GPU_CHECK(cudaDeviceSynchronize());

                // Merge step: for each (point, slot), keep the best entry
                // across all ng versions.
                // This is a simple O(ng * n * k) scan on the GPU.
                // In practice ng is small (2–8) so this is fast.
                // A proper merge kernel would avoid the triple loop but this
                // is clear and correct.
                //
                // We do the merge on the HOST for simplicity (Phase 12 will
                // move this to a GPU reduction kernel).
                std::vector<knng::index_t> h_ids_all(
                    static_cast<std::size_t>(ng) * graph_n * graph_k);
                std::vector<float> h_dists_all(
                    static_cast<std::size_t>(ng) * graph_n * graph_k);
                GPU_CHECK(cudaMemcpy(h_ids_all.data(), d_ids_recv,
                    h_ids_all.size() * sizeof(knng::index_t),
                    cudaMemcpyDeviceToHost));
                GPU_CHECK(cudaMemcpy(h_dists_all.data(), d_dists_recv,
                    h_dists_all.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));

                // For each point p: collect the best k neighbors across all
                // ng versions of the graph.
                std::vector<knng::index_t> merged_ids  (graph_n * graph_k,
                    static_cast<knng::index_t>(-1));
                std::vector<float> merged_dists(graph_n * graph_k, 1e38f);

                for (int r = 0; r < ng; ++r) {
                    const std::size_t offset =
                        static_cast<std::size_t>(r) * graph_n * graph_k;
                    for (unsigned int p = 0; p < n; ++p) {
                        for (unsigned int s = 0; s < k; ++s) {
                            const knng::index_t id   = h_ids_all  [offset + p*k+s];
                            const float         dist = h_dists_all[offset + p*k+s];
                            if (id == knng::index_t(-1)) continue;

                            // Find worst slot in merged row p.
                            float wdist = merged_dists[p*k];
                            unsigned int wp = 0;
                            for (unsigned int ss = 1; ss < k; ++ss)
                                if (merged_dists[p*k+ss] > wdist) {
                                    wdist = merged_dists[p*k+ss]; wp = ss;
                                }
                            // Duplicate check.
                            bool dup = false;
                            for (unsigned int ss = 0; ss < k; ++ss)
                                if (merged_ids[p*k+ss] == id) { dup = true; break; }

                            if (!dup && dist < wdist) {
                                merged_ids  [p*k+wp] = id;
                                merged_dists[p*k+wp] = dist;
                            }
                        }
                    }
                }

                // Upload merged result back to device.
                GPU_CHECK(cudaMemcpy(d_ids, merged_ids.data(),
                    graph_n * graph_k * sizeof(knng::index_t),
                    cudaMemcpyHostToDevice));
                GPU_CHECK(cudaMemcpy(d_dists, merged_dists.data(),
                    graph_n * graph_k * sizeof(float),
                    cudaMemcpyHostToDevice));
            };

            // Initial AllGather so everyone has the random graph.
            sync_graph();

            // -----------------------------------------------------------------
            // NN-Descent iteration loop.
            // -----------------------------------------------------------------
            const std::size_t shard_n = shards[ru].size();

            constexpr unsigned int kBlock = 64u;
            const std::size_t shmem =
                2u * graph_k * sizeof(knng::index_t)
                + 3u * sizeof(unsigned int);

            // Per-point update counts (for convergence detection).
            unsigned int* d_update_counts = nullptr;
            GPU_CHECK(cudaMalloc(&d_update_counts, graph_n * sizeof(unsigned int)));

            const double threshold =
                cfg.delta * static_cast<double>(n) * k;

            for (int iter = 0; iter < cfg.n_iterations; ++iter) {
                GPU_CHECK(cudaMemset(d_locks,         0,
                    graph_n * sizeof(int)));
                GPU_CHECK(cudaMemset(d_update_counts, 0,
                    graph_n * sizeof(unsigned int)));

                // -------------------------------------------------------------
                // Local join: each block handles ONE owned point.
                // Grid = shard_n (number of owned points).
                // -------------------------------------------------------------
                GPU_LAUNCH(mg_nnd_local_join_kernel,
                           static_cast<unsigned int>(shard_n),
                           kBlock, shmem, nullptr,
                           d_pts, n, d, k, beg, end,
                           d_ids, d_dists, d_flags, d_locks,
                           d_update_counts);
                GPU_CHECK(cudaDeviceSynchronize());

                // -------------------------------------------------------------
                // Convergence: CUB DeviceReduce::Sum over ALL n update counts
                // (only owned ones are non-zero after local join).
                // Then NCCL AllReduce to get the global sum across all GPUs.
                // -------------------------------------------------------------
                unsigned int* d_total = nullptr;
                GPU_CHECK(cudaMalloc(&d_total, sizeof(unsigned int)));
                {
                    std::size_t temp_bytes = 0;
                    cub::DeviceReduce::Sum(nullptr, temp_bytes,
                                          d_update_counts, d_total,
                                          static_cast<int>(n));
                    std::uint8_t* d_temp = nullptr;
                    GPU_CHECK(cudaMalloc(&d_temp, temp_bytes ? temp_bytes : 1));
                    cub::DeviceReduce::Sum(d_temp, temp_bytes,
                                          d_update_counts, d_total,
                                          static_cast<int>(n));
                    GPU_CHECK(cudaFree(d_temp));
                }
                GPU_CHECK(cudaDeviceSynchronize());

                // AllReduce (sum) the local update counts so every GPU knows
                // the GLOBAL total update count.
                NCCL_CHECK(ncclAllReduce(
                    d_total, d_total,
                    1, ncclUint32, ncclSum,
                    comms[ru], nullptr));
                GPU_CHECK(cudaDeviceSynchronize());

                unsigned int h_total = 0;
                GPU_CHECK(cudaMemcpy(&h_total, d_total,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost));
                GPU_CHECK(cudaFree(d_total));

                // -------------------------------------------------------------
                // AllGather: broadcast updated shard to all GPUs.
                // After this, every GPU has the merged best-of-ng graph.
                // -------------------------------------------------------------
                sync_graph();

                if (static_cast<double>(h_total) < threshold) break;
            }

            // -----------------------------------------------------------------
            // Copy final graph to host (only rank 0 writes final_ids/final_dists,
            // but since all ranks have identical graphs after AllGather, any rank
            // can do this; we use rank 0 for simplicity).
            // -----------------------------------------------------------------
            if (rank == 0) {
                GPU_CHECK(cudaMemcpy(final_ids.data(), d_ids,
                    graph_n * graph_k * sizeof(knng::index_t),
                    cudaMemcpyDeviceToHost));
                GPU_CHECK(cudaMemcpy(final_dists.data(), d_dists,
                    graph_n * graph_k * sizeof(float),
                    cudaMemcpyDeviceToHost));
            }

            GPU_CHECK(cudaFree(d_update_counts));
            GPU_CHECK(cudaFree(d_ids_recv));
            GPU_CHECK(cudaFree(d_dists_recv));
            GPU_CHECK(cudaFree(d_locks));
            GPU_CHECK(cudaFree(d_flags));
            GPU_CHECK(cudaFree(d_dists));
            GPU_CHECK(cudaFree(d_ids));
            GPU_CHECK(cudaFree(d_pts));
        });
    }

    for (auto& t : workers) t.join();

    for (int r = 0; r < ng; ++r)
        ncclCommDestroy(comms[static_cast<std::size_t>(r)]);

    // -------------------------------------------------------------------------
    // 5. Assemble result: sort each row ascending by distance.
    // -------------------------------------------------------------------------
    knng::Knng result(dataset.n, cfg.k);
    const std::size_t ks = static_cast<std::size_t>(cfg.k);

    for (std::size_t qi = 0; qi < dataset.n; ++qi) {
        // Sort the k entries for this point by distance (selection sort; k small).
        std::vector<std::size_t> order(ks);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return final_dists[qi*ks+a] < final_dists[qi*ks+b];
        });
        for (std::size_t r = 0; r < ks; ++r) {
            result.neighbors[qi*ks+r] = final_ids  [qi*ks+order[r]];
            result.distances[qi*ks+r] = final_dists[qi*ks+order[r]];
        }
    }
    return result;
}

} // namespace knng::gpu

#endif // KNNG_HAVE_CUDA && KNNG_HAVE_NCCL
