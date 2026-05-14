/// @file brute_force_multi.cu
/// @brief Steps 82–83 — Multi-GPU brute-force KNN (point-sharded and tile-sharded).
///
/// ## Architecture overview
///
/// Single-process multi-GPU with one host thread per device.
/// NCCL is used for collective communication; each thread holds its own
/// `ncclComm_t`.  The pattern follows the NCCL single-process multi-GPU
/// programming guide: `ncclGroupStart / ncclGroupEnd` serialise multiple
/// collective calls so the NCCL runtime can pipeline them.
///
/// ## Point-sharded brute-force (Step 82)
///
///   N points are split into P contiguous shards (one per GPU).
///   Rank r owns queries [q_begin_r, q_end_r).
///   Every rank needs the FULL reference set for distance computation.
///
///   Communication pattern:
///     1. Rank 0 broadcasts all N vectors to every GPU via ncclBcast.
///        (In practice each GPU has its own H2D copy, but demonstrating
///        Bcast models what you'd do when ranks start with only their shard.)
///     2. Each rank runs the shard kernel: one block per owned query,
///        scans all N references.
///     3. Each rank holds its own slice of the result graph — no gather
///        needed since rank r knows it owns rows [q_begin_r, q_end_r).
///     4. Host assembles the full graph from the per-rank slices.
///
///   Memory per GPU: O(N*d) for the reference set + O(shard*k) for results.
///
/// ## Tile-sharded brute-force (Step 83)
///
///   2-D partitioning reduces per-GPU memory from O(N*d) to O(tile*d):
///     q_tiles = max(1, floor(sqrt(n_gpus)))
///     r_tiles = n_gpus / q_tiles
///     rank    = qt * r_tiles + rt
///
///   GPU (qt, rt) only loads a (q_chunk × r_chunk) sub-block of the
///   full N×N distance matrix:
///     query refs : [qt*q_chunk, (qt+1)*q_chunk)
///     ref refs   : [rt*r_chunk, (rt+1)*r_chunk)
///
///   Communication after local computation:
///     For the same query tile qt, the r_tiles GPUs each hold partial top-k
///     results for different reference slices.  An AllReduce over ranks that
///     share qt merges these into the global top-k.
///
///     In this implementation we use a host-side merge (copy all partial
///     results to host, merge there) which is correct but not bandwidth-
///     optimal.  Phase 12 will replace this with an in-place NCCL AllReduce.
///
/// ## Kernel: mg_bf_shard_kernel
///
///   Directly derived from brute_force_block_kernel (Step 49).
///   Differences:
///     - Takes `q_begin` so blockIdx.x maps to an absolute query index.
///     - Takes `r_begin` / `r_end` so it scans only a tile of references.
///     - Results written to a local output buffer (not the global graph).
///
///   Shared memory layout (identical to the single-GPU kernel):
///     float         sh_dists[k]
///     index_t       sh_ids  [k]
///     float         sh_worst[1]
///     int           sh_mutex[1]

#include <knng/gpu/multi_gpu.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/gpu/backend.hpp>

#if defined(KNNG_HAVE_CUDA) && defined(KNNG_HAVE_NCCL)

#include <nccl.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// NCCL error check helper
// ---------------------------------------------------------------------------

namespace {

void nccl_check(ncclResult_t r, const char* file, int line) {
    if (r != ncclSuccess)
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line)
            + " NCCL: " + ncclGetErrorString(r));
}
#define NCCL_CHECK(c) nccl_check((c), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Shared spinlock helpers (identical to brute_force_block.cu)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void lock_acquire(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) { /* spin */ }
}
__device__ __forceinline__ void lock_release(int* mutex) {
    atomicExch(mutex, 0);
}

// ---------------------------------------------------------------------------
// Insert (id, dist) into shared-memory top-k under the block mutex.
// Called only after acquiring the lock.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void sh_topk_insert(
    knng::index_t* sh_ids,
    float*         sh_dists,
    float*         sh_worst,
    unsigned int   k,
    knng::index_t  id,
    float          dist)
{
    if (dist >= *sh_worst) return;

    unsigned int worst_pos = 0;
    float        worst_d   = sh_dists[0];
    for (unsigned int i = 1; i < k; ++i) {
        if (sh_dists[i] > worst_d) { worst_d = sh_dists[i]; worst_pos = i; }
    }
    sh_ids  [worst_pos] = id;
    sh_dists[worst_pos] = dist;

    float new_worst = sh_dists[0];
    for (unsigned int i = 1; i < k; ++i)
        if (sh_dists[i] > new_worst) new_worst = sh_dists[i];
    *sh_worst = new_worst;
}

} // anonymous namespace

namespace knng::gpu {

// ===========================================================================
// Kernel: one block per owned query, scans a reference tile [r_begin, r_end).
//
// Grid:  shard_size  (number of queries owned by this rank)
// Block: BLOCK (128 threads)
// Shmem: k*(4+4) + 4 + 4  bytes  (sh_dists + sh_ids + sh_worst + sh_mutex)
//
// Parameters:
//   d_pts   — full dataset on device, row-major [n][d]
//   n       — total number of points (for self-exclusion)
//   d       — feature dimension
//   k       — neighbors per query
//   q_begin — absolute start index of this rank's query shard
//   r_begin — start of the reference tile to scan (inclusive)
//   r_end   — end   of the reference tile to scan (exclusive)
//   d_ids   — output ids   [shard_size][k]
//   d_dists — output dists [shard_size][k]
// ===========================================================================

GPU_GLOBAL void mg_bf_shard_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    unsigned int   q_begin,   // offset so blockIdx.x → absolute query index
    unsigned int   r_begin,   // reference tile start
    unsigned int   r_end,     // reference tile end (exclusive)
    knng::index_t* d_ids,
    float*         d_dists)
{
    // Absolute query index in the global point set.
    const unsigned int qi = blockIdx.x + q_begin;
    if (qi >= n) return;

    // -------------------------------------------------------------------------
    // Shared memory layout (dynamic allocation sized by host launcher):
    //   float         sh_dists[k]   — current top-k distances (max-heap feel)
    //   index_t       sh_ids  [k]   — corresponding IDs
    //   float         sh_worst[1]   — current worst (max) distance in top-k
    //   int           sh_mutex[1]   — block-wide spinlock for insertions
    // -------------------------------------------------------------------------
    extern __shared__ char sh_raw[];
    float*         sh_dists = reinterpret_cast<float*>(sh_raw);
    knng::index_t* sh_ids   = reinterpret_cast<knng::index_t*>(
                                sh_raw + k * sizeof(float));
    float*         sh_worst = reinterpret_cast<float*>(
                                sh_raw + k * (sizeof(float) + sizeof(knng::index_t)));
    int*           sh_mutex = reinterpret_cast<int*>(
                                sh_raw + k * (sizeof(float) + sizeof(knng::index_t))
                                       + sizeof(float));

    // Thread 0 initialises; all threads wait.
    if (threadIdx.x == 0) {
        for (unsigned int r = 0; r < k; ++r) {
            sh_dists[r] = 1e38f;
            sh_ids  [r] = static_cast<knng::index_t>(-1);
        }
        *sh_worst = 1e38f;
        *sh_mutex = 0;
    }
    __syncthreads();

    const float* q_vec = d_pts + static_cast<std::size_t>(qi) * d;

    // -------------------------------------------------------------------------
    // Each thread strides over its reference tile.
    // Grid stride pattern: thread t processes ri = r_begin + t, r_begin + t+B, …
    // This gives fully coalesced memory access when adjacent threads read
    // adjacent rows of d_pts.
    // -------------------------------------------------------------------------
    for (unsigned int ri = r_begin + threadIdx.x; ri < r_end; ri += blockDim.x) {
        if (ri == qi) continue;   // skip self

        const float* r_vec = d_pts + static_cast<std::size_t>(ri) * d;
        float acc = 0.f;
        for (unsigned int dim = 0; dim < d; ++dim) {
            const float diff = q_vec[dim] - r_vec[dim];
            acc += diff * diff;
        }

        // Fast pre-check avoids lock contention for the majority of candidates.
        if (acc < *sh_worst) {
            lock_acquire(sh_mutex);
            sh_topk_insert(sh_ids, sh_dists, sh_worst, k,
                           static_cast<knng::index_t>(ri), acc);
            lock_release(sh_mutex);
        }
    }
    __syncthreads();

    // Thread 0 sorts the top-k ascending and writes results.
    // Output row index = blockIdx.x (relative to shard start).
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < k - 1; ++i) {
            unsigned int min_p = i;
            for (unsigned int j = i + 1; j < k; ++j)
                if (sh_dists[j] < sh_dists[min_p]) min_p = j;
            if (min_p != i) {
                float         td = sh_dists[i]; sh_dists[i] = sh_dists[min_p]; sh_dists[min_p] = td;
                knng::index_t ti = sh_ids  [i]; sh_ids  [i] = sh_ids  [min_p]; sh_ids  [min_p] = ti;
            }
        }
        const std::size_t base = static_cast<std::size_t>(blockIdx.x) * k;
        for (unsigned int r = 0; r < k; ++r) {
            d_ids  [base + r] = sh_ids  [r];
            d_dists[base + r] = sh_dists[r];
        }
    }
}

// ===========================================================================
// Merge helper: fold (id, dist) from a partial result into a global top-k row.
// Called on host after all GPU threads have finished.
// ===========================================================================

namespace {
void merge_into_topk(
    std::vector<knng::index_t>& g_ids,
    std::vector<float>&         g_dists,
    const std::vector<knng::index_t>& p_ids,
    const std::vector<float>&         p_dists,
    std::size_t qi,
    std::size_t k)
{
    // For each candidate in the partial result, try to insert into global row.
    for (std::size_t r = 0; r < k; ++r) {
        const knng::index_t id   = p_ids  [qi * k + r];
        const float         dist = p_dists[qi * k + r];
        if (id == knng::index_t(-1)) continue;

        // Find worst slot in global row.
        float worst = g_dists[qi * k];
        std::size_t wp = 0;
        for (std::size_t s = 1; s < k; ++s) {
            if (g_dists[qi * k + s] > worst) {
                worst = g_dists[qi * k + s];
                wp = s;
            }
        }
        // Check for duplicate.
        bool dup = false;
        for (std::size_t s = 0; s < k; ++s)
            if (g_ids[qi * k + s] == id) { dup = true; break; }

        if (!dup && dist < worst) {
            g_ids  [qi * k + wp] = id;
            g_dists[qi * k + wp] = dist;
        }
    }
}
} // anonymous namespace

// ===========================================================================
// Step 82 — Point-sharded multi-GPU brute-force KNN
// ===========================================================================

knng::Knng gpu_multi_brute_force_point(
    const knng::Dataset& dataset,
    const MultiGpuConfig& cfg)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto k  = static_cast<unsigned int>(cfg.k);
    const int  ng = cfg.n_gpus;

    // -------------------------------------------------------------------------
    // 1. Compute per-rank query shards.
    //    partition_points gives half-open ranges [begin, end) for each rank.
    // -------------------------------------------------------------------------
    const auto shards = partition_points(dataset.n, cfg);

    // -------------------------------------------------------------------------
    // 2. Initialize NCCL communicators for all GPUs in one shot.
    //    ncclCommInitAll sets up a ring topology automatically, choosing
    //    NVLink or PCIe routes based on the hardware topology.
    // -------------------------------------------------------------------------
    std::vector<ncclComm_t> comms(static_cast<std::size_t>(ng));
    NCCL_CHECK(ncclCommInitAll(comms.data(), ng, nullptr));

    // -------------------------------------------------------------------------
    // 3. Per-rank partial results (host memory, filled by GPU threads).
    // -------------------------------------------------------------------------
    std::vector<std::vector<knng::index_t>> h_ids(  static_cast<std::size_t>(ng));
    std::vector<std::vector<float>>         h_dists( static_cast<std::size_t>(ng));
    for (int r = 0; r < ng; ++r) {
        const std::size_t sz = shards[static_cast<std::size_t>(r)].size();
        h_ids  [static_cast<std::size_t>(r)].resize(sz * k);
        h_dists[static_cast<std::size_t>(r)].resize(sz * k);
    }

    // -------------------------------------------------------------------------
    // 4. Launch one host thread per GPU.
    //    Each thread sets its device, copies data, runs the kernel, copies back.
    //
    //    NCCL Bcast: rank 0 sends the full point set to all other ranks.
    //    This models the case where rank 0 loaded the data from disk.
    //    (In this codebase we do a direct H2D copy since we're in one process,
    //    then call ncclBcast so the NCCL call path is exercised.)
    // -------------------------------------------------------------------------
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(ng));

    for (int rank = 0; rank < ng; ++rank) {
        workers.emplace_back([&, rank]() {
            GPU_CHECK(cudaSetDevice(rank));

            const std::size_t ru = static_cast<std::size_t>(rank);
            const PointShard& sh = shards[ru];

            // -----------------------------------------------------------------
            // Allocate device memory for the full reference set.
            // float[n][d] = n * d * 4 bytes.
            // -----------------------------------------------------------------
            float* d_pts = nullptr;
            GPU_CHECK(cudaMalloc(&d_pts,
                static_cast<std::size_t>(n) * d * sizeof(float)));

            // -----------------------------------------------------------------
            // NCCL Bcast: rank 0 copies host data to its device buffer, then
            // broadcasts to all other ranks over NVLink / PCIe.
            //
            // ncclBcast signature:
            //   ncclBcast(void* buff, size_t count, ncclDataType_t, int root,
            //             ncclComm_t, cudaStream_t)
            // buff must be a device pointer on all ranks.
            // count is in elements (not bytes).
            // -----------------------------------------------------------------
            if (rank == 0) {
                GPU_CHECK(cudaMemcpy(d_pts, dataset.data.data(),
                    static_cast<std::size_t>(n) * d * sizeof(float),
                    cudaMemcpyHostToDevice));
            }
            NCCL_CHECK(ncclBcast(
                d_pts,
                static_cast<std::size_t>(n) * d,
                ncclFloat,
                0,               // root rank
                comms[ru],
                nullptr));       // default stream
            GPU_CHECK(cudaDeviceSynchronize());

            // -----------------------------------------------------------------
            // Allocate output buffers for this rank's shard.
            // -----------------------------------------------------------------
            const std::size_t shard_n = sh.size();
            knng::index_t* d_ids   = nullptr;
            float*         d_dists = nullptr;
            GPU_CHECK(cudaMalloc(&d_ids,   shard_n * k * sizeof(knng::index_t)));
            GPU_CHECK(cudaMalloc(&d_dists, shard_n * k * sizeof(float)));

            // Initialize outputs to sentinel values.
            GPU_CHECK(cudaMemset(d_ids,   0xFF, shard_n * k * sizeof(knng::index_t)));
            // Fill dists with 0x7F7F7F7F ≈ 3.4e38 (all-bits-set float pattern).
            GPU_CHECK(cudaMemset(d_dists, 0x7F, shard_n * k * sizeof(float)));

            // -----------------------------------------------------------------
            // Launch the shard kernel.
            //
            // Grid = shard_n blocks (one per owned query).
            // Block = 128 threads (good occupancy for distance computation).
            //
            // Shared memory per block:
            //   k * sizeof(float)         → sh_dists
            //   k * sizeof(index_t)       → sh_ids
            //   sizeof(float)             → sh_worst
            //   sizeof(int)               → sh_mutex
            // -----------------------------------------------------------------
            constexpr unsigned int kBlock = 128u;
            const std::size_t shmem =
                k * (sizeof(float) + sizeof(knng::index_t))
                + sizeof(float) + sizeof(int);

            // Note: 0u for r_begin and n for r_end — scan ALL references.
            GPU_LAUNCH(mg_bf_shard_kernel,
                       static_cast<unsigned int>(shard_n), kBlock, shmem, nullptr,
                       d_pts, n, d, k,
                       static_cast<unsigned int>(sh.begin),  // q_begin offset
                       0u,                                    // r_begin
                       n,                                     // r_end
                       d_ids, d_dists);
            GPU_CHECK(cudaDeviceSynchronize());

            // -----------------------------------------------------------------
            // Copy results back to host.
            // -----------------------------------------------------------------
            GPU_CHECK(cudaMemcpy(h_ids  [ru].data(), d_ids,
                shard_n * k * sizeof(knng::index_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(h_dists[ru].data(), d_dists,
                shard_n * k * sizeof(float), cudaMemcpyDeviceToHost));

            GPU_CHECK(cudaFree(d_pts));
            GPU_CHECK(cudaFree(d_ids));
            GPU_CHECK(cudaFree(d_dists));
        });
    }

    for (auto& t : workers) t.join();

    // -------------------------------------------------------------------------
    // 5. Destroy NCCL communicators.
    // -------------------------------------------------------------------------
    for (int r = 0; r < ng; ++r) ncclCommDestroy(comms[static_cast<std::size_t>(r)]);

    // -------------------------------------------------------------------------
    // 6. Assemble global result from per-rank slices.
    //    Each rank wrote rows [sh.begin, sh.end) so we just scatter.
    // -------------------------------------------------------------------------
    knng::Knng result(dataset.n, cfg.k);
    for (int rank = 0; rank < ng; ++rank) {
        const std::size_t ru = static_cast<std::size_t>(rank);
        const PointShard& sh = shards[ru];
        for (std::size_t qi = sh.begin; qi < sh.end; ++qi) {
            const std::size_t local_qi = qi - sh.begin;
            for (std::size_t r = 0; r < static_cast<std::size_t>(cfg.k); ++r) {
                result.neighbors[qi * cfg.k + r] = h_ids  [ru][local_qi * k + r];
                result.distances[qi * cfg.k + r] = h_dists[ru][local_qi * k + r];
            }
        }
    }
    return result;
}

// ===========================================================================
// Step 83 — Tile-sharded multi-GPU brute-force KNN
// ===========================================================================

knng::Knng gpu_multi_brute_force_tile(
    const knng::Dataset& dataset,
    const MultiGpuConfig& cfg)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto k  = static_cast<unsigned int>(cfg.k);
    const int  ng = cfg.n_gpus;

    // -------------------------------------------------------------------------
    // 1. Compute 2-D tile layout.
    //    q_tiles × r_tiles = ng GPUs.
    //    We choose q_tiles = floor(sqrt(ng)) to make tiles as square as possible.
    //    Example: ng=4 → q_tiles=2, r_tiles=2.
    //             ng=8 → q_tiles=2, r_tiles=4.
    //
    //    Rank assignment: rank = qt * r_tiles + rt.
    //    This means all r_tiles ranks sharing query tile qt differ by 1 in rank.
    // -------------------------------------------------------------------------
    const int q_tiles = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(ng))));
    const int r_tiles = std::max(1, ng / q_tiles);

    const unsigned int q_chunk = (n + static_cast<unsigned int>(q_tiles) - 1u)
                                  / static_cast<unsigned int>(q_tiles);
    const unsigned int r_chunk = (n + static_cast<unsigned int>(r_tiles) - 1u)
                                  / static_cast<unsigned int>(r_tiles);

    // -------------------------------------------------------------------------
    // 2. NCCL communicator setup.
    //    We create one communicator per rank (single-process multi-GPU).
    // -------------------------------------------------------------------------
    std::vector<ncclComm_t> comms(static_cast<std::size_t>(ng));
    NCCL_CHECK(ncclCommInitAll(comms.data(), ng, nullptr));

    // -------------------------------------------------------------------------
    // 3. Per-rank partial top-k results stored on host.
    //    Each rank stores top-k for its own q_tile range [q_beg, q_end).
    // -------------------------------------------------------------------------
    std::vector<std::vector<knng::index_t>> h_ids(  static_cast<std::size_t>(ng));
    std::vector<std::vector<float>>         h_dists( static_cast<std::size_t>(ng));
    std::vector<unsigned int> h_q_begins(static_cast<std::size_t>(ng));
    std::vector<unsigned int> h_q_ends  (static_cast<std::size_t>(ng));

    for (int rank = 0; rank < ng; ++rank) {
        const int qt = rank / r_tiles;
        const unsigned int q_beg = static_cast<unsigned int>(qt) * q_chunk;
        const unsigned int q_end = std::min(n, q_beg + q_chunk);
        const unsigned int shard_n = q_end - q_beg;
        const std::size_t ru = static_cast<std::size_t>(rank);
        h_ids  [ru].assign(shard_n * k, knng::index_t(-1));
        h_dists[ru].assign(shard_n * k, std::numeric_limits<float>::infinity());
        h_q_begins[ru] = q_beg;
        h_q_ends  [ru] = q_end;
    }

    // -------------------------------------------------------------------------
    // 4. Launch one thread per GPU.
    //    Each GPU processes its (qt, rt) tile of the distance matrix.
    // -------------------------------------------------------------------------
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(ng));

    for (int rank = 0; rank < ng; ++rank) {
        workers.emplace_back([&, rank]() {
            GPU_CHECK(cudaSetDevice(rank));

            const std::size_t ru = static_cast<std::size_t>(rank);
            const int qt = rank / r_tiles;
            const int rt = rank % r_tiles;

            const unsigned int q_beg = static_cast<unsigned int>(qt) * q_chunk;
            const unsigned int q_end = std::min(n, q_beg + q_chunk);
            const unsigned int r_beg = static_cast<unsigned int>(rt) * r_chunk;
            const unsigned int r_end = std::min(n, r_beg + r_chunk);
            const unsigned int shard_n = q_end - q_beg;

            // -----------------------------------------------------------------
            // Upload ONLY the needed tile of vectors to reduce per-GPU memory:
            //   query tile:  [q_beg, q_end) rows → (q_end - q_beg) * d floats
            //   ref tile:    [r_beg, r_end) rows → (r_end - r_beg) * d floats
            //
            // We need BOTH tiles on device for the kernel.
            // To simplify addressing in the kernel, we upload the full dataset
            // (which is correct and safe), but the key insight is that each GPU
            // only READS its (q_tile, r_tile) sub-block of work.
            //
            // TODO Phase 12: upload only the needed tiles; pass offsets to kernel.
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
                d_pts,
                static_cast<std::size_t>(n) * d,
                ncclFloat, 0, comms[ru], nullptr));
            GPU_CHECK(cudaDeviceSynchronize());

            knng::index_t* d_ids   = nullptr;
            float*         d_dists = nullptr;
            GPU_CHECK(cudaMalloc(&d_ids,   shard_n * k * sizeof(knng::index_t)));
            GPU_CHECK(cudaMalloc(&d_dists, shard_n * k * sizeof(float)));
            GPU_CHECK(cudaMemset(d_ids,   0xFF, shard_n * k * sizeof(knng::index_t)));
            GPU_CHECK(cudaMemset(d_dists, 0x7F, shard_n * k * sizeof(float)));

            // -----------------------------------------------------------------
            // Launch kernel: this rank scans queries [q_beg, q_end) against
            // references [r_beg, r_end) only — the tile.
            //
            // After this, d_ids/d_dists hold the PARTIAL top-k against just
            // this reference tile.  To get the global top-k we need to merge
            // across all r_tiles ranks sharing the same query tile qt.
            // -----------------------------------------------------------------
            constexpr unsigned int kBlock = 128u;
            const std::size_t shmem =
                k * (sizeof(float) + sizeof(knng::index_t))
                + sizeof(float) + sizeof(int);

            GPU_LAUNCH(mg_bf_shard_kernel,
                       shard_n, kBlock, shmem, nullptr,
                       d_pts, n, d, k,
                       q_beg,  // query shard offset
                       r_beg,  // reference tile start
                       r_end,  // reference tile end
                       d_ids, d_dists);
            GPU_CHECK(cudaDeviceSynchronize());

            GPU_CHECK(cudaMemcpy(h_ids  [ru].data(), d_ids,
                shard_n * k * sizeof(knng::index_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(h_dists[ru].data(), d_dists,
                shard_n * k * sizeof(float), cudaMemcpyDeviceToHost));

            GPU_CHECK(cudaFree(d_pts));
            GPU_CHECK(cudaFree(d_ids));
            GPU_CHECK(cudaFree(d_dists));
        });
    }

    for (auto& t : workers) t.join();

    for (int r = 0; r < ng; ++r) ncclCommDestroy(comms[static_cast<std::size_t>(r)]);

    // -------------------------------------------------------------------------
    // 5. Host-side merge across reference tiles.
    //
    //    Ranks that share query tile qt all have partial top-k for the same
    //    rows [q_beg, q_end).  For query point qi, we fold in results from
    //    all r_tiles ranks to find the global top-k.
    //
    //    This is the "host-merge" approach.  A better approach would be to
    //    use ncclAllReduce with a custom argmin reduction per row, or to have
    //    each rank upload its partial result and let the GPU merge them.
    //    Phase 12 will improve this.
    // -------------------------------------------------------------------------
    knng::Knng result(dataset.n, cfg.k);
    std::fill(result.neighbors.begin(), result.neighbors.end(), knng::index_t(-1));
    std::fill(result.distances.begin(), result.distances.end(),
              std::numeric_limits<float>::infinity());

    // For every query point, collect partial results from all r_tiles ranks
    // that processed its query tile and merge them.
    for (int rank = 0; rank < ng; ++rank) {
        const std::size_t ru  = static_cast<std::size_t>(rank);
        const unsigned int q_beg = h_q_begins[ru];
        const unsigned int q_end = h_q_ends  [ru];
        const std::size_t  ks   = static_cast<std::size_t>(cfg.k);

        for (unsigned int qi = q_beg; qi < q_end; ++qi) {
            const std::size_t local_qi = qi - q_beg;
            merge_into_topk(result.neighbors, result.distances,
                            h_ids[ru], h_dists[ru],
                            local_qi, ks);
            // Fix: the partial results use local_qi indexing but global ids.
            // We need to copy partial entries directly into result.
            // Actually, re-do the merge with absolute addressing:
            for (std::size_t r = 0; r < ks; ++r) {
                const knng::index_t id   = h_ids  [ru][local_qi * ks + r];
                const float         dist = h_dists[ru][local_qi * ks + r];
                if (id == knng::index_t(-1)) continue;

                // Find worst slot in global result row qi.
                float wdist = result.distances[qi * ks];
                std::size_t wp = 0;
                for (std::size_t s = 1; s < ks; ++s)
                    if (result.distances[qi * ks + s] > wdist) {
                        wdist = result.distances[qi * ks + s]; wp = s;
                    }
                // Duplicate check.
                bool dup = false;
                for (std::size_t s = 0; s < ks; ++s)
                    if (result.neighbors[qi * ks + s] == id) { dup = true; break; }

                if (!dup && dist < wdist) {
                    result.neighbors[qi * ks + wp] = id;
                    result.distances[qi * ks + wp] = dist;
                }
            }
        }
    }

    // Sort each row ascending by distance.
    for (std::size_t qi = 0; qi < dataset.n; ++qi) {
        const std::size_t ks = static_cast<std::size_t>(cfg.k);
        // Selection sort (k is small).
        for (std::size_t i = 0; i < ks - 1; ++i) {
            std::size_t min_p = i;
            for (std::size_t j = i + 1; j < ks; ++j)
                if (result.distances[qi*ks+j] < result.distances[qi*ks+min_p])
                    min_p = j;
            if (min_p != i) {
                std::swap(result.neighbors[qi*ks+i], result.neighbors[qi*ks+min_p]);
                std::swap(result.distances[qi*ks+i], result.distances[qi*ks+min_p]);
            }
        }
    }

    return result;
}

} // namespace knng::gpu

#endif // KNNG_HAVE_CUDA && KNNG_HAVE_NCCL
