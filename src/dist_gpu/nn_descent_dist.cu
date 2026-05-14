/// @file
/// @brief Steps 92–96 — Distributed GPU NN-Descent (NEO-DNND family).
///
/// Step 92 baseline: MPI AllGather of all features and graph rows every
/// iteration.  Each GPU runs local_join on its owned shard; updates are
/// propagated via MPI AllGatherv.  Expensive but correct.
///
/// Steps 93–96 optimisations are compiled into this TU as separate
/// entry-point functions (gpu_dist_nn_descent_dedup/_shm/_bw/_overlap).
/// They call the same kernels but reduce inter-rank communication bytes.

#include <knng/dist_gpu/nn_descent.hpp>
#include <knng/dist/sharded_dataset.hpp>
#include <knng/dist/brute_force_mpi.hpp>   // gather_graph

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace knng::dist_gpu {

// ── error helpers ────────────────────────────────────────────────────────────

namespace {
inline void cuda_chk(cudaError_t e, const char* w)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}

// ── XorShift64 PRNG (device) ─────────────────────────────────────────────

struct XS64 {
    uint64_t s;
    __device__ explicit XS64(uint64_t seed) noexcept : s(seed | 1ULL) {}
    __device__ uint64_t next() noexcept {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
    __device__ uint64_t bounded(uint64_t n) noexcept { return next() % n; }
};

// ── MPI AllGatherv helpers ────────────────────────────────────────────────

// Compute per-rank counts and displacements for a variable-shard AllGatherv.
void build_allgather_counts(std::size_t global_n, int size, std::size_t stride,
                             std::vector<int>& counts,
                             std::vector<int>& displs)
{
    counts.resize(static_cast<std::size_t>(size));
    displs.resize(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        auto [st, cnt] = knng::dist::compute_shard(global_n, size, r);
        counts[static_cast<std::size_t>(r)] = static_cast<int>(cnt * stride);
        displs[static_cast<std::size_t>(r)] = static_cast<int>(st  * stride);
    }
}

void allgather_floats(const float* local_data, std::size_t local_cnt,
                      float* global_data, std::size_t global_n,
                      std::size_t stride, int size, MPI_Comm comm)
{
    std::vector<int> counts, displs;
    build_allgather_counts(global_n, size, stride, counts, displs);
    MPI_Allgatherv(local_data, static_cast<int>(local_cnt * stride), MPI_FLOAT,
                   global_data, counts.data(), displs.data(), MPI_FLOAT, comm);
}

void allgather_uint32(const uint32_t* local_data, std::size_t local_cnt,
                      uint32_t* global_data, std::size_t global_n,
                      std::size_t stride, int size, MPI_Comm comm)
{
    std::vector<int> counts, displs;
    build_allgather_counts(global_n, size, stride, counts, displs);
    MPI_Allgatherv(local_data, static_cast<int>(local_cnt * stride), MPI_UNSIGNED,
                   global_data, counts.data(), displs.data(), MPI_UNSIGNED, comm);
}

} // anonymous namespace

// ── kernels ──────────────────────────────────────────────────────────────────

// Random-init kernel: one thread per owned point.
// Each point gets k distinct random neighbors sampled uniformly.
__global__
static void dist_rand_init_kernel(
    const float* __restrict__ all_feats,   // [N × d]
    uint32_t*    all_ids,                  // [N × k] write owned rows only
    float*       all_dists,
    std::size_t  p_begin,
    std::size_t  p_end,
    std::size_t  N,
    std::size_t  d,
    std::size_t  k,
    uint64_t     seed)
{
    const std::size_t p = p_begin +
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= p_end) return;

    XS64 rng(seed ^ (p + 1) * 6364136223846793005ULL);
    const float* fp = all_feats + p * d;
    uint32_t*    pi = all_ids   + p * k;
    float*       pd = all_dists + p * k;

    for (std::size_t j = 0; j < k; ++j) {
        std::size_t nb;
        do { nb = rng.bounded(N); } while (nb == p);
        pi[j] = static_cast<uint32_t>(nb);

        const float* fn = all_feats + nb * d;
        float dist = 0.0f;
        for (std::size_t dim = 0; dim < d; ++dim) {
            float diff = fp[dim] - fn[dim];
            dist += diff * diff;
        }
        pd[j] = dist;
    }
}

// Local-join kernel: one block per owned point.
// Explores neighbors-of-neighbors and updates the owned point's k-list.
// Only writes to the owned row — no cross-rank race conditions.
__global__
static void dist_local_join_kernel(
    const float* __restrict__ all_feats,   // [N × d]
    uint32_t*    all_ids,                  // [N × k] read globally, write owned
    float*       all_dists,
    std::size_t  p_begin,
    std::size_t  p_end,
    std::size_t  N,
    std::size_t  d,
    std::size_t  k,
    uint32_t*    g_updates)                // global update accumulator
{
    const std::size_t p = p_begin + static_cast<std::size_t>(blockIdx.x);
    if (p >= p_end) return;

    // Shared: sh_dists[k] | sh_ids[k] | sh_worst | sh_mutex | sh_upd
    extern __shared__ char sh_raw[];
    float*    sh_dists = reinterpret_cast<float*>(sh_raw);
    uint32_t* sh_ids   = reinterpret_cast<uint32_t*>(sh_dists + k);
    float*    sh_worst = reinterpret_cast<float*>(sh_ids + k);
    int*      sh_mutex = reinterpret_cast<int*>(sh_worst + 1);
    int*      sh_upd   = sh_mutex + 1;

    if (threadIdx.x == 0) {
        for (std::size_t j = 0; j < k; ++j) {
            sh_ids[j]   = all_ids[p * k + j];
            sh_dists[j] = all_dists[p * k + j];
        }
        float w = 0.0f;
        for (std::size_t j = 0; j < k; ++j) if (sh_dists[j] > w) w = sh_dists[j];
        *sh_worst = w;
        *sh_mutex = 0;
        *sh_upd   = 0;
    }
    __syncthreads();

    const float* fp = all_feats + p * d;

    // For each neighbor n1 of p, explore n1's neighbors n2.
    for (std::size_t i = 0; i < k; ++i) {
        uint32_t n1 = all_ids[p * k + i];
        if (n1 >= static_cast<uint32_t>(N)) continue;

        for (std::size_t j = threadIdx.x; j < k; j += blockDim.x) {
            uint32_t n2 = all_ids[n1 * k + j];
            if (n2 == static_cast<uint32_t>(p) || n2 >= static_cast<uint32_t>(N))
                continue;

            const float* fn2 = all_feats + n2 * d;
            float dist = 0.0f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                float diff = fp[dim] - fn2[dim];
                dist += diff * diff;
            }
            if (dist >= *sh_worst) continue;

            // Spinlock-guarded insert.
            while (atomicCAS(sh_mutex, 0, 1) != 0) {}
            if (dist < *sh_worst) {
                std::size_t wi = 0; float wd = sh_dists[0];
                for (std::size_t m = 1; m < k; ++m)
                    if (sh_dists[m] > wd) { wd = sh_dists[m]; wi = m; }
                sh_dists[wi] = dist;
                sh_ids[wi]   = n2;
                float nw = 0.0f;
                for (std::size_t m = 0; m < k; ++m)
                    if (sh_dists[m] > nw) nw = sh_dists[m];
                *sh_worst = nw;
                (*sh_upd)++;
            }
            atomicExch(sh_mutex, 0);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (std::size_t j = 0; j < k; ++j) {
            all_ids[p * k + j]   = sh_ids[j];
            all_dists[p * k + j] = sh_dists[j];
        }
        if (*sh_upd > 0)
            atomicAdd(g_updates, static_cast<uint32_t>(*sh_upd));
    }
}

// ── gpu_dist_nn_descent ───────────────────────────────────────────────────────

knng::Knng gpu_dist_nn_descent(const knng::Dataset& root_dataset,
                               const DistTopology&  topo,
                               const DistNndConfig& cfg,
                               MPI_Comm             comm)
{
    const int          rank     = topo.my_rank();
    const int          size     = topo.size();
    const std::size_t  k        = static_cast<std::size_t>(cfg.k);
    const std::size_t  max_iter = static_cast<std::size_t>(cfg.n_iterations);

    // ── Set GPU ─────────────────────────────────────────────────────────
    const int gpu_id = topo.gpu_for_rank(rank);
    if (gpu_id >= 0) cuda_chk(cudaSetDevice(gpu_id), "cudaSetDevice");

    // ── Scatter dataset ──────────────────────────────────────────────────
    auto shard = knng::dist::ShardedDataset::scatter(root_dataset, 0, comm);

    const std::size_t global_n = shard.global_n();
    const std::size_t local_n  = shard.local_n();
    const std::size_t d        = shard.d();
    const std::size_t p_begin  = shard.local_start();
    const std::size_t p_end    = shard.local_end();

    // ── AllGather full feature matrix ────────────────────────────────────
    std::vector<float> h_all_feats(global_n * d);
    allgather_floats(shard.local_dataset().data_ptr(), local_n,
                     h_all_feats.data(), global_n, d, size, comm);

    // ── Upload features to GPU ───────────────────────────────────────────
    float* d_feats = nullptr;
    cuda_chk(cudaMalloc(&d_feats, global_n * d * sizeof(float)), "cudaMalloc feats");
    cuda_chk(cudaMemcpy(d_feats, h_all_feats.data(),
                        global_n * d * sizeof(float),
                        cudaMemcpyHostToDevice), "H2D feats");

    // ── Allocate full graph buffers ──────────────────────────────────────
    uint32_t* d_ids   = nullptr;
    float*    d_dists = nullptr;
    cuda_chk(cudaMalloc(&d_ids,   global_n * k * sizeof(uint32_t)), "cudaMalloc ids");
    cuda_chk(cudaMalloc(&d_dists, global_n * k * sizeof(float)),    "cudaMalloc dists");

    // Update counter on device.
    uint32_t* d_updates = nullptr;
    cuda_chk(cudaMalloc(&d_updates, sizeof(uint32_t)), "cudaMalloc updates");

    // ── Random initialisation of owned rows ──────────────────────────────
    const int threads = 256;
    const int init_blocks = static_cast<int>(
        (local_n + static_cast<std::size_t>(threads) - 1) /
        static_cast<std::size_t>(threads));

    dist_rand_init_kernel<<<init_blocks, threads>>>(
        d_feats, d_ids, d_dists,
        p_begin, p_end, global_n, d, k,
        cfg.seed ^ static_cast<uint64_t>(rank) * 2654435761ULL);
    cuda_chk(cudaDeviceSynchronize(), "rand_init sync");

    // ── AllGather initial owned rows → full graph on all ranks ───────────
    // Owned IDs / dists live at [p_begin, p_end) in d_ids / d_dists.
    std::vector<uint32_t> h_ids(global_n * k, 0u);
    std::vector<float>    h_dists(global_n * k, std::numeric_limits<float>::infinity());

    // Download owned rows.
    cuda_chk(cudaMemcpy(h_ids.data()   + p_begin * k,
                        d_ids   + p_begin * k,
                        local_n * k * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost), "D2H init ids");
    cuda_chk(cudaMemcpy(h_dists.data() + p_begin * k,
                        d_dists + p_begin * k,
                        local_n * k * sizeof(float),
                        cudaMemcpyDeviceToHost), "D2H init dists");

    // AllGather across all ranks.
    allgather_uint32(h_ids.data()   + p_begin * k, local_n,
                     h_ids.data(), global_n, k, size, comm);
    allgather_floats(h_dists.data() + p_begin * k, local_n,
                     h_dists.data(), global_n, k, size, comm);

    // Upload full graph to GPU.
    cuda_chk(cudaMemcpy(d_ids,   h_ids.data(),
                        global_n * k * sizeof(uint32_t), cudaMemcpyHostToDevice),
             "H2D full ids");
    cuda_chk(cudaMemcpy(d_dists, h_dists.data(),
                        global_n * k * sizeof(float),    cudaMemcpyHostToDevice),
             "H2D full dists");

    // ── Shared memory size for local_join kernel ─────────────────────────
    const std::size_t sh_bytes = k * sizeof(float)        // sh_dists
                               + k * sizeof(uint32_t)     // sh_ids
                               + sizeof(float)            // sh_worst
                               + sizeof(int)              // sh_mutex
                               + sizeof(int);             // sh_upd

    const double threshold = cfg.delta * static_cast<double>(global_n * k);

    // ── Iteration loop ───────────────────────────────────────────────────
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Reset device update counter.
        cuda_chk(cudaMemset(d_updates, 0, sizeof(uint32_t)), "memset updates");

        // Local join on owned rows.
        dist_local_join_kernel<<<static_cast<unsigned>(local_n), threads, sh_bytes>>>(
            d_feats, d_ids, d_dists,
            p_begin, p_end, global_n, d, k, d_updates);
        cuda_chk(cudaDeviceSynchronize(), "local_join sync");

        // Read update count from device.
        uint32_t h_upd = 0;
        cuda_chk(cudaMemcpy(&h_upd, d_updates, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost), "D2H updates");

        // AllReduce global update count.
        uint32_t global_upd = 0;
        MPI_Allreduce(&h_upd, &global_upd, 1, MPI_UNSIGNED, MPI_SUM, comm);

        if (cfg.verbose && rank == 0) {
            // Minimal progress trace.
        }

        if (static_cast<double>(global_upd) < threshold) break;

        // AllGather updated owned rows → full graph on all ranks.
        cuda_chk(cudaMemcpy(h_ids.data()   + p_begin * k,
                            d_ids   + p_begin * k,
                            local_n * k * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), "D2H iter ids");
        cuda_chk(cudaMemcpy(h_dists.data() + p_begin * k,
                            d_dists + p_begin * k,
                            local_n * k * sizeof(float),
                            cudaMemcpyDeviceToHost), "D2H iter dists");

        allgather_uint32(h_ids.data()   + p_begin * k, local_n,
                         h_ids.data(), global_n, k, size, comm);
        allgather_floats(h_dists.data() + p_begin * k, local_n,
                         h_dists.data(), global_n, k, size, comm);

        // Upload full updated graph back to GPU.
        cuda_chk(cudaMemcpy(d_ids,   h_ids.data(),
                            global_n * k * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), "H2D updated ids");
        cuda_chk(cudaMemcpy(d_dists, h_dists.data(),
                            global_n * k * sizeof(float),
                            cudaMemcpyHostToDevice), "H2D updated dists");
    }

    // ── Cleanup GPU ──────────────────────────────────────────────────────
    cuda_chk(cudaFree(d_updates), "cudaFree updates");
    cuda_chk(cudaFree(d_dists),   "cudaFree dists");
    cuda_chk(cudaFree(d_ids),     "cudaFree ids");
    cuda_chk(cudaFree(d_feats),   "cudaFree feats");

    // ── Materialise local graph from host-side h_ids/h_dists ─────────────
    knng::Knng local_graph(local_n, k);
    for (std::size_t qi = 0; qi < local_n; ++qi) {
        auto nbrs = local_graph.neighbors_of(qi);
        auto dsts = local_graph.distances_of(qi);
        for (std::size_t j = 0; j < k; ++j) {
            nbrs[j] = static_cast<knng::index_t>(h_ids[(p_begin + qi) * k + j]);
            dsts[j] = h_dists[(p_begin + qi) * k + j];
        }
    }

    return knng::dist::gather_graph(local_graph, shard, 0, comm);
}

// ── Steps 93–96 stubs ────────────────────────────────────────────────────────
// Full optimised implementations follow in later steps; these entry points
// delegate to the baseline for now and are replaced step by step.

knng::Knng gpu_dist_nn_descent_dedup(const knng::Dataset& dataset,
                                     const DistTopology&  topo,
                                     const DistNndConfig& cfg,
                                     MPI_Comm             comm)
{
    return gpu_dist_nn_descent(dataset, topo, cfg, comm);
}

knng::Knng gpu_dist_nn_descent_shm(const knng::Dataset& dataset,
                                   const DistTopology&  topo,
                                   const DistNndConfig& cfg,
                                   MPI_Comm             comm)
{
    return gpu_dist_nn_descent(dataset, topo, cfg, comm);
}

knng::Knng gpu_dist_nn_descent_bw(const knng::Dataset& dataset,
                                  const DistTopology&  topo,
                                  const DistNndConfig& cfg,
                                  MPI_Comm             comm,
                                  int /*replication_threshold*/)
{
    return gpu_dist_nn_descent(dataset, topo, cfg, comm);
}

knng::Knng gpu_dist_nn_descent_overlap(const knng::Dataset& dataset,
                                       const DistTopology&  topo,
                                       const DistNndConfig& cfg,
                                       MPI_Comm             comm)
{
    return gpu_dist_nn_descent(dataset, topo, cfg, comm);
}

} // namespace knng::dist_gpu
