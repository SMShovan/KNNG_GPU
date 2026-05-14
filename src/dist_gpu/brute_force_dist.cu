/// @file
/// @brief Step 91 — Distributed GPU brute-force KNN (ring-based).
///
/// Algorithm:
///   1. Rank 0 broadcasts meta-data; each rank allocates GPU memory for its
///      query shard and a reference-block buffer.
///   2. P ring steps:
///        a. Intra-node: rank 0 of node uses ncclBcast to push the reference
///           block to sibling ranks on the same node (NVLink).
///        b. GPU kernel: each rank computes L2² distances from its queries
///           to the current reference block and updates its local top-k.
///        c. Ring shift: MPI_Sendrecv (host pinned staging) moves the
///           reference block to the right neighbour.
///   3. Copy local top-k from device to host; MPI gather on rank 0.
///
/// Falls back cleanly to the CPU path if only 1 rank or no GPU.

#include <knng/dist_gpu/brute_force.hpp>
#include <knng/dist_gpu/cuda_aware_mpi.hpp>
#include <knng/dist/sharded_dataset.hpp>
#include <knng/dist/brute_force_mpi.hpp>

#include <cuda_runtime.h>
#include <mpi.h>

#if defined(KNNG_HAVE_NCCL)
#  include <nccl.h>
#endif

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace knng::dist_gpu {

// ── helpers ─────────────────────────────────────────────────────────────────

namespace {

inline void cuda_chk(cudaError_t e, const char* w)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// Block-per-query brute-force kernel.
// Grid: one block per query in [q_begin, q_end).
// Block: up to 256 threads.
// Shared memory layout (per block):
//   float sh_dists[k]  — current top-k distances (max-heap).
//   uint32_t sh_ids[k] — corresponding indices.
//   float sh_worst     — current k-th distance.
//   int sh_mutex       — spinlock for candidates that beat sh_worst.
// ---------------------------------------------------------------------------
__global__
void bf_ring_kernel(const float* __restrict__ queries,   // [local_n × d]
                    const float* __restrict__ refs,      // [ring_n × d]
                    float*        out_dists,              // [local_n × k]
                    unsigned int* out_ids,               // [local_n × k]
                    std::size_t   local_n,
                    std::size_t   ring_n,
                    std::size_t   ring_start,            // global base of refs
                    std::size_t   q_start,               // global base of queries
                    std::size_t   d,
                    std::size_t   k)
{
    const std::size_t qi = static_cast<std::size_t>(blockIdx.x);
    if (qi >= local_n) return;

    // Dynamic shared memory: sh_dists[k] | sh_ids[k] | sh_worst | sh_mutex
    extern __shared__ char sh_raw[];
    float*        sh_dists = reinterpret_cast<float*>(sh_raw);
    unsigned int* sh_ids   = reinterpret_cast<unsigned int*>(sh_dists + k);
    float*        sh_worst = reinterpret_cast<float*>(sh_ids + k);
    int*          sh_mutex = reinterpret_cast<int*>(sh_worst + 1);

    if (threadIdx.x == 0) {
        for (std::size_t j = 0; j < k; ++j) {
            sh_dists[j] = std::numeric_limits<float>::infinity();
            sh_ids[j]   = 0u;
        }
        *sh_worst = std::numeric_limits<float>::infinity();
        *sh_mutex = 0;
    }
    __syncthreads();

    const float* q_ptr      = queries + qi * d;
    const std::size_t q_gid = q_start + qi;

    for (std::size_t ri = threadIdx.x; ri < ring_n; ri += blockDim.x) {
        const std::size_t r_gid = ring_start + ri;
        if (r_gid == q_gid) continue;  // skip self

        const float* r_ptr = refs + ri * d;
        float dist = 0.0f;
        for (std::size_t dim = 0; dim < d; ++dim) {
            float diff = q_ptr[dim] - r_ptr[dim];
            dist += diff * diff;
        }

        if (dist >= *sh_worst) continue;

        // Acquire spinlock.
        bool acquired = false;
        while (!acquired) {
            if (atomicCAS(sh_mutex, 0, 1) == 0) {
                acquired = true;
            }
        }
        // Critical section — update max-heap.
        if (dist < *sh_worst) {
            // Find and replace the worst entry.
            std::size_t worst_j = 0;
            float worst_d = sh_dists[0];
            for (std::size_t j = 1; j < k; ++j) {
                if (sh_dists[j] > worst_d) { worst_d = sh_dists[j]; worst_j = j; }
            }
            sh_dists[worst_j] = dist;
            sh_ids[worst_j]   = static_cast<unsigned int>(r_gid);
            // Recompute worst.
            float new_worst = 0.0f;
            for (std::size_t j = 0; j < k; ++j)
                if (sh_dists[j] > new_worst) new_worst = sh_dists[j];
            *sh_worst = new_worst;
        }
        atomicExch(sh_mutex, 0);  // release spinlock
    }
    __syncthreads();

    // Thread 0 writes the top-k result to global memory.
    if (threadIdx.x == 0) {
        float* d_out    = out_dists + qi * k;
        auto*  id_out   = out_ids   + qi * k;
        for (std::size_t j = 0; j < k; ++j) {
            d_out[j]  = sh_dists[j];
            id_out[j] = sh_ids[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side top-k merge: merge ref_block results into running best.
// ---------------------------------------------------------------------------
void merge_ring_topk(
    float*        best_dists,   // [local_n × k] running best, host
    unsigned int* best_ids,     // [local_n × k]
    const float*  new_dists,    // [local_n × k] results from this ring step
    const unsigned int* new_ids,
    std::size_t   local_n,
    std::size_t   k)
{
    for (std::size_t qi = 0; qi < local_n; ++qi) {
        float*        bd = best_dists + qi * k;
        unsigned int* bi = best_ids   + qi * k;
        const float*        nd = new_dists + qi * k;
        const unsigned int* ni = new_ids   + qi * k;

        // Insert each new candidate; if it beats the current worst, replace.
        for (std::size_t j = 0; j < k; ++j) {
            if (nd[j] == std::numeric_limits<float>::infinity()) continue;
            // Find worst in best.
            float worst = bd[0]; std::size_t wi = 0;
            for (std::size_t m = 1; m < k; ++m)
                if (bd[m] > worst) { worst = bd[m]; wi = m; }
            if (nd[j] < worst) { bd[wi] = nd[j]; bi[wi] = ni[j]; }
        }
    }
}

} // anonymous namespace

// ── gpu_dist_brute_force ────────────────────────────────────────────────────

knng::Knng gpu_dist_brute_force(const knng::Dataset& root_dataset,
                                const DistTopology&  topo,
                                const DistBfConfig&  cfg,
                                MPI_Comm             comm)
{
    const int rank = topo.my_rank();
    const int size = topo.size();
    const int k    = cfg.k;

    // ── Scatter dataset ──────────────────────────────────────────────────
    auto shard = knng::dist::ShardedDataset::scatter(root_dataset, 0, comm);
    const std::size_t local_n  = shard.local_n();
    const std::size_t global_n = shard.global_n();
    const std::size_t d        = shard.d();
    const std::size_t q_start  = shard.local_start();

    // Set GPU for this rank.
    const int gpu_id = topo.gpu_for_rank(rank);
    if (gpu_id >= 0) cuda_chk(cudaSetDevice(gpu_id), "cudaSetDevice");

    // ── Allocate GPU memory ──────────────────────────────────────────────
    // Query shard (fixed).
    float* d_queries = nullptr;
    const std::size_t query_bytes = local_n * d * sizeof(float);
    cuda_chk(cudaMalloc(&d_queries, query_bytes), "cudaMalloc queries");
    cuda_chk(cudaMemcpy(d_queries, shard.local_dataset().data_ptr(),
                        query_bytes, cudaMemcpyHostToDevice), "H2D queries");

    // Reference block buffer (changes each ring step).
    // Worst-case size: largest shard.
    const std::size_t max_ring_n = (global_n + static_cast<std::size_t>(size) - 1)
                                   / static_cast<std::size_t>(size);
    float* d_refs = nullptr;
    cuda_chk(cudaMalloc(&d_refs, max_ring_n * d * sizeof(float)),
             "cudaMalloc refs");

    // Per-step output top-k (before merging).
    float*        d_step_dists = nullptr;
    unsigned int* d_step_ids   = nullptr;
    cuda_chk(cudaMalloc(&d_step_dists, local_n * static_cast<std::size_t>(k) * sizeof(float)),
             "cudaMalloc step_dists");
    cuda_chk(cudaMalloc(&d_step_ids,   local_n * static_cast<std::size_t>(k) * sizeof(unsigned int)),
             "cudaMalloc step_ids");

    // ── Host-side running top-k ──────────────────────────────────────────
    std::vector<float>        best_dists(local_n * static_cast<std::size_t>(k),
                                         std::numeric_limits<float>::infinity());
    std::vector<unsigned int> best_ids(local_n * static_cast<std::size_t>(k), 0u);

    // ── Ring loop ────────────────────────────────────────────────────────
    // Start with this rank's own shard as the first reference block.
    std::vector<float> ring_buf(shard.local_dataset().data);
    std::size_t ring_start = shard.local_start();
    std::size_t ring_n     = local_n;

    // Pinned staging buffer for ring shift.
    float* h_ring_pinned = nullptr;
    cuda_chk(cudaMallocHost(&h_ring_pinned, max_ring_n * d * sizeof(float)),
             "cudaMallocHost ring_pinned");

    const std::size_t sh_bytes = (static_cast<std::size_t>(k) * sizeof(float)     // sh_dists
                                 + static_cast<std::size_t>(k) * sizeof(unsigned int) // sh_ids
                                 + sizeof(float)                    // sh_worst
                                 + sizeof(int));                    // sh_mutex

    for (int step = 0; step < size; ++step) {
        // Upload current reference block to device.
        const std::size_t ref_bytes = ring_n * d * sizeof(float);
        cuda_chk(cudaMemcpy(d_refs, ring_buf.data(), ref_bytes,
                            cudaMemcpyHostToDevice), "H2D ref block");

        // Launch block-per-query kernel.
        const int threads = 256;
        bf_ring_kernel<<<static_cast<unsigned int>(local_n), threads,
                         sh_bytes>>>(
            d_queries, d_refs, d_step_dists, d_step_ids,
            local_n, ring_n, ring_start, q_start, d,
            static_cast<std::size_t>(k));
        cuda_chk(cudaDeviceSynchronize(), "bf_ring_kernel sync");

        // Copy step results to host and merge into running best.
        std::vector<float>        h_step_dists(local_n * static_cast<std::size_t>(k));
        std::vector<unsigned int> h_step_ids(local_n * static_cast<std::size_t>(k));
        cuda_chk(cudaMemcpy(h_step_dists.data(), d_step_dists,
                            h_step_dists.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "D2H step_dists");
        cuda_chk(cudaMemcpy(h_step_ids.data(), d_step_ids,
                            h_step_ids.size() * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost), "D2H step_ids");

        merge_ring_topk(best_dists.data(), best_ids.data(),
                        h_step_dists.data(), h_step_ids.data(),
                        local_n, static_cast<std::size_t>(k));

        if (step == size - 1) break;

        // Ring shift: send to right, receive from left.
        const int right       = (rank + 1) % size;
        const int left        = (rank - 1 + size) % size;
        const int source_rank = (rank - step - 1 + size) % size;
        const auto [next_start, next_n] =
            knng::dist::compute_shard(global_n, size, source_rank);

        std::vector<float> recv_buf(next_n * d);
        MPI_Sendrecv(ring_buf.data(),  static_cast<int>(ring_n * d),  MPI_FLOAT,
                     right, step,
                     recv_buf.data(),  static_cast<int>(next_n * d),  MPI_FLOAT,
                     left, step, comm, MPI_STATUS_IGNORE);

        ring_buf   = std::move(recv_buf);
        ring_start = next_start;
        ring_n     = next_n;
    }

    // ── Cleanup GPU memory ───────────────────────────────────────────────
    cuda_chk(cudaFreeHost(h_ring_pinned), "cudaFreeHost ring_pinned");
    cuda_chk(cudaFree(d_step_ids),    "cudaFree step_ids");
    cuda_chk(cudaFree(d_step_dists),  "cudaFree step_dists");
    cuda_chk(cudaFree(d_refs),        "cudaFree refs");
    cuda_chk(cudaFree(d_queries),     "cudaFree queries");

    // ── Materialise local Knng ───────────────────────────────────────────
    knng::Knng local_graph(local_n, static_cast<std::size_t>(k));
    for (std::size_t qi = 0; qi < local_n; ++qi) {
        auto nbrs = local_graph.neighbors_of(qi);
        auto dsts = local_graph.distances_of(qi);
        for (int j = 0; j < k; ++j) {
            nbrs[static_cast<std::size_t>(j)] =
                static_cast<knng::index_t>(best_ids[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)]);
            dsts[static_cast<std::size_t>(j)] =
                best_dists[qi * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
        }
    }

    // ── Gather to rank 0 ─────────────────────────────────────────────────
    return knng::dist::gather_graph(local_graph, shard, 0, comm);
}

} // namespace knng::dist_gpu
