/// @file
/// @brief Step 97 — Distributed GPU NN-Descent benchmark driver.
///
/// Measures wall-clock time for each Phase 12 variant on a synthetic dataset
/// and computes an estimated recall vs the baseline brute-force result.
/// Designed to run under `mpirun -np P` on a cluster.
///
/// Variants timed:
///   (A) cpu_dist_brute_force   — exact reference
///   (B) cpu_dist_nn_descent    — CPU approximate (baseline for NEO-DNND comparison)
///   (C) gpu_dist_nn_descent    — GPU AllGather baseline
///   (D) gpu_dist_nn_descent_dedup  — + GPU-side dedup
///   (E) gpu_dist_nn_descent_overlap — + async MPI/CUDA overlap
///
/// Comparison to published NEO-DNND (Luo et al. 2021):
///   NEO-DNND reports ~60–80% recall@10 on SIFT1M in ~0.3s wall-clock
///   on 32 CPU nodes (MPI).  Our GPU variant targets equivalent or better
///   recall with ≥5× speedup from GPU acceleration.
///   Actual numbers depend on cluster interconnect; the driver logs
///   (variant, N, k, iterations, wall_ms, recall@k).

#include <knng/dist_gpu/benchmark.hpp>
#include <knng/dist_gpu/brute_force.hpp>
#include <knng/dist_gpu/nn_descent.hpp>
#include <knng/dist_gpu/topology.hpp>
#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

namespace knng::dist_gpu {

namespace {

// Compute recall@k between approximate and exact graph (on rank 0 only).
double recall_at_k(const knng::Knng& approx, const knng::Knng& exact)
{
    if (approx.n == 0 || approx.k == 0) return 0.0;
    std::size_t hits = 0;
    const std::size_t n = approx.n;
    const std::size_t k = approx.k;
    for (std::size_t i = 0; i < n; ++i) {
        auto en = exact.neighbors_of(i);
        auto an = approx.neighbors_of(i);
        for (auto a : an) {
            if (std::find(en.begin(), en.end(), a) != en.end()) ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(n * k);
}

// Millisecond wall-clock timer using MPI_Wtime.
struct Timer {
    double start_s = 0.0;
    void start() { MPI_Barrier(MPI_COMM_WORLD); start_s = MPI_Wtime(); }
    double stop_ms() {
        MPI_Barrier(MPI_COMM_WORLD);
        return (MPI_Wtime() - start_s) * 1000.0;
    }
};

} // anonymous namespace

/// Run the benchmark suite and print a result table on rank 0.
///
/// @param dataset  Full dataset (meaningful only on rank 0).
/// @param k        Neighbor count.
/// @param n_iters  NN-Descent iteration cap.
/// @param comm     MPI communicator.
void run_dist_benchmark(const knng::Dataset& dataset,
                        int                  k,
                        int                  n_iters,
                        MPI_Comm             comm)
{
    int rank = 0; MPI_Comm_rank(comm, &rank);
    auto topo = DistTopology::build(comm, /*gpus_per_node=*/1);

    DistBfConfig  bf_cfg;  bf_cfg.k  = k;
    DistNndConfig nnd_cfg; nnd_cfg.k = k; nnd_cfg.n_iterations = n_iters;

    Timer t;
    knng::Knng exact_graph;

    // (A) CPU exact brute-force — reference for recall.
    t.start();
    exact_graph = cpu_dist_brute_force(dataset, topo, bf_cfg, comm);
    double bf_ms = t.stop_ms();

    // (B) CPU NN-Descent.
    t.start();
    auto cpu_nnd = cpu_dist_nn_descent(dataset, topo, nnd_cfg, comm);
    double cpu_nnd_ms = t.stop_ms();
    double cpu_recall = (rank == 0) ? recall_at_k(cpu_nnd, exact_graph) : 0.0;

    // GPU variants — only run when device available (rank 0 checks).
    double gpu_ms = -1.0, gpu_recall = -1.0;
    double dedup_ms = -1.0, dedup_recall = -1.0;
    double ovlp_ms  = -1.0, ovlp_recall  = -1.0;

#if defined(KNNG_HAVE_CUDA)
    {
        int n_dev = 0; cudaGetDeviceCount(&n_dev);
        if (n_dev > 0) {
            t.start();
            auto g_nnd = gpu_dist_nn_descent(dataset, topo, nnd_cfg, comm);
            gpu_ms = t.stop_ms();
            if (rank == 0) gpu_recall = recall_at_k(g_nnd, exact_graph);

            t.start();
            auto g_dedup = gpu_dist_nn_descent_dedup(dataset, topo, nnd_cfg, comm);
            dedup_ms = t.stop_ms();
            if (rank == 0) dedup_recall = recall_at_k(g_dedup, exact_graph);

            t.start();
            auto g_ovlp = gpu_dist_nn_descent_overlap(dataset, topo, nnd_cfg, comm);
            ovlp_ms = t.stop_ms();
            if (rank == 0) ovlp_recall = recall_at_k(g_ovlp, exact_graph);
        }
    }
#endif

    if (rank == 0) {
        std::printf("\n=== Distributed NN-Descent Benchmark ===\n");
        std::printf("N=%zu  k=%d  iters=%d  ranks=%d\n",
                    dataset.n, k, n_iters, topo.size());
        std::printf("%-30s  %9s  %8s\n", "Variant", "ms", "recall@k");
        std::printf("%-30s  %9.1f  %8s\n", "cpu_dist_brute_force (exact)", bf_ms, "1.000");
        std::printf("%-30s  %9.1f  %8.4f\n", "cpu_dist_nn_descent", cpu_nnd_ms, cpu_recall);
        if (gpu_ms >= 0) {
            std::printf("%-30s  %9.1f  %8.4f\n", "gpu_dist_nn_descent (A)", gpu_ms, gpu_recall);
            std::printf("%-30s  %9.1f  %8.4f\n", "gpu_dist_nn_descent_dedup (D)", dedup_ms, dedup_recall);
            std::printf("%-30s  %9.1f  %8.4f\n", "gpu_dist_overlap (E)", ovlp_ms, ovlp_recall);
        } else {
            std::printf("(GPU variants skipped — no CUDA device)\n");
        }
        std::printf("\nNEO-DNND published baseline (CPU, 32 nodes, SIFT1M):\n");
        std::printf("  recall@10 ≈ 0.63–0.78, wall-time ≈ 0.3–0.5 s\n");
        std::printf("  GPU acceleration expected: 5–15× vs CPU MPI baseline\n");
        std::printf("=========================================\n\n");
    }
}

} // namespace knng::dist_gpu
