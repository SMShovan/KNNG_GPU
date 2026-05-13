#pragma once

/// @file
/// @brief Steps 72–79 (plan Steps 71–78) — CAGRA-style graph refinement.
///
/// Post-processes the NN-Descent output from Phase 9 to improve ANN search
/// quality.  Steps are added incrementally across commits 72–79; the full
/// API is assembled in Step 77.

#include <knng/gpu/device_graph.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <cstddef>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Step 77 — Ablation configuration (populated as steps are added)
// ---------------------------------------------------------------------------

/// @brief Controls which CAGRA-style refinement steps are applied.
struct GraphRefinementConfig {
    bool enforce_out_degree = true;  ///< Step 72.
    bool rank_reorder       = true;  ///< Step 73.
    bool add_reverse_edges  = true;  ///< Step 74.
    bool prune_detour       = true;  ///< Step 75.
    bool merge_components   = true;  ///< Step 76.
};

// ---------------------------------------------------------------------------
// Step 72 — Fixed out-degree enforcement
// ---------------------------------------------------------------------------

/// @brief Sort each row by distance ascending; sentinels last.
///
/// After NN-Descent atomic insertions the per-point lists are unsorted.
/// This step establishes the invariant that position 0 is always the closest
/// neighbor, which subsequent refinement steps depend on.
///
/// @param graph  In/out SoA graph (mutated in place).
void cpu_enforce_out_degree(CpuDeviceGraph& graph);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU: sort each row by distance ascending (one block per point).
void gpu_enforce_out_degree(DeviceGraph& graph);
#endif

// ---------------------------------------------------------------------------
// Step 73 — Rank-based reordering
// ---------------------------------------------------------------------------

/// @brief Reorder each point's neighbor list so that neighbors with lower
/// average rank (i.e., they appear close to position 0 in their own lists)
/// come first.  Valid entries only; sentinels stay at the tail.
///
/// For each candidate neighbor j of point i, its rank score is the mean
/// list-position at which i appears in j's own neighbor list (lower = better).
/// The row is then stably sorted by rank score ascending.
///
/// @param graph  In/out SoA graph (mutated in place).
void cpu_rank_reorder(CpuDeviceGraph& graph);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU: build per-neighbor rank scores then sort each row (one block per point).
void gpu_rank_reorder(DeviceGraph& graph);
#endif

// ---------------------------------------------------------------------------
// Step 74 — Add reverse edges
// ---------------------------------------------------------------------------

/// @brief For every directed edge i → j, attempt to add the reverse edge j → i
/// (with the same distance) into j's neighbor list, replacing the worst
/// current entry if j's list is full.  Sentinels are overwritten first.
///
/// A reverse edge improves recall when the original graph is asymmetric:
/// points that only appear *as* neighbors (but do not *have* i in their own
/// list) become reachable from both directions.
///
/// @param graph  In/out SoA graph (mutated in place).
void cpu_add_reverse_edges(CpuDeviceGraph& graph);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU: scatter reverse edges using atomic distance comparisons.
void gpu_add_reverse_edges(DeviceGraph& graph);
#endif

// ---------------------------------------------------------------------------
// Step 75 — Detourable-edge pruning (MRNG rule)
// ---------------------------------------------------------------------------

/// @brief Remove edge i → j if there exists another neighbor m of i such that
/// d(i,m) < d(i,j) and d(m,j) < d(i,j) — i.e., the path i→m→j is a detour
/// that makes the direct edge redundant.
///
/// This is the Monotonic Relative Neighborhood Graph (MRNG) pruning rule
/// described in the CAGRA paper (Ootomo et al., IPDPS 2023).  Pruned slots
/// are replaced with sentinels.
///
/// @param graph    In/out SoA graph (mutated in place).
/// @param vectors  Row-major float matrix, shape [n × dim], used to compute
///                 d(m, j) between neighbors.
/// @param dim      Feature dimension.
void cpu_prune_detour_edges(CpuDeviceGraph& graph,
                             const float* vectors,
                             std::size_t dim);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU: mark detourable edges as sentinels (one block per point).
void gpu_prune_detour_edges(DeviceGraph& graph,
                             const float* d_vectors,
                             std::size_t dim);
#endif

// ---------------------------------------------------------------------------
// Step 76 — Strong-component merging
// ---------------------------------------------------------------------------

/// @brief Detect weakly connected components (union-find on the undirected
/// adjacency) and, for each isolated component, add a bridge edge from its
/// most central node to the closest node in the largest component.
///
/// After pruning (Step 75), some nodes may lose all valid neighbors and
/// become unreachable singletons.  This step reconnects them so the final
/// graph is a single weakly connected component.
///
/// @param graph    In/out SoA graph (mutated in place).
/// @param vectors  Row-major float matrix, shape [n × dim].
/// @param dim      Feature dimension.
void cpu_merge_components(CpuDeviceGraph& graph,
                           const float* vectors,
                           std::size_t dim);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU: label propagation to find components then bridge singletons.
void gpu_merge_components(DeviceGraph& graph,
                           const float* d_vectors,
                           std::size_t dim);
#endif

// ---------------------------------------------------------------------------
// Step 77 — Full pipeline driver
// ---------------------------------------------------------------------------

/// @brief Run the complete CAGRA-style refinement pipeline on a CPU graph.
///
/// Steps are applied in order 72 → 73 → 74 → 75 → 76, each gated by the
/// corresponding flag in `cfg`.  Steps 75 and 76 require `vectors` to be
/// non-null.
///
/// @param graph    In/out SoA graph.
/// @param cfg      Ablation flags — set false to skip individual steps.
/// @param vectors  Row-major float matrix [n × dim]; may be nullptr if
///                 neither `prune_detour` nor `merge_components` is set.
/// @param dim      Feature dimension (ignored when vectors is nullptr).
void cpu_refine_graph(CpuDeviceGraph& graph,
                       const GraphRefinementConfig& cfg,
                       const float* vectors = nullptr,
                       std::size_t dim = 0);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU pipeline driver — chains Steps 72–76 with config gating.
void gpu_refine_graph(DeviceGraph& graph,
                       const GraphRefinementConfig& cfg,
                       const float* d_vectors = nullptr,
                       std::size_t dim = 0);
#endif

} // namespace knng::gpu
