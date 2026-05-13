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

} // namespace knng::gpu
