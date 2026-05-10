#pragma once

/// @file
/// @brief Step 55 — Register-tiled brute-force KNN (query-side register tiling).
///
/// Previous kernels assign one query per thread (Step 48) or one query per
/// block (Steps 49/53).  This step assigns Q=4 queries per thread: the
/// thread loops over all references once and accumulates distances to all Q
/// queries simultaneously.  The cost of loading each reference vector is
/// amortized over Q computations rather than one.
///
/// ### Arithmetic intensity improvement
/// Naive (Q=1): 1 reference load → 2d FLOPs → intensity = 2d / (d×4) = 0.5 FLOP/byte.
/// Register-tiled (Q=4): 1 reference load → 2d×4 FLOPs → intensity = 2 FLOP/byte.
///
/// For SIFT-128 at fp32 this moves from ~64 to ~256 FLOP/byte — well above
/// the typical GPU memory-bandwidth ceiling (~80 FLOP/byte on A100 HBM2e),
/// making the kernel compute-bound rather than bandwidth-bound.
///
/// ### Constraints
/// Each query occupies `d` register floats.  For Q=4 and d=128 that is 512
/// additional registers per thread — the total register file on Volta
/// (65536 per SM, 255 per thread) becomes the hard limit.  Q=4 fits
/// comfortably (4×128 = 512 < 255 is *not* satisfied: 128 registers per
/// query exceeds the per-thread limit).  In practice `nvcc` will register-
/// spill to L1 for d>60 approximately.  The kernel is most effective for
/// smaller d; for large d (e.g., SIFT-128) the per-thread budget forces
/// Q=2 or Q=1 anyway.  The plan exposes Q as a compile-time template
/// parameter so Phase 9 can re-tune for each dataset dimensionality.
///
/// @tparam Q  Queries per thread (default 2 for safety across all d).

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <cstddef>

namespace knng::gpu {

/// @brief Compile-time queries-per-thread parameter.
inline constexpr std::size_t kRegTileQ = 2;

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Register-tiled brute-force KNN CPU reference.
///
/// Processes queries in groups of `kRegTileQ`: for each reference, computes
/// distances to all `kRegTileQ` queries before advancing.  Output must match
/// `cpu_brute_force_knn`.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query.
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng cpu_register_tiled_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

// ---------------------------------------------------------------------------
// GPU path
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Register-tiled brute-force KNN: kRegTileQ queries per thread.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query.
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng gpu_register_tiled_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

#endif

} // namespace knng::gpu
