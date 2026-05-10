#pragma once

/// @file
/// @brief Step 53 — Shared-memory reference-tiling brute-force KNN.
///
/// The Step-49 block-per-query kernel scans references with a stride of
/// `blockDim.x` per thread.  Each thread independently loads its reference
/// slice from global memory; there is zero reuse across threads in the block.
///
/// This step adds a *tiled* variant: threads cooperatively load a tile of
/// `TILE_W` reference vectors into shared memory, then every thread in the
/// block computes its query distance against all `TILE_W` references in the
/// tile.  The net effect: each reference float is loaded from global memory
/// once and used `blockDim.x` times — reducing global traffic by
/// `blockDim.x / TILE_W`.
///
/// ### TILE_W choice
/// `TILE_W = 32` (one warp) is the natural choice: the tile load is a
/// single coalesced warp transaction (32 floats × `d` dimensions), and the
/// tile fits in shared memory without bank conflicts when loaded in the
/// canonical pattern (thread `t` loads reference `tile_base + t`).
///
/// ### CPU reference
/// `cpu_tiled_brute_force_knn` implements the same tiling loop in portable
/// C++ (tile size = `TILE_W`).  Its output must be identical to
/// `cpu_brute_force_knn` from Step 48.

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <cstddef>

namespace knng::gpu {

/// Tile width used by the shared-memory brute-force kernel.
inline constexpr std::size_t kTileW = 32;

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Tiled brute-force KNN on the CPU — same output as cpu_brute_force_knn.
///
/// Processes references in tiles of `kTileW` to mirror the GPU access pattern
/// and make the tiling logic separately testable.
///
/// @param dataset  Input point set.
/// @param k        Number of nearest neighbours.
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng cpu_tiled_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

// ---------------------------------------------------------------------------
// GPU path
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Shared-memory tiled brute-force KNN.
///
/// One block per query; threads cooperate to load tiles of `kTileW`
/// references into shared memory and scan the query against each tile.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query.
/// @return         Knng sorted by distance.
[[nodiscard]] knng::Knng gpu_tiled_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k);

#endif

} // namespace knng::gpu
