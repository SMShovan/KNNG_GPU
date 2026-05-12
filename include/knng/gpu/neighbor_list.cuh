#pragma once

/// @file
/// @brief Step 61 — Device-side AoS neighbor list struct + layout analysis.
///
/// ## The `DeviceNeighborList<K_MAX>` struct
///
/// Fixed-width per-point neighbor list: `ids[K_MAX]`, `dists[K_MAX]`,
/// `flags` — matching the plan's specification exactly.
/// Declared with `alignas(128)` so every instance is 128-byte aligned.
///
/// ### Size calculation (K_MAX = 16)
/// ```
///   ids[16]    = 16 × 4 =  64 bytes
///   dists[16]  = 16 × 4 =  64 bytes
///   flags      =       1 ×  4 bytes
///              ─────────────────────
///   data total =        132 bytes
///   alignment  =        128 bytes
///   sizeof     = ceil(132/128) × 128 = 256 bytes
/// ```
/// `static_assert(sizeof(DeviceNeighborList<16>) == 256)` passes. ✓
///
/// For K_MAX = 32: data = 128 + 128 + 4 = 260 bytes → sizeof = 384.
/// For K_MAX = 15: data = 60  + 60  + 4 = 124 bytes → sizeof = 128 (exact!).
///
/// ## AoS vs SoA layout analysis
///
/// The live implementation uses **SoA** (`DeviceGraph` in `device_graph.hpp`)
/// rather than AoS.  The reasoning:
///
/// ### Per-point access (sequential within one point)
/// Both AoS and SoA perform identically: reading `ids[0..k-1]` and
/// `dists[0..k-1]` for one point is a stride-1 access in both layouts.
///
/// ### Batch access (warp reads same field for 32 consecutive points)
/// - **SoA**: `dists[p*k + r]` for p in {warp_base, …, warp_base+31}
///   → 32 consecutive 4-byte words → **1 L2 cache transaction**.
/// - **AoS**: `list[p].dists[r]` strides by `sizeof(DeviceNeighborList)` = 256
///   bytes between consecutive threads → **up to 32 L2 transactions**.
///
/// For the local-join kernel, which processes multiple points per warp
/// (Step 64 batch kernel), the batch access pattern is critical.
/// SoA gives 32× better coalescing efficiency → SoA wins.
///
/// `DeviceNeighborList` is retained for:
/// 1. Pedagogical completeness (plan requirement).
/// 2. Shared-memory staging: loading one point's neighbor data into a
///    shared-memory AoS struct enables intra-warp access without bank
///    conflicts (each 4-byte field maps to a different bank).

#include <knng/gpu/backend.hpp>
#include <knng/core/types.hpp>

#include <cstddef>
#include <cstdint>

namespace knng::gpu {

/// @brief Default maximum k for the AoS neighbor list (gives 256-byte struct).
inline constexpr unsigned int kNeighborListMaxK = 16u;

/// @brief Fixed-width AoS per-point neighbor list, `alignas(128)`.
///
/// @tparam K_MAX  Compile-time maximum number of neighbors (≤ 32 recommended).
template <unsigned int K_MAX = kNeighborListMaxK>
struct alignas(128) DeviceNeighborList {
    knng::index_t ids  [K_MAX];   ///< Neighbor IDs,  K_MAX × 4 bytes.
    float         dists[K_MAX];   ///< Distances,      K_MAX × 4 bytes.
    std::uint32_t flags;          ///< Bit `r` = is_new for neighbor slot `r`.

    /// @brief Mark slot `r` as new (thread-safe only under warp-exclusive access).
    GPU_HOST_DEVICE void set_new(unsigned int r) noexcept { flags |=  (1u << r); }
    /// @brief Mark slot `r` as old.
    GPU_HOST_DEVICE void set_old(unsigned int r) noexcept { flags &= ~(1u << r); }
    /// @brief Query is-new for slot `r`.
    GPU_HOST_DEVICE bool is_new(unsigned int r) const noexcept
        { return (flags >> r) & 1u; }
    /// @brief Mark all slots old.
    GPU_HOST_DEVICE void mark_all_old() noexcept { flags = 0u; }
};

// Layout assertion: K_MAX=16 → 132 bytes data, alignas(128) → 256-byte struct.
static_assert(sizeof(DeviceNeighborList<16>) == 256,
              "DeviceNeighborList<16> must be 256 bytes (128-byte alignment, 132 bytes data)");

// ---------------------------------------------------------------------------
// AoS ↔ SoA conversion (host-side; not callable from GPU kernels)
// ---------------------------------------------------------------------------

/// @brief Copy one AoS entry into SoA row `qi` of flat arrays.
template <unsigned int K_MAX>
inline void aos_to_soa(
    const DeviceNeighborList<K_MAX>& src,
    unsigned int                     qi,
    unsigned int                     k,
    knng::index_t*                   soa_ids,
    float*                           soa_dists,
    std::uint32_t*                   soa_flags)
{
    for (unsigned int r = 0; r < k && r < K_MAX; ++r) {
        soa_ids  [qi * k + r] = src.ids  [r];
        soa_dists[qi * k + r] = src.dists[r];
        soa_flags[qi * k + r] = src.is_new(r) ? 1u : 0u;
    }
}

/// @brief Copy SoA row `qi` into an AoS `DeviceNeighborList<K_MAX>`.
template <unsigned int K_MAX>
inline void soa_to_aos(
    unsigned int                 qi,
    unsigned int                 k,
    const knng::index_t*         soa_ids,
    const float*                 soa_dists,
    const std::uint32_t*         soa_flags,
    DeviceNeighborList<K_MAX>&   dst)
{
    dst.flags = 0u;
    for (unsigned int r = 0; r < k && r < K_MAX; ++r) {
        dst.ids  [r] = soa_ids  [qi * k + r];
        dst.dists[r] = soa_dists[qi * k + r];
        if (soa_flags[qi * k + r] & 1u) { dst.set_new(r); }
    }
}

} // namespace knng::gpu
