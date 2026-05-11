#pragma once

/// @file
/// @brief Step 62 — GPU-side SoA neighbor graph for NN-Descent.
///
/// The CPU NN-Descent uses `NnDescentGraph` (an array of `NeighborList`
/// objects, each a heap-allocated `std::vector<Neighbor>`).  That shape
/// is fundamentally incompatible with GPU execution: pointer-chasing into
/// per-point heap allocations destroys coalescing and cannot be represented
/// in device memory at all.
///
/// This step introduces `DeviceGraph`: a **Structure-of-Arrays** (SoA)
/// representation where neighbor IDs, distances, and is-new flags live in
/// three flat contiguous device buffers of shape `[n × k]` (row-major).
///
/// ### Layout choice: SoA vs AoS
/// The NN-Descent kernels access neighbor data in two patterns:
/// 1. **Per-point**: read all `k` (id, dist) pairs for one point.
///    Both SoA and AoS are equally cache-friendly here.
/// 2. **Batch**: one warp reads the same field (e.g., distances) for
///    32 consecutive points simultaneously (in the local-join).
///    SoA gives a fully coalesced warp transaction; AoS would stride
///    by `sizeof(DeviceNeighborList)` bytes between threads.
///
/// SoA wins for batch access and is equal or better for per-point access,
/// so SoA is the canonical layout throughout Phase 9.
///
/// ### Flags encoding
/// `flags[n*k]` stores per-entry is-new bits.  Bit 0 = 1 → the entry is
/// "new" (inserted since the last local-join iteration) and participates in
/// the next local-join.  Bit 0 = 0 → "old".  Using a full `uint32_t` per
/// entry (rather than a packed bitfield) avoids sub-word atomic races and
/// lets future steps add more flag bits without ABI changes.
///
/// ### CPU mirror: `CpuDeviceGraph`
/// Mirrors the SoA layout in host memory so all algorithmic logic can be
/// tested on Mac without a GPU.  The conversion helpers `from_knng()` and
/// `to_knng()` are shared between the GPU and CPU variants.

#include <knng/gpu/device_buffer.hpp>
#include <knng/core/graph.hpp>
#include <knng/core/types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace knng::gpu {

/// Maximum k supported by the GPU NN-Descent kernels.
/// Higher k exceeds the per-thread register budget for the local-join.
inline constexpr std::size_t kNndMaxK = 64;

/// Sentinel distance meaning "no neighbor assigned yet."
inline constexpr float kSentinelDist = 1e38f;

// ---------------------------------------------------------------------------
// CPU mirror (always compiled — used by all tests on Mac)
// ---------------------------------------------------------------------------

/// @brief CPU-side mirror of the GPU SoA neighbor graph.
///
/// Uses `std::vector` instead of `DeviceBuffer` so it compiles and runs
/// without CUDA.  All GPU NN-Descent kernels have a CPU reference that
/// operates on `CpuDeviceGraph` and produces bit-identical output.
struct CpuDeviceGraph {
    std::vector<knng::index_t> ids;    ///< Row-major [n][k] neighbor IDs.
    std::vector<float>         dists;  ///< Row-major [n][k] distances.
    std::vector<std::uint32_t> flags;  ///< Row-major [n][k]; bit 0 = is_new.
    std::size_t n = 0;
    std::size_t k = 0;

    CpuDeviceGraph() = default;

    /// @brief Allocate an empty graph for `n` points and `k` neighbors.
    ///
    /// All IDs initialised to `index_t(-1)`, distances to `kSentinelDist`,
    /// flags to 0 (old).
    CpuDeviceGraph(std::size_t n, std::size_t k);

    /// @brief Construct from a `knng::Knng`; marks all entries as new.
    [[nodiscard]] static CpuDeviceGraph from_knng(const knng::Knng& g);

    /// @brief Convert to `knng::Knng` (strips flags, copies ids/dists).
    [[nodiscard]] knng::Knng to_knng() const;

    /// @brief Accessor helpers.
    knng::index_t& id   (std::size_t qi, std::size_t r) noexcept
        { return ids  [qi * k + r]; }
    float&         dist (std::size_t qi, std::size_t r) noexcept
        { return dists[qi * k + r]; }
    std::uint32_t& flag (std::size_t qi, std::size_t r) noexcept
        { return flags[qi * k + r]; }

    const knng::index_t& id   (std::size_t qi, std::size_t r) const noexcept
        { return ids  [qi * k + r]; }
    const float&         dist (std::size_t qi, std::size_t r) const noexcept
        { return dists[qi * k + r]; }
    const std::uint32_t& flag (std::size_t qi, std::size_t r) const noexcept
        { return flags[qi * k + r]; }

    /// @brief True if entry (qi, r) is flagged as new.
    bool is_new(std::size_t qi, std::size_t r) const noexcept
        { return (flags[qi * k + r] & 1u) != 0; }

    /// @brief Try to insert (id, d) into row qi, returning true if changed.
    ///
    /// Rejects if id is already present or d >= current worst.
    /// Newly inserted entries are marked is_new=1.
    bool try_insert(std::size_t qi, knng::index_t id, float d);

    /// @brief Worst distance in row qi (max over all k entries).
    float worst_dist(std::size_t qi) const noexcept;

    /// @brief Mark all entries in row qi as old (flags &= ~1).
    void mark_old(std::size_t qi) noexcept;
};

// ---------------------------------------------------------------------------
// GPU DeviceGraph (CUDA/HIP or CPU stub via DeviceBuffer)
// ---------------------------------------------------------------------------

/// @brief GPU-resident SoA neighbor graph.
///
/// On CUDA/HIP builds, the three `DeviceBuffer` members allocate on
/// the device.  On the CPU stub (Mac dev), they allocate from the heap —
/// so all host-side code compiles and tests run without a GPU.
struct DeviceGraph {
    DeviceBuffer<knng::index_t> ids;    ///< [n][k] neighbor IDs on device.
    DeviceBuffer<float>         dists;  ///< [n][k] distances on device.
    DeviceBuffer<std::uint32_t> flags;  ///< [n][k] is-new flags on device.
    std::size_t n = 0;
    std::size_t k = 0;

    DeviceGraph() = default;

    /// @brief Allocate device buffers for `n` points and `k` neighbors.
    DeviceGraph(std::size_t n, std::size_t k);

    /// @brief Upload a `CpuDeviceGraph` to the device.
    explicit DeviceGraph(const CpuDeviceGraph& cpu);

    /// @brief Download to host as `CpuDeviceGraph`.
    [[nodiscard]] CpuDeviceGraph to_cpu() const;

    /// @brief Convenience: construct from `knng::Knng`, upload to device.
    [[nodiscard]] static DeviceGraph from_knng(const knng::Knng& g);

    /// @brief Download and convert to `knng::Knng`.
    [[nodiscard]] knng::Knng to_knng() const;
};

} // namespace knng::gpu
