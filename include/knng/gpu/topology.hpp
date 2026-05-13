#pragma once

/// @file
/// @brief Step 81 — Device discovery + P2P topology probe.
///
/// `knng::gpu::Topology` stores the inter-device link type for every (i,j)
/// pair on a single node.  Always compiled so tests on Mac can create a
/// simulated topology without CUDA.

#include <cstddef>
#include <string>
#include <vector>

namespace knng::gpu {

/// @brief Classification of the communication path between two GPU devices.
enum class PeerLinkType : unsigned char {
    Same      = 0,  ///< i == j
    NVLink    = 1,  ///< Direct NVLink (fastest)
    PCIe      = 2,  ///< PCIe peer-to-peer (cudaDeviceCanAccessPeer)
    HostBridge = 3, ///< Same node, no P2P — route through host memory
    Simulated = 4,  ///< CPU simulation (no real GPU hardware)
};

/// @brief Returns a human-readable name for a `PeerLinkType`.
const char* peer_link_type_name(PeerLinkType t) noexcept;

/// @brief Node-local GPU topology: n_gpus × n_gpus matrix of `PeerLinkType`.
///
/// Constructed by `Topology::probe()` (GPU path) or `Topology::simulate()`
/// (CPU path).  Immutable after construction.
class Topology {
public:
    /// @brief Build a simulated topology for n_gpus virtual "GPUs".
    ///
    /// All off-diagonal pairs get `PeerLinkType::Simulated`.
    static Topology simulate(int n_gpus);

#if defined(KNNG_HAVE_CUDA)
    /// @brief Probe actual CUDA device topology via `cudaDeviceCanAccessPeer`.
    ///
    /// NVLink detection requires querying `cudaDevAttrGpuOverlapCount` or
    /// NVML; when NVML is unavailable this conservatively reports PCIe for
    /// all P2P-capable pairs.
    static Topology probe();
#endif

    int           n_gpus()                        const noexcept { return n_; }
    PeerLinkType  link(int i, int j)              const noexcept;
    bool          has_p2p(int i, int j)           const noexcept;

    /// @brief Format a human-readable topology table.
    std::string to_string() const;

private:
    explicit Topology(int n_gpus);
    int                       n_ = 0;
    std::vector<PeerLinkType> matrix_; ///< Row-major n_ × n_
};

} // namespace knng::gpu
