/// @file topology_gpu.cu
/// @brief Topology::probe() — query real CUDA device P2P capabilities.
///
/// ## How peer-to-peer topology detection works
///
/// CUDA exposes three P2P attributes via `cudaDeviceGetP2PAttribute`:
///
///   cudaP2PAttrAccessSupported
///     1 if the two GPUs can do peer memory access at all.
///     Same as `cudaDeviceCanAccessPeer()`.
///
///   cudaP2PAttrNativeAtomicSupported
///     1 if NVLink is the interconnect.  NVLink supports native
///     GPU-to-GPU atomics; PCIe does not (on pre-Hopper hardware).
///     This is the most reliable NVLink indicator without NVML.
///
///   cudaP2PAttrPerformanceRank
///     Lower = faster.  0 = best performance (NVLink on DGX).
///     Used as a fallback on platforms where native atomics return 0
///     even for NVLink (e.g. some configurations of Grace Hopper).
///
/// Decision tree for (i, j):
///   cudaDeviceCanAccessPeer == 0  → HostBridge  (all traffic via host RAM)
///   NativeAtomicSupported   == 1  → NVLink      (fastest, native atomics)
///   PerformanceRank         == 0  → NVLink      (fallback for GH200)
///   else                          → PCIe        (P2P but slower)
///
/// ## Why this matters for scheduling
/// The Topology matrix drives the AllGather strategy:
///   NVLink pairs     → cudaMemcpyPeerAsync (device-to-device, full BW)
///   PCIe pairs       → stage through pinned host memory (two async copies)
///   HostBridge pairs → cudaMemcpy via host (serialize on default stream)

#include <knng/gpu/topology.hpp>
#include <knng/gpu/backend.hpp>

#if defined(KNNG_HAVE_CUDA)

#include <cuda_runtime.h>
#include <stdexcept>

namespace knng::gpu {

Topology Topology::probe() {
    int n = 0;
    GPU_CHECK(cudaGetDeviceCount(&n));
    if (n == 0)
        throw std::runtime_error("Topology::probe: no CUDA devices found");

    Topology t(n);

    for (int i = 0; i < n; ++i) {
        // Diagonal: device can always "access" itself.
        t.matrix_[static_cast<std::size_t>(i * n + i)] = PeerLinkType::Same;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            // ----------------------------------------------------------------
            // Step 1: Can these two GPUs access each other's memory at all?
            // Without P2P the only route is host RAM (cudaMemcpy bounces
            // through the CPU — expensive and high latency).
            // ----------------------------------------------------------------
            int can_access = 0;
            GPU_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));

            if (!can_access) {
                t.matrix_[static_cast<std::size_t>(i * n + j)] =
                    PeerLinkType::HostBridge;
                continue;
            }

            // ----------------------------------------------------------------
            // Step 2: Check for NVLink via native-atomic support.
            // NVLink supports GPU-to-GPU atomic operations (e.g. atomicAdd
            // to the peer's memory); PCIe P2P emulates atomics through the
            // host and does not set this attribute.
            // ----------------------------------------------------------------
            int native_atomics = 0;
            GPU_CHECK(cudaDeviceGetP2PAttribute(
                &native_atomics,
                cudaP2PAttrNativeAtomicSupported,
                i, j));

            if (native_atomics) {
                t.matrix_[static_cast<std::size_t>(i * n + j)] =
                    PeerLinkType::NVLink;
                continue;
            }

            // ----------------------------------------------------------------
            // Step 3: Fallback NVLink detection for Grace-Hopper and other
            // platforms where native-atomic query is unreliable.
            // PerformanceRank == 0 means "best available link" — on NVLink
            // systems this indicates the NVLink fabric; on PCIe-only systems
            // all pairs report the same non-zero rank.
            // ----------------------------------------------------------------
            int perf_rank = -1;
            GPU_CHECK(cudaDeviceGetP2PAttribute(
                &perf_rank,
                cudaP2PAttrPerformanceRank,
                i, j));

            t.matrix_[static_cast<std::size_t>(i * n + j)] =
                (perf_rank == 0) ? PeerLinkType::NVLink : PeerLinkType::PCIe;
        }
    }

    return t;
}

} // namespace knng::gpu

#endif // KNNG_HAVE_CUDA
