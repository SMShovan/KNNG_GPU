#pragma once

/// @file
/// @brief Steps 63–70 — GPU NN-Descent public API.
///
/// This header declares all GPU NN-Descent components built up across
/// Steps 63–70.  Each function has a CPU-reference counterpart that runs
/// on `CpuDeviceGraph` and is always compiled; the GPU kernels are gated
/// on `KNNG_HAVE_CUDA`.
///
/// ### GPU NN-Descent algorithm (high level)
/// 1. **Init** (Step 63): random k-NN graph using per-thread XorShift.
/// 2. **Local-join** (Steps 64–65): for each point p, enumerate (u,v) pairs
///    where at least one is "new"; compute dist(u,v); atomically try-insert
///    into u's and v's lists.
/// 3. **Reverse graph** (Step 67): build CSR reverse-neighbor lists so every
///    point's local-join considers both forward and reverse neighbors.
/// 4. **Sampling** (Step 68): per-iteration downsampling to `rho*k` candidates.
/// 5. **Convergence** (Step 69): `DeviceReduce::Sum` of per-point update counts.
/// 6. **Driver** (Step 69): host-side iteration loop until convergence.

#include <knng/gpu/device_graph.hpp>
#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <cstddef>
#include <cstdint>

namespace knng::gpu {

// ===========================================================================
// Step 63 — Random graph initialization
// ===========================================================================

/// @brief CPU reference: random k-NN graph in SoA layout.
///
/// Each point gets k distinct non-self random neighbors.  Distances are
/// computed exactly (squared L2).  Uses XorShift64 seeded per-point.
///
/// @param dataset  Input point set.
/// @param k        Neighbours per query.
/// @param seed     Global RNG seed (non-zero).
/// @return         CpuDeviceGraph with all entries marked is_new=1.
[[nodiscard]] CpuDeviceGraph cpu_init_random_graph(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::uint64_t        seed = 42);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU random k-NN graph initialization (Step 62).
[[nodiscard]] DeviceGraph gpu_init_random_graph(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::uint64_t        seed = 42);
#endif

// ===========================================================================
// Step 64 — Naive local-join kernel (one block per point)
// ===========================================================================

/// @brief CPU reference: one iteration of the NN-Descent local-join.
///
/// For each point p, enumerates all (u,v) pairs from p's neighbor list
/// where at least one entry is is_new.  Computes dist(u,v) and tries to
/// insert into both u's and v's rows.
///
/// @param dataset  Input point set.
/// @param graph    In/out SoA graph (mutated in place).
/// @return         Number of insertions that changed a row.
std::size_t cpu_local_join(
    const knng::Dataset& dataset,
    CpuDeviceGraph&      graph);

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU local-join (Steps 63–65): selects batch or naive variant.
///
/// Batch kernel (Step 64): 4 points per block (one warp per point) when k≤32.
/// Naive kernel (Step 63): 1 block per point for larger k.
/// Both use per-point spinlocks (Step 65) and CUB DeviceReduce (Step 68).
///
/// @return  Number of insertions that changed a row.
std::size_t gpu_local_join(
    const knng::Dataset& dataset,
    DeviceGraph&         graph);

/// @brief GPU reverse-neighbor graph via CUB DeviceScan::ExclusiveSum (Step 66).
void gpu_build_reverse_graph(
    const DeviceGraph&           fwd,
    DeviceBuffer<knng::index_t>& rev_ids,
    DeviceBuffer<unsigned int>&  rev_offsets);

/// @brief GPU reservoir sampling: mark ceil(rho*k) new entries per point (Step 67).
void gpu_sample_new_neighbors(
    DeviceGraph&  graph,
    double        rho,
    std::uint64_t iter_seed);

/// @brief GPU NN-Descent with fp16 distance storage (Step 69).
[[nodiscard]] knng::Knng gpu_nn_descent_fp16(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg = {});
#endif

// ===========================================================================
// Step 67 — Reverse-neighbor accumulation (CSR)
// ===========================================================================

/// @brief CPU reference: build the reverse-neighbor graph in CSR format.
///
/// @param graph        Forward SoA graph (read-only).
/// @param rev_ids      Output: flat CSR neighbor IDs [sum of in-degrees].
/// @param rev_offsets  Output: CSR row offsets [n+1].
void cpu_build_reverse_graph(
    const CpuDeviceGraph&          graph,
    std::vector<knng::index_t>&    rev_ids,
    std::vector<std::uint32_t>&    rev_offsets);

// ===========================================================================
// Step 68 — Sampling
// ===========================================================================

/// @brief CPU reference: reservoir-sample each row to ceil(rho * k) entries.
///
/// Resets is_new on unsampled entries (they remain eligible next iteration).
///
/// @param graph      In/out graph.
/// @param rho        Sampling rate in (0, 1].
/// @param iter_seed  Per-iteration RNG seed.
void cpu_sample_new_neighbors(
    CpuDeviceGraph& graph,
    double          rho,
    std::uint64_t   iter_seed);

// ===========================================================================
// Step 69 — Convergence reduction + full driver
// ===========================================================================

/// @brief Tuning knobs for the GPU NN-Descent driver.
struct GpuNNDConfig {
    std::size_t   max_iters  = 50;
    double        delta      = 0.001;
    double        rho        = 1.0;
    std::uint64_t seed       = 42;
    bool          use_reverse= true;
};

/// @brief CPU reference: full NN-Descent driver on CpuDeviceGraph.
[[nodiscard]] knng::Knng cpu_gpu_nn_descent(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg = {});

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
/// @brief GPU NN-Descent end-to-end driver.
[[nodiscard]] knng::Knng gpu_nn_descent(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg = {});
#endif

// ===========================================================================
// Step 70 — Mixed-precision NN-Descent (fp16 distances)
// ===========================================================================

/// @brief CPU reference: NN-Descent with fp16 distance storage.
///
/// Distances are stored as fp16 bits (uint16_t) in the graph and
/// converted to fp32 only for comparison / insertion.
[[nodiscard]] knng::Knng cpu_fp16_nn_descent(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg = {});

} // namespace knng::gpu
