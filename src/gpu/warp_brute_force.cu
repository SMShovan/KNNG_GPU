/// @file
/// @brief Step 54 — Warp-level top-k GPU brute-force KNN.
///
/// One warp (32 threads) per query.  Each lane scans every 32nd reference,
/// maintains a register-resident partial top-k, then shuffle-merges with
/// the other lanes to produce the final top-k.

#include <knng/gpu/warp_top_k.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/core/types.hpp>

namespace knng::gpu {

static constexpr unsigned int kWarp = 32u;
static constexpr unsigned int kMaxK = 32u;   // register budget limit

// ---------------------------------------------------------------------------
// Warp-level reduction helpers
// ---------------------------------------------------------------------------

/// @brief Warp-wide maximum distance using shuffle reduction.
__device__ GPU_INLINE float warp_max(float v)
{
    for (unsigned int offset = kWarp / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        v = (other > v) ? other : v;
    }
    return v;
}

/// @brief Insert (id, dist) into a register-resident sorted partial list.
///
/// The list occupies arrays `ids[0..k-1]` and `dists[0..k-1]` in the
/// calling thread's register file, sorted ascending by distance.
/// `worst` is `dists[k-1]` (maintained by the caller for the fast reject).
__device__ GPU_INLINE void reg_insert(
    knng::index_t* ids,
    float*         dists,
    unsigned int   k,
    knng::index_t  id,
    float          dist,
    float&         worst)
{
    if (dist >= worst) { return; }
    unsigned int pos = k - 1u;
    while (pos > 0 && dists[pos - 1u] > dist) {
        ids  [pos] = ids  [pos - 1u];
        dists[pos] = dists[pos - 1u];
        --pos;
    }
    ids  [pos] = id;
    dists[pos] = dist;
    worst = dists[k - 1u];
}

// ---------------------------------------------------------------------------
// Kernel — one warp per query
// ---------------------------------------------------------------------------

GPU_GLOBAL void warp_brute_force_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    float*         d_dists)
{
    // Warp index = query index; lane = position within warp.
    const unsigned int qi   = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    const unsigned int lane = threadIdx.x % kWarp;

    if (qi >= n) { return; }

    // Register-resident partial top-k.
    knng::index_t top_ids  [kMaxK];
    float         top_dists[kMaxK];
    for (unsigned int r = 0; r < k; ++r) {
        top_ids  [r] = static_cast<knng::index_t>(-1);
        top_dists[r] = 1e38f;
    }
    float worst = 1e38f;

    const float* q = d_pts + static_cast<std::size_t>(qi) * d;

    // Each lane processes every kWarp-th reference starting at `lane`.
    for (unsigned int ri = lane; ri < n; ri += kWarp) {
        if (ri == qi) { continue; }
        const float* r = d_pts + static_cast<std::size_t>(ri) * d;
        float acc = 0.f;
        for (unsigned int dim = 0; dim < d; ++dim) {
            const float diff = q[dim] - r[dim];
            acc += diff * diff;
        }
        reg_insert(top_ids, top_dists, k, static_cast<knng::index_t>(ri), acc, worst);
    }

    // Merge partial top-k lists across lanes: lane 0 collects all candidates.
    // Strategy: shuffle each lane's partial top-k to lane 0, which inserts
    // candidates one by one.
    for (unsigned int src_lane = 1; src_lane < kWarp; ++src_lane) {
        for (unsigned int r = 0; r < k; ++r) {
            const float          d_val = __shfl_sync(0xFFFFFFFFu, top_dists[r], src_lane);
            const knng::index_t  i_val = static_cast<knng::index_t>(
                __shfl_sync(0xFFFFFFFFu,
                            static_cast<int>(top_ids[r]), src_lane));
            if (lane == 0 && d_val < 1e37f) {
                reg_insert(top_ids, top_dists, k, i_val, d_val, worst);
            }
        }
    }

    // Lane 0 writes the final sorted result.
    if (lane == 0) {
        const std::size_t base = static_cast<std::size_t>(qi) * k;
        for (unsigned int r = 0; r < k; ++r) {
            d_ids  [base + r] = top_ids  [r];
            d_dists[base + r] = top_dists[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

knng::Knng gpu_warp_top_k_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    const auto n  = static_cast<unsigned int>(dataset.n);
    const auto d  = static_cast<unsigned int>(dataset.d);
    const auto ku = static_cast<unsigned int>(k);

    DeviceBuffer<float>         d_pts(dataset.n * dataset.d);
    DeviceBuffer<knng::index_t> d_ids(dataset.n * k);
    DeviceBuffer<float>         d_dists(dataset.n * k);

    d_pts.copy_from_host(dataset.data.data(), dataset.n * dataset.d);

    // warps_per_block * kWarp threads per block; 4 warps per block is typical.
    constexpr unsigned int kWarpsPerBlock = 4u;
    constexpr unsigned int kBlock = kWarpsPerBlock * kWarp;
    const unsigned int grid = (n + kWarpsPerBlock - 1u) / kWarpsPerBlock;

    GPU_LAUNCH(warp_brute_force_kernel, grid, kBlock, 0, nullptr,
               d_pts.data(), n, d, ku, d_ids.data(), d_dists.data());
    GPU_SYNC();

    knng::Knng graph(n, ku);
    d_ids.copy_to_host(graph.neighbors);
    d_dists.copy_to_host(graph.distances);
    return graph;
}

} // namespace knng::gpu
