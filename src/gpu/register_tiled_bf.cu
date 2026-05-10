/// @file
/// @brief Step 55 — Register-tiled GPU brute-force KNN.
///
/// Each thread processes Q=kRegTileQ=2 queries: for every reference, both
/// distances are computed before the reference registers are retired.  This
/// halves the number of global reference loads relative to Q=1.

#include <knng/gpu/register_tiled_bf.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/core/types.hpp>

namespace knng::gpu {

static constexpr unsigned int kQ = static_cast<unsigned int>(kRegTileQ);  // 2
static constexpr unsigned int kMaxK = 32u;

// Insertion-sort helper (same as Step 48).
__device__ GPU_INLINE static void reg_insert2(
    knng::index_t* ids, float* dists, unsigned int k,
    knng::index_t id, float dist, float& worst)
{
    if (dist >= worst) { return; }
    unsigned int pos = k - 1u;
    while (pos > 0 && dists[pos-1u] > dist) {
        ids  [pos] = ids  [pos-1u];
        dists[pos] = dists[pos-1u];
        --pos;
    }
    ids[pos] = id; dists[pos] = dist;
    worst = dists[k-1u];
}

/// @brief Register-tiled kernel: kQ queries per thread.
///
/// Thread index encodes (query_group, sub_query) via:
///   global_query = blockIdx.x * (blockDim.x / kQ) * kQ + threadIdx.x
/// but for simplicity, thread `t` owns query indices
///   qi0 = t * kQ, qi1 = t * kQ + 1
GPU_GLOBAL void register_tiled_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    float*         d_dists)
{
    const unsigned int ti  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int qi0 = ti * kQ;
    const unsigned int qi1 = qi0 + 1u;

    if (qi0 >= n) { return; }

    knng::index_t ids0[kMaxK], ids1[kMaxK];
    float         dst0[kMaxK], dst1[kMaxK];
    for (unsigned int r = 0; r < k; ++r) {
        ids0[r] = ids1[r] = static_cast<knng::index_t>(-1);
        dst0[r] = dst1[r] = 1e38f;
    }
    float worst0 = 1e38f, worst1 = 1e38f;

    const float* q0 = d_pts + static_cast<std::size_t>(qi0) * d;
    const float* q1 = (qi1 < n) ? (d_pts + static_cast<std::size_t>(qi1) * d) : nullptr;

    for (unsigned int ri = 0; ri < n; ++ri) {
        const float* r = d_pts + static_cast<std::size_t>(ri) * d;
        // Compute distance to q0.
        if (ri != qi0) {
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = q0[dim] - r[dim];
                acc += diff * diff;
            }
            reg_insert2(ids0, dst0, k, static_cast<knng::index_t>(ri), acc, worst0);
        }
        // Compute distance to q1 (if valid).
        if (q1 && ri != qi1) {
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = q1[dim] - r[dim];
                acc += diff * diff;
            }
            reg_insert2(ids1, dst1, k, static_cast<knng::index_t>(ri), acc, worst1);
        }
    }

    // Write q0.
    {
        const std::size_t base = static_cast<std::size_t>(qi0) * k;
        for (unsigned int r = 0; r < k; ++r) {
            d_ids  [base + r] = ids0[r];
            d_dists[base + r] = dst0[r];
        }
    }
    // Write q1.
    if (qi1 < n) {
        const std::size_t base = static_cast<std::size_t>(qi1) * k;
        for (unsigned int r = 0; r < k; ++r) {
            d_ids  [base + r] = ids1[r];
            d_dists[base + r] = dst1[r];
        }
    }
}

knng::Knng gpu_register_tiled_knn(
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

    constexpr unsigned int kBlock = 64u;   // 64 threads, each handling 2 queries
    const unsigned int queries_per_block = kBlock * kQ;
    const unsigned int grid = (n + queries_per_block - 1u) / queries_per_block;

    GPU_LAUNCH(register_tiled_kernel, grid, kBlock, 0, nullptr,
               d_pts.data(), n, d, ku, d_ids.data(), d_dists.data());
    GPU_SYNC();

    knng::Knng graph(n, ku);
    d_ids.copy_to_host(graph.neighbors);
    d_dists.copy_to_host(graph.distances);
    return graph;
}

} // namespace knng::gpu
