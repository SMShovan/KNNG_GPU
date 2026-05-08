/// @file
/// @brief Step 48 — Thread-per-query GPU brute-force KNN kernel.
///
/// Each CUDA thread owns one query point.  It iterates over all `n`
/// references, maintains a register-resident top-k list using insertion
/// sort, and writes the sorted result to global memory.
///
/// Register budget: the top-k list occupies 2×k float registers.  For k=32
/// that is 64 registers, well within the 255-register limit per thread on
/// Volta+.  Above k=32 register spill starts to dominate; use
/// `brute_force_block_knn` for larger k.

#include <knng/gpu/brute_force.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/core/types.hpp>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// @brief Maximum k supported by the thread-per-query kernel (register limit).
static constexpr unsigned int kMaxKThread = 32u;

/// @brief Insert (id, dist) into an insertion-sorted top-k list in registers.
///
/// The list is stored as parallel arrays of length `k`.  On entry the list
/// holds the current k best candidates (worst last).  If `dist` is better
/// than `worst_dist` the worst element is evicted and `(id, dist)` is
/// inserted in sorted position.
///
/// @param ids        Register array of k neighbor IDs.
/// @param dists      Register array of k distances (ascending).
/// @param k          Current list capacity.
/// @param id         Candidate neighbor ID.
/// @param dist       Candidate distance.
/// @param worst_dist In/out: current maximum distance in the list.
__device__ GPU_INLINE void insert_top_k(
    knng::index_t* ids,
    float*         dists,
    unsigned int   k,
    knng::index_t  id,
    float          dist,
    float&         worst_dist)
{
    if (dist >= worst_dist) {
        return;
    }
    // Find insertion position (linear scan; k is small).
    unsigned int pos = k - 1u;
    while (pos > 0 && dists[pos - 1u] > dist) {
        ids  [pos] = ids  [pos - 1u];
        dists[pos] = dists[pos - 1u];
        --pos;
    }
    ids  [pos] = id;
    dists[pos] = dist;
    worst_dist = dists[k - 1u];
}

/// @brief Thread-per-query brute-force KNN kernel.
///
/// Grid: 1-D, one thread per query point.
/// Register usage: 2 × k floats/ints for the top-k list.
///
/// @param d_pts      Device pointer, row-major `float[n][d]`.
/// @param n          Number of points.
/// @param d          Feature dimensionality.
/// @param k          Neighbours per query (must be ≤ kMaxKThread = 32).
/// @param d_ids      Device output, row-major `index_t[n][k]`.
/// @param d_dists    Device output, row-major `float[n][k]`.
GPU_GLOBAL void brute_force_thread_kernel(
    const float*        d_pts,
    unsigned int        n,
    unsigned int        d,
    unsigned int        k,
    knng::index_t*      d_ids,
    float*              d_dists)
{
    const unsigned int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n) {
        return;
    }

    // Register-resident top-k arrays (fixed size at compile time).
    knng::index_t top_ids  [kMaxKThread];
    float         top_dists[kMaxKThread];

    // Initialise with sentinels.
    for (unsigned int r = 0; r < k; ++r) {
        top_ids  [r] = static_cast<knng::index_t>(-1);
        top_dists[r] = 1e38f;
    }
    float worst = 1e38f;

    const float* q = d_pts + static_cast<std::size_t>(qi) * d;

    for (unsigned int ri = 0; ri < n; ++ri) {
        if (ri == qi) {
            continue;
        }
        const float* r = d_pts + static_cast<std::size_t>(ri) * d;
        float acc = 0.f;
        for (unsigned int dim = 0; dim < d; ++dim) {
            const float diff = q[dim] - r[dim];
            acc += diff * diff;
        }
        insert_top_k(top_ids, top_dists, k, static_cast<knng::index_t>(ri), acc, worst);
    }

    // Write results.
    const std::size_t base = static_cast<std::size_t>(qi) * k;
    for (unsigned int r = 0; r < k; ++r) {
        d_ids  [base + r] = top_ids  [r];
        d_dists[base + r] = top_dists[r];
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

knng::Knng brute_force_thread_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    const auto n = static_cast<unsigned int>(dataset.n);
    const auto d = static_cast<unsigned int>(dataset.d);
    const auto ku = static_cast<unsigned int>(k);

    DeviceBuffer<float>        d_pts(dataset.n * dataset.d);
    DeviceBuffer<knng::index_t> d_ids(dataset.n * k);
    DeviceBuffer<float>        d_dists(dataset.n * k);

    d_pts.copy_from_host(dataset.data.data(), dataset.n * dataset.d);

    constexpr unsigned int kBlock = 128u;
    const unsigned int grid = (n + kBlock - 1u) / kBlock;
    GPU_LAUNCH(brute_force_thread_kernel, grid, kBlock, 0, nullptr,
               d_pts.data(), n, d, ku,
               d_ids.data(), d_dists.data());
    GPU_SYNC();

    knng::Knng graph(n, ku);
    d_ids.copy_to_host(graph.neighbors);
    d_dists.copy_to_host(graph.distances);
    return graph;
}

} // namespace knng::gpu
