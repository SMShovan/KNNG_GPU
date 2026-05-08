/// @file
/// @brief Step 49 — Block-per-query GPU brute-force KNN kernel.
///
/// One thread block per query.  All threads in the block cooperate to scan
/// the reference set, accumulating a shared-memory top-k buffer.
///
/// ### Algorithm
/// 1. Each thread loads a `BLOCK_SIZE`-strided slice of the reference set and
///    computes distances to the assigned query.
/// 2. Partial results (id, dist) are inserted into a per-block shared-memory
///    top-k list that all threads see.
/// 3. After the reference scan, the shared top-k is sorted and written to
///    global memory.
///
/// ### Shared memory layout
/// ```
///   float         sh_dists[k]      // top-k distances
///   index_t       sh_ids  [k]      // top-k ids (parallel to sh_dists)
///   float         sh_worst[1]      // current worst (max) distance in top-k
/// ```
/// Total: `k * (sizeof(float) + sizeof(index_t)) + sizeof(float)` bytes.
///
/// ### Synchronisation
/// Insertions into the shared top-k are serialised by an `atomicCAS`-based
/// mutex (one 32-bit lock per block, stored in shared memory).  For k ≤ 128
/// (typical ANN workload) the contention is low because only the threads
/// whose distance beats the current worst attempt the locked insert.
///
/// This is the layout that scales to arbitrary k and large n.  Phase 8 will
/// replace the mutex with a warp-level reduction.

#include <knng/gpu/brute_force.hpp>
#include <knng/gpu/device_buffer.hpp>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// @brief Attempt to acquire a block-scope spinlock stored in shared memory.
__device__ GPU_INLINE void lock_acquire(int* mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0) {
        /* spin */
    }
}

/// @brief Release the block-scope spinlock.
__device__ GPU_INLINE void lock_release(int* mutex)
{
    atomicExch(mutex, 0);
}

/// @brief Insert (id, dist) into the shared-memory top-k list.
///
/// Assumes the caller holds the block mutex.
__device__ GPU_INLINE void sh_insert(
    knng::index_t* sh_ids,
    float*         sh_dists,
    float*         sh_worst,
    unsigned int   k,
    knng::index_t  id,
    float          dist)
{
    if (dist >= *sh_worst) {
        return;
    }
    // Find the worst element and overwrite it.
    unsigned int worst_pos = 0;
    float        worst_d   = sh_dists[0];
    for (unsigned int i = 1; i < k; ++i) {
        if (sh_dists[i] > worst_d) {
            worst_d   = sh_dists[i];
            worst_pos = i;
        }
    }
    sh_ids  [worst_pos] = id;
    sh_dists[worst_pos] = dist;

    // Recompute worst.
    float new_worst = sh_dists[0];
    for (unsigned int i = 1; i < k; ++i) {
        if (sh_dists[i] > new_worst) {
            new_worst = sh_dists[i];
        }
    }
    *sh_worst = new_worst;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// @brief Block-per-query brute-force KNN.
///
/// Grid: 1-D, one block per query point.
/// Block: 1-D, `blockDim.x` threads.
/// Shared memory: `k*(sizeof(float)+sizeof(index_t)) + sizeof(float) + sizeof(int)`.
///
/// @param d_pts    Device pointer, row-major `float[n][d]`.
/// @param n        Number of points.
/// @param d        Feature dimensionality.
/// @param k        Neighbours per query.
/// @param d_ids    Device output, row-major `index_t[n][k]`.
/// @param d_dists  Device output, row-major `float[n][k]`.
GPU_GLOBAL void brute_force_block_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    float*         d_dists)
{
    const unsigned int qi = blockIdx.x;   // query index = block index
    if (qi >= n) {
        return;
    }

    // Shared memory layout (dynamic allocation from host launcher):
    //   float         sh_dists[k]
    //   index_t       sh_ids  [k]
    //   float         sh_worst[1]
    //   int           mutex   [1]
    extern __shared__ char sh_raw[];
    float*         sh_dists = reinterpret_cast<float*>(sh_raw);
    knng::index_t* sh_ids   = reinterpret_cast<knng::index_t*>(sh_raw + k * sizeof(float));
    float*         sh_worst = reinterpret_cast<float*>(sh_raw + k * (sizeof(float) + sizeof(knng::index_t)));
    int*           sh_mutex = reinterpret_cast<int*>(  sh_raw + k * (sizeof(float) + sizeof(knng::index_t)) + sizeof(float));

    // Initialise shared memory (thread 0 only, then syncthreads).
    if (threadIdx.x == 0) {
        for (unsigned int r = 0; r < k; ++r) {
            sh_dists[r] = 1e38f;
            sh_ids  [r] = static_cast<knng::index_t>(-1);
        }
        *sh_worst = 1e38f;
        *sh_mutex = 0;
    }
    __syncthreads();

    const float* q = d_pts + static_cast<std::size_t>(qi) * d;

    // Each thread processes references strided by blockDim.x.
    for (unsigned int ri = threadIdx.x; ri < n; ri += blockDim.x) {
        if (ri == qi) {
            continue;
        }
        const float* r = d_pts + static_cast<std::size_t>(ri) * d;
        float acc = 0.f;
        for (unsigned int dim = 0; dim < d; ++dim) {
            const float diff = q[dim] - r[dim];
            acc += diff * diff;
        }
        if (acc < *sh_worst) {  // cheap pre-check before locking
            lock_acquire(sh_mutex);
            sh_insert(sh_ids, sh_dists, sh_worst, k,
                      static_cast<knng::index_t>(ri), acc);
            lock_release(sh_mutex);
        }
    }
    __syncthreads();

    // Sort the shared top-k list ascending by distance (selection sort;
    // k is small so O(k²) is fine here).
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < k - 1; ++i) {
            unsigned int min_pos = i;
            for (unsigned int j = i + 1; j < k; ++j) {
                if (sh_dists[j] < sh_dists[min_pos]) {
                    min_pos = j;
                }
            }
            if (min_pos != i) {
                float         td = sh_dists[i]; sh_dists[i] = sh_dists[min_pos]; sh_dists[min_pos] = td;
                knng::index_t ti = sh_ids  [i]; sh_ids  [i] = sh_ids  [min_pos]; sh_ids  [min_pos] = ti;
            }
        }
        // Write results.
        const std::size_t base = static_cast<std::size_t>(qi) * k;
        for (unsigned int r = 0; r < k; ++r) {
            d_ids  [base + r] = sh_ids  [r];
            d_dists[base + r] = sh_dists[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

knng::Knng brute_force_block_knn(
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

    constexpr unsigned int kBlock = 128u;
    const std::size_t shmem =
        ku * (sizeof(float) + sizeof(knng::index_t)) +   // sh_dists + sh_ids
        sizeof(float) +                                   // sh_worst
        sizeof(int);                                      // mutex

    GPU_LAUNCH(brute_force_block_kernel, n, kBlock, shmem, nullptr,
               d_pts.data(), n, d, ku,
               d_ids.data(), d_dists.data());
    GPU_SYNC();

    knng::Knng graph(n, ku);
    d_ids.copy_to_host(graph.neighbors);
    d_dists.copy_to_host(graph.distances);
    return graph;
}

// ---------------------------------------------------------------------------
// Step 50 — auto-strategy launcher (selects block-per-query for all k)
// ---------------------------------------------------------------------------

knng::Knng gpu_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    return brute_force_block_knn(dataset, k);
}

} // namespace knng::gpu
