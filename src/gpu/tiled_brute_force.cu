/// @file
/// @brief Step 53 — Shared-memory reference-tiling GPU brute-force KNN.
///
/// One block per query; threads cooperatively load `TILE_W`-wide tiles of
/// the reference matrix into shared memory, then each thread computes its
/// query distance against all references in the tile.  The per-block top-k
/// mechanism reuses the atomicCAS spinlock from Step 49.

#include <knng/gpu/tiled_brute_force.hpp>
#include <knng/gpu/device_buffer.hpp>
#include <knng/core/types.hpp>

namespace knng::gpu {

static constexpr unsigned int kTile = static_cast<unsigned int>(kTileW);

// Spinlock helpers (same pattern as brute_force_block.cu).
__device__ GPU_INLINE static void tile_lock(int* m) {
    while (atomicCAS(m, 0, 1) != 0) {}
}
__device__ GPU_INLINE static void tile_unlock(int* m) {
    atomicExch(m, 0);
}

/// @brief Find & evict the worst entry (no sort — used inside locked section).
__device__ GPU_INLINE static void sh_evict_insert(
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
    float new_worst = sh_dists[0];
    for (unsigned int i = 1; i < k; ++i) {
        if (sh_dists[i] > new_worst) {
            new_worst = sh_dists[i];
        }
    }
    *sh_worst = new_worst;
}

/// @brief Tiled block-per-query brute-force KNN kernel.
///
/// Shared memory layout:
///   float  sh_tile  [kTile][d_max]  — tile of kTile reference vectors
///   float  sh_dists [k]             — top-k distances
///   index_t sh_ids  [k]             — top-k ids
///   float  sh_worst [1]             — current worst distance
///   int    sh_mutex [1]             — spinlock
///
/// Dynamic shmem size computed by host launcher.
GPU_GLOBAL void tiled_brute_force_kernel(
    const float*   d_pts,
    unsigned int   n,
    unsigned int   d,
    unsigned int   k,
    knng::index_t* d_ids,
    float*         d_dists)
{
    const unsigned int qi = blockIdx.x;
    if (qi >= n) { return; }

    // Shared memory layout (byte-offsets computed below).
    extern __shared__ char sh_raw[];

    float*         sh_tile  = reinterpret_cast<float*>(sh_raw);
    float*         sh_dists = sh_tile + kTile * d;
    knng::index_t* sh_ids   = reinterpret_cast<knng::index_t*>(sh_dists + k);
    float*         sh_worst = reinterpret_cast<float*>(sh_ids + k);
    int*           sh_mutex = reinterpret_cast<int*>(sh_worst + 1);

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

    for (unsigned int tile_base = 0; tile_base < n; tile_base += kTile) {
        // Cooperatively load the tile into shared memory.
        // Thread t loads reference (tile_base + t) if in-bounds.
        const unsigned int ri_load = tile_base + threadIdx.x;
        if (threadIdx.x < kTile && ri_load < n) {
            const float* r = d_pts + static_cast<std::size_t>(ri_load) * d;
            for (unsigned int dim = 0; dim < d; ++dim) {
                sh_tile[threadIdx.x * d + dim] = r[dim];
            }
        }
        __syncthreads();

        // Each thread scans all kTile references in the tile.
        const unsigned int tile_end =
            (tile_base + kTile < n) ? kTile : (n - tile_base);

        for (unsigned int t = threadIdx.x; t < tile_end; t += blockDim.x) {
            const unsigned int ri = tile_base + t;
            if (ri == qi) { continue; }

            const float* r = sh_tile + t * d;
            float acc = 0.f;
            for (unsigned int dim = 0; dim < d; ++dim) {
                const float diff = q[dim] - r[dim];
                acc += diff * diff;
            }
            if (acc < *sh_worst) {
                tile_lock(sh_mutex);
                sh_evict_insert(sh_ids, sh_dists, sh_worst, k,
                                static_cast<knng::index_t>(ri), acc);
                tile_unlock(sh_mutex);
            }
        }
        __syncthreads();
    }

    // Sort and write (thread 0).
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < k - 1; ++i) {
            unsigned int min_pos = i;
            for (unsigned int j = i + 1; j < k; ++j) {
                if (sh_dists[j] < sh_dists[min_pos]) { min_pos = j; }
            }
            if (min_pos != i) {
                float         td = sh_dists[i]; sh_dists[i] = sh_dists[min_pos]; sh_dists[min_pos] = td;
                knng::index_t ti = sh_ids  [i]; sh_ids  [i] = sh_ids  [min_pos]; sh_ids  [min_pos] = ti;
            }
        }
        const std::size_t base = static_cast<std::size_t>(qi) * k;
        for (unsigned int r = 0; r < k; ++r) {
            d_ids  [base + r] = sh_ids  [r];
            d_dists[base + r] = sh_dists[r];
        }
    }
}

knng::Knng gpu_tiled_brute_force_knn(
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
        kTile * d * sizeof(float) +          // sh_tile
        ku * sizeof(float) +                 // sh_dists
        ku * sizeof(knng::index_t) +         // sh_ids
        sizeof(float) +                      // sh_worst
        sizeof(int);                         // mutex

    GPU_LAUNCH(tiled_brute_force_kernel, n, kBlock, shmem, nullptr,
               d_pts.data(), n, d, ku, d_ids.data(), d_dists.data());
    GPU_SYNC();

    knng::Knng graph(n, ku);
    d_ids.copy_to_host(graph.neighbors);
    d_dists.copy_to_host(graph.distances);
    return graph;
}

} // namespace knng::gpu
