/// @file
/// @brief Step 53 — CPU reference: tiled brute-force KNN.

#include <knng/gpu/tiled_brute_force.hpp>
#include <knng/top_k.hpp>

#include <algorithm>
#include <vector>

namespace knng::gpu {

knng::Knng cpu_tiled_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    const std::size_t n    = dataset.n;
    const std::size_t d    = dataset.d;
    const std::size_t tile = kTileW;

    knng::Knng graph(n, k);

    for (std::size_t qi = 0; qi < n; ++qi) {
        TopK heap(k);
        const float* q = dataset.data.data() + qi * d;

        for (std::size_t tile_base = 0; tile_base < n; tile_base += tile) {
            const std::size_t tile_end = std::min(tile_base + tile, n);

            for (std::size_t ri = tile_base; ri < tile_end; ++ri) {
                if (ri == qi) {
                    continue;
                }
                const float* r = dataset.data.data() + ri * d;
                float acc = 0.f;
                for (std::size_t dim = 0; dim < d; ++dim) {
                    const float diff = q[dim] - r[dim];
                    acc += diff * diff;
                }
                heap.push(static_cast<knng::index_t>(ri), acc);
            }
        }

        auto nbrs = heap.extract_sorted();
        for (std::size_t r = 0; r < nbrs.size(); ++r) {
            graph.neighbors[qi * k + r] = nbrs[r].first;
            graph.distances[qi * k + r] = nbrs[r].second;
        }
    }
    return graph;
}

} // namespace knng::gpu
