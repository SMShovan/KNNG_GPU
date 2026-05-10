/// @file
/// @brief Step 54 — CPU reference: warp-level top-k KNN.
///
/// Simulates the "32-lane partial top-k + merge" strategy in plain C++.
/// Each "lane" processes every 32nd reference starting from its lane ID.
/// After all lanes finish, the partial top-k lists are merged into one.

#include <knng/gpu/warp_top_k.hpp>
#include <knng/top_k.hpp>

#include <vector>

namespace knng::gpu {

static constexpr std::size_t kWarpSize = 32;

knng::Knng cpu_warp_top_k_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    const std::size_t n = dataset.n;
    const std::size_t d = dataset.d;
    knng::Knng graph(n, k);

    for (std::size_t qi = 0; qi < n; ++qi) {
        const float* q = dataset.data.data() + qi * d;

        // Each lane builds a partial top-k over its strided slice.
        std::vector<TopK> lanes;
        lanes.reserve(kWarpSize);
        for (std::size_t lane = 0; lane < kWarpSize; ++lane) {
            lanes.emplace_back(k);
        }

        for (std::size_t ri = 0; ri < n; ++ri) {
            if (ri == qi) { continue; }
            const float* r = dataset.data.data() + ri * d;
            float acc = 0.f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                const float diff = q[dim] - r[dim];
                acc += diff * diff;
            }
            lanes[ri % kWarpSize].push(static_cast<knng::index_t>(ri), acc);
        }

        // Merge all lanes into a single top-k.
        TopK merged(k);
        for (std::size_t lane = 0; lane < kWarpSize; ++lane) {
            for (auto& [id, dist] : lanes[lane].extract_sorted()) {
                merged.push(id, dist);
            }
        }

        auto nbrs = merged.extract_sorted();
        for (std::size_t r = 0; r < nbrs.size(); ++r) {
            graph.neighbors[qi * k + r] = nbrs[r].first;
            graph.distances[qi * k + r] = nbrs[r].second;
        }
    }
    return graph;
}

} // namespace knng::gpu
