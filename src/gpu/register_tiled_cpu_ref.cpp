/// @file
/// @brief Step 55 — CPU reference: register-tiled brute-force KNN.

#include <knng/gpu/register_tiled_bf.hpp>
#include <knng/top_k.hpp>

#include <algorithm>
#include <vector>

namespace knng::gpu {

knng::Knng cpu_register_tiled_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    const std::size_t n  = dataset.n;
    const std::size_t d  = dataset.d;
    const std::size_t Q  = kRegTileQ;

    knng::Knng graph(n, k);

    // Process queries in groups of Q.
    for (std::size_t qi_base = 0; qi_base < n; qi_base += Q) {
        const std::size_t qi_end = std::min(qi_base + Q, n);
        const std::size_t q_count = qi_end - qi_base;

        // One TopK per query in the group.
        std::vector<TopK> heaps;
        heaps.reserve(Q);
        for (std::size_t q = 0; q < q_count; ++q) {
            heaps.emplace_back(k);
        }

        // For each reference, compute distances to all Q queries at once.
        for (std::size_t ri = 0; ri < n; ++ri) {
            const float* r = dataset.data.data() + ri * d;
            for (std::size_t q = 0; q < q_count; ++q) {
                const std::size_t qi = qi_base + q;
                if (ri == qi) { continue; }
                const float* qptr = dataset.data.data() + qi * d;
                float acc = 0.f;
                for (std::size_t dim = 0; dim < d; ++dim) {
                    const float diff = qptr[dim] - r[dim];
                    acc += diff * diff;
                }
                heaps[q].push(static_cast<knng::index_t>(ri), acc);
            }
        }

        // Write results.
        for (std::size_t q = 0; q < q_count; ++q) {
            const std::size_t qi = qi_base + q;
            auto nbrs = heaps[q].extract_sorted();
            for (std::size_t r = 0; r < nbrs.size(); ++r) {
                graph.neighbors[qi * k + r] = nbrs[r].first;
                graph.distances[qi * k + r] = nbrs[r].second;
            }
        }
    }
    return graph;
}

} // namespace knng::gpu
