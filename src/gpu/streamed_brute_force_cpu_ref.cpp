/// @file
/// @brief Step 59 — CPU reference: chunked brute-force KNN (sequential).

#include <knng/gpu/streamed_brute_force.hpp>
#include <knng/top_k.hpp>

#include <algorithm>
#include <vector>

namespace knng::gpu {

knng::Knng cpu_streamed_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::size_t          chunk_sz)
{
    const std::size_t n = dataset.n;
    const std::size_t d = dataset.d;
    knng::Knng graph(n, k);

    // Process queries in chunks, references always as the full set.
    for (std::size_t qi_base = 0; qi_base < n; qi_base += chunk_sz) {
        const std::size_t qi_end = std::min(qi_base + chunk_sz, n);

        for (std::size_t qi = qi_base; qi < qi_end; ++qi) {
            const float* q = dataset.data.data() + qi * d;
            TopK heap(k);

            for (std::size_t ri = 0; ri < n; ++ri) {
                if (ri == qi) { continue; }
                const float* r = dataset.data.data() + ri * d;
                float acc = 0.f;
                for (std::size_t dim = 0; dim < d; ++dim) {
                    const float diff = q[dim] - r[dim];
                    acc += diff * diff;
                }
                heap.push(static_cast<knng::index_t>(ri), acc);
            }

            auto nbrs = heap.extract_sorted();
            for (std::size_t r = 0; r < nbrs.size(); ++r) {
                graph.neighbors[qi * k + r] = nbrs[r].first;
                graph.distances[qi * k + r] = nbrs[r].second;
            }
        }
    }
    return graph;
}

} // namespace knng::gpu
