/// @file
/// @brief Steps 48–50 — CPU-reference brute-force KNN implementation.
///
/// Always compiled (no CUDA dependency).  Used as the correctness oracle
/// in unit tests for all three GPU brute-force tiers.

#include <knng/gpu/brute_force.hpp>
#include <knng/top_k.hpp>

namespace knng::gpu {

knng::Knng cpu_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k)
{
    const std::size_t n = dataset.n;
    const std::size_t d = dataset.d;
    knng::Knng graph(n, k);

    for (std::size_t qi = 0; qi < n; ++qi) {
        TopK heap(k);
        const float* q = dataset.data.data() + qi * d;

        for (std::size_t ri = 0; ri < n; ++ri) {
            if (ri == qi) {
                continue;
            }
            const float* r = dataset.data.data() + ri * d;
            float acc = 0.f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                const float diff = q[dim] - r[dim];
                acc += diff * diff;
            }
            heap.push(static_cast<index_t>(ri), acc);
        }

        auto neighbours = heap.extract_sorted();
        for (std::size_t rank = 0; rank < neighbours.size(); ++rank) {
            graph.neighbors[qi * k + rank] = neighbours[rank].first;
            graph.distances[qi * k + rank] = neighbours[rank].second;
        }
    }
    return graph;
}

} // namespace knng::gpu
