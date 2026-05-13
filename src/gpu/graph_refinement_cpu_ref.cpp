/// @file
/// @brief Steps 72–77 — CPU reference implementations for CAGRA-style refinement.
///
/// Always compiled (no CUDA dependency).  Each function mirrors the GPU kernel.

#include <knng/gpu/graph_refinement.hpp>
#include <knng/gpu/device_graph.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Step 72 — Fixed out-degree enforcement (CPU reference)
// ---------------------------------------------------------------------------

void cpu_enforce_out_degree(CpuDeviceGraph& graph) {
    const std::size_t n = graph.n;
    const std::size_t k = graph.k;
    const auto kSentinel = static_cast<knng::index_t>(-1);

    std::vector<std::size_t>    order(k);
    std::vector<knng::index_t>  tmp_ids(k);
    std::vector<float>          tmp_dists(k);
    std::vector<std::uint32_t>  tmp_flags(k);

    for (std::size_t qi = 0; qi < n; ++qi) {
        std::iota(order.begin(), order.end(), 0u);

        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const bool av = (graph.ids[qi * k + a] != kSentinel);
            const bool bv = (graph.ids[qi * k + b] != kSentinel);
            if (av != bv) return static_cast<int>(av) > static_cast<int>(bv);
            return graph.dists[qi * k + a] < graph.dists[qi * k + b];
        });

        for (std::size_t r = 0; r < k; ++r) {
            tmp_ids  [r] = graph.ids  [qi * k + order[r]];
            tmp_dists[r] = graph.dists[qi * k + order[r]];
            tmp_flags[r] = graph.flags[qi * k + order[r]];
        }
        for (std::size_t r = 0; r < k; ++r) {
            graph.ids  [qi * k + r] = tmp_ids  [r];
            graph.dists[qi * k + r] = tmp_dists[r];
            graph.flags[qi * k + r] = tmp_flags[r];
        }
    }
}

// ---------------------------------------------------------------------------
// Step 73 — Rank-based reordering (CPU reference)
// ---------------------------------------------------------------------------

void cpu_rank_reorder(CpuDeviceGraph& graph) {
    const std::size_t n = graph.n;
    const std::size_t k = graph.k;
    const auto kSentinel = static_cast<knng::index_t>(-1);

    // For each point j, record the list-position at which each of its neighbors
    // appears: rank_of[j][neighbor_id] = position (0-based).
    // We store this as a flat vector indexed [j * n + id] — feasible for small n.
    // For production n, a hash map or CSR would replace this.
    std::vector<float> rank_of(n * n, std::numeric_limits<float>::infinity());
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t r = 0; r < k; ++r) {
            const knng::index_t id = graph.ids[j * k + r];
            if (id == kSentinel) break;
            rank_of[j * n + static_cast<std::size_t>(id)] = static_cast<float>(r);
        }
    }

    std::vector<std::size_t>   order(k);
    std::vector<knng::index_t> tmp_ids(k);
    std::vector<float>         tmp_dists(k);
    std::vector<std::uint32_t> tmp_flags(k);

    for (std::size_t qi = 0; qi < n; ++qi) {
        // Compute rank score for each neighbor of qi:
        //   score[r] = rank_of[neighbor][qi]  (how early qi appears in neighbor's list)
        // Sentinels get infinity so they always sort last.
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const knng::index_t ia = graph.ids[qi * k + a];
            const knng::index_t ib = graph.ids[qi * k + b];
            const bool av = (ia != kSentinel);
            const bool bv = (ib != kSentinel);
            if (av != bv) return static_cast<int>(av) > static_cast<int>(bv);
            if (!av) return false;
            const float ra = rank_of[static_cast<std::size_t>(ia) * n + qi];
            const float rb = rank_of[static_cast<std::size_t>(ib) * n + qi];
            return ra < rb;
        });

        for (std::size_t r = 0; r < k; ++r) {
            tmp_ids  [r] = graph.ids  [qi * k + order[r]];
            tmp_dists[r] = graph.dists[qi * k + order[r]];
            tmp_flags[r] = graph.flags[qi * k + order[r]];
        }
        for (std::size_t r = 0; r < k; ++r) {
            graph.ids  [qi * k + r] = tmp_ids  [r];
            graph.dists[qi * k + r] = tmp_dists[r];
            graph.flags[qi * k + r] = tmp_flags[r];
        }
    }
}

} // namespace knng::gpu
