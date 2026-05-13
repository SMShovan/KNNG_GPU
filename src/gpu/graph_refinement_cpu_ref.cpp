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

// ---------------------------------------------------------------------------
// Step 74 — Add reverse edges (CPU reference)
// ---------------------------------------------------------------------------

void cpu_add_reverse_edges(CpuDeviceGraph& graph) {
    const std::size_t n = graph.n;
    const std::size_t k = graph.k;
    const auto kSentinel = static_cast<knng::index_t>(-1);

    // For each edge i → j, try to insert (j → i, dist) into j's list.
    // Replace a sentinel slot if available; otherwise replace the worst valid
    // entry if dist is better.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < k; ++r) {
            const knng::index_t j = graph.ids[i * k + r];
            if (j == kSentinel) break;
            const float dist = graph.dists[i * k + r];

            // Check if i is already in j's list
            bool already_present = false;
            std::size_t sentinel_slot = k;      // first sentinel slot in j's list
            std::size_t worst_slot    = k;      // slot with largest distance in j's list
            float       worst_dist    = -1.f;

            for (std::size_t s = 0; s < k; ++s) {
                const knng::index_t jid = graph.ids[j * k + s];
                if (jid == static_cast<knng::index_t>(i)) { already_present = true; break; }
                if (jid == kSentinel) {
                    if (sentinel_slot == k) sentinel_slot = s;
                } else if (graph.dists[j * k + s] > worst_dist) {
                    worst_dist = graph.dists[j * k + s];
                    worst_slot = s;
                }
            }
            if (already_present) continue;

            if (sentinel_slot != k) {
                // Fill a sentinel slot
                graph.ids  [j * k + sentinel_slot] = static_cast<knng::index_t>(i);
                graph.dists[j * k + sentinel_slot] = dist;
                graph.flags[j * k + sentinel_slot] = 0u;
            } else if (worst_slot != k && dist < worst_dist) {
                // Evict worst entry
                graph.ids  [j * k + worst_slot] = static_cast<knng::index_t>(i);
                graph.dists[j * k + worst_slot] = dist;
                graph.flags[j * k + worst_slot] = 0u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 75 — Detourable-edge pruning, MRNG rule (CPU reference)
// ---------------------------------------------------------------------------

namespace {
inline float sq_l2(const float* a, const float* b, std::size_t dim) {
    float s = 0.f;
    for (std::size_t d = 0; d < dim; ++d) { const float diff = a[d] - b[d]; s += diff * diff; }
    return s;
}
} // anonymous namespace

void cpu_prune_detour_edges(CpuDeviceGraph& graph,
                             const float* vectors,
                             std::size_t dim) {
    const std::size_t n = graph.n;
    const std::size_t k = graph.k;
    const auto kSentinel = static_cast<knng::index_t>(-1);

    for (std::size_t i = 0; i < n; ++i) {
        (void)(vectors + i * dim); // vi not needed: dij already in graph.dists
        for (std::size_t r = 0; r < k; ++r) {
            const knng::index_t j = graph.ids[i * k + r];
            if (j == kSentinel) continue;
            const float dij = graph.dists[i * k + r]; // d(i,j)²
            const float* vj = vectors + static_cast<std::size_t>(j) * dim;

            // Check if any other neighbor m provides a detour
            bool detourable = false;
            for (std::size_t s = 0; s < k && !detourable; ++s) {
                if (s == r) continue;
                const knng::index_t m = graph.ids[i * k + s];
                if (m == kSentinel) continue;
                const float dim2 = graph.dists[i * k + s]; // d(i,m)²
                if (dim2 >= dij) continue;                  // m is not closer than j
                const float* vm = vectors + static_cast<std::size_t>(m) * dim;
                const float dmj = sq_l2(vm, vj, dim);       // d(m,j)²
                if (dmj < dij) detourable = true;
            }

            if (detourable) {
                graph.ids  [i * k + r] = kSentinel;
                graph.dists[i * k + r] = kSentinelDist;
                graph.flags[i * k + r] = 0u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step 76 — Strong-component merging (CPU reference)
// ---------------------------------------------------------------------------

namespace {

std::size_t uf_find(std::vector<std::size_t>& parent, std::size_t x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}

void uf_union(std::vector<std::size_t>& parent,
              std::vector<std::size_t>& rank_uf,
              std::size_t a, std::size_t b) {
    a = uf_find(parent, a); b = uf_find(parent, b);
    if (a == b) return;
    if (rank_uf[a] < rank_uf[b]) std::swap(a, b);
    parent[b] = a;
    if (rank_uf[a] == rank_uf[b]) ++rank_uf[a];
}

} // anonymous namespace

void cpu_merge_components(CpuDeviceGraph& graph,
                           const float* vectors,
                           std::size_t dim) {
    const std::size_t n = graph.n;
    const std::size_t k = graph.k;
    const auto kSentinel = static_cast<knng::index_t>(-1);

    // Build union-find on the undirected graph
    std::vector<std::size_t> parent(n), rank_uf(n, 0u);
    std::iota(parent.begin(), parent.end(), 0u);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < k; ++r) {
            const knng::index_t j = graph.ids[i * k + r];
            if (j == kSentinel) break;
            uf_union(parent, rank_uf, i, static_cast<std::size_t>(j));
        }
    }

    // Find the largest component (by root count)
    std::vector<std::size_t> comp_size(n, 0u);
    for (std::size_t i = 0; i < n; ++i) ++comp_size[uf_find(parent, i)];
    const std::size_t main_root = static_cast<std::size_t>(
        std::max_element(comp_size.begin(), comp_size.end()) - comp_size.begin());

    // For each node not in the main component, find its closest node in the
    // main component and add a bridge edge (replacing the worst slot or a sentinel).
    for (std::size_t i = 0; i < n; ++i) {
        if (uf_find(parent, i) == main_root) continue;

        float   best_dist = std::numeric_limits<float>::infinity();
        std::size_t best_j = n;
        for (std::size_t j = 0; j < n; ++j) {
            if (uf_find(parent, j) != main_root) continue;
            const float d = sq_l2(vectors + i * dim, vectors + j * dim, dim);
            if (d < best_dist) { best_dist = d; best_j = j; }
        }
        if (best_j == n) continue;

        // Find a slot: prefer sentinel, else evict worst
        std::size_t target_slot = k;
        float       worst_d     = -1.f;
        std::size_t worst_slot  = k;
        for (std::size_t r = 0; r < k; ++r) {
            const knng::index_t id = graph.ids[i * k + r];
            if (id == kSentinel) { target_slot = r; break; }
            if (graph.dists[i * k + r] > worst_d) {
                worst_d = graph.dists[i * k + r]; worst_slot = r;
            }
        }
        if (target_slot == k) target_slot = worst_slot;
        if (target_slot == k) continue;

        graph.ids  [i * k + target_slot] = static_cast<knng::index_t>(best_j);
        graph.dists[i * k + target_slot] = best_dist;
        graph.flags[i * k + target_slot] = 0u;

        // Merge the two components so subsequent isolated nodes see the updated graph
        uf_union(parent, rank_uf, i, best_j);
    }
}

} // namespace knng::gpu
