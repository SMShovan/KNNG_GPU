/// @file
/// @brief Steps 72–77 — CAGRA-style graph refinement GPU kernels.
///
/// ## Step coverage
///   Step 72: `enforce_out_degree_kernel`  — insertion-sort rows in shared mem
///   Step 73: `build_ranks_kernel`, `rank_sort_kernel`
///   Step 74: `add_reverse_edges_kernel`
///   Step 75: `prune_detour_kernel`
///   Step 76: `label_propagate_kernel`, `bridge_components_kernel`
///   Step 77: `gpu_refine_graph` pipeline driver

#include <knng/gpu/graph_refinement.hpp>
#include <knng/gpu/device_graph.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace knng::gpu {

// ===========================================================================
// Step 72 — Fixed out-degree enforcement kernel
// ===========================================================================

GPU_GLOBAL void enforce_out_degree_kernel(
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags,
    unsigned int   n,
    unsigned int   k)
{
    const unsigned int p = blockIdx.x;
    if (p >= n) return;

    // Shared layout: ids[k] | dists[k] | flags[k]
    extern __shared__ char sh[];
    auto* sh_ids   = reinterpret_cast<knng::index_t*>(sh);
    auto* sh_dists = reinterpret_cast<float*>(sh_ids + k);
    auto* sh_flags = reinterpret_cast<std::uint32_t*>(sh_dists + k);

    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        sh_ids  [r] = d_ids  [p * k + r];
        sh_dists[r] = d_dists[p * k + r];
        sh_flags[r] = d_flags[p * k + r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const auto kSentinel = static_cast<knng::index_t>(-1);
        for (unsigned int i = 1; i < k; ++i) {
            const knng::index_t ki   = sh_ids  [i];
            const float         kd   = sh_dists[i];
            const std::uint32_t kf   = sh_flags[i];
            const bool          kv   = (ki != kSentinel);
            int j = static_cast<int>(i) - 1;
            while (j >= 0) {
                const auto ju = static_cast<unsigned int>(j);
                const bool jv = (sh_ids[ju] != kSentinel);
                const bool swap_needed =
                    (kv && !jv) ||
                    (kv && jv && kd < sh_dists[ju]);
                if (!swap_needed) break;
                sh_ids  [ju + 1u] = sh_ids  [ju];
                sh_dists[ju + 1u] = sh_dists[ju];
                sh_flags[ju + 1u] = sh_flags[ju];
                --j;
            }
            const auto ju1 = static_cast<unsigned int>(j) + 1u;
            sh_ids  [ju1] = ki;
            sh_dists[ju1] = kd;
            sh_flags[ju1] = kf;
        }
    }
    __syncthreads();

    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        d_ids  [p * k + r] = sh_ids  [r];
        d_dists[p * k + r] = sh_dists[r];
        d_flags[p * k + r] = sh_flags[r];
    }
}

void gpu_enforce_out_degree(DeviceGraph& graph) {
    const auto n  = static_cast<unsigned int>(graph.n);
    const auto ku = static_cast<unsigned int>(graph.k);
    const std::size_t shmem = static_cast<std::size_t>(ku) *
        (sizeof(knng::index_t) + sizeof(float) + sizeof(std::uint32_t));
    GPU_LAUNCH(enforce_out_degree_kernel, n, 1u, shmem, nullptr,
               graph.ids.data(), graph.dists.data(), graph.flags.data(), n, ku);
    GPU_SYNC();
}

// ===========================================================================
// Step 73 — Rank-based reordering kernel
// ===========================================================================

/// Build rank table: d_ranks[j * n + id] = position of id in j's neighbor list.
GPU_GLOBAL void build_ranks_kernel(
    const knng::index_t* d_ids,
    float*               d_ranks,   // n * n floats, pre-filled with FLT_MAX
    unsigned int         n,
    unsigned int         k)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    for (unsigned int r = 0; r < k; ++r) {
        const knng::index_t id = d_ids[j * k + r];
        if (id == static_cast<knng::index_t>(-1)) break;
        d_ranks[static_cast<std::size_t>(j) * n + id] = static_cast<float>(r);
    }
}

/// Sort each row by rank score ascending (valid entries first, sentinels last).
GPU_GLOBAL void rank_sort_kernel(
    knng::index_t*       d_ids,
    float*               d_dists,
    std::uint32_t*       d_flags,
    const float*         d_ranks,
    unsigned int         n,
    unsigned int         k)
{
    const unsigned int qi = blockIdx.x;
    if (qi >= n) return;

    extern __shared__ char sh[];
    auto* sh_ids   = reinterpret_cast<knng::index_t*>(sh);
    auto* sh_dists = reinterpret_cast<float*>(sh_ids + k);
    auto* sh_flags = reinterpret_cast<std::uint32_t*>(sh_dists + k);
    auto* sh_ranks = reinterpret_cast<float*>(sh_flags + k);

    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        const knng::index_t id = d_ids[qi * k + r];
        sh_ids  [r] = id;
        sh_dists[r] = d_dists[qi * k + r];
        sh_flags[r] = d_flags[qi * k + r];
        sh_ranks[r] = (id != static_cast<knng::index_t>(-1))
            ? d_ranks[static_cast<std::size_t>(id) * n + qi]
            : 3.402823466e+38f; // FLT_MAX
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (unsigned int i = 1; i < k; ++i) {
            const knng::index_t ki = sh_ids  [i];
            const float         kd = sh_dists[i];
            const std::uint32_t kf = sh_flags[i];
            const float         kr = sh_ranks[i];
            const bool kv = (ki != static_cast<knng::index_t>(-1));
            int j = static_cast<int>(i) - 1;
            while (j >= 0) {
                const auto ju = static_cast<unsigned int>(j);
                const bool jv = (sh_ids[ju] != static_cast<knng::index_t>(-1));
                const bool swap_needed =
                    (kv && !jv) ||
                    (kv && jv && kr < sh_ranks[ju]);
                if (!swap_needed) break;
                sh_ids  [ju + 1u] = sh_ids  [ju];
                sh_dists[ju + 1u] = sh_dists[ju];
                sh_flags[ju + 1u] = sh_flags[ju];
                sh_ranks[ju + 1u] = sh_ranks[ju];
                --j;
            }
            const auto ju1 = static_cast<unsigned int>(j) + 1u;
            sh_ids  [ju1] = ki;
            sh_dists[ju1] = kd;
            sh_flags[ju1] = kf;
            sh_ranks[ju1] = kr;
        }
    }
    __syncthreads();

    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        d_ids  [qi * k + r] = sh_ids  [r];
        d_dists[qi * k + r] = sh_dists[r];
        d_flags[qi * k + r] = sh_flags[r];
    }
}

void gpu_rank_reorder(DeviceGraph& graph) {
    const auto n  = static_cast<unsigned int>(graph.n);
    const auto ku = static_cast<unsigned int>(graph.k);

    // Allocate rank table (n * n floats, initialised to FLT_MAX)
    DeviceBuffer<float> d_ranks(static_cast<std::size_t>(n) * n);
    const float fmax = 3.402823466e+38f;
    GPU_CHECK(cudaMemset(d_ranks.data(), 0xFF, d_ranks.size() * sizeof(float)));
    // 0xFF bytes → 0xFFFFFFFF as float = NaN, use explicit fill instead
    // Fill with fmax via a simple kernel:
    {
        const std::size_t total = static_cast<std::size_t>(n) * n;
        const unsigned int threads = 256;
        const unsigned int blocks  = static_cast<unsigned int>((total + threads - 1) / threads);
        // inline fill kernel — just set every element
        // We use thrust::fill or a small custom kernel; here we reuse cudaMemset workaround:
        // cudaMemset sets bytes; 0x7F7F7F7F = 2139095039 = 3.4028e38 (close enough to FLT_MAX)
        GPU_CHECK(cudaMemset(d_ranks.data(), 0x7f, d_ranks.size() * sizeof(float)));
        (void)blocks; (void)fmax;
    }

    const unsigned int threads_build = 256u;
    const unsigned int blocks_build  = (n + threads_build - 1u) / threads_build;
    GPU_LAUNCH(build_ranks_kernel, blocks_build, threads_build, 0, nullptr,
               graph.ids.data(), d_ranks.data(), n, ku);

    const std::size_t shmem = static_cast<std::size_t>(ku) *
        (sizeof(knng::index_t) + 2 * sizeof(float) + sizeof(std::uint32_t));
    GPU_LAUNCH(rank_sort_kernel, n, 1u, shmem, nullptr,
               graph.ids.data(), graph.dists.data(), graph.flags.data(),
               d_ranks.data(), n, ku);
    GPU_SYNC();
}

// ===========================================================================
// Step 74 — Add reverse edges kernel
// ===========================================================================

/// For each edge i → j, atomically scatter the reverse edge (j → i, dist)
/// into j's neighbor list.  We use atomicMin on d_dists to claim a slot:
/// each thread walks j's list looking for a sentinel or an entry worse than
/// dist, then does an atomic CAS on the id slot to claim it.
GPU_GLOBAL void add_reverse_edges_kernel(
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags,
    unsigned int   n,
    unsigned int   k)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const auto kSentinel = static_cast<knng::index_t>(-1);

    for (unsigned int r = 0; r < k; ++r) {
        const knng::index_t j = d_ids[i * k + r];
        if (j == kSentinel) break;
        const float dist = d_dists[i * k + r];
        const unsigned int ju = static_cast<unsigned int>(j);

        // Scan j's list for a sentinel or a worse-distance slot
        for (unsigned int s = 0; s < k; ++s) {
            const knng::index_t cur = d_ids[ju * k + s];
            if (cur == static_cast<knng::index_t>(i)) break; // already present
            if (cur == kSentinel) {
                // CAS: try to claim this sentinel slot
                const knng::index_t old = static_cast<knng::index_t>(
                    atomicCAS(reinterpret_cast<unsigned int*>(&d_ids[ju * k + s]),
                              static_cast<unsigned int>(kSentinel),
                              static_cast<unsigned int>(i)));
                if (old == kSentinel) {
                    d_dists[ju * k + s] = dist;
                    d_flags[ju * k + s] = 0u;
                }
                break;
            } else if (d_dists[ju * k + s] > dist) {
                // Try to evict: CAS on id, then update dist
                const knng::index_t old = static_cast<knng::index_t>(
                    atomicCAS(reinterpret_cast<unsigned int*>(&d_ids[ju * k + s]),
                              static_cast<unsigned int>(cur),
                              static_cast<unsigned int>(i)));
                if (old == cur) {
                    d_dists[ju * k + s] = dist;
                    d_flags[ju * k + s] = 0u;
                }
                break;
            }
        }
    }
}

void gpu_add_reverse_edges(DeviceGraph& graph) {
    const auto n  = static_cast<unsigned int>(graph.n);
    const auto ku = static_cast<unsigned int>(graph.k);
    const unsigned int threads = 256u;
    const unsigned int blocks  = (n + threads - 1u) / threads;
    GPU_LAUNCH(add_reverse_edges_kernel, blocks, threads, 0, nullptr,
               graph.ids.data(), graph.dists.data(), graph.flags.data(), n, ku);
    GPU_SYNC();
}

// ===========================================================================
// Step 75 — Detourable-edge pruning kernel (MRNG rule)
// ===========================================================================

GPU_GLOBAL void prune_detour_kernel(
    knng::index_t* d_ids,
    float*         d_dists,
    std::uint32_t* d_flags,
    const float*   d_vectors,
    unsigned int   n,
    unsigned int   k,
    unsigned int   dim)
{
    const unsigned int i = blockIdx.x;
    if (i >= n) return;

    const auto kSentinel  = static_cast<knng::index_t>(-1);
    const float kSentDist = 3.402823466e+38f;

    // Each thread handles one candidate edge i → j
    for (unsigned int r = threadIdx.x; r < k; r += blockDim.x) {
        const knng::index_t j = d_ids[i * k + r];
        if (j == kSentinel) continue;
        const float dij = d_dists[i * k + r];
        const float* vj = d_vectors + static_cast<std::size_t>(j) * dim;

        bool detourable = false;
        for (unsigned int s = 0; s < k && !detourable; ++s) {
            if (s == r) continue;
            const knng::index_t m = d_ids[i * k + s];
            if (m == kSentinel) continue;
            const float dim2 = d_dists[i * k + s];
            if (dim2 >= dij) continue;
            const float* vm = d_vectors + static_cast<std::size_t>(m) * dim;
            float dmj = 0.f;
            for (unsigned int dd = 0; dd < dim; ++dd) {
                const float diff = vm[dd] - vj[dd];
                dmj += diff * diff;
            }
            if (dmj < dij) detourable = true;
        }

        if (detourable) {
            d_ids  [i * k + r] = kSentinel;
            d_dists[i * k + r] = kSentDist;
            d_flags[i * k + r] = 0u;
        }
    }
}

void gpu_prune_detour_edges(DeviceGraph& graph,
                             const float* d_vectors,
                             std::size_t dim) {
    const auto n   = static_cast<unsigned int>(graph.n);
    const auto ku  = static_cast<unsigned int>(graph.k);
    const auto dmu = static_cast<unsigned int>(dim);
    GPU_LAUNCH(prune_detour_kernel, n, 32u, 0, nullptr,
               graph.ids.data(), graph.dists.data(), graph.flags.data(),
               d_vectors, n, ku, dmu);
    GPU_SYNC();
}

// ===========================================================================
// Step 76 — Strong-component merging (label propagation + bridge)
// ===========================================================================

/// Label propagation: each node adopts the minimum label among itself and its
/// neighbors.  Converges in O(diameter) iterations.
GPU_GLOBAL void label_propagate_kernel(
    const knng::index_t* d_ids,
    unsigned int*        d_labels,
    unsigned int         n,
    unsigned int         k,
    unsigned int*        d_changed)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto kSentinel = static_cast<knng::index_t>(-1);

    unsigned int my_label = d_labels[i];
    for (unsigned int r = 0; r < k; ++r) {
        const knng::index_t j = d_ids[i * k + r];
        if (j == kSentinel) break;
        const unsigned int jl = d_labels[static_cast<unsigned int>(j)];
        if (jl < my_label) my_label = jl;
    }
    if (my_label != d_labels[i]) {
        d_labels[i] = my_label;
        atomicOr(d_changed, 1u);
    }
}

/// For each node not in the main component, find its nearest neighbor in the
/// main component (brute-force, one thread per isolated node) and add a bridge.
GPU_GLOBAL void bridge_components_kernel(
    knng::index_t*       d_ids,
    float*               d_dists,
    std::uint32_t*       d_flags,
    const unsigned int*  d_labels,
    const float*         d_vectors,
    unsigned int         main_label,
    unsigned int         n,
    unsigned int         k,
    unsigned int         dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_labels[i] == main_label) return;

    const auto kSentinel = static_cast<knng::index_t>(-1);
    float best_dist = 3.402823466e+38f;
    unsigned int best_j = n;

    for (unsigned int j = 0; j < n; ++j) {
        if (d_labels[j] != main_label) continue;
        const float* vi = d_vectors + static_cast<std::size_t>(i) * dim;
        const float* vj = d_vectors + static_cast<std::size_t>(j) * dim;
        float d = 0.f;
        for (unsigned int dd = 0; dd < dim; ++dd) {
            const float diff = vi[dd] - vj[dd];
            d += diff * diff;
        }
        if (d < best_dist) { best_dist = d; best_j = j; }
    }
    if (best_j == n) return;

    // Find a sentinel slot; if none, overwrite index 0 (worst eviction heuristic)
    unsigned int target = 0u;
    for (unsigned int r = 0; r < k; ++r) {
        if (d_ids[i * k + r] == kSentinel) { target = r; break; }
    }
    d_ids  [i * k + target] = static_cast<knng::index_t>(best_j);
    d_dists[i * k + target] = best_dist;
    d_flags[i * k + target] = 0u;
}

void gpu_merge_components(DeviceGraph& graph,
                           const float* d_vectors,
                           std::size_t dim) {
    const auto n   = static_cast<unsigned int>(graph.n);
    const auto ku  = static_cast<unsigned int>(graph.k);
    const auto dmu = static_cast<unsigned int>(dim);
    const unsigned int threads = 256u;
    const unsigned int blocks  = (n + threads - 1u) / threads;

    DeviceBuffer<unsigned int> d_labels(n);
    // Initialize labels[i] = i
    {
        std::vector<unsigned int> h(n);
        std::iota(h.begin(), h.end(), 0u);
        GPU_CHECK(cudaMemcpy(d_labels.data(), h.data(), n * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));
    }

    // Label propagation until convergence
    DeviceBuffer<unsigned int> d_changed(1);
    for (int iter = 0; iter < static_cast<int>(n); ++iter) {
        GPU_CHECK(cudaMemset(d_changed.data(), 0, sizeof(unsigned int)));
        GPU_LAUNCH(label_propagate_kernel, blocks, threads, 0, nullptr,
                   graph.ids.data(), d_labels.data(), n, ku, d_changed.data());
        GPU_SYNC();
        unsigned int changed = 0;
        GPU_CHECK(cudaMemcpy(&changed, d_changed.data(), sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
        if (!changed) break;
    }

    // Find main component (label with most members)
    std::vector<unsigned int> h_labels(n);
    GPU_CHECK(cudaMemcpy(h_labels.data(), d_labels.data(), n * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
    std::vector<unsigned int> cnt(n, 0u);
    for (auto l : h_labels) ++cnt[l];
    const unsigned int main_label = static_cast<unsigned int>(
        std::max_element(cnt.begin(), cnt.end()) - cnt.begin());

    GPU_LAUNCH(bridge_components_kernel, blocks, threads, 0, nullptr,
               graph.ids.data(), graph.dists.data(), graph.flags.data(),
               d_labels.data(), d_vectors, main_label, n, ku, dmu);
    GPU_SYNC();
}

// ===========================================================================
// Step 77 — GPU pipeline driver
// ===========================================================================

void gpu_refine_graph(DeviceGraph& graph,
                       const GraphRefinementConfig& cfg,
                       const float* d_vectors,
                       std::size_t dim) {
    if (cfg.enforce_out_degree) gpu_enforce_out_degree(graph);
    if (cfg.rank_reorder)       gpu_rank_reorder(graph);
    if (cfg.add_reverse_edges)  gpu_add_reverse_edges(graph);
    if (cfg.prune_detour && d_vectors)
        gpu_prune_detour_edges(graph, d_vectors, dim);
    if (cfg.merge_components && d_vectors)
        gpu_merge_components(graph, d_vectors, dim);
}

} // namespace knng::gpu
