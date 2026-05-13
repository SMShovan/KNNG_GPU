/// @file
/// @brief Steps 72–77 — CAGRA-style graph refinement GPU kernels.
///
/// ## Step coverage
///   Step 72: `enforce_out_degree_kernel`  — insertion-sort rows in shared mem

#include <knng/gpu/graph_refinement.hpp>
#include <knng/gpu/device_graph.hpp>

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

} // namespace knng::gpu
