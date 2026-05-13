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

} // namespace knng::gpu
