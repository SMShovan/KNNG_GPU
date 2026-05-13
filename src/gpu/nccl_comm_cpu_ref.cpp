/// @file
/// @brief Step 80 — CPU simulation of NCCL-style collective operations.
///
/// Always compiled (no CUDA/NCCL dependency).  Each function mirrors the
/// semantics of the corresponding NCCL collective exactly so tests running
/// on Mac exercise the same data-flow as the real GPU path.

#include <knng/gpu/nccl_comm.hpp>

#include <algorithm>
#include <cassert>

namespace knng::gpu {

void CpuSimCollective::allreduce_sum(std::vector<std::vector<float>>& per_rank,
                                      std::size_t count) {
    const std::size_t n = per_rank.size();
    if (n == 0 || count == 0) return;

    std::vector<float> acc(count, 0.f);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t i = 0; i < count; ++i)
            acc[i] += per_rank[r][i];
    for (std::size_t r = 0; r < n; ++r)
        std::copy(acc.begin(), acc.end(), per_rank[r].begin());
}

void CpuSimCollective::bcast(std::vector<std::vector<float>>& per_rank,
                               int root, std::size_t count) {
    const std::size_t n = per_rank.size();
    if (n == 0 || count == 0) return;
    assert(root >= 0 && static_cast<std::size_t>(root) < n);
    const std::size_t ru = static_cast<std::size_t>(root);
    for (std::size_t r = 0; r < n; ++r) {
        if (r == ru) continue;
        std::copy(per_rank[ru].begin(),
                  per_rank[ru].begin() + static_cast<std::ptrdiff_t>(count),
                  per_rank[r].begin());
    }
}

void CpuSimCollective::allgather(const std::vector<std::vector<float>>& per_rank,
                                   std::vector<float>& result,
                                   std::size_t count_per_rank) {
    const std::size_t n = per_rank.size();
    result.resize(n * count_per_rank);
    for (std::size_t r = 0; r < n; ++r)
        std::copy(per_rank[r].begin(),
                  per_rank[r].begin() + static_cast<std::ptrdiff_t>(count_per_rank),
                  result.begin() + static_cast<std::ptrdiff_t>(r * count_per_rank));
}

void CpuSimCollective::reduce_scatter(std::vector<std::vector<float>>& per_rank,
                                        std::size_t count) {
    const std::size_t n = per_rank.size();
    if (n == 0 || count == 0) return;
    assert(count % n == 0);
    const std::size_t chunk = count / n;

    std::vector<float> acc(count, 0.f);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t i = 0; i < count; ++i)
            acc[i] += per_rank[r][i];

    for (std::size_t r = 0; r < n; ++r) {
        per_rank[r].resize(chunk);
        std::copy(acc.begin() + static_cast<std::ptrdiff_t>(r * chunk),
                  acc.begin() + static_cast<std::ptrdiff_t>((r + 1) * chunk),
                  per_rank[r].begin());
    }
}

} // namespace knng::gpu
