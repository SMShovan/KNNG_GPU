/// @file
/// @brief Steps 93 & 95 — NEO-DNND replication policy CPU implementations.
///
/// cpu_dedup_requests: sort + unique to remove duplicate global IDs.
/// BandwidthAwarePolicy::update: frequency counter updated from request set;
///   returns IDs whose frequency has reached `threshold`.

#include <knng/dist_gpu/replication_policy.hpp>

#include <algorithm>
#include <numeric>

namespace knng::dist_gpu {

// ── Step 93 — CPU dedup ──────────────────────────────────────────────────────

GpuDedupStats cpu_dedup_requests(std::vector<knng::index_t>& requests,
                                 std::size_t                 /*global_n*/)
{
    const std::size_t raw = requests.size();

    std::sort(requests.begin(), requests.end());
    auto end_it = std::unique(requests.begin(), requests.end());
    requests.erase(end_it, requests.end());

    return {raw, requests.size()};
}

// ── Step 95 — Bandwidth-aware policy ─────────────────────────────────────────

std::vector<knng::index_t>
BandwidthAwarePolicy::update(const std::vector<knng::index_t>& requests,
                             std::size_t                        global_n)
{
    if (freq_.size() < global_n)
        freq_.assign(global_n, 0);

    for (auto id : requests) {
        if (static_cast<std::size_t>(id) < global_n)
            ++freq_[static_cast<std::size_t>(id)];
    }

    std::vector<knng::index_t> replicate;
    for (std::size_t i = 0; i < global_n; ++i) {
        if (freq_[i] >= threshold)
            replicate.push_back(static_cast<knng::index_t>(i));
    }
    return replicate;
}

} // namespace knng::dist_gpu
