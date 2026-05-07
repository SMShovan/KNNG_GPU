/// @file
/// @brief NEO-DNND Optimisation 1 — duplicate-request deduplication.

#include "knng/dist/request_dedup.hpp"

#include <algorithm>
#include <numeric>

namespace knng::dist {

DeduplicationStats
dedup_requests(std::vector<std::vector<knng::index_t>>& requests)
{
    std::size_t raw_total   = 0;
    std::size_t dedup_total = 0;

    for (auto& req : requests) {
        raw_total += req.size();
        std::sort(req.begin(), req.end());
        req.erase(std::unique(req.begin(), req.end()), req.end());
        dedup_total += req.size();
    }

    return {raw_total, dedup_total};
}

DeduplicationStats
allreduce_dedup_stats(const DeduplicationStats& local_stats, MPI_Comm comm)
{
    // Pack into a 2-element array for a single Allreduce call.
    std::size_t local_arr[2] = {local_stats.raw_count,
                                local_stats.dedup_count};
    std::size_t global_arr[2] = {0, 0};
    MPI_Allreduce(local_arr, global_arr, 2, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    return {global_arr[0], global_arr[1]};
}

} // namespace knng::dist
