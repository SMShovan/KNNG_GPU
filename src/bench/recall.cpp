/// @file
/// @brief Implementation of `knng::bench::recall_at_k`.
///
/// The algorithm is straightforward: for each point, intersect the
/// set of `approx` neighbor IDs with the set of `truth` neighbor IDs
/// and accumulate `|intersection| / k` averaged over all points.
///
/// The implementation choice that matters is the per-row data
/// structure for the `truth` set. For small `k` (≤ 1024 in every
/// benchmark we care about) a sorted `std::vector<index_t>` plus
/// `std::binary_search` is comparable to a `std::unordered_set` but
/// allocates one block instead of `O(k)` and has predictable cache
/// behaviour. For large `k` we would switch to an unordered set, but
/// recall@1000 is already at the upper end of practical evaluation
/// — the `std::vector` path is the right default for now.

#include "knng/bench/recall.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "knng/core/types.hpp"

namespace knng::bench {

namespace {

void check_shapes(const Knng& approx, const Knng& truth)
{
    if (approx.n != truth.n) {
        throw std::invalid_argument(
            "knng::bench::recall: approx.n (" + std::to_string(approx.n)
            + ") != truth.n (" + std::to_string(truth.n) + ")");
    }
    if (approx.k != truth.k) {
        throw std::invalid_argument(
            "knng::bench::recall: approx.k (" + std::to_string(approx.k)
            + ") != truth.k (" + std::to_string(truth.k) + ")");
    }
}

/// Count the number of unique IDs in `approx_row` that also appear
/// in `truth_row`. Both rows are scanned linearly; `truth_sorted` is
/// the row's IDs sorted ascending so binary search is `O(log k)` per
/// lookup. A tiny `std::vector<index_t> seen` deduplicates the
/// approx row so a malformed builder cannot inflate its score by
/// repeating a single correct neighbor `k` times.
[[nodiscard]] std::size_t row_intersection(
    std::span<const index_t> approx_row,
    std::span<const index_t> truth_row)
{
    std::vector<index_t> truth_sorted(truth_row.begin(), truth_row.end());
    std::sort(truth_sorted.begin(), truth_sorted.end());

    std::vector<index_t> seen;
    seen.reserve(approx_row.size());

    std::size_t hits = 0;
    for (const index_t id : approx_row) {
        if (std::find(seen.begin(), seen.end(), id) != seen.end()) {
            continue;
        }
        seen.push_back(id);
        if (std::binary_search(truth_sorted.begin(),
                               truth_sorted.end(), id))
        {
            ++hits;
        }
    }
    return hits;
}

} // namespace

double recall_at_k(const Knng& approx, const Knng& truth)
{
    check_shapes(approx, truth);

    if (approx.n == 0) {
        return 1.0;
    }

    std::size_t total_hits = 0;
    for (std::size_t q = 0; q < approx.n; ++q) {
        total_hits += row_intersection(
            approx.neighbors_of(q), truth.neighbors_of(q));
    }

    const double denom = static_cast<double>(approx.n)
                       * static_cast<double>(approx.k);
    return static_cast<double>(total_hits) / denom;
}

std::size_t recall_at_k_row(const Knng& approx,
                            const Knng& truth,
                            std::size_t row)
{
    check_shapes(approx, truth);
    if (row >= approx.n) {
        throw std::invalid_argument(
            "knng::bench::recall_at_k_row: row " + std::to_string(row)
            + " out of range (n=" + std::to_string(approx.n) + ")");
    }
    return row_intersection(approx.neighbors_of(row),
                            truth.neighbors_of(row));
}

} // namespace knng::bench
