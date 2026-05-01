#pragma once

/// @file
/// @brief Bounded-size top-k buffer keyed on distance.
///
/// `TopK` is the small per-query container that every nearest-neighbor
/// builder pushes candidates into and reads its final answer out of.
/// The contract is intentionally narrow:
///
///   * Capacity `k` is fixed at construction time. The buffer admits a
///     candidate `(id, dist)` whenever `dist` is strictly less than the
///     current worst (largest) distance held — or whenever the buffer is
///     not yet full.
///   * On equal-distance ties, the smaller `id` wins. This makes the
///     output deterministic at every step that has not yet introduced a
///     seeded RNG (the brute-force builder at Step 10 relies on this).
///   * `extract_sorted()` drains the buffer and returns the held entries
///     in ascending distance order. After a call, the buffer is empty.
///
/// Why a bounded max-heap? The natural admission test is "is the new
/// candidate better than the current worst?" — which is exactly
/// `dist < top()` if the heap is a max-heap on distance. `O(log k)` per
/// admission, `O(k log k)` for the final sort. For the small-k regime
/// (k ≤ 100 in every benchmark we care about), this is already
/// competitive with linear-scan partial-sort tactics, and it keeps the
/// API one-element-at-a-time so callers do not need to materialise a
/// dense distance array. Phase 3 will benchmark a `std::partial_sort`
/// alternative against this; for Phase 1 it is the right baseline.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "knng/core/types.hpp"

namespace knng {

/// Bounded-size top-k buffer over `(index_t, float)` candidates. See
/// the file-level documentation for the admission and tie-break rules.
class TopK {
public:
    /// Construct an empty buffer with capacity `k`. A zero-capacity
    /// buffer is permitted: every `push` is a no-op, `extract_sorted`
    /// always returns the empty vector. This degenerate case lets
    /// callers parameterise on `k` without special-casing.
    explicit TopK(std::size_t k) : k_{k} {}

    /// Offer a candidate to the buffer. Admitted iff (a) the buffer is
    /// not yet at capacity, or (b) `dist` is strictly smaller than the
    /// current worst distance, or (c) `dist` ties the current worst
    /// distance and `id` is strictly smaller than the current worst's
    /// id (deterministic tie-break — see file-level docs).
    void push(index_t id, float dist)
    {
        if (k_ == 0) {
            return;
        }
        if (heap_.size() < k_) {
            heap_.push(Entry{dist, id});
            return;
        }
        const Entry& worst = heap_.top();
        if (dist < worst.dist
            || (dist == worst.dist && id < worst.id))
        {
            heap_.pop();
            heap_.push(Entry{dist, id});
        }
    }

    /// Drain the buffer into a vector sorted ascending by distance,
    /// breaking distance ties by ascending `id`. The returned vector
    /// has length `min(size_at_call_time, k)`. The buffer is empty
    /// after this call.
    [[nodiscard]] std::vector<std::pair<index_t, float>> extract_sorted()
    {
        std::vector<std::pair<index_t, float>> out;
        out.reserve(heap_.size());
        while (!heap_.empty()) {
            const Entry& e = heap_.top();
            out.emplace_back(e.id, e.dist);
            heap_.pop();
        }
        // The pops emitted entries worst-distance-first; flip for the
        // documented ascending-distance order.
        std::reverse(out.begin(), out.end());
        return out;
    }

    /// Number of admitted candidates currently held; never exceeds `k`.
    [[nodiscard]] std::size_t size() const noexcept { return heap_.size(); }

    /// Maximum number of candidates the buffer will retain.
    [[nodiscard]] std::size_t capacity() const noexcept { return k_; }

    /// True iff no candidates have been admitted (or the last call was
    /// `extract_sorted`).
    [[nodiscard]] bool empty() const noexcept { return heap_.empty(); }

private:
    /// Heap entry. Distance is the primary key, id the tie-breaker.
    struct Entry {
        float   dist;
        index_t id;
    };

    /// Strict-weak-ordering comparator that yields a max-heap on
    /// `(dist, id)`: the worst-by-distance / worst-by-id-on-tie entry
    /// is at the top, ready for eviction.
    struct WorseFirst {
        bool operator()(const Entry& a, const Entry& b) const noexcept
        {
            if (a.dist != b.dist) {
                return a.dist < b.dist;
            }
            return a.id < b.id;
        }
    };

    using Heap = std::priority_queue<Entry, std::vector<Entry>, WorseFirst>;

    std::size_t k_;
    Heap        heap_;
};

} // namespace knng
