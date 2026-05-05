#pragma once

/// @file
/// @brief Per-point neighbor list with `is_new` flag (Step 30).
///
/// The data structure NN-Descent builds during each iteration. Every
/// point keeps a sorted-by-distance list of its `k` best-known
/// neighbours plus a `is_new` flag per entry. The flag is the
/// algorithmic discriminator that turns NN-Descent from `O(N²)`
/// brute-force into a tractable approximate builder:
///
///   * On graph initialisation (Step 31), every neighbour is
///     marked `is_new = true`.
///   * In each local-join iteration (Step 32+), the algorithm
///     considers only `(u, v)` pairs *where at least one is new*
///     — old × old pairs were already considered in a previous
///     iteration and re-comparing them would do redundant work.
///   * After a point's neighbours have been processed, the flag
///     is flipped to `is_new = false` so the next iteration
///     skips them; freshly-inserted neighbours start as `true`
///     and are picked up next time.
///
/// Without the flag, the local-join would re-examine every pair
/// every iteration; with the flag, the per-iteration work shrinks
/// monotonically as the graph stabilises and convergence becomes
/// the natural stopping condition. This is the single most
/// important *constant-factor* optimisation NN-Descent ships, and
/// it is referenced directly in Wang et al. 2012 §4.1 and the
/// NEO-DNND paper §3.1.
///
/// Container choice: a flat `std::vector<Neighbor>` kept sorted
/// ascending by distance. The list size is bounded by `k`
/// (typically 10–50), so a linear scan for insertion / containment
/// outperforms tree-based or hash-based structures at every
/// realistic input. Insertion is O(k); containment is O(k);
/// `mark_all_old` is O(k). All scale with the (small, fixed) `k`,
/// not with `n`.
///
/// Tie-breaking matches the `TopK` from Step 09: equal distances
/// are ordered by ascending neighbour id, so the output is
/// deterministic across runs without an RNG. This is what the
/// per-iteration regression test will rely on.

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "knng/core/types.hpp"

namespace knng {

/// One entry in a point's neighbour list. The `is_new` flag is the
/// NN-Descent local-join's redundancy filter — see file-level
/// docs in `neighbor_list.hpp` for the full motivation.
struct Neighbor {
    index_t id;
    float   dist;
    bool    is_new;
};

} // namespace knng

namespace knng::cpu {

/// Bounded-size, sorted-by-distance neighbour list with `is_new`
/// tracking. The container is the building block every
/// NN-Descent iteration mutates; its public surface is the
/// minimum needed for Step 32's local-join kernel to be
/// straightforward.
class NeighborList {
public:
    /// Sentinel `id` used by callers that need to denote
    /// "no neighbour assigned yet." Matches the convention from
    /// `knng/core/types.hpp` (`numeric_limits<index_t>::max()`).
    static constexpr index_t kEmptyId =
        std::numeric_limits<index_t>::max();

    /// Construct an empty list with capacity `k`. A `k = 0` list
    /// is permitted: every `insert` is a no-op, every accessor
    /// returns the empty span. This degenerate case lets callers
    /// parameterise on `k` without special-casing the boundary.
    explicit NeighborList(std::size_t k) : k_{k} { entries_.reserve(k); }

    /// Try to insert `(id, dist, is_new)` while preserving the
    /// sorted-ascending-by-distance invariant.
    ///
    /// Decision matrix:
    ///   * If `id` is already present: keep whichever copy has the
    ///     smaller distance; if `dist` ties, keep the existing
    ///     one's `is_new` flag (a duplicate insertion is not a
    ///     "new" event). Returns `false` because the list's
    ///     contents did not change.
    ///   * If the list is below capacity: insert in sorted order,
    ///     return `true`.
    ///   * If the list is at capacity and `dist` (with the same
    ///     `(dist, id)` lexicographic tie-break as `TopK`) is
    ///     better than the worst entry: evict the worst, insert
    ///     in sorted order, return `true`.
    ///   * Otherwise: reject, return `false`.
    ///
    /// The `bool` return value is what Step 33's convergence
    /// check counts: `n_updates += list.insert(...) ? 1 : 0`
    /// across every neighbour list, then divide by `n*k` to get
    /// the per-iteration update fraction.
    bool insert(index_t id, float dist, bool is_new);

    /// Linear scan checking whether `id` is already present.
    /// Used by callers that want to short-circuit a redundant
    /// distance computation.
    [[nodiscard]] bool contains(index_t id) const noexcept;

    /// Flip every entry's `is_new` flag to `false`. Called by the
    /// local-join after a point's neighbours have been processed
    /// so the next iteration skips them.
    void mark_all_old() noexcept;

    /// Read-only / mutable view of the sorted entries. The span's
    /// length is `size()`, never larger than `k`.
    [[nodiscard]] std::span<const Neighbor> view() const noexcept
    {
        return {entries_.data(), entries_.size()};
    }
    [[nodiscard]] std::span<Neighbor> view() noexcept
    {
        return {entries_.data(), entries_.size()};
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return entries_.size();
    }
    [[nodiscard]] std::size_t capacity() const noexcept { return k_; }
    [[nodiscard]] bool empty() const noexcept
    {
        return entries_.empty();
    }
    [[nodiscard]] bool full() const noexcept
    {
        return entries_.size() >= k_;
    }

    /// Worst (largest) distance currently held; `+inf` when the
    /// list is empty. Used by the local-join to early-reject any
    /// candidate that cannot improve the worst slot.
    [[nodiscard]] float worst_dist() const noexcept
    {
        if (entries_.empty()) {
            return std::numeric_limits<float>::infinity();
        }
        return entries_.back().dist;
    }

private:
    std::size_t           k_;
    std::vector<Neighbor> entries_;  // size <= k_, sorted ascending by dist
};

} // namespace knng::cpu
