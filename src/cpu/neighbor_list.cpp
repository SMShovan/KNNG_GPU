/// @file
/// @brief Out-of-line members for `knng::cpu::NeighborList` (Step 30).
///
/// The header keeps the small accessor inline; the non-trivial
/// members — `insert`, `contains`, `mark_all_old` — live here so a
/// future SIMD / GPU specialisation can swap the implementation
/// without touching consumer code. Today everything is plain scalar
/// loops over a `std::vector<Neighbor>`.

#include "knng/cpu/neighbor_list.hpp"

#include <algorithm>
#include <cstddef>

namespace knng::cpu {

bool NeighborList::insert(index_t id, float dist, bool is_new)
{
    if (k_ == 0) {
        return false;
    }

    // Reject duplicate ids. Two cases:
    //   * existing dist <= new dist: keep the existing entry
    //     (it is at least as good and its is_new flag carries
    //     the canonical answer).
    //   * existing dist > new dist: replace the dist + is_new on
    //     the existing entry. We re-sort by lifting it out and
    //     reinserting via the normal path.
    auto existing = std::find_if(
        entries_.begin(), entries_.end(),
        [id](const Neighbor& e) { return e.id == id; });
    if (existing != entries_.end()) {
        if (existing->dist <= dist) {
            // Existing copy already at least as good — no change.
            return false;
        }
        // Lift, then fall through to the standard sorted-insert.
        // The list size briefly drops by 1 so the
        // capacity-and-eviction logic below works unchanged.
        entries_.erase(existing);
    }

    // Find the insertion position by lexicographic `(dist, id)`,
    // matching the `TopK` tie-break: smaller dist first; on tie,
    // smaller id first.
    auto lt = [](const Neighbor& e, std::pair<float, index_t> key) {
        if (e.dist != key.first) {
            return e.dist < key.first;
        }
        return e.id < key.second;
    };
    const std::pair<float, index_t> key{dist, id};
    auto pos = std::lower_bound(
        entries_.begin(), entries_.end(), key, lt);

    if (entries_.size() < k_) {
        entries_.insert(pos, Neighbor{id, dist, is_new});
        return true;
    }

    // At capacity. The new candidate beats the worst iff it sorts
    // strictly before the last element under the same comparator.
    const Neighbor& worst = entries_.back();
    const bool dominates =
        (dist < worst.dist) ||
        (dist == worst.dist && id < worst.id);
    if (!dominates) {
        return false;
    }

    // Evict the worst and insert. `pos` was computed before the
    // pop_back; both operations happen on slots strictly before
    // the popped element, so `pos` remains a valid iterator into
    // the shrunken vector unless it pointed to the soon-to-be-
    // popped tail. Recompute defensively.
    entries_.pop_back();
    pos = std::lower_bound(entries_.begin(), entries_.end(), key, lt);
    entries_.insert(pos, Neighbor{id, dist, is_new});
    return true;
}

bool NeighborList::contains(index_t id) const noexcept
{
    for (const Neighbor& n : entries_) {
        if (n.id == id) {
            return true;
        }
    }
    return false;
}

void NeighborList::mark_all_old() noexcept
{
    for (Neighbor& n : entries_) {
        n.is_new = false;
    }
}

} // namespace knng::cpu
