#pragma once

/// @file
/// @brief CPU NN-Descent: random init, local join, convergence driver.
///
/// This is the public interface for the CPU NN-Descent builder
/// (Wang et al. 2012; NEO-DNND paper). The header is built up
/// across Phase-5 steps:
///
///   * Step 31 (this file's first commit) — `NnDescentGraph` + the
///     random-graph initialiser.
///   * Step 32 — adds the local-join kernel.
///   * Step 33 — adds the convergence-driven driver.
///   * Step 34 — adds reverse-neighbour-list support.
///   * Step 35 — adds the `rho` sampling parameter.
///   * Step 36 — adds the OpenMP-parallel variant.
///
/// The container `NnDescentGraph` is a thin owner of
/// `std::vector<NeighborList>`. It exists so the public API can
/// hand a single object around (vs the algorithm taking
/// `std::vector<NeighborList>&` everywhere) and so a future GPU
/// port can ship its own `gpu::NnDescentGraph` against the same
/// shape. Conversion to the canonical `knng::Knng` (Step 03)
/// happens via `to_knng()`; consumers that just want the final
/// flat-and-sorted result never need to interact with the
/// per-point list type directly.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/cpu/neighbor_list.hpp"

namespace knng::cpu {

/// `n × k` collection of per-point `NeighborList`s. Owned by
/// value; copies are deep. The internal storage is a contiguous
/// `std::vector<NeighborList>` so iteration is cache-friendly and
/// `at(i)` is one bounds-checked subscript.
class NnDescentGraph {
public:
    /// Construct an `n`-row graph where every list has capacity
    /// `k`. All lists start empty; the caller is expected to
    /// populate them (`init_random_graph` is the canonical caller).
    NnDescentGraph(std::size_t n, std::size_t k);

    /// Mutable / read-only access to row `i`.
    [[nodiscard]] NeighborList& at(std::size_t i) noexcept;
    [[nodiscard]] const NeighborList& at(std::size_t i) const noexcept;

    /// Number of rows (points) in the graph.
    [[nodiscard]] std::size_t n() const noexcept { return n_; }

    /// Per-row capacity (neighbours per point).
    [[nodiscard]] std::size_t k() const noexcept { return k_; }

    /// Direct access to the underlying vector. Useful for the
    /// local-join kernel which iterates over every list, and for
    /// future SIMD / GPU variants that want raw bulk access.
    [[nodiscard]] std::vector<NeighborList>& lists() noexcept
    {
        return lists_;
    }
    [[nodiscard]] const std::vector<NeighborList>& lists() const noexcept
    {
        return lists_;
    }

    /// Convert to a flat `(n, k)` row-major `knng::Knng`. Each row
    /// is the corresponding `NeighborList::view()` — already
    /// sorted ascending by distance, tie-broken by ascending id.
    /// The conversion drops the `is_new` flag (the canonical
    /// `Knng` shape does not carry it). If a list has fewer than
    /// `k` entries, the missing slots are filled with the
    /// `NeighborList::kEmptyId` sentinel and `+inf` distance.
    [[nodiscard]] Knng to_knng() const;

private:
    std::size_t                n_;
    std::size_t                k_;
    std::vector<NeighborList>  lists_;
};

/// Random k-NN graph: every point gets `k` random distinct
/// non-self neighbours under the supplied distance functor, all
/// flagged `is_new = true`.
///
/// Deterministic for a given `seed` (uses
/// `knng::random::XorShift64` from Step 17). Same `(ds, k, seed)`
/// triple ⇒ bit-identical output across runs across platforms.
///
/// Sampling strategy: rejection sampling. For each point, draw
/// uniform random ids in `[0, n)`, skip self-matches and
/// duplicates (which `NeighborList::insert` would silently reject
/// anyway, but the explicit check saves the distance computation).
/// Expected attempts per slot: `1 + k / (n - k)`, which for the
/// `k ≪ n` regime NN-Descent runs in is ~1. A defensive bound on
/// total attempts per point prevents infinite loops on
/// pathologically-shaped inputs.
///
/// @param ds Reference / query set. Must be contiguous.
/// @param k Per-point neighbour count. Must satisfy
///          `1 ≤ k ≤ n - 1`.
/// @param seed RNG seed. Must be non-zero (XorShift64 fixed
///          point); zero seeds throw `std::invalid_argument` from
///          the underlying `XorShift64` constructor.
/// @param distance Distance functor satisfying `knng::Distance`.
///          Defaults to a default-constructed `D` when callable.
/// @return `NnDescentGraph` of shape `(ds.n, k)` with every row
///          filled with `k` distinct non-self neighbours, sorted
///          ascending by distance with ties broken by ascending
///          neighbour id, all `is_new = true`.
/// @throws std::invalid_argument on malformed inputs.
template <Distance D>
[[nodiscard]] NnDescentGraph init_random_graph(
    const Dataset& ds,
    std::size_t k,
    std::uint64_t seed,
    D distance = D{});

/// One iteration of NN-Descent's local-join kernel.
///
/// The algorithmic core of the entire builder. For each point `p`,
/// every pair `(u, v)` of `p`'s neighbours where at least one is
/// flagged `is_new = true` becomes a candidate distance computation;
/// the result is offered to both `u`'s and `v`'s lists. The `is_new`
/// flag (set up by Step 30 and seeded `true` by Step 31's random
/// initialiser) is what prunes the work — old × old pairs were
/// considered in a previous iteration and re-comparing them adds
/// no information.
///
/// Iteration shape (Wang et al. 2012 §4.1):
///
///   1. **Snapshot.** Walk every point. For each, partition its
///      current list into `new[p] = {id : entry.is_new}` and
///      `old[p] = {id : !entry.is_new}`. Then call
///      `mark_all_old()` so the *next* iteration sees only the
///      entries that get newly inserted *during* this one.
///   2. **Local-join.** For each point `p`, compute and insert
///      every `(u, v)` where:
///        * `u` and `v` are both in `new[p]` and `u < v` (the
///          `<` avoids visiting the same pair twice within `p`);
///        * `u ∈ new[p]` and `v ∈ old[p]` (no duplication concern
///          because the two sets are disjoint).
///      Old × old is *deliberately omitted* — that is the
///      optimisation Step 30's `is_new` flag exists to enable.
///
/// Inserts during phase 2 are flagged `is_new = true` so they are
/// picked up by the next iteration's snapshot.
///
/// Convergence-counting hook: returns the total number of
/// `NeighborList::insert` calls that *changed* a list (the bool
/// from Step 30). Step 33's driver compares the returned count to
/// `delta * n * k`; below the threshold means the graph has
/// stabilised.
///
/// Single-threaded by design; Step 36 will introduce the
/// OpenMP-parallel variant with per-point locks. Even single-
/// threaded the kernel is `O(n * k²)` per iteration vs
/// brute-force's `O(n²)`, so for `k ≪ n` the asymptotic win is
/// substantial.
///
/// @param ds Reference dataset. Must satisfy `ds.is_contiguous()`.
/// @param graph In/out graph; rows are mutated in place.
/// @param distance Distance functor satisfying `knng::Distance`.
/// @return Number of inserts that changed a row this iteration.
///         Step 33 uses this to decide convergence.
template <Distance D>
[[nodiscard]] std::size_t local_join(const Dataset& ds,
                                     NnDescentGraph& graph,
                                     D distance = D{});

} // namespace knng::cpu
