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

/// One iteration of NN-Descent's local-join kernel *with reverse
/// neighbour lists* added to the candidate set (Step 34).
///
/// Wang et al. 2012 §4.2 / NEO-DNND §3.1 observation: a point's
/// neighbour-of-neighbour relation is *not* symmetric under
/// finite `k`. If `q` lists `p` in its top-`k`, that does not
/// guarantee `p` lists `q` in its own top-`k`. The plain
/// local-join (Step 32) only walks `p`'s forward neighbours; if
/// `p` happens to have an excellent candidate sitting in some
/// other point's list (where `p` is the candidate), the plain
/// local-join can miss it for many iterations.
///
/// The reverse-list variant fixes this by tracking, for each
/// point `p`, the *reverse* graph
/// `R(p) = { q : p ∈ neighbours(q) }`. The local-join then
/// processes `(u, v)` pairs drawn from
/// `neighbours(p) ∪ reverse_neighbours(p)`, which is the set of
/// "every point that knows about `p` or that `p` knows about" —
/// the algorithmic equivalent of "ask both sides of the
/// neighbour relation."
///
/// The cost is one extra `O(n * k)` pass per iteration to build
/// the reverse lists from the per-point new/old snapshots. The
/// payoff is a substantial recall acceleration: a graph that
/// took 12 iterations under plain local-join often converges in
/// 5–6 with reverse lists.
///
/// Implementation: builds two reverse-list arrays
/// (`rev_new`, `rev_old`) from the snapshots, unions each into
/// the local-join's candidate sets per point, deduplicates by
/// sort + unique (the union can contain duplicates because
/// `p ↔ q` mutual neighbours appear from both directions), then
/// runs the same `(new × new, u < v)` + `(new × old)` pair
/// enumeration as Step 32.
///
/// @return Number of `NeighborList::insert` calls that changed a
///          row this iteration. Same convergence-counter
///          contract as `local_join`.
template <Distance D>
[[nodiscard]] std::size_t local_join_with_reverse(
    const Dataset& ds,
    NnDescentGraph& graph,
    D distance = D{});

/// Tunable knobs for the convergence-driven NN-Descent driver.
struct NnDescentConfig {
    /// Hard cap on iterations. The convergence criterion is the
    /// usual stopping condition; `max_iters` is the safety bound
    /// that prevents an unstable input from spinning forever.
    /// Default `50` is generous — typical SIFT1M-scale runs
    /// converge in 10–15 iterations.
    std::size_t max_iters = 50;

    /// Convergence threshold on the per-iteration update fraction
    /// `n_updates / (n * k)`. Below this, the graph is declared
    /// stable and the driver returns. Default `0.001` matches
    /// Wang et al. 2012 §4.1; it corresponds to "fewer than 0.1%
    /// of the total `(n*k)` neighbour slots changed in the last
    /// iteration."
    ///
    /// Smaller `delta` → tighter convergence → higher recall at
    /// the cost of more iterations. Larger `delta` → looser
    /// convergence → lower recall but faster build. The
    /// recall@k vs `delta` curve is approximately monotone, with
    /// diminishing returns below ~0.001.
    double delta = 0.001;

    /// RNG seed for `init_random_graph` (passed straight through).
    /// Same seed → bit-identical output graph; the driver never
    /// introduces additional randomness beyond initialisation.
    std::uint64_t seed = 42;

    /// When `true` (default), the driver runs `local_join_with_reverse`
    /// each iteration; when `false`, the plain `local_join`. Reverse
    /// lists are the NEO-DNND recall-acceleration headline; turning
    /// them off is rarely the right call but is supported for
    /// pedagogy and ablation studies.
    bool use_reverse = true;
};

/// Per-iteration statistics emitted by `nn_descent_with_log`. The
/// shape is what the bench harness JSON wants to consume: a list
/// of `(iteration, updates, update_fraction)` rows that captures
/// the convergence curve for plotting.
struct NnDescentIterationLog {
    std::size_t iteration;        ///< 1-based iteration index.
    std::size_t updates;          ///< `local_join` return value.
    double      update_fraction;  ///< `updates / (n * k)`.
};

/// Run NN-Descent to convergence and return the resulting
/// `(n × k)` `Knng`. Wraps `init_random_graph` + iterated
/// `local_join` with the configured stopping rule.
///
/// @param ds Reference dataset. Must be contiguous.
/// @param k Per-point neighbour count.
/// @param cfg Tuning knobs (`max_iters`, `delta`, `seed`).
/// @param distance Distance functor.
/// @return Final `Knng`. Rows sorted ascending by distance.
template <Distance D>
[[nodiscard]] Knng nn_descent(const Dataset& ds,
                              std::size_t k,
                              const NnDescentConfig& cfg = {},
                              D distance = D{});

/// Same as `nn_descent` but additionally writes a per-iteration
/// log to `log_out`. The vector is `clear()`-ed first; on return
/// it has one entry per iteration actually run (so its size is
/// `≤ cfg.max_iters`). Intended for the bench harness's
/// "convergence curve" JSON output and for tests that want to
/// pin `iterations_run`.
template <Distance D>
[[nodiscard]] Knng nn_descent_with_log(
    const Dataset& ds,
    std::size_t k,
    const NnDescentConfig& cfg,
    std::vector<NnDescentIterationLog>& log_out,
    D distance = D{});

} // namespace knng::cpu
