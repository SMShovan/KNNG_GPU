/// @file
/// @brief Implementation of the NN-Descent driver and helpers.
///
/// Step 31 ships:
///   * `NnDescentGraph`'s out-of-line members (constructor,
///     `at`, `to_knng`).
///   * `init_random_graph<D>` — explicitly instantiated for the
///     two built-in distance functors.
///
/// Subsequent Phase-5 steps (32–36) will add to this TU:
///   * The local-join kernel
///   * The convergence-driven driver
///   * Reverse neighbour lists
///   * Sampling
///   * The OpenMP-parallel variant

#include "knng/cpu/nn_descent.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "knng/random.hpp"

namespace knng::cpu {

NnDescentGraph::NnDescentGraph(std::size_t n, std::size_t k)
    : n_{n}, k_{k}
{
    lists_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        lists_.emplace_back(k);
    }
}

NeighborList& NnDescentGraph::at(std::size_t i) noexcept
{
    assert(i < n_);
    return lists_[i];
}

const NeighborList& NnDescentGraph::at(std::size_t i) const noexcept
{
    assert(i < n_);
    return lists_[i];
}

Knng NnDescentGraph::to_knng() const
{
    Knng out(n_, k_);
    for (std::size_t i = 0; i < n_; ++i) {
        const auto view = lists_[i].view();
        auto neighbors_row = out.neighbors_of(i);
        auto distances_row = out.distances_of(i);
        for (std::size_t j = 0; j < k_; ++j) {
            if (j < view.size()) {
                neighbors_row[j] = view[j].id;
                distances_row[j] = view[j].dist;
            } else {
                neighbors_row[j] = NeighborList::kEmptyId;
                distances_row[j] = std::numeric_limits<float>::infinity();
            }
        }
    }
    return out;
}

template <Distance D>
NnDescentGraph init_random_graph(const Dataset& ds,
                                 std::size_t k,
                                 std::uint64_t seed,
                                 D distance)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::init_random_graph: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::init_random_graph: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::init_random_graph: k ("
            + std::to_string(k) + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }
    assert(ds.is_contiguous());

    NnDescentGraph graph(ds.n, k);
    knng::random::XorShift64 rng{seed};

    // Defensive cap: at most `4*k + 16` attempts per point. For the
    // common `k ≪ n` regime the expected attempts are ~k; the cap
    // keeps a degenerate input from looping forever and a future
    // contributor from accidentally introducing one.
    const std::size_t max_attempts = 4 * k + 16;

    for (std::size_t p = 0; p < ds.n; ++p) {
        NeighborList& list = graph.at(p);
        const auto a = ds.row(p);
        std::size_t attempts = 0;
        while (list.size() < k && attempts < max_attempts) {
            ++attempts;
            const auto raw =
                static_cast<index_t>(rng.next_below(ds.n));
            if (raw == static_cast<index_t>(p)) {
                continue;  // skip self
            }
            if (list.contains(raw)) {
                continue;  // duplicate; saves the distance call
            }
            const float d = distance(a, ds.row(raw));
            list.insert(raw, d, /*is_new=*/true);
        }
        // The `n - 1 >= k` precondition makes the loop terminate
        // with `list.size() == k` in expectation; the assert
        // documents the invariant for the regression tests.
        assert(list.size() == k);
    }

    return graph;
}

template NnDescentGraph init_random_graph<L2Squared>(
    const Dataset&, std::size_t, std::uint64_t, L2Squared);

template NnDescentGraph init_random_graph<NegativeInnerProduct>(
    const Dataset&, std::size_t, std::uint64_t, NegativeInnerProduct);

namespace {

/// Snapshot every point's new / old ids and flip every list
/// entry to `is_new = false`. Shared by `local_join` and
/// `local_join_with_reverse`.
void snapshot_and_age(NnDescentGraph& graph,
                      std::vector<std::vector<index_t>>& new_ids,
                      std::vector<std::vector<index_t>>& old_ids)
{
    const std::size_t n = graph.n();
    new_ids.assign(n, {});
    old_ids.assign(n, {});
    for (std::size_t p = 0; p < n; ++p) {
        const auto view = graph.at(p).view();
        new_ids[p].reserve(view.size());
        old_ids[p].reserve(view.size());
        for (const Neighbor& nb : view) {
            if (nb.is_new) {
                new_ids[p].push_back(nb.id);
            } else {
                old_ids[p].push_back(nb.id);
            }
        }
        graph.at(p).mark_all_old();
    }
}

/// Enumerate `(new × new, u < v)` and `(new × old)` pairs and
/// offer each distance to both endpoints' lists. Returns the
/// number of `insert` calls that changed a row. Shared by both
/// local-join variants — the reverse-list variant just hands a
/// larger `nv` / `ov` after merging the snapshots.
template <Distance D>
std::size_t join_pairs(const Dataset& ds,
                        NnDescentGraph& graph,
                        const std::vector<index_t>& nv,
                        const std::vector<index_t>& ov,
                        D distance)
{
    std::size_t updates = 0;

    // new × new — only u < v to avoid double-visiting a pair
    // within this point's candidate set.
    for (std::size_t i = 0; i + 1 < nv.size(); ++i) {
        const index_t u = nv[i];
        const auto    a = ds.row(u);
        for (std::size_t j = i + 1; j < nv.size(); ++j) {
            const index_t v = nv[j];
            if (u == v) {
                continue;
            }
            const float d = distance(a, ds.row(v));
            if (graph.at(u).insert(v, d, /*is_new=*/true)) {
                ++updates;
            }
            if (graph.at(v).insert(u, d, /*is_new=*/true)) {
                ++updates;
            }
        }
    }

    // new × old — disjoint by definition (an entry is either
    // is_new or not), so no duplication concern.
    for (const index_t u : nv) {
        const auto a = ds.row(u);
        for (const index_t v : ov) {
            if (u == v) {
                continue;
            }
            const float d = distance(a, ds.row(v));
            if (graph.at(u).insert(v, d, /*is_new=*/true)) {
                ++updates;
            }
            if (graph.at(v).insert(u, d, /*is_new=*/true)) {
                ++updates;
            }
        }
    }

    return updates;
}

/// Sort + `unique` an id vector in place. Used to deduplicate
/// the merged forward / reverse candidate sets.
inline void sort_unique(std::vector<index_t>& v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// Convert a `rho ∈ (0, 1]` rate to an effective sample size at
/// most `k`. Always returns at least 1 when `k > 0` so a
/// pathologically small `rho` does not silently zero out the
/// per-iteration work.
[[nodiscard]] std::size_t rho_to_sample_size(double rho,
                                              std::size_t k) noexcept
{
    if (k == 0) {
        return 0;
    }
    const double scaled = rho * static_cast<double>(k);
    auto out = static_cast<std::size_t>(scaled);
    if (out == 0 && rho > 0.0) {
        out = 1;
    }
    if (out > k) {
        out = k;
    }
    return out;
}

/// Partial Fisher-Yates over a vector of positions. Picks `m`
/// distinct entries uniformly at random; the picked entries
/// land in the prefix `[0, m)` of `pool` after the call. The
/// rest of `pool` is left in an arbitrary state (we do not
/// rely on it). `O(m)` work; `O(1)` extra memory.
void partial_fisher_yates(std::vector<std::size_t>& pool,
                          std::size_t m,
                          knng::random::XorShift64& rng) noexcept
{
    const std::size_t n = pool.size();
    if (m >= n) {
        return;  // pool already represents the full sample
    }
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t j =
            i + static_cast<std::size_t>(rng.next_below(n - i));
        std::swap(pool[i], pool[j]);
    }
}

/// Snapshot every point's new / old ids with `rho`-sampling and
/// flip *only the sampled new* entries to `is_new = false`. The
/// unsampled new entries remain `is_new = true` so they are
/// eligible for sampling in subsequent iterations. `old` entries
/// are also subsampled but their `is_new` state is unchanged
/// (they are already old).
void snapshot_and_age_sampled(NnDescentGraph& graph,
                               double rho,
                               knng::random::XorShift64& rng,
                               std::vector<std::vector<index_t>>& new_ids,
                               std::vector<std::vector<index_t>>& old_ids)
{
    const std::size_t n = graph.n();
    const std::size_t k = graph.k();
    const std::size_t sample_size = rho_to_sample_size(rho, k);

    new_ids.assign(n, {});
    old_ids.assign(n, {});

    // Scratch position buffers, reused across points.
    std::vector<std::size_t> new_positions;
    std::vector<std::size_t> old_positions;
    new_positions.reserve(k);
    old_positions.reserve(k);

    for (std::size_t p = 0; p < n; ++p) {
        new_positions.clear();
        old_positions.clear();
        // We can't hold a span across the mutation below, so
        // capture positions and ids up front.
        {
            const auto view = graph.at(p).view();
            for (std::size_t i = 0; i < view.size(); ++i) {
                if (view[i].is_new) {
                    new_positions.push_back(i);
                } else {
                    old_positions.push_back(i);
                }
            }
        }

        // Subsample positions.
        partial_fisher_yates(new_positions, sample_size, rng);
        partial_fisher_yates(old_positions, sample_size, rng);
        const std::size_t new_take =
            std::min(sample_size, new_positions.size());
        const std::size_t old_take =
            std::min(sample_size, old_positions.size());

        // Capture sampled ids.
        new_ids[p].reserve(new_take);
        old_ids[p].reserve(old_take);
        const auto view = graph.at(p).view();
        for (std::size_t i = 0; i < new_take; ++i) {
            new_ids[p].push_back(view[new_positions[i]].id);
        }
        for (std::size_t i = 0; i < old_take; ++i) {
            old_ids[p].push_back(view[old_positions[i]].id);
        }

        // Flip only the sampled-new positions to `is_new = false`.
        // Unsampled-new entries remain new and may be picked by a
        // later iteration's sampler.
        auto mut_view = graph.at(p).view();
        for (std::size_t i = 0; i < new_take; ++i) {
            mut_view[new_positions[i]].is_new = false;
        }
    }
}

} // namespace

template <Distance D>
std::size_t local_join(const Dataset& ds,
                        NnDescentGraph& graph,
                        D distance)
{
    const std::size_t n = ds.n;
    if (graph.n() != n) {
        throw std::invalid_argument(
            "knng::cpu::local_join: graph.n != ds.n");
    }
    assert(ds.is_contiguous());

    std::vector<std::vector<index_t>> new_ids;
    std::vector<std::vector<index_t>> old_ids;
    snapshot_and_age(graph, new_ids, old_ids);

    std::size_t updates = 0;
    for (std::size_t p = 0; p < n; ++p) {
        updates += join_pairs(ds, graph, new_ids[p], old_ids[p],
                              distance);
    }
    return updates;
}

template <Distance D>
std::size_t local_join_with_reverse(const Dataset& ds,
                                     NnDescentGraph& graph,
                                     D distance)
{
    const std::size_t n = ds.n;
    if (graph.n() != n) {
        throw std::invalid_argument(
            "knng::cpu::local_join_with_reverse: graph.n != ds.n");
    }
    assert(ds.is_contiguous());

    // Phase 1: snapshot + age, identical to plain local-join.
    std::vector<std::vector<index_t>> new_ids;
    std::vector<std::vector<index_t>> old_ids;
    snapshot_and_age(graph, new_ids, old_ids);

    // Phase 2: build per-point reverse-new and reverse-old lists.
    // For every entry `q ∈ new_ids[p]`, push `p` into
    // `rev_new[q]`. Same for old. The result: for any `q`,
    // `rev_new[q]` is the set of points whose *new* neighbour
    // list (this iteration's snapshot) contains `q`.
    std::vector<std::vector<index_t>> rev_new(n);
    std::vector<std::vector<index_t>> rev_old(n);
    for (std::size_t p = 0; p < n; ++p) {
        const index_t pi = static_cast<index_t>(p);
        for (const index_t u : new_ids[p]) {
            rev_new[u].push_back(pi);
        }
        for (const index_t u : old_ids[p]) {
            rev_old[u].push_back(pi);
        }
    }

    // Phase 3: per-point local-join over the unioned candidate
    // sets. Scratch vectors live outside the loop so the
    // allocator can amortise capacity across points.
    std::size_t updates = 0;
    std::vector<index_t> nv_total;
    std::vector<index_t> ov_total;
    for (std::size_t p = 0; p < n; ++p) {
        nv_total.clear();
        ov_total.clear();
        nv_total.insert(nv_total.end(),
                        new_ids[p].begin(), new_ids[p].end());
        nv_total.insert(nv_total.end(),
                        rev_new[p].begin(), rev_new[p].end());
        ov_total.insert(ov_total.end(),
                        old_ids[p].begin(), old_ids[p].end());
        ov_total.insert(ov_total.end(),
                        rev_old[p].begin(), rev_old[p].end());
        // Mutual-neighbour pairs (`p ↔ q`) appear from both
        // directions; the union can also accidentally place an
        // id in both new and old totals when one direction had
        // it as new and the other as old. Both cases need
        // deduplication.
        sort_unique(nv_total);
        sort_unique(ov_total);
        // Items in both totals: keep them in `new` only (the
        // new-flagged side wins so a fresh insertion from this
        // iteration's exchange propagates).
        if (!nv_total.empty() && !ov_total.empty()) {
            std::vector<index_t> dedup_old;
            dedup_old.reserve(ov_total.size());
            std::size_t i = 0;
            std::size_t j = 0;
            while (i < ov_total.size() && j < nv_total.size()) {
                if (ov_total[i] < nv_total[j]) {
                    dedup_old.push_back(ov_total[i]);
                    ++i;
                } else if (ov_total[i] > nv_total[j]) {
                    ++j;
                } else {
                    ++i;
                    ++j;  // present in both → drop from old
                }
            }
            for (; i < ov_total.size(); ++i) {
                dedup_old.push_back(ov_total[i]);
            }
            ov_total = std::move(dedup_old);
        }

        updates += join_pairs(ds, graph, nv_total, ov_total,
                              distance);
    }
    return updates;
}

template std::size_t local_join<L2Squared>(
    const Dataset&, NnDescentGraph&, L2Squared);

template std::size_t local_join<NegativeInnerProduct>(
    const Dataset&, NnDescentGraph&, NegativeInnerProduct);

template std::size_t local_join_with_reverse<L2Squared>(
    const Dataset&, NnDescentGraph&, L2Squared);

template std::size_t local_join_with_reverse<NegativeInnerProduct>(
    const Dataset&, NnDescentGraph&, NegativeInnerProduct);

template <Distance D>
std::size_t local_join_sampled(const Dataset& ds,
                                NnDescentGraph& graph,
                                double rho,
                                std::uint64_t iter_seed,
                                D distance)
{
    if (rho <= 0.0) {
        throw std::invalid_argument(
            "knng::cpu::local_join_sampled: rho must be > 0.0");
    }
    if (graph.n() != ds.n) {
        throw std::invalid_argument(
            "knng::cpu::local_join_sampled: graph.n != ds.n");
    }
    assert(ds.is_contiguous());

    knng::random::XorShift64 rng{iter_seed};
    std::vector<std::vector<index_t>> new_ids;
    std::vector<std::vector<index_t>> old_ids;
    snapshot_and_age_sampled(graph, rho, rng, new_ids, old_ids);

    std::size_t updates = 0;
    for (std::size_t p = 0; p < ds.n; ++p) {
        updates += join_pairs(ds, graph, new_ids[p], old_ids[p],
                              distance);
    }
    return updates;
}

template <Distance D>
std::size_t local_join_with_reverse_sampled(const Dataset& ds,
                                              NnDescentGraph& graph,
                                              double rho,
                                              std::uint64_t iter_seed,
                                              D distance)
{
    if (rho <= 0.0) {
        throw std::invalid_argument(
            "knng::cpu::local_join_with_reverse_sampled: rho must be > 0.0");
    }
    if (graph.n() != ds.n) {
        throw std::invalid_argument(
            "knng::cpu::local_join_with_reverse_sampled: graph.n != ds.n");
    }
    assert(ds.is_contiguous());

    knng::random::XorShift64 rng{iter_seed};
    const std::size_t n = ds.n;

    std::vector<std::vector<index_t>> new_ids;
    std::vector<std::vector<index_t>> old_ids;
    snapshot_and_age_sampled(graph, rho, rng, new_ids, old_ids);

    // Reverse lists are built from the (sampled) snapshots, so
    // they too are proportionally smaller. The same `O(n*k)`
    // walk as the unsampled variant, with a smaller constant.
    std::vector<std::vector<index_t>> rev_new(n);
    std::vector<std::vector<index_t>> rev_old(n);
    for (std::size_t p = 0; p < n; ++p) {
        const index_t pi = static_cast<index_t>(p);
        for (const index_t u : new_ids[p]) {
            rev_new[u].push_back(pi);
        }
        for (const index_t u : old_ids[p]) {
            rev_old[u].push_back(pi);
        }
    }

    // The reverse lists may themselves be larger than `rho * k`
    // when many points list `q` as a (sampled) neighbour. The
    // standard NN-Descent practice is to subsample reverse to
    // `rho * k` too. We pick uniformly via the same partial
    // Fisher-Yates as the forward sampler.
    const std::size_t sample_size =
        rho_to_sample_size(rho, graph.k());

    std::size_t updates = 0;
    std::vector<index_t> nv_total;
    std::vector<index_t> ov_total;
    std::vector<std::size_t> rev_positions;  // scratch
    auto subsample_into = [&](std::vector<index_t>& dst,
                              std::vector<index_t>& src)
    {
        if (src.size() > sample_size) {
            rev_positions.resize(src.size());
            for (std::size_t i = 0; i < src.size(); ++i) {
                rev_positions[i] = i;
            }
            partial_fisher_yates(rev_positions, sample_size, rng);
            for (std::size_t i = 0; i < sample_size; ++i) {
                dst.push_back(src[rev_positions[i]]);
            }
        } else {
            dst.insert(dst.end(), src.begin(), src.end());
        }
    };

    for (std::size_t p = 0; p < n; ++p) {
        nv_total.clear();
        ov_total.clear();
        nv_total.insert(nv_total.end(),
                        new_ids[p].begin(), new_ids[p].end());
        subsample_into(nv_total, rev_new[p]);
        ov_total.insert(ov_total.end(),
                        old_ids[p].begin(), old_ids[p].end());
        subsample_into(ov_total, rev_old[p]);

        sort_unique(nv_total);
        sort_unique(ov_total);
        if (!nv_total.empty() && !ov_total.empty()) {
            std::vector<index_t> dedup_old;
            dedup_old.reserve(ov_total.size());
            std::size_t i = 0;
            std::size_t j = 0;
            while (i < ov_total.size() && j < nv_total.size()) {
                if (ov_total[i] < nv_total[j]) {
                    dedup_old.push_back(ov_total[i]);
                    ++i;
                } else if (ov_total[i] > nv_total[j]) {
                    ++j;
                } else {
                    ++i;
                    ++j;
                }
            }
            for (; i < ov_total.size(); ++i) {
                dedup_old.push_back(ov_total[i]);
            }
            ov_total = std::move(dedup_old);
        }

        updates += join_pairs(ds, graph, nv_total, ov_total,
                              distance);
    }
    return updates;
}

template std::size_t local_join_sampled<L2Squared>(
    const Dataset&, NnDescentGraph&, double, std::uint64_t, L2Squared);

template std::size_t local_join_sampled<NegativeInnerProduct>(
    const Dataset&, NnDescentGraph&, double, std::uint64_t,
    NegativeInnerProduct);

template std::size_t local_join_with_reverse_sampled<L2Squared>(
    const Dataset&, NnDescentGraph&, double, std::uint64_t, L2Squared);

template std::size_t local_join_with_reverse_sampled<NegativeInnerProduct>(
    const Dataset&, NnDescentGraph&, double, std::uint64_t,
    NegativeInnerProduct);

namespace {

/// The shared driver body used by both `nn_descent` and
/// `nn_descent_with_log`. `log_out` is optional — when non-null,
/// the function appends one entry per iteration run.
template <Distance D>
Knng nn_descent_impl(const Dataset& ds,
                      std::size_t k,
                      const NnDescentConfig& cfg,
                      D distance,
                      std::vector<NnDescentIterationLog>* log_out)
{
    if (cfg.delta < 0.0) {
        throw std::invalid_argument(
            "knng::cpu::nn_descent: cfg.delta must be non-negative");
    }
    if (cfg.rho <= 0.0) {
        throw std::invalid_argument(
            "knng::cpu::nn_descent: cfg.rho must be > 0.0");
    }
    // `init_random_graph` validates `(ds, k)`; further checks not
    // needed here.

    NnDescentGraph graph =
        init_random_graph(ds, k, cfg.seed, distance);

    if (log_out != nullptr) {
        log_out->clear();
        log_out->reserve(cfg.max_iters);
    }

    const double denom = static_cast<double>(ds.n)
                       * static_cast<double>(k);
    const double threshold_updates =
        cfg.delta * denom;  // updates count below which we stop

    // The driver picks one of four kernel variants based on
    // `(use_reverse, rho < 1.0)`. The `rho < 1.0` branch
    // computes a per-iteration seed by mixing `cfg.seed` with the
    // iteration index — the multiplier is the 64-bit
    // golden-ratio constant used by `splitmix64`-style hashers,
    // which gives good spread without a full hash.
    constexpr std::uint64_t kPhi = 0x9E3779B97F4A7C15ULL;
    const bool use_sampling = cfg.rho < 1.0;

    for (std::size_t it = 0; it < cfg.max_iters; ++it) {
        std::size_t updates = 0;
        if (use_sampling) {
            const std::uint64_t iter_seed =
                cfg.seed ^ (static_cast<std::uint64_t>(it + 1) * kPhi);
            updates = cfg.use_reverse
                ? local_join_with_reverse_sampled(
                      ds, graph, cfg.rho, iter_seed, distance)
                : local_join_sampled(
                      ds, graph, cfg.rho, iter_seed, distance);
        } else {
            updates = cfg.use_reverse
                ? local_join_with_reverse(ds, graph, distance)
                : local_join(ds, graph, distance);
        }
        const double fraction = (denom > 0.0)
            ? static_cast<double>(updates) / denom
            : 0.0;
        if (log_out != nullptr) {
            log_out->push_back({it + 1, updates, fraction});
        }
        // Stop when the absolute update count drops below the
        // configured threshold. Comparing against the *count*
        // rather than the *fraction* avoids a redundant
        // floating-point op per iteration; the two are equivalent
        // because `denom > 0`.
        if (static_cast<double>(updates) < threshold_updates) {
            break;
        }
    }

    return graph.to_knng();
}

} // namespace

template <Distance D>
Knng nn_descent(const Dataset& ds,
                 std::size_t k,
                 const NnDescentConfig& cfg,
                 D distance)
{
    return nn_descent_impl(ds, k, cfg, distance, nullptr);
}

template <Distance D>
Knng nn_descent_with_log(
    const Dataset& ds,
    std::size_t k,
    const NnDescentConfig& cfg,
    std::vector<NnDescentIterationLog>& log_out,
    D distance)
{
    return nn_descent_impl(ds, k, cfg, distance, &log_out);
}

template Knng nn_descent<L2Squared>(
    const Dataset&, std::size_t, const NnDescentConfig&, L2Squared);

template Knng nn_descent<NegativeInnerProduct>(
    const Dataset&, std::size_t, const NnDescentConfig&,
    NegativeInnerProduct);

template Knng nn_descent_with_log<L2Squared>(
    const Dataset&, std::size_t, const NnDescentConfig&,
    std::vector<NnDescentIterationLog>&, L2Squared);

template Knng nn_descent_with_log<NegativeInnerProduct>(
    const Dataset&, std::size_t, const NnDescentConfig&,
    std::vector<NnDescentIterationLog>&, NegativeInnerProduct);

} // namespace knng::cpu
