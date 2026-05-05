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

#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

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

    // Phase 1: snapshot per-point new / old id sets, then flip
    // every entry to is_new=false. The snapshots live in vectors
    // on the call frame; we accept the `O(n * k)` allocation
    // because it avoids the alternative (mutating during the
    // local-join, which is order-dependent and harder to
    // parallelise).
    std::vector<std::vector<index_t>> new_ids(n);
    std::vector<std::vector<index_t>> old_ids(n);
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

    // Phase 2: local-join. For each point p, enumerate the
    // new × new and new × old pairs in p's neighbourhood and
    // offer the `d(u, v)` distance to both endpoints' lists.
    std::size_t updates = 0;
    for (std::size_t p = 0; p < n; ++p) {
        const auto& nv = new_ids[p];
        const auto& ov = old_ids[p];

        // new × new — only u < v to avoid double-visiting a
        // pair within p's neighbourhood.
        for (std::size_t i = 0; i + 1 < nv.size(); ++i) {
            const index_t u = nv[i];
            const auto    a = ds.row(u);
            for (std::size_t j = i + 1; j < nv.size(); ++j) {
                const index_t v = nv[j];
                if (u == v) {
                    continue;  // defensive; the snapshot is
                               // distinct so this cannot fire,
                               // but it lets the assertion
                               // simplify.
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

        // new × old. The two sets are disjoint by construction
        // (an entry is either is_new or not), so we visit every
        // (u, v) pair exactly once with no duplication concern.
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
    }

    return updates;
}

template std::size_t local_join<L2Squared>(
    const Dataset&, NnDescentGraph&, L2Squared);

template std::size_t local_join<NegativeInnerProduct>(
    const Dataset&, NnDescentGraph&, NegativeInnerProduct);

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

    for (std::size_t it = 0; it < cfg.max_iters; ++it) {
        const std::size_t updates = local_join(ds, graph, distance);
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
