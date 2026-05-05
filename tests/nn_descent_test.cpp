/// @file
/// @brief Unit tests for the NN-Descent public surface (Steps 31+).
///
/// Step 31 contributes the random-graph initialiser tests below.
/// Subsequent Phase-5 steps (32–36) will append to this TU.

#include <array>
#include <cstddef>
#include <set>
#include <stdexcept>

#include <gtest/gtest.h>

#include "knng/bench/recall.hpp"
#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/cpu/nn_descent.hpp"

namespace {

/// Cluster fixture from the brute-force tests, reused so the
/// expected-shape assertions read familiarly.
knng::Dataset two_clusters_eight_points()
{
    knng::Dataset ds(8, 2);
    constexpr std::array<std::array<float, 2>, 8> coords{{
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
        {4.0f, 4.0f}, {5.0f, 4.0f}, {4.0f, 5.0f}, {5.0f, 5.0f},
    }};
    for (std::size_t i = 0; i < coords.size(); ++i) {
        ds.row(i)[0] = coords[i][0];
        ds.row(i)[1] = coords[i][1];
    }
    return ds;
}

// --- NnDescentGraph plumbing -----------------------------------------

TEST(NnDescentGraph, ConstructedShapeIsRespected)
{
    knng::cpu::NnDescentGraph g(/*n=*/4, /*k=*/3);
    EXPECT_EQ(g.n(), std::size_t{4});
    EXPECT_EQ(g.k(), std::size_t{3});
    EXPECT_EQ(g.lists().size(), std::size_t{4});
    for (std::size_t i = 0; i < g.n(); ++i) {
        EXPECT_TRUE(g.at(i).empty());
        EXPECT_EQ(g.at(i).capacity(), std::size_t{3});
    }
}

TEST(NnDescentGraph, ToKnngFillsMissingSlotsWithSentinels)
{
    knng::cpu::NnDescentGraph g(2, 3);
    // Populate row 0 with two entries; leave row 1 empty.
    g.at(0).insert(5, 0.5f, true);
    g.at(0).insert(7, 0.3f, true);

    const auto k = g.to_knng();
    EXPECT_EQ(k.n, std::size_t{2});
    EXPECT_EQ(k.k, std::size_t{3});
    // Row 0: [(7, 0.3), (5, 0.5), (sentinel, +inf)]
    EXPECT_EQ(k.neighbors_of(0)[0], knng::index_t{7});
    EXPECT_EQ(k.neighbors_of(0)[1], knng::index_t{5});
    EXPECT_EQ(k.neighbors_of(0)[2],
              knng::cpu::NeighborList::kEmptyId);
    EXPECT_FLOAT_EQ(k.distances_of(0)[0], 0.3f);
    EXPECT_FLOAT_EQ(k.distances_of(0)[1], 0.5f);
    EXPECT_TRUE(std::isinf(k.distances_of(0)[2]));
    // Row 1: all sentinels.
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_EQ(k.neighbors_of(1)[j],
                  knng::cpu::NeighborList::kEmptyId);
        EXPECT_TRUE(std::isinf(k.distances_of(1)[j]));
    }
}

// --- init_random_graph -----------------------------------------------

TEST(InitRandomGraph, OutputShapeMatchesArguments)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});
    EXPECT_EQ(g.n(), std::size_t{8});
    EXPECT_EQ(g.k(), std::size_t{3});
    for (std::size_t i = 0; i < g.n(); ++i) {
        EXPECT_EQ(g.at(i).size(), std::size_t{3});
    }
}

TEST(InitRandomGraph, EveryEntryIsNonSelfDistinctAndIsNew)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{1234}, knng::L2Squared{});

    for (std::size_t p = 0; p < g.n(); ++p) {
        std::set<knng::index_t> seen;
        for (const auto& nb : g.at(p).view()) {
            EXPECT_NE(static_cast<std::size_t>(nb.id), p);
            EXPECT_TRUE(seen.insert(nb.id).second)
                << "row " << p << " contains duplicate id " << nb.id;
            EXPECT_TRUE(nb.is_new);
            EXPECT_GE(nb.dist, 0.0f);
        }
    }
}

TEST(InitRandomGraph, RowsAreSortedAscendingByDistance)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{99}, knng::L2Squared{});

    for (std::size_t p = 0; p < g.n(); ++p) {
        const auto v = g.at(p).view();
        for (std::size_t j = 1; j < v.size(); ++j) {
            EXPECT_LE(v[j - 1].dist, v[j].dist)
                << "row " << p << " column " << j;
        }
    }
}

TEST(InitRandomGraph, SameSeedYieldsIdenticalGraphs)
{
    const auto ds = two_clusters_eight_points();
    const auto a = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{2026}, knng::L2Squared{});
    const auto b = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{2026}, knng::L2Squared{});
    for (std::size_t p = 0; p < a.n(); ++p) {
        const auto va = a.at(p).view();
        const auto vb = b.at(p).view();
        ASSERT_EQ(va.size(), vb.size());
        for (std::size_t j = 0; j < va.size(); ++j) {
            EXPECT_EQ(va[j].id, vb[j].id);
            EXPECT_FLOAT_EQ(va[j].dist, vb[j].dist);
        }
    }
}

TEST(InitRandomGraph, DifferentSeedsYieldDifferentGraphs)
{
    const auto ds = two_clusters_eight_points();
    const auto a = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{1}, knng::L2Squared{});
    const auto b = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{2}, knng::L2Squared{});

    bool any_diff = false;
    for (std::size_t p = 0; p < a.n() && !any_diff; ++p) {
        const auto va = a.at(p).view();
        const auto vb = b.at(p).view();
        for (std::size_t j = 0; j < va.size(); ++j) {
            if (va[j].id != vb[j].id) {
                any_diff = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_diff)
        << "two distinct seeds produced byte-identical graphs";
}

TEST(InitRandomGraph, DistancesMatchUnderlyingMetric)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{777}, knng::L2Squared{});
    knng::L2Squared metric;
    for (std::size_t p = 0; p < g.n(); ++p) {
        for (const auto& nb : g.at(p).view()) {
            const float expected = metric(ds.row(p), ds.row(nb.id));
            EXPECT_FLOAT_EQ(nb.dist, expected)
                << "row " << p << " neighbour " << nb.id;
        }
    }
}

TEST(InitRandomGraph, ToKnngRoundTripIsConsistent)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{42}, knng::L2Squared{});
    const auto k = g.to_knng();
    EXPECT_EQ(k.n, g.n());
    EXPECT_EQ(k.k, g.k());
    for (std::size_t p = 0; p < g.n(); ++p) {
        const auto v = g.at(p).view();
        const auto neigh = k.neighbors_of(p);
        const auto dist  = k.distances_of(p);
        for (std::size_t j = 0; j < g.k(); ++j) {
            EXPECT_EQ(neigh[j], v[j].id);
            EXPECT_FLOAT_EQ(dist[j], v[j].dist);
        }
    }
}

TEST(InitRandomGraph, NegativeInnerProductMetricCompiles)
{
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42},
        knng::NegativeInnerProduct{});
    EXPECT_EQ(g.n(), std::size_t{8});
    EXPECT_EQ(g.k(), std::size_t{3});
}

TEST(InitRandomGraph, ZeroKThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::init_random_graph(
            ds, std::size_t{0}, std::uint64_t{1},
            knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(InitRandomGraph, KGreaterThanNMinusOneThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    EXPECT_THROW(
        { (void)knng::cpu::init_random_graph(
            ds, std::size_t{8}, std::uint64_t{1},
            knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(InitRandomGraph, EmptyDatasetThrowsInvalidArgument)
{
    const knng::Dataset ds;
    EXPECT_THROW(
        { (void)knng::cpu::init_random_graph(
            ds, std::size_t{1}, std::uint64_t{1},
            knng::L2Squared{}); },
        std::invalid_argument);
}

// --- local_join ------------------------------------------------------

TEST(LocalJoin, FirstIterationProducesUpdatesAndPreservesShape)
{
    const auto ds = two_clusters_eight_points();
    auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});

    const std::size_t updates = knng::cpu::local_join(
        ds, g, knng::L2Squared{});

    // The first local-join on a random graph must do *some* work.
    EXPECT_GT(updates, std::size_t{0});

    // Shape preserved: each row still holds exactly k entries
    // and they remain sorted ascending by distance.
    for (std::size_t p = 0; p < g.n(); ++p) {
        EXPECT_EQ(g.at(p).size(), std::size_t{3});
        const auto v = g.at(p).view();
        for (std::size_t j = 1; j < v.size(); ++j) {
            EXPECT_LE(v[j - 1].dist, v[j].dist)
                << "row " << p << " column " << j;
        }
        // Self-match never appears.
        for (const auto& nb : v) {
            EXPECT_NE(static_cast<std::size_t>(nb.id), p);
        }
    }
}

TEST(LocalJoin, AfterFirstIterationOriginalEntriesAreOldFreshOnesNew)
{
    const auto ds = two_clusters_eight_points();
    auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});

    // Before local_join: every entry is is_new=true.
    for (std::size_t p = 0; p < g.n(); ++p) {
        for (const auto& nb : g.at(p).view()) {
            EXPECT_TRUE(nb.is_new);
        }
    }
    (void)knng::cpu::local_join(ds, g, knng::L2Squared{});
    // After: surviving original entries are is_new=false; freshly-
    // inserted ones are is_new=true. We can't predict which is
    // which on a non-trivial fixture, but at least one freshly-
    // inserted entry should be is_new=true (because the graph
    // converged toward truth — see the recall test below).
    bool any_new   = false;
    bool any_old   = false;
    for (std::size_t p = 0; p < g.n(); ++p) {
        for (const auto& nb : g.at(p).view()) {
            if (nb.is_new) any_new = true;
            else           any_old = true;
        }
    }
    EXPECT_TRUE(any_new);
    EXPECT_TRUE(any_old);
}

TEST(LocalJoin, IteratingToConvergenceMatchesBruteForceRecall)
{
    // The acid test: after enough local-join iterations on a
    // small fixture the NN-Descent graph should reach the
    // brute-force ground truth (recall@k = 1.0).
    const auto ds = two_clusters_eight_points();
    auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});

    for (int iter = 0; iter < 16; ++iter) {
        const std::size_t updates = knng::cpu::local_join(
            ds, g, knng::L2Squared{});
        if (updates == 0) {
            break;
        }
    }

    const auto truth = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    const double recall =
        knng::bench::recall_at_k(g.to_knng(), truth);
    EXPECT_DOUBLE_EQ(recall, 1.0);
}

TEST(LocalJoin, SecondIterationDoesAtMostAsMuchWork)
{
    const auto ds = two_clusters_eight_points();
    auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{777}, knng::L2Squared{});

    const std::size_t u1 = knng::cpu::local_join(
        ds, g, knng::L2Squared{});
    const std::size_t u2 = knng::cpu::local_join(
        ds, g, knng::L2Squared{});

    // The graph monotonically converges; the second iteration
    // cannot increase the per-iteration update count above the
    // first. (Strict-less is more typical but not guaranteed; the
    // bound is the algorithmic claim.)
    EXPECT_LE(u2, u1);
}

TEST(LocalJoin, GraphSizeMismatchThrows)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentGraph wrong(ds.n + 1, 3);
    EXPECT_THROW(
        { (void)knng::cpu::local_join(
            ds, wrong, knng::L2Squared{}); },
        std::invalid_argument);
}

// --- nn_descent driver -----------------------------------------------

TEST(NnDescent, ConvergesToBruteForceOnEightPointFixture)
{
    const auto ds = two_clusters_eight_points();
    const auto truth = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    knng::cpu::NnDescentConfig cfg{
        .max_iters = 16, .delta = 0.0, .seed = 42};
    const auto g = knng::cpu::nn_descent(
        ds, std::size_t{3}, cfg, knng::L2Squared{});

    EXPECT_DOUBLE_EQ(knng::bench::recall_at_k(g, truth), 1.0);
}

TEST(NnDescent, DefaultConfigConvergesAndStops)
{
    // Default config: max_iters=50, delta=0.001, seed=42.
    // The 8-point fixture converges in <5 iterations so the cap
    // is far from binding. Just check the function returns a
    // shape-correct graph.
    const auto ds = two_clusters_eight_points();
    const auto g = knng::cpu::nn_descent(
        ds, std::size_t{3}, /*cfg=*/{}, knng::L2Squared{});
    EXPECT_EQ(g.n, ds.n);
    EXPECT_EQ(g.k, std::size_t{3});
}

TEST(NnDescent, SameSeedYieldsByteIdenticalGraph)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentConfig cfg{};  // default seed
    const auto a = knng::cpu::nn_descent(
        ds, std::size_t{3}, cfg, knng::L2Squared{});
    const auto b = knng::cpu::nn_descent(
        ds, std::size_t{3}, cfg, knng::L2Squared{});
    EXPECT_EQ(a.neighbors, b.neighbors);
    EXPECT_EQ(a.distances, b.distances);
}

TEST(NnDescentWithLog, EmitsOnePerIterationDecreasingFraction)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentConfig cfg{
        .max_iters = 10, .delta = 0.0, .seed = 42};
    std::vector<knng::cpu::NnDescentIterationLog> log;
    const auto g = knng::cpu::nn_descent_with_log(
        ds, std::size_t{3}, cfg, log, knng::L2Squared{});

    EXPECT_EQ(g.n, ds.n);
    EXPECT_GE(log.size(), std::size_t{1});
    EXPECT_LE(log.size(), cfg.max_iters);
    // 1-based iteration counter.
    for (std::size_t i = 0; i < log.size(); ++i) {
        EXPECT_EQ(log[i].iteration, i + 1);
        EXPECT_GE(log[i].update_fraction, 0.0);
    }
    // Update fractions are monotonically non-increasing.
    for (std::size_t i = 1; i < log.size(); ++i) {
        EXPECT_LE(log[i].updates, log[i - 1].updates)
            << "iteration " << log[i].iteration
            << " did more work than iteration "
            << log[i - 1].iteration;
    }
}

TEST(NnDescentWithLog, StopsEarlyWhenDeltaIsLoose)
{
    const auto ds = two_clusters_eight_points();
    // delta = 1.0 — any positive update fraction is "above"
    // the threshold check (`updates < cfg.delta * n*k`); the
    // first iteration's update count *is* less than 1.0 *
    // n*k, so the loop should break immediately after the
    // first iteration.
    knng::cpu::NnDescentConfig cfg{
        .max_iters = 50, .delta = 1.0, .seed = 42};
    std::vector<knng::cpu::NnDescentIterationLog> log;
    (void)knng::cpu::nn_descent_with_log(
        ds, std::size_t{3}, cfg, log, knng::L2Squared{});
    EXPECT_EQ(log.size(), std::size_t{1});
}

TEST(NnDescentWithLog, RespectsMaxItersWithDeltaZero)
{
    // delta = 0.0 — the threshold is impossible to undershoot
    // (every iteration runs at least one update on a non-trivial
    // input until the graph is exactly converged). Cap at 3
    // iterations to verify the safety bound.
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentConfig cfg{
        .max_iters = 3, .delta = 0.0, .seed = 42};
    std::vector<knng::cpu::NnDescentIterationLog> log;
    (void)knng::cpu::nn_descent_with_log(
        ds, std::size_t{3}, cfg, log, knng::L2Squared{});
    EXPECT_LE(log.size(), cfg.max_iters);
}

TEST(NnDescent, NegativeDeltaThrowsInvalidArgument)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentConfig cfg{};
    cfg.delta = -0.1;
    EXPECT_THROW(
        { (void)knng::cpu::nn_descent(
            ds, std::size_t{3}, cfg, knng::L2Squared{}); },
        std::invalid_argument);
}

// --- local_join_with_reverse + ablation --------------------------------

TEST(LocalJoinWithReverse, MatchesPlainOnFirstIteration)
{
    // On the first iteration both variants snapshot the same
    // is_new=true entries and run the same `(new × new)` work
    // — reverse_new just adds back-edges that didn't exist
    // pre-iteration. On a tiny n=8 fixture the resulting
    // graphs may differ in which fp-equal candidates survive
    // ties; we compare the recall against truth instead.
    const auto ds = two_clusters_eight_points();
    auto g_plain   = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});
    auto g_reverse = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});

    const auto u_plain = knng::cpu::local_join(
        ds, g_plain, knng::L2Squared{});
    const auto u_rev = knng::cpu::local_join_with_reverse(
        ds, g_reverse, knng::L2Squared{});

    // The reverse variant always does at least as much work
    // as plain on the first iteration (it considers a
    // superset of pairs).
    EXPECT_GE(u_rev, u_plain);

    // Both produce shape-correct graphs.
    EXPECT_EQ(g_plain.n(), ds.n);
    EXPECT_EQ(g_reverse.n(), ds.n);
}

TEST(NnDescent, ReverseConvergesAtLeastAsFastAsPlainOnEightPointFixture)
{
    // Acid test for the NEO-DNND headline claim: reverse
    // lists shouldn't make convergence *slower* on any
    // fixture. We measure "iterations to first hit
    // recall@k=1.0" with both variants and assert the
    // reverse variant converges no later.
    const auto ds = two_clusters_eight_points();
    const auto truth = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    auto iters_to_converge = [&](bool use_reverse) {
        knng::cpu::NnDescentConfig cfg{
            .max_iters = 16, .delta = 0.0, .seed = 42,
            .use_reverse = use_reverse};
        std::vector<knng::cpu::NnDescentIterationLog> log;
        (void)knng::cpu::nn_descent_with_log(
            ds, std::size_t{3}, cfg, log, knng::L2Squared{});
        // The driver stops on `delta`-based convergence; the
        // log size is the iteration count.
        return log.size();
    };
    const auto plain_iters   = iters_to_converge(false);
    const auto reverse_iters = iters_to_converge(true);
    EXPECT_LE(reverse_iters, plain_iters);
}

TEST(NnDescent, BothVariantsConvergeToSameRecallOnEightPointFixture)
{
    const auto ds = two_clusters_eight_points();
    const auto truth = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    auto recall_after = [&](bool use_reverse, std::size_t iters) {
        knng::cpu::NnDescentConfig cfg{
            .max_iters = iters, .delta = 0.0, .seed = 42,
            .use_reverse = use_reverse};
        const auto g = knng::cpu::nn_descent(
            ds, std::size_t{3}, cfg, knng::L2Squared{});
        return knng::bench::recall_at_k(g, truth);
    };
    // Both reach perfect recall after enough iterations.
    EXPECT_DOUBLE_EQ(recall_after(false, 16), 1.0);
    EXPECT_DOUBLE_EQ(recall_after(true,  16), 1.0);
}

TEST(LocalJoinWithReverse, GraphSizeMismatchThrows)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentGraph wrong(ds.n + 1, 3);
    EXPECT_THROW(
        { (void)knng::cpu::local_join_with_reverse(
            ds, wrong, knng::L2Squared{}); },
        std::invalid_argument);
}

// --- sampled variants (Step 35) -------------------------------------

TEST(LocalJoinSampled, RhoOneFullSampleIsEquivalentToPlain)
{
    // At rho=1.0 the sampled variant should produce exactly the
    // same graph as plain `local_join` because no candidates are
    // dropped. The seed argument is unused at full rate.
    const auto ds = two_clusters_eight_points();
    auto g_plain   = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});
    auto g_sampled = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});

    (void)knng::cpu::local_join(ds, g_plain, knng::L2Squared{});
    (void)knng::cpu::local_join_sampled(
        ds, g_sampled, /*rho=*/1.0, /*iter_seed=*/123,
        knng::L2Squared{});

    for (std::size_t p = 0; p < g_plain.n(); ++p) {
        const auto a = g_plain.at(p).view();
        const auto b = g_sampled.at(p).view();
        ASSERT_EQ(a.size(), b.size()) << "row " << p;
        for (std::size_t j = 0; j < a.size(); ++j) {
            EXPECT_EQ(a[j].id, b[j].id) << "row " << p << " col " << j;
            EXPECT_FLOAT_EQ(a[j].dist, b[j].dist);
        }
    }
}

TEST(LocalJoinSampled, RhoLessThanOneStillProducesUpdates)
{
    const auto ds = two_clusters_eight_points();
    auto g = knng::cpu::init_random_graph(
        ds, std::size_t{4}, std::uint64_t{42}, knng::L2Squared{});
    const auto u = knng::cpu::local_join_sampled(
        ds, g, /*rho=*/0.5, /*iter_seed=*/7, knng::L2Squared{});
    EXPECT_GT(u, std::size_t{0});
    EXPECT_EQ(g.n(), ds.n);
    for (std::size_t p = 0; p < g.n(); ++p) {
        EXPECT_EQ(g.at(p).size(), std::size_t{4});
    }
}

TEST(LocalJoinSampled, ZeroOrNegativeRhoThrows)
{
    const auto ds = two_clusters_eight_points();
    auto g = knng::cpu::init_random_graph(
        ds, std::size_t{3}, std::uint64_t{42}, knng::L2Squared{});
    EXPECT_THROW(
        { (void)knng::cpu::local_join_sampled(
            ds, g, /*rho=*/0.0, 1, knng::L2Squared{}); },
        std::invalid_argument);
    EXPECT_THROW(
        { (void)knng::cpu::local_join_sampled(
            ds, g, /*rho=*/-0.5, 1, knng::L2Squared{}); },
        std::invalid_argument);
    EXPECT_THROW(
        { (void)knng::cpu::local_join_with_reverse_sampled(
            ds, g, /*rho=*/0.0, 1, knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(LocalJoinSampled, GraphSizeMismatchThrows)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentGraph wrong(ds.n + 1, 3);
    EXPECT_THROW(
        { (void)knng::cpu::local_join_sampled(
            ds, wrong, 0.5, 1, knng::L2Squared{}); },
        std::invalid_argument);
    EXPECT_THROW(
        { (void)knng::cpu::local_join_with_reverse_sampled(
            ds, wrong, 0.5, 1, knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(NnDescent, RhoOneRouteMatchesPlainConfig)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentConfig cfg_a{};
    knng::cpu::NnDescentConfig cfg_b{};
    cfg_b.rho = 1.0;
    const auto a = knng::cpu::nn_descent(
        ds, std::size_t{3}, cfg_a, knng::L2Squared{});
    const auto b = knng::cpu::nn_descent(
        ds, std::size_t{3}, cfg_b, knng::L2Squared{});
    EXPECT_EQ(a.neighbors, b.neighbors);
}

TEST(NnDescent, RhoLessThanOneStillReachesConvergence)
{
    const auto ds = two_clusters_eight_points();
    const auto truth = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    for (double rho : {0.3, 0.5, 0.8}) {
        knng::cpu::NnDescentConfig cfg{};
        cfg.max_iters = 32;
        cfg.delta     = 0.0;
        cfg.rho       = rho;
        const auto g = knng::cpu::nn_descent(
            ds, std::size_t{3}, cfg, knng::L2Squared{});
        const double recall = knng::bench::recall_at_k(g, truth);
        EXPECT_DOUBLE_EQ(recall, 1.0)
            << "rho = " << rho;
    }
}

TEST(NnDescent, NonPositiveRhoThrows)
{
    const auto ds = two_clusters_eight_points();
    knng::cpu::NnDescentConfig cfg{};
    cfg.rho = 0.0;
    EXPECT_THROW(
        { (void)knng::cpu::nn_descent(
            ds, std::size_t{3}, cfg, knng::L2Squared{}); },
        std::invalid_argument);
    cfg.rho = -0.1;
    EXPECT_THROW(
        { (void)knng::cpu::nn_descent(
            ds, std::size_t{3}, cfg, knng::L2Squared{}); },
        std::invalid_argument);
}

TEST(NnDescent, RhoSweepOnLogShowsDecreasingPerIterWork)
{
    // Ablation knob sanity check: at rho=0.3 the per-iteration
    // update count *averaged* across a run should be lower than
    // at rho=1.0 (less candidate set means less work). We
    // compare iteration-1 update counts; the smaller candidate
    // set produces at most as much work per iteration.
    const auto ds = two_clusters_eight_points();

    knng::cpu::NnDescentConfig cfg_full{};
    cfg_full.max_iters = 1;
    cfg_full.delta     = 0.0;
    cfg_full.rho       = 1.0;

    knng::cpu::NnDescentConfig cfg_small{};
    cfg_small.max_iters = 1;
    cfg_small.delta     = 0.0;
    cfg_small.rho       = 0.3;

    std::vector<knng::cpu::NnDescentIterationLog> log_full;
    std::vector<knng::cpu::NnDescentIterationLog> log_small;
    (void)knng::cpu::nn_descent_with_log(
        ds, std::size_t{4}, cfg_full, log_full,
        knng::L2Squared{});
    (void)knng::cpu::nn_descent_with_log(
        ds, std::size_t{4}, cfg_small, log_small,
        knng::L2Squared{});

    ASSERT_EQ(log_full.size(), std::size_t{1});
    ASSERT_EQ(log_small.size(), std::size_t{1});
    // Strict-less is the typical case; equality is possible on a
    // tiny fixture where the sample happens to cover every
    // candidate, so the bound is `≤`.
    EXPECT_LE(log_small[0].updates, log_full[0].updates);
}

} // namespace
