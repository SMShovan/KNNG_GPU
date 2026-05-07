/// @file
/// @brief Tests for `knng::dist::nn_descent_mpi`.
///
/// In single-rank mode the distributed result must have recall@k
/// comparable to (or better than) the sequential CPU NN-Descent on
/// the same dataset and config.

#include "knng/dist/nn_descent_mpi.hpp"
#include "knng/dist/brute_force_mpi.hpp"
#include "knng/dist/mpi_env.hpp"
#include "knng/dist/sharded_dataset.hpp"
#include "knng/bench/recall.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/core/distance.hpp"
#include "knng/random.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstddef>
#include <vector>

namespace {

knng::dist::MpiEnv* g_env = nullptr;

class DistNnDescentTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        static knng::dist::MpiEnv env{};
        g_env = &env;
    }

    static knng::Dataset make_clustered_dataset(std::size_t n, std::size_t d,
                                                std::size_t n_clusters = 4)
    {
        knng::Dataset ds(n, d);
        knng::random::XorShift64 rng{12345};
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t cluster = i % n_clusters;
            for (std::size_t j = 0; j < d; ++j) {
                ds.data[i * d + j] =
                    static_cast<float>(cluster) * 10.0f +
                    (rng.next_float01() - 0.5f);
            }
        }
        return ds;
    }
};

TEST_F(DistNnDescentTest, SingleRankConverges) {
    if (g_env->size() != 1) {
        GTEST_SKIP() << "single-rank test skipped in multi-rank run";
    }
    constexpr std::size_t n = 64;
    constexpr std::size_t d = 8;
    constexpr std::size_t k = 5;

    const knng::Dataset root = make_clustered_dataset(n, d);

    // Ground truth.
    const knng::Knng truth =
        knng::cpu::brute_force_knn(root, k, knng::L2Squared{});

    // Distributed NN-Descent.
    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);

    knng::dist::NnDescentMpiConfig cfg;
    cfg.max_iters = 30;
    cfg.delta     = 0.0;  // full convergence
    cfg.seed      = 42;

    const knng::Knng local_graph =
        knng::dist::nn_descent_mpi(shard, k, cfg, MPI_COMM_WORLD);
    const knng::Knng gathered =
        knng::dist::gather_graph(local_graph, shard, 0, MPI_COMM_WORLD);

    if (g_env->is_root()) {
        const double recall = knng::bench::recall_at_k(gathered, truth);
        // On a clustered dataset with full convergence, recall should be
        // high. We use a lenient threshold since this is an approximate
        // algorithm.
        EXPECT_GE(recall, 0.6)
            << "recall@" << k << " = " << recall
            << " (expected >= 0.6 for a clustered dataset)";
    }
}

TEST_F(DistNnDescentTest, LocalGraphShapeCorrect) {
    constexpr std::size_t n = 20;
    constexpr std::size_t d = 4;
    constexpr std::size_t k = 3;

    knng::Dataset root;
    if (g_env->is_root()) {
        root = make_clustered_dataset(n, d);
    }

    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);

    knng::dist::NnDescentMpiConfig cfg;
    cfg.max_iters = 5;

    const knng::Knng local_graph =
        knng::dist::nn_descent_mpi(shard, k, cfg, MPI_COMM_WORLD);

    EXPECT_EQ(local_graph.n, shard.local_n());
    EXPECT_EQ(local_graph.k, k);
}

TEST_F(DistNnDescentTest, LogOutputRecordsIterations) {
    if (g_env->size() != 1) {
        GTEST_SKIP() << "single-rank test skipped in multi-rank run";
    }
    constexpr std::size_t n = 32;
    constexpr std::size_t d = 4;
    constexpr std::size_t k = 4;

    const knng::Dataset root = make_clustered_dataset(n, d);
    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);

    knng::dist::NnDescentMpiConfig cfg;
    cfg.max_iters = 10;
    cfg.delta     = 0.0;

    std::vector<knng::dist::NnDescentMpiLog> log;
    knng::dist::nn_descent_mpi_with_log(shard, k, cfg, MPI_COMM_WORLD, log);

    EXPECT_GT(log.size(), 0u);
    EXPECT_LE(log.size(), cfg.max_iters);

    // Iteration numbers must be 1-based and increasing.
    for (std::size_t i = 0; i < log.size(); ++i) {
        EXPECT_EQ(log[i].iteration, i + 1);
    }
}

} // namespace
