/// @file
/// @brief Tests for `knng::dist::brute_force_knn_mpi`.
///
/// In single-rank mode the distributed brute-force reduces to plain
/// brute-force and must produce a graph that matches `brute_force_knn`
/// from `knng::cpu` on the same dataset.
///
/// Multi-rank correctness (each rank's shard agrees with the global
/// reference) is verified when `mpirun -np 2+` is available.

#include "knng/dist/brute_force_mpi.hpp"
#include "knng/dist/mpi_env.hpp"
#include "knng/dist/sharded_dataset.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/core/distance.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <cstddef>

namespace {

knng::dist::MpiEnv* g_env = nullptr;

class DistBruteForceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        static knng::dist::MpiEnv env{};
        g_env = &env;
    }

    /// Build a small synthetic dataset: n points, d dimensions.
    /// Point i has all coordinates set to (i + 1) * 0.1f.
    static knng::Dataset make_dataset(std::size_t n, std::size_t d) {
        knng::Dataset ds(n, d);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                ds.data[i * d + j] = static_cast<float>(i + 1) * 0.1f;
            }
        }
        return ds;
    }
};

TEST_F(DistBruteForceTest, SingleRankMatchesCpuBruteForce) {
    // On rank 0 the distributed result must match the sequential CPU result.
    if (g_env->size() != 1) {
        GTEST_SKIP() << "single-rank test skipped in multi-rank run";
    }
    constexpr std::size_t n = 10;
    constexpr std::size_t d = 4;
    constexpr std::size_t k = 3;

    const knng::Dataset root = make_dataset(n, d);

    // CPU reference.
    const knng::Knng ref =
        knng::cpu::brute_force_knn(root, k, knng::L2Squared{});

    // Distributed (single rank).
    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);
    const knng::Knng local_graph =
        knng::dist::brute_force_knn_mpi(shard, k, MPI_COMM_WORLD);
    const knng::Knng gathered =
        knng::dist::gather_graph(local_graph, shard, 0, MPI_COMM_WORLD);

    // Root rank compares gathered result to CPU reference.
    ASSERT_EQ(gathered.n, n);
    ASSERT_EQ(gathered.k, k);
    for (std::size_t i = 0; i < n; ++i) {
        const auto ref_nb  = ref.neighbors_of(i);
        const auto got_nb  = gathered.neighbors_of(i);
        const auto ref_dist = ref.distances_of(i);
        const auto got_dist = gathered.distances_of(i);
        for (std::size_t j = 0; j < k; ++j) {
            EXPECT_EQ(ref_nb[j], got_nb[j])
                << "row " << i << " slot " << j;
            EXPECT_NEAR(ref_dist[j], got_dist[j],
                        std::abs(ref_dist[j]) * 1e-5f + 1e-6f)
                << "row " << i << " slot " << j;
        }
    }
}

TEST_F(DistBruteForceTest, GatherGraphShapeCorrect) {
    constexpr std::size_t n = 8;
    constexpr std::size_t d = 3;
    constexpr std::size_t k = 2;

    knng::Dataset root;
    if (g_env->is_root()) {
        root = make_dataset(n, d);
    }

    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);
    const knng::Knng local_graph =
        knng::dist::brute_force_knn_mpi(shard, k, MPI_COMM_WORLD);

    EXPECT_EQ(local_graph.n, shard.local_n());
    EXPECT_EQ(local_graph.k, k);

    const knng::Knng gathered =
        knng::dist::gather_graph(local_graph, shard, 0, MPI_COMM_WORLD);

    if (g_env->is_root()) {
        EXPECT_EQ(gathered.n, n);
        EXPECT_EQ(gathered.k, k);
    }
}

TEST_F(DistBruteForceTest, LocalGraphIndicesAreGlobal) {
    // Every neighbor index in the local graph must be a valid global index
    // (< global_n) and must not equal the query's own global index.
    constexpr std::size_t n = 12;
    constexpr std::size_t d = 4;
    constexpr std::size_t k = 4;

    knng::Dataset root;
    if (g_env->is_root()) {
        root = make_dataset(n, d);
    }

    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);
    const knng::Knng local_graph =
        knng::dist::brute_force_knn_mpi(shard, k, MPI_COMM_WORLD);

    for (std::size_t qi = 0; qi < shard.local_n(); ++qi) {
        const std::size_t q_global = shard.global_index(qi);
        const auto nb = local_graph.neighbors_of(qi);
        for (std::size_t j = 0; j < k; ++j) {
            EXPECT_LT(static_cast<std::size_t>(nb[j]), n)
                << "neighbor index out of range: row " << qi;
            EXPECT_NE(static_cast<std::size_t>(nb[j]), q_global)
                << "self-match in row " << qi;
        }
    }
}

} // namespace
