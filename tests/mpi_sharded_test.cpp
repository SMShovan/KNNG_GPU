/// @file
/// @brief Tests for `knng::dist::ShardedDataset` and `compute_shard`.
///
/// Covers:
///   * `compute_shard` arithmetic for uniform and non-uniform divisions.
///   * Single-rank scatter: trivially returns the full dataset as a shard.
///   * Single-rank gather: round-trip through scatter → gather.
///   * Local accessors: `local_n`, `global_n`, `local_start`, `local_end`,
///     `global_index`, `d`, `rank`, `size`.

#include "knng/dist/mpi_env.hpp"
#include "knng/dist/sharded_dataset.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstddef>
#include <stdexcept>

namespace {

knng::dist::MpiEnv* g_env = nullptr;

class ShardedDatasetTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        static knng::dist::MpiEnv env{};
        g_env = &env;
    }
};

// ---- compute_shard unit tests (no MPI communication needed) ----

TEST_F(ShardedDatasetTest, ComputeShardExactDivision) {
    // 8 points, 4 ranks → 2 per rank, no remainder.
    const auto b0 = knng::dist::compute_shard(8, 4, 0);
    EXPECT_EQ(b0.start, 0u);
    EXPECT_EQ(b0.count, 2u);

    const auto b1 = knng::dist::compute_shard(8, 4, 1);
    EXPECT_EQ(b1.start, 2u);
    EXPECT_EQ(b1.count, 2u);

    const auto b3 = knng::dist::compute_shard(8, 4, 3);
    EXPECT_EQ(b3.start, 6u);
    EXPECT_EQ(b3.count, 2u); // last rank, no remainder
}

TEST_F(ShardedDatasetTest, ComputeShardWithRemainder) {
    // 10 points, 3 ranks → base=3, remainder=1 → [0,3), [3,6), [6,10).
    const auto b0 = knng::dist::compute_shard(10, 3, 0);
    EXPECT_EQ(b0.start, 0u);
    EXPECT_EQ(b0.count, 3u);

    const auto b1 = knng::dist::compute_shard(10, 3, 1);
    EXPECT_EQ(b1.start, 3u);
    EXPECT_EQ(b1.count, 3u);

    const auto b2 = knng::dist::compute_shard(10, 3, 2); // last rank gets +1
    EXPECT_EQ(b2.start, 6u);
    EXPECT_EQ(b2.count, 4u);
}

TEST_F(ShardedDatasetTest, ComputeShardSingleRank) {
    const auto b = knng::dist::compute_shard(7, 1, 0);
    EXPECT_EQ(b.start, 0u);
    EXPECT_EQ(b.count, 7u);
}

// ---- scatter + gather round-trip (single-rank pass) ----

TEST_F(ShardedDatasetTest, ScatterSingleRankIdentity) {
    // With one rank, scatter is a no-op: the shard IS the full dataset.
    constexpr std::size_t n = 6;
    constexpr std::size_t d = 4;
    knng::Dataset root(n, d);
    for (std::size_t i = 0; i < n * d; ++i) {
        root.data[i] = static_cast<float>(i) * 0.5f;
    }

    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);

    EXPECT_EQ(shard.global_n(),    n);
    EXPECT_EQ(shard.local_n(),     n);
    EXPECT_EQ(shard.local_start(), 0u);
    EXPECT_EQ(shard.local_end(),   n);
    EXPECT_EQ(shard.d(),           d);
    EXPECT_EQ(shard.rank(),        0);
    EXPECT_EQ(shard.size(),        1);

    // Data content preserved.
    for (std::size_t i = 0; i < n * d; ++i) {
        EXPECT_FLOAT_EQ(shard.local_dataset().data[i], root.data[i]);
    }
}

TEST_F(ShardedDatasetTest, GatherRoundTrip) {
    constexpr std::size_t n = 6;
    constexpr std::size_t d = 4;
    knng::Dataset root(n, d);
    for (std::size_t i = 0; i < n * d; ++i) {
        root.data[i] = static_cast<float>(i) + 1.0f;
    }

    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);

    const knng::Dataset gathered = shard.gather(0, MPI_COMM_WORLD);

    if (g_env->is_root()) {
        ASSERT_EQ(gathered.n, n);
        ASSERT_EQ(gathered.d, d);
        for (std::size_t i = 0; i < n * d; ++i) {
            EXPECT_FLOAT_EQ(gathered.data[i], root.data[i]);
        }
    }
}

TEST_F(ShardedDatasetTest, GlobalIndexMatchesLocalStart) {
    constexpr std::size_t n = 5;
    constexpr std::size_t d = 3;
    knng::Dataset root(n, d);

    const auto shard =
        knng::dist::ShardedDataset::scatter(root, 0, MPI_COMM_WORLD);

    EXPECT_EQ(shard.global_index(0), shard.local_start());
    if (shard.local_n() > 0) {
        EXPECT_EQ(shard.global_index(shard.local_n() - 1),
                  shard.local_end() - 1);
    }
}

TEST_F(ShardedDatasetTest, ScatterEmptyDatasetThrows) {
    knng::Dataset empty{};
    EXPECT_THROW(
        knng::dist::ShardedDataset::scatter(empty, 0, MPI_COMM_WORLD),
        std::runtime_error);
}

} // namespace
