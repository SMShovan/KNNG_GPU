/// @file
/// @brief Tests for `knng::dist::dedup_requests` and `DeduplicationStats`.
///
/// These tests exercise the deduplication primitive in isolation — no MPI
/// communication needed.  The `allreduce_dedup_stats` function is tested
/// in an MPI context (single-rank so it reduces to a trivial identity).

#include "knng/dist/mpi_env.hpp"
#include "knng/dist/request_dedup.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstddef>
#include <vector>

namespace {

knng::dist::MpiEnv* g_env = nullptr;

class RequestDedupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        static knng::dist::MpiEnv env{};
        g_env = &env;
    }
};

// ---- dedup_requests (no MPI) ----

TEST_F(RequestDedupTest, EmptyRequestsUnchanged) {
    std::vector<std::vector<knng::index_t>> requests(3);
    const auto stats = knng::dist::dedup_requests(requests);
    EXPECT_EQ(stats.raw_count,   0u);
    EXPECT_EQ(stats.dedup_count, 0u);
    EXPECT_DOUBLE_EQ(stats.reduction_fraction(), 0.0);
}

TEST_F(RequestDedupTest, NoDuplicatesUnchanged) {
    std::vector<std::vector<knng::index_t>> requests = {
        {1, 2, 3},
        {4, 5},
        {6}
    };
    const auto stats = knng::dist::dedup_requests(requests);
    EXPECT_EQ(stats.raw_count,   6u);
    EXPECT_EQ(stats.dedup_count, 6u);
    EXPECT_DOUBLE_EQ(stats.reduction_fraction(), 0.0);
}

TEST_F(RequestDedupTest, AllDuplicatesWithinOneRank) {
    std::vector<std::vector<knng::index_t>> requests = {
        {5, 5, 5, 5},
        {}
    };
    const auto stats = knng::dist::dedup_requests(requests);
    EXPECT_EQ(stats.raw_count,   4u);
    EXPECT_EQ(stats.dedup_count, 1u);
    // After dedup, each list has unique sorted elements.
    EXPECT_EQ(requests[0].size(), 1u);
    EXPECT_EQ(requests[0][0], 5u);
}

TEST_F(RequestDedupTest, MixedDuplicates) {
    // 3 ranks; rank 0 requests {10, 10, 20, 20, 30}
    //          rank 1 requests {1, 1, 2}
    //          rank 2 empty
    std::vector<std::vector<knng::index_t>> requests = {
        {10, 10, 20, 20, 30},
        {1, 1, 2},
        {}
    };
    const auto stats = knng::dist::dedup_requests(requests);
    EXPECT_EQ(stats.raw_count,   8u);
    EXPECT_EQ(stats.dedup_count, 5u); // {10,20,30} + {1,2} + {}
    EXPECT_NEAR(stats.reduction_fraction(), 3.0 / 8.0, 1e-9);
    // Check sorted order.
    ASSERT_EQ(requests[0].size(), 3u);
    EXPECT_EQ(requests[0][0], 10u);
    EXPECT_EQ(requests[0][1], 20u);
    EXPECT_EQ(requests[0][2], 30u);
    ASSERT_EQ(requests[1].size(), 2u);
    EXPECT_EQ(requests[1][0], 1u);
    EXPECT_EQ(requests[1][1], 2u);
}

TEST_F(RequestDedupTest, AllreduceStatsIdentityOnSingleRank) {
    const knng::dist::DeduplicationStats local{10, 7};
    const auto global =
        knng::dist::allreduce_dedup_stats(local, MPI_COMM_WORLD);

    if (g_env->size() == 1) {
        EXPECT_EQ(global.raw_count,   10u);
        EXPECT_EQ(global.dedup_count, 7u);
    } else {
        // In multi-rank, totals should be sum of all ranks' counts.
        EXPECT_GE(global.raw_count,   10u);
        EXPECT_GE(global.dedup_count, 7u);
    }
}

TEST_F(RequestDedupTest, ReductionFractionZeroDivision) {
    const knng::dist::DeduplicationStats zero{0, 0};
    EXPECT_DOUBLE_EQ(zero.reduction_fraction(), 0.0);
}

} // namespace
