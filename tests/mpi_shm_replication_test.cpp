/// @file
/// @brief Tests for `knng::dist::ShmRegion` (Step 43).
///
/// In single-rank mode ShmRegion degenerates: the window covers only the
/// local rank's data, `intra_size() == 1`, and `read_remote_row(0, i, d)`
/// returns the same data as the original buffer. All correctness invariants
/// can be verified without multi-rank.

#include "knng/dist/mpi_env.hpp"
#include "knng/dist/shm_replication.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cstddef>
#include <vector>

namespace {

knng::dist::MpiEnv* g_env = nullptr;

class ShmRegionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        static knng::dist::MpiEnv env{};
        g_env = &env;
    }
};

TEST_F(ShmRegionTest, ConstructAndDestruct) {
    constexpr std::size_t n = 4;
    constexpr std::size_t d = 3;
    std::vector<float> data(n * d);
    for (std::size_t i = 0; i < n * d; ++i) {
        data[i] = static_cast<float>(i) + 1.0f;
    }
    // Should not throw.
    knng::dist::ShmRegion region(data.data(), n, d, MPI_COMM_WORLD);
    EXPECT_GE(region.intra_size(), 1);
    EXPECT_GE(region.intra_rank(), 0);
    EXPECT_LT(region.intra_rank(), region.intra_size());
}

TEST_F(ShmRegionTest, LocalDataReadableViaWindow) {
    constexpr std::size_t n = 3;
    constexpr std::size_t d = 4;
    std::vector<float> data(n * d);
    for (std::size_t i = 0; i < n * d; ++i) {
        data[i] = static_cast<float>(i) * 2.0f;
    }

    knng::dist::ShmRegion region(data.data(), n, d, MPI_COMM_WORLD);

    // On a single node, intra_rank() == 0 when intra_size() == 1.
    // Reading row 0 should give data[0..d-1].
    const float* row0 = region.read_remote_row(region.intra_rank(), 0, d);
    ASSERT_NE(row0, nullptr);
    for (std::size_t j = 0; j < d; ++j) {
        EXPECT_FLOAT_EQ(row0[j], data[j]) << "row 0, col " << j;
    }

    // Reading row 1.
    const float* row1 = region.read_remote_row(region.intra_rank(), 1, d);
    ASSERT_NE(row1, nullptr);
    for (std::size_t j = 0; j < d; ++j) {
        EXPECT_FLOAT_EQ(row1[j], data[d + j]) << "row 1, col " << j;
    }
}

TEST_F(ShmRegionTest, IntraRowCountsMatchLocalN) {
    constexpr std::size_t n = 5;
    constexpr std::size_t d = 2;
    std::vector<float> data(n * d, 1.0f);

    knng::dist::ShmRegion region(data.data(), n, d, MPI_COMM_WORLD);

    const auto& counts = region.intra_row_counts();
    ASSERT_EQ(counts.size(),
              static_cast<std::size_t>(region.intra_size()));

    // This rank's own count must equal n.
    EXPECT_EQ(counts[static_cast<std::size_t>(region.intra_rank())], n);
}

TEST_F(ShmRegionTest, LocalBasePointerNotNull) {
    constexpr std::size_t n = 2;
    constexpr std::size_t d = 3;
    std::vector<float> data(n * d, 0.5f);

    knng::dist::ShmRegion region(data.data(), n, d, MPI_COMM_WORLD);
    EXPECT_NE(region.local_base(), nullptr);
}

} // namespace
