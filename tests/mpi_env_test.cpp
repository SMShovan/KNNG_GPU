/// @file
/// @brief Tests for `knng::dist::MpiEnv`.
///
/// Exercises the RAII MPI wrapper. Designed to pass in both single-rank
/// (`mpirun -np 1`) and multi-rank (`mpirun -np 2`) launches; CTest
/// runs it single-rank via `mpirun -np 1` (or directly when mpirun
/// is not needed for single-process MPI).
///
/// Test coverage:
///   * `rank()` is in `[0, size())`.
///   * `is_root()` is true iff `rank() == 0`.
///   * Allreduce sum: each rank contributes `rank`; total is
///     `size * (size - 1) / 2`.  Correct for any process count.
///   * `barrier()` does not deadlock (observational test).
///   * Double-init guard: constructing a second `MpiEnv` while the
///     first lives throws `std::runtime_error`.

#include "knng/dist/mpi_env.hpp"

#include <gtest/gtest.h>
#include <mpi.h>
#include <stdexcept>

namespace {

knng::dist::MpiEnv* g_env = nullptr; // Set by MpiEnvTest::SetUpTestSuite.

class MpiEnvTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        static knng::dist::MpiEnv env{};
        g_env = &env;
    }
};

TEST_F(MpiEnvTest, RankInRange) {
    ASSERT_NE(g_env, nullptr);
    EXPECT_GE(g_env->rank(), 0);
    EXPECT_LT(g_env->rank(), g_env->size());
}

TEST_F(MpiEnvTest, IsRootMatchesRank) {
    EXPECT_EQ(g_env->is_root(), g_env->rank() == 0);
}

TEST_F(MpiEnvTest, SizePositive) {
    EXPECT_GT(g_env->size(), 0);
}

TEST_F(MpiEnvTest, AllreduceSumCorrect) {
    // Each rank contributes its rank value. Sum = size*(size-1)/2.
    int value  = g_env->rank();
    int result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int expected = g_env->size() * (g_env->size() - 1) / 2;
    EXPECT_EQ(result, expected);
}

TEST_F(MpiEnvTest, BarrierDoesNotDeadlock) {
    g_env->barrier(); // All ranks must reach this.
    SUCCEED();
}

TEST_F(MpiEnvTest, DoubleInitThrows) {
    // MPI is already initialised by g_env; a second MpiEnv must throw.
    EXPECT_THROW(
        { knng::dist::MpiEnv second{}; },
        std::runtime_error
    );
}

} // namespace
