/// @file
/// @brief Unit tests for `knng::random::XorShift64` (Step 17).
///
/// Pins the project-wide RNG contract: deterministic, reproducible,
/// non-zero state at all times, drop-in compatible with the C++
/// `<random>` distributions, period long enough to be irrelevant
/// at the n×k counts we care about.

#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "knng/random.hpp"

namespace {

TEST(XorShift64, SameSeedYieldsSameSequence)
{
    knng::random::XorShift64 a{42};
    knng::random::XorShift64 b{42};
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(a(), b());
    }
}

TEST(XorShift64, DifferentSeedsYieldDifferentSequences)
{
    knng::random::XorShift64 a{42};
    knng::random::XorShift64 b{43};
    // Two different seeds should diverge inside the first handful
    // of steps. We collect 8 values and demand at least one
    // mismatch — far more permissive than the truth (every value
    // mismatches), but robust to a future reshift if the algorithm
    // is ever swapped.
    std::array<std::uint64_t, 8> as{};
    std::array<std::uint64_t, 8> bs{};
    for (auto& x : as) x = a();
    for (auto& x : bs) x = b();
    EXPECT_NE(as, bs);
}

TEST(XorShift64, ZeroSeedIsRejected)
{
    EXPECT_THROW({ knng::random::XorShift64 rng{0}; },
                 std::invalid_argument);
}

TEST(XorShift64, ResettingToZeroIsRejected)
{
    knng::random::XorShift64 rng{1};
    EXPECT_THROW(rng.seed(0), std::invalid_argument);
}

TEST(XorShift64, StateIsNonZeroAfterEveryStep)
{
    knng::random::XorShift64 rng{1};
    for (int i = 0; i < 10000; ++i) {
        EXPECT_NE(rng(), std::uint64_t{0});
    }
    EXPECT_NE(rng.state(), std::uint64_t{0});
}

TEST(XorShift64, NextFloat01IsInRange)
{
    knng::random::XorShift64 rng{12345};
    for (int i = 0; i < 100000; ++i) {
        const float f = rng.next_float01();
        EXPECT_GE(f, 0.0f);
        EXPECT_LT(f, 1.0f);
    }
}

TEST(XorShift64, NextFloat01CoversTheUnitInterval)
{
    // 100k samples should easily hit every histogram bucket of
    // width 0.05. The threshold is loose enough to absorb random
    // variation while still failing if next_float01 is constant
    // or stuck in a sub-range.
    knng::random::XorShift64 rng{77777};
    constexpr int kBuckets = 20;
    std::array<int, kBuckets> hist{};
    for (int i = 0; i < 100000; ++i) {
        const float f = rng.next_float01();
        const int bucket = static_cast<int>(f * kBuckets);
        ASSERT_GE(bucket, 0);
        ASSERT_LT(bucket, kBuckets);
        ++hist[static_cast<std::size_t>(bucket)];
    }
    for (int count : hist) {
        EXPECT_GT(count, 1000)
            << "histogram bucket count too low — RNG output likely biased";
    }
}

TEST(XorShift64, NextBelowZeroOrOneReturnsZero)
{
    knng::random::XorShift64 rng{1};
    EXPECT_EQ(rng.next_below(0), std::uint64_t{0});
    EXPECT_EQ(rng.next_below(1), std::uint64_t{0});
}

TEST(XorShift64, NextBelowStaysInRange)
{
    knng::random::XorShift64 rng{3};
    for (int i = 0; i < 100000; ++i) {
        const auto v = rng.next_below(7);
        EXPECT_LT(v, std::uint64_t{7});
    }
}

TEST(XorShift64, ConformsToUniformRandomBitGenerator)
{
    // Drop the type into a standard distribution — this fails to
    // compile if the named requirement is not met (min/max not
    // constexpr, result_type wrong, operator() not callable).
    knng::random::XorShift64 rng{1234};
    std::uniform_int_distribution<int> dist(0, 9);

    std::set<int> seen;
    for (int i = 0; i < 1000; ++i) {
        seen.insert(dist(rng));
    }
    // 1000 draws from [0, 9] must hit every bucket — if any value
    // is missing, either the distribution or the RNG is broken.
    EXPECT_EQ(seen.size(), std::size_t{10});
}

TEST(XorShift64, StateSnapshotAndRestore)
{
    knng::random::XorShift64 rng{42};
    for (int i = 0; i < 100; ++i) {
        (void)rng();
    }
    const std::uint64_t snapshot = rng.state();
    std::vector<std::uint64_t> seq;
    for (int i = 0; i < 50; ++i) {
        seq.push_back(rng());
    }

    // Restore via seed() and replay.
    rng.seed(snapshot);
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(rng(), seq[static_cast<std::size_t>(i)]);
    }
}

} // namespace
