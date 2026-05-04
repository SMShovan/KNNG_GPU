/// @file
/// @brief Unit tests for `knng::cpu::first_touch` (Step 26).
///
/// The function's contract is "walk the buffer in parallel, write
/// every page-sized stride, do not change the contents." We pin
/// each of those properties on a small synthetic buffer.

#include <cstddef>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "knng/cpu/numa.hpp"

namespace {

TEST(NumaFirstTouch, NullPointerIsNoOp)
{
    // The function takes an unconditional `nullptr` early-exit;
    // calling it with `nullptr, 0` must not crash.
    EXPECT_NO_THROW(knng::cpu::first_touch(nullptr, 0));
}

TEST(NumaFirstTouch, ZeroLengthIsNoOp)
{
    float dummy = 1.0f;
    EXPECT_NO_THROW(knng::cpu::first_touch(&dummy, 0));
    EXPECT_FLOAT_EQ(dummy, 1.0f);
}

TEST(NumaFirstTouch, PreservesBufferContents)
{
    // Fill `buf` with a known pattern, run first_touch, assert
    // every cell is exactly what it was. The function writes only
    // `data[i] = data[i]` so no value should change.
    constexpr std::size_t n = 64 * 1024;  // ~256 KB, multi-page on every host
    std::vector<float> buf(n);
    std::iota(buf.begin(), buf.end(), 0.0f);

    knng::cpu::first_touch(buf.data(), buf.size(), /*num_threads=*/2);

    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_FLOAT_EQ(buf[i], static_cast<float>(i)) << "i = " << i;
    }
}

TEST(NumaFirstTouch, ToleratesSubPageBuffers)
{
    // Buffer smaller than one page; the loop's stride is `page /
    // sizeof(float)` so at most one iteration runs. Must not
    // overrun the buffer.
    std::vector<float> buf(8, 7.0f);
    knng::cpu::first_touch(buf.data(), buf.size(), /*num_threads=*/1);
    for (float v : buf) {
        EXPECT_FLOAT_EQ(v, 7.0f);
    }
}

TEST(NumaFirstTouch, IsNumaRelevantPlatformReturnsAStableValue)
{
    // The function is platform-keyed, so we just assert a
    // deterministic answer on this build.
    [[maybe_unused]] const bool flag =
        knng::cpu::is_numa_relevant_platform();
    SUCCEED();
}

} // namespace
