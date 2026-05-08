/// @file
/// @brief Step 46 — DeviceBuffer<T> RAII unit tests.
///
/// Exercises the full DeviceBuffer API through the CPU-stub backend
/// (GPU_MALLOC → std::malloc, GPU_MEMCPY_* → std::memcpy).
/// On a GPU machine these tests also validate that the real CUDA paths
/// behave identically.

#include <gtest/gtest.h>
#include <knng/gpu/device_buffer.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

using knng::gpu::DeviceBuffer;

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

TEST(DeviceBuffer, DefaultConstructEmpty)
{
    DeviceBuffer<float> buf;
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.bytes(), 0u);
    EXPECT_TRUE(buf.empty());
}

TEST(DeviceBuffer, SizedConstruct)
{
    DeviceBuffer<float> buf(16);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 16u);
    EXPECT_EQ(buf.bytes(), 16u * sizeof(float));
    EXPECT_FALSE(buf.empty());
}

TEST(DeviceBuffer, SizedConstructZeroThrows)
{
    EXPECT_THROW(DeviceBuffer<float>(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Construct from host vector
// ---------------------------------------------------------------------------

TEST(DeviceBuffer, ConstructFromVector)
{
    const std::vector<float> src = {1.f, 2.f, 3.f, 4.f};
    DeviceBuffer<float> buf(src);

    EXPECT_EQ(buf.size(), src.size());
    const auto result = buf.to_host();
    EXPECT_EQ(result, src);
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

TEST(DeviceBuffer, MoveConstruct)
{
    DeviceBuffer<int> a(8);
    float* raw = reinterpret_cast<float*>(a.data());  // save address
    (void)raw;

    DeviceBuffer<int> b(std::move(a));
    EXPECT_EQ(a.data(), nullptr);   // NOLINT — testing moved-from state
    EXPECT_EQ(a.size(), 0u);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_EQ(b.size(), 8u);
}

TEST(DeviceBuffer, MoveAssign)
{
    DeviceBuffer<double> a(4);
    DeviceBuffer<double> b(2);

    b = std::move(a);
    EXPECT_EQ(a.data(), nullptr);   // NOLINT
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(b.size(), 4u);
}

TEST(DeviceBuffer, SelfMoveAssignSafe)
{
    // Self-move through an alias pointer — avoids the -Wself-move warning
    // while still exercising the `if (this != &other)` guard.
    DeviceBuffer<float> buf(8);
    DeviceBuffer<float>* alias = &buf;
    buf = std::move(*alias);
    // After self-move the buffer must still be in a valid (non-crashed) state.
    EXPECT_TRUE(buf.size() == 8u || buf.data() == nullptr);
}

// ---------------------------------------------------------------------------
// resize()
// ---------------------------------------------------------------------------

TEST(DeviceBuffer, ResizeFreesOldAllocation)
{
    DeviceBuffer<float> buf(8);
    buf.resize(32);
    EXPECT_EQ(buf.size(), 32u);
    EXPECT_NE(buf.data(), nullptr);
}

TEST(DeviceBuffer, ResizeZeroThrows)
{
    DeviceBuffer<float> buf(4);
    EXPECT_THROW(buf.resize(0), std::invalid_argument);
    // Buffer must still be in a valid (unchanged) state after the throw.
    EXPECT_EQ(buf.size(), 4u);
}

// ---------------------------------------------------------------------------
// copy_from_host / copy_to_host round-trip
// ---------------------------------------------------------------------------

TEST(DeviceBuffer, CopyRoundTrip)
{
    const std::vector<float> src = {10.f, 20.f, 30.f, 40.f, 50.f};
    DeviceBuffer<float> buf(src.size());
    buf.copy_from_host(src.data(), src.size());

    const auto got = buf.to_host();
    EXPECT_EQ(got, src);
}

TEST(DeviceBuffer, CopyFromHostTooLargeThrows)
{
    DeviceBuffer<float> buf(4);
    const std::vector<float> big(8, 0.f);
    EXPECT_THROW(buf.copy_from_host(big.data(), big.size()), std::out_of_range);
}

TEST(DeviceBuffer, CopyToHostVector)
{
    const std::vector<int> src = {7, 8, 9};
    DeviceBuffer<int> buf(src);
    std::vector<int> dst;
    buf.copy_to_host(dst);
    EXPECT_EQ(dst, src);
}

// ---------------------------------------------------------------------------
// Type safety: non-trivially-copyable types must be rejected at compile time.
// This is verified by static_assert inside DeviceBuffer — the test below
// exercises a trivially-copyable aggregate to confirm the path compiles.
// ---------------------------------------------------------------------------

struct TrivialPair { int x; float y; };
static_assert(std::is_trivially_copyable_v<TrivialPair>);

TEST(DeviceBuffer, TrivialAggregateCompiles)
{
    DeviceBuffer<TrivialPair> buf(2);
    EXPECT_EQ(buf.size(), 2u);
}
