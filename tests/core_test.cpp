/// @file
/// @brief Unit tests for the `knng::core` public API surface.
///
/// These tests pin down the small behavioral contracts of the types
/// introduced in Step 03:
///   * `L2Squared` returns 0 for identical vectors and the sum-of-
///     squares of componentwise deltas otherwise.
///   * `NegativeInnerProduct` agrees with minus the mathematical dot
///     product (which also witnesses the "lower is closer" convention).
///   * `Knng` rows are contiguous, correctly sized, and laid out
///     row-major with stride `k`.
///   * Both concrete metrics satisfy the `knng::Distance` concept.
///
/// Secondary purpose: prove that a test target can link
/// `knng::core` + `GTest::gtest_main` without either the library's
/// warning policy or GoogleTest's templates emitting a single
/// diagnostic under `-Wall -Wextra -Wpedantic -Wconversion -Werror`.

#include <array>
#include <cstddef>
#include <span>

#include <gtest/gtest.h>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"

namespace {

TEST(SquaredL2Free, ZeroForIdenticalPointers)
{
    constexpr std::array<float, 4> a{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(knng::squared_l2(a.data(), a.data(), a.size()), 0.0f);
}

TEST(SquaredL2Free, HandVerifiedThreeFourPair)
{
    // (3-0)² + (4-0)² + (0-0)² == 9 + 16 == 25
    constexpr std::array<float, 3> a{0.0f, 0.0f, 0.0f};
    constexpr std::array<float, 3> b{3.0f, 4.0f, 0.0f};
    EXPECT_FLOAT_EQ(knng::squared_l2(a.data(), b.data(), 3), 25.0f);
}

TEST(SquaredL2Free, DimZeroIsEmptySum)
{
    // The pointers must remain valid even though no element is read.
    // A nullptr pair would also be defensible, but the contract is
    // "points to at least dim floats", so we exercise the dim == 0
    // case with real (unread) storage.
    constexpr std::array<float, 1> a{42.0f};
    constexpr std::array<float, 1> b{-7.0f};
    EXPECT_FLOAT_EQ(knng::squared_l2(a.data(), b.data(), 0), 0.0f);
}

TEST(SquaredL2Free, DimOneIsScalarSquaredDifference)
{
    constexpr float a = 5.0f;
    constexpr float b = 2.0f;
    EXPECT_FLOAT_EQ(knng::squared_l2(&a, &b, 1), 9.0f);
}

TEST(SquaredL2Free, AgreesWithFunctor)
{
    // Cross-check: the functor delegates to this function, so the two
    // must agree byte-for-byte on the same inputs.
    constexpr std::array<float, 5> a{ 1.0f,  2.0f,  3.0f,  4.0f,  5.0f};
    constexpr std::array<float, 5> b{-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
    const knng::L2Squared functor;
    EXPECT_FLOAT_EQ(
        knng::squared_l2(a.data(), b.data(), a.size()),
        functor(std::span<const float>{a}, std::span<const float>{b}));
}

TEST(L2Squared, ZeroForIdenticalVectors)
{
    constexpr std::array<float, 4> a{1.0f, 2.0f, 3.0f, 4.0f};
    const knng::L2Squared d;
    EXPECT_FLOAT_EQ(d(std::span<const float>{a}, std::span<const float>{a}), 0.0f);
}

TEST(L2Squared, SumOfSquaredDeltas)
{
    constexpr std::array<float, 3> a{0.0f, 0.0f, 0.0f};
    constexpr std::array<float, 3> b{3.0f, 4.0f, 0.0f};
    const knng::L2Squared d;
    EXPECT_FLOAT_EQ(
        d(std::span<const float>{a}, std::span<const float>{b}),
        25.0f);
}

TEST(NegativeInnerProduct, EqualsMinusDotProduct)
{
    constexpr std::array<float, 3> a{1.0f, 2.0f, 3.0f};
    constexpr std::array<float, 3> b{4.0f, 5.0f, 6.0f};
    const knng::NegativeInnerProduct d;
    // dot(a, b) = 4 + 10 + 18 = 32  →  negated = -32.
    EXPECT_FLOAT_EQ(
        d(std::span<const float>{a}, std::span<const float>{b}),
        -32.0f);
}

TEST(DistanceConcept, BuiltinMetricsSatisfyConcept)
{
    static_assert(knng::Distance<knng::L2Squared>);
    static_assert(knng::Distance<knng::NegativeInnerProduct>);
    SUCCEED();
}

TEST(Knng, ConstructedShapeMatchesArguments)
{
    const knng::Knng g(5, 3);
    EXPECT_EQ(g.n, std::size_t{5});
    EXPECT_EQ(g.k, std::size_t{3});
    EXPECT_EQ(g.neighbors.size(), std::size_t{15});
    EXPECT_EQ(g.distances.size(), std::size_t{15});
}

TEST(Knng, RowViewsAreContiguousWithStrideK)
{
    knng::Knng g(4, 3);
    EXPECT_EQ(g.neighbors_of(0).size(), std::size_t{3});
    EXPECT_EQ(g.distances_of(3).size(), std::size_t{3});

    // Row i+1 begins exactly k elements after row i.
    EXPECT_EQ(g.neighbors_of(1).data(), g.neighbors_of(0).data() + 3);
    EXPECT_EQ(g.distances_of(2).data(), g.distances_of(0).data() + 6);
}

TEST(Knng, MutatingRowViewIsReflectedInUnderlyingStorage)
{
    knng::Knng g(2, 2);
    auto row = g.neighbors_of(1);
    row[0] = knng::index_t{7};
    row[1] = knng::index_t{9};

    EXPECT_EQ(g.neighbors[2], knng::index_t{7});
    EXPECT_EQ(g.neighbors[3], knng::index_t{9});
}

TEST(Dataset, ConstructedShapeMatchesArguments)
{
    const knng::Dataset ds(5, 4);
    EXPECT_EQ(ds.n, std::size_t{5});
    EXPECT_EQ(ds.d, std::size_t{4});
    EXPECT_EQ(ds.data.size(), std::size_t{20});
}

TEST(Dataset, RowViewsAreContiguousWithStrideD)
{
    knng::Dataset ds(3, 4);
    EXPECT_EQ(ds.row(0).size(), std::size_t{4});
    EXPECT_EQ(ds.row(2).size(), std::size_t{4});

    // Row i+1 begins exactly d floats after row i.
    EXPECT_EQ(ds.row(1).data(), ds.row(0).data() + 4);
    EXPECT_EQ(ds.row(2).data(), ds.row(0).data() + 8);
}

TEST(Dataset, MutatingRowViewIsReflectedInUnderlyingStorage)
{
    knng::Dataset ds(2, 3);
    auto row = ds.row(1);
    row[0] = 1.5f;
    row[1] = 2.5f;
    row[2] = 3.5f;

    EXPECT_FLOAT_EQ(ds.data[3], 1.5f);
    EXPECT_FLOAT_EQ(ds.data[4], 2.5f);
    EXPECT_FLOAT_EQ(ds.data[5], 3.5f);
}

TEST(Dataset, StrideHelpersReturnRowMajorD)
{
    const knng::Dataset ds(7, 16);
    EXPECT_EQ(ds.stride(), std::size_t{16});
    EXPECT_EQ(ds.byte_stride(), std::size_t{16 * sizeof(float)});
    EXPECT_EQ(ds.size(), std::size_t{7 * 16});
}

TEST(Dataset, DataPtrAddressesUnderlyingBuffer)
{
    knng::Dataset ds(3, 4);
    ds.data_ptr()[0] = 1.0f;
    ds.data_ptr()[1] = 2.0f;
    EXPECT_FLOAT_EQ(ds.data[0], 1.0f);
    EXPECT_FLOAT_EQ(ds.data[1], 2.0f);

    const knng::Dataset& cds = ds;
    EXPECT_EQ(cds.data_ptr(), cds.data.data());
}

TEST(Dataset, IsContiguousHoldsForFreshlyConstructedDataset)
{
    knng::Dataset ds(5, 8);
    EXPECT_TRUE(ds.is_contiguous());
}

TEST(Dataset, IsContiguousFlagsSizeMismatch)
{
    // The contract is `data.size() == n * d`. If a caller manually
    // resizes the buffer (or, more realistically, miscomputes a
    // shape after deserialisation), is_contiguous flags it.
    knng::Dataset ds(5, 8);
    ds.data.resize(20);  // 5 * 8 == 40 expected
    EXPECT_FALSE(ds.is_contiguous());
}

TEST(Dataset, EmptyDatasetIsContiguous)
{
    const knng::Dataset ds;  // 0×0
    EXPECT_TRUE(ds.is_contiguous());
    EXPECT_EQ(ds.size(), std::size_t{0});
    EXPECT_EQ(ds.stride(), std::size_t{0});
}

TEST(Dataset, RowsAddressesDeriveFromStride)
{
    knng::Dataset ds(4, 5);
    // Row i must start exactly stride() * i floats into the buffer.
    for (std::size_t i = 0; i < ds.n; ++i) {
        EXPECT_EQ(ds.row(i).data(),
                  ds.data_ptr() + i * ds.stride());
    }
}

} // namespace
