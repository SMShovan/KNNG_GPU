/// @file
/// @brief Step 59 — Stream-pipelined brute-force KNN tests.

#include <gtest/gtest.h>
#include <knng/gpu/streamed_brute_force.hpp>
#include <knng/gpu/brute_force.hpp>   // cpu_brute_force_knn (oracle)
#include <knng/core/dataset.hpp>
#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::cpu_brute_force_knn;
using knng::gpu::cpu_streamed_brute_force_knn;

static Dataset make_ds(std::vector<float> d, std::size_t n, std::size_t dim)
{ Dataset ds; ds.data=std::move(d); ds.n=n; ds.d=dim; return ds; }

static void expect_eq(const Knng& ref, const Knng& got)
{
    ASSERT_EQ(ref.neighbors.size(), got.neighbors.size());
    for (std::size_t i=0;i<ref.neighbors.size();++i) {
        EXPECT_EQ(ref.neighbors[i], got.neighbors[i]) << "id at " << i;
        EXPECT_NEAR(ref.distances[i], got.distances[i], 1e-5f) << "dist at " << i;
    }
}

// ---------------------------------------------------------------------------
// CPU chunked == CPU reference
// ---------------------------------------------------------------------------

TEST(StreamedBF, OneChunk)
{
    auto ds = make_ds({0,0, 1,0, 0,1, 1,1}, 4, 2);
    expect_eq(cpu_brute_force_knn(ds,2),
              cpu_streamed_brute_force_knn(ds,2,4));  // chunk_sz >= n
}

TEST(StreamedBF, MultipleChunks)
{
    std::vector<float> data(20*2);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds=make_ds(data,20,2);
    // chunk_sz=7 → 3 chunks (7, 7, 6)
    expect_eq(cpu_brute_force_knn(ds,3),
              cpu_streamed_brute_force_knn(ds,3,7));
}

TEST(StreamedBF, ExactlyTwoChunks)
{
    std::vector<float> data(16*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%11);
    auto ds=make_ds(data,16,3);
    // chunk_sz=8 → exactly 2 chunks
    expect_eq(cpu_brute_force_knn(ds,4),
              cpu_streamed_brute_force_knn(ds,4,8));
}

TEST(StreamedBF, SortedDistances)
{
    std::vector<float> data(12*2);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i);
    auto ds=make_ds(data,12,2);
    Knng g=cpu_streamed_brute_force_knn(ds,3,5);
    for (std::size_t qi=0;qi<12;++qi) {
        for (std::size_t r=1;r<3;++r) {
            EXPECT_LE(g.distances[qi*3+r-1], g.distances[qi*3+r]);
        }
    }
}
