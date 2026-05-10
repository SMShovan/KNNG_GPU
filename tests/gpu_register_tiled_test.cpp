/// @file
/// @brief Step 55 — Register-tiled brute-force KNN tests.

#include <gtest/gtest.h>
#include <knng/gpu/register_tiled_bf.hpp>
#include <knng/gpu/brute_force.hpp>
#include <knng/core/dataset.hpp>
#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::cpu_brute_force_knn;
using knng::gpu::cpu_register_tiled_knn;

static Dataset make_ds(std::vector<float> d, std::size_t n, std::size_t dim)
{ Dataset ds; ds.data=std::move(d); ds.n=n; ds.d=dim; return ds; }

static void expect_eq(const Knng& ref, const Knng& got)
{
    ASSERT_EQ(ref.neighbors.size(), got.neighbors.size());
    for (std::size_t i = 0; i < ref.neighbors.size(); ++i) {
        EXPECT_EQ(ref.neighbors[i], got.neighbors[i]) << "id at " << i;
        EXPECT_NEAR(ref.distances[i], got.distances[i], 1e-5f) << "dist at " << i;
    }
}

TEST(RegTiled, TwoPoints) {
    auto ds = make_ds({0.f, 1.f}, 2, 1);
    expect_eq(cpu_brute_force_knn(ds,1), cpu_register_tiled_knn(ds,1));
}
TEST(RegTiled, OddN) {
    // n=5 — the last group has only 1 query (kRegTileQ=2, so 5%2=1).
    std::vector<float> data(5*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds = make_ds(data,5,3);
    expect_eq(cpu_brute_force_knn(ds,3), cpu_register_tiled_knn(ds,3));
}
TEST(RegTiled, Clusters) {
    std::vector<float> data={0,0, 0.1f,0, 0,0.1f, 0.1f,0.1f,
                             9,9, 9.1f,9, 9,9.1f, 9.1f,9.1f};
    auto ds=make_ds(data,8,2);
    expect_eq(cpu_brute_force_knn(ds,3), cpu_register_tiled_knn(ds,3));
}
TEST(RegTiled, LargeN) {
    std::vector<float> pts(40*4);
    for (std::size_t i=0;i<pts.size();++i) pts[i]=static_cast<float>(i%13);
    auto ds=make_ds(pts,40,4);
    expect_eq(cpu_brute_force_knn(ds,4), cpu_register_tiled_knn(ds,4));
}

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)
TEST(RegTiled, GpuMatchesCpu) {
    std::vector<float> pts(32*4);
    for (std::size_t i=0;i<pts.size();++i) pts[i]=static_cast<float>(i%9);
    auto ds=make_ds(pts,32,4);
    expect_eq(cpu_brute_force_knn(ds,4), knng::gpu::gpu_register_tiled_knn(ds,4));
}
#endif
