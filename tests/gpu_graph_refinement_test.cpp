/// @file
/// @brief Steps 72–77 — CAGRA-style graph refinement tests.
///
/// Tests are added alongside each step.  All tests exercise CPU reference
/// implementations (Mac-compatible).  GPU tests guarded by KNNG_HAVE_CUDA.

#include <gtest/gtest.h>

#include <knng/gpu/graph_refinement.hpp>
#include <knng/gpu/gpu_nn_descent.hpp>
#include <knng/gpu/device_graph.hpp>
#include <knng/core/dataset.hpp>

#include <vector>

using knng::gpu::CpuDeviceGraph;
using knng::gpu::cpu_enforce_out_degree;
using knng::gpu::kSentinelDist;

static CpuDeviceGraph make_graph(std::size_t n, std::size_t k,
                                  std::vector<knng::index_t> ids,
                                  std::vector<float>         dists = {})
{
    CpuDeviceGraph g(n, k);
    g.ids   = std::move(ids);
    if (!dists.empty()) g.dists = std::move(dists);
    return g;
}

// ===========================================================================
// Step 72 — Fixed out-degree enforcement
// ===========================================================================

TEST(GraphRefinement, EnforceOutDegree_SortsByDistanceAscending) {
    CpuDeviceGraph g = make_graph(1, 3,
        {2u, 0u, 1u},
        {9.f, 1.f, 4.f});
    cpu_enforce_out_degree(g);
    EXPECT_EQ(g.ids[0], 0u);
    EXPECT_FLOAT_EQ(g.dists[0], 1.f);
    EXPECT_EQ(g.ids[1], 1u);
    EXPECT_FLOAT_EQ(g.dists[1], 4.f);
    EXPECT_EQ(g.ids[2], 2u);
    EXPECT_FLOAT_EQ(g.dists[2], 9.f);
}

TEST(GraphRefinement, EnforceOutDegree_SentinelsLast) {
    const auto S = static_cast<knng::index_t>(-1);
    CpuDeviceGraph g = make_graph(1, 4,
        {S, 1u, S, 0u},
        {kSentinelDist, 3.f, kSentinelDist, 1.f});
    cpu_enforce_out_degree(g);
    EXPECT_NE(g.ids[0], S);
    EXPECT_NE(g.ids[1], S);
    EXPECT_EQ(g.ids[2], S);
    EXPECT_EQ(g.ids[3], S);
    EXPECT_LT(g.dists[0], g.dists[1]);
}

TEST(GraphRefinement, EnforceOutDegree_AlreadySorted_NoChange) {
    CpuDeviceGraph g = make_graph(1, 3,
        {0u, 1u, 2u},
        {1.f, 2.f, 3.f});
    const auto before = g.ids;
    cpu_enforce_out_degree(g);
    EXPECT_EQ(g.ids, before);
}

TEST(GraphRefinement, EnforceOutDegree_MultipleRows) {
    CpuDeviceGraph g = make_graph(3, 2,
        {1u, 0u, 2u, 0u, 0u, 1u},
        {5.f, 1.f, 3.f, 2.f, 7.f, 4.f});
    cpu_enforce_out_degree(g);
    EXPECT_EQ(g.ids[0], 0u); EXPECT_FLOAT_EQ(g.dists[0], 1.f);
    EXPECT_EQ(g.ids[1], 1u); EXPECT_FLOAT_EQ(g.dists[1], 5.f);
    EXPECT_EQ(g.ids[2], 0u); EXPECT_FLOAT_EQ(g.dists[2], 2.f);
    EXPECT_EQ(g.ids[3], 2u); EXPECT_FLOAT_EQ(g.dists[3], 3.f);
    EXPECT_EQ(g.ids[4], 1u); EXPECT_FLOAT_EQ(g.dists[4], 4.f);
    EXPECT_EQ(g.ids[5], 0u); EXPECT_FLOAT_EQ(g.dists[5], 7.f);
}

// ===========================================================================
// GPU (CUDA only)
// ===========================================================================

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

TEST(GraphRefinement, GpuEnforceMatchesCpu) {
    std::vector<float> data(16 * 3);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 7);
    auto ds = make_ds(data, 16, 3);

    knng::gpu::CpuDeviceGraph cpu_g = knng::gpu::cpu_init_random_graph(ds, 4, 42);
    knng::gpu::DeviceGraph gpu_g(cpu_g);

    cpu_enforce_out_degree(cpu_g);
    knng::gpu::gpu_enforce_out_degree(gpu_g);

    const knng::gpu::CpuDeviceGraph from_gpu = gpu_g.to_cpu();
    for (std::size_t i = 0; i < 16 * 4; ++i) {
        EXPECT_EQ(cpu_g.ids[i], from_gpu.ids[i]);
    }
}

#endif
