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
using knng::gpu::cpu_rank_reorder;
using knng::gpu::cpu_add_reverse_edges;
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
// Step 73 — Rank-based reordering
// ===========================================================================

// Build a 3-point graph where point 0 has neighbors [1, 2].
// Point 1's list: [0, 2] — so qi=0 appears at rank 0 in neighbor 1's list.
// Point 2's list: [1, 0] — so qi=0 appears at rank 1 in neighbor 2's list.
// After rank_reorder, neighbor 1 (rank 0) should come before neighbor 2 (rank 1).
TEST(GraphRefinement, RankReorder_PrefersLowerRankNeighbor) {
    // 3 points, k=2
    CpuDeviceGraph g(3, 2);
    // Row 0: neighbors 2 (dist 1.0), 1 (dist 2.0) — stored distance-ascending
    g.ids   = {2u, 1u,   // row 0 — we'll reorder by rank, not distance
               0u, 2u,   // row 1: [0 at rank0, 2 at rank1]
               1u, 0u};  // row 2: [1 at rank0, 0 at rank1]
    g.dists = {1.f, 2.f,
               1.f, 2.f,
               1.f, 2.f};
    // ranks for qi=0: neighbor 2's list has 0 at position 1 → rank=1
    //                 neighbor 1's list has 0 at position 0 → rank=0
    // so after reorder: ids[0][0]=1 (rank 0), ids[0][1]=2 (rank 1)
    cpu_rank_reorder(g);
    EXPECT_EQ(g.ids[0], 1u);
    EXPECT_EQ(g.ids[1], 2u);
}

TEST(GraphRefinement, RankReorder_SentinelsStayLast) {
    const auto S = static_cast<knng::index_t>(-1);
    // 2 points, k=3, one sentinel
    CpuDeviceGraph g(2, 3);
    g.ids   = {1u, S,  S,   // row 0: one valid neighbor, two sentinels
               0u, S,  S};  // row 1
    g.dists = {1.f, kSentinelDist, kSentinelDist,
               1.f, kSentinelDist, kSentinelDist};
    cpu_rank_reorder(g);
    EXPECT_EQ(g.ids[0], 1u);
    EXPECT_EQ(g.ids[1], S);
    EXPECT_EQ(g.ids[2], S);
}

TEST(GraphRefinement, RankReorder_AllSentinels_NoChange) {
    const auto S = static_cast<knng::index_t>(-1);
    CpuDeviceGraph g(1, 3);
    g.ids   = {S, S, S};
    g.dists = {kSentinelDist, kSentinelDist, kSentinelDist};
    cpu_rank_reorder(g);
    EXPECT_EQ(g.ids[0], S);
    EXPECT_EQ(g.ids[1], S);
    EXPECT_EQ(g.ids[2], S);
}

// ===========================================================================
// Step 74 — Add reverse edges
// ===========================================================================

// Graph: 0→1 (dist 3), 1→0 (dist 3) already exists — no change expected.
// After add_reverse_edges, neither list should duplicate entries.
TEST(GraphRefinement, AddReverseEdges_AlreadyPresent_NoChange) {
    // 2 points, k=2; fully mutual
    CpuDeviceGraph g(2, 2);
    const auto S = static_cast<knng::index_t>(-1);
    g.ids   = {1u, S,   0u, S};
    g.dists = {3.f, kSentinelDist, 3.f, kSentinelDist};
    cpu_add_reverse_edges(g);
    EXPECT_EQ(g.ids[0], 1u);
    EXPECT_EQ(g.ids[1], S);
    EXPECT_EQ(g.ids[2], 0u);
    EXPECT_EQ(g.ids[3], S);
}

// Graph: 0→1 (dist 2), 1 has no edge to 0.
// After add_reverse_edges, 1's list should contain 0 at the sentinel slot.
TEST(GraphRefinement, AddReverseEdges_FillsSentinelSlot) {
    CpuDeviceGraph g(2, 2);
    const auto S = static_cast<knng::index_t>(-1);
    g.ids   = {1u, S,   S, S};           // 0→1; 1 has no edges
    g.dists = {2.f, kSentinelDist, kSentinelDist, kSentinelDist};
    cpu_add_reverse_edges(g);
    // 1 should now have 0 in its list
    bool found = (g.ids[2] == 0u || g.ids[3] == 0u);
    EXPECT_TRUE(found);
}

// Graph: 0→1 (dist 2.0), 1→2 (dist 10.0, bad), 2 has no edges.
// Reverse of 0→1 (dist 2.0): should evict 1's entry (2, dist 10.0) since 2.0 < 10.0.
TEST(GraphRefinement, AddReverseEdges_EvictsWorstEntry) {
    const auto S = static_cast<knng::index_t>(-1);
    CpuDeviceGraph g(3, 1);
    g.ids   = {1u, 2u, S};
    g.dists = {2.f, 10.f, kSentinelDist};
    cpu_add_reverse_edges(g);
    // 1's slot should now be 0 (dist 2.0 replaced dist 10.0)
    EXPECT_EQ(g.ids[1], 0u);
    EXPECT_FLOAT_EQ(g.dists[1], 2.f);
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
