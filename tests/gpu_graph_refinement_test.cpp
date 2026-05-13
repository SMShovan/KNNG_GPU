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
using knng::gpu::GraphRefinementConfig;
using knng::gpu::cpu_enforce_out_degree;
using knng::gpu::cpu_rank_reorder;
using knng::gpu::cpu_add_reverse_edges;
using knng::gpu::cpu_prune_detour_edges;
using knng::gpu::cpu_merge_components;
using knng::gpu::cpu_refine_graph;
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
// Step 75 — Detourable-edge pruning
// ===========================================================================

// 3 collinear points: A=(0,0), B=(1,0), C=(4,0).
// B→C: d(B,C)²=9.  d(B,A)²=1 < 9, d(A,C)²=16 > 9 — A does NOT provide detour.
// So B→C should be KEPT.
TEST(GraphRefinement, PruneDetour_KeepsNecessaryEdge) {
    // 3 points: A=0=(0,0), B=1=(1,0), C=2=(4,0)
    // Point 1 (B) has neighbors: A (dist²=1), C (dist²=9)
    std::vector<float> vecs = {0.f, 0.f,  1.f, 0.f,  4.f, 0.f};
    CpuDeviceGraph g(3, 2);
    g.ids   = {1u, 2u,   // row 0 (A): not tested
               0u, 2u,   // row 1 (B): A at dist²=1, C at dist²=9
               0u, 1u};  // row 2 (C): not tested
    g.dists = {1.f, 16.f,
               1.f, 9.f,
               16.f, 9.f};
    cpu_prune_detour_edges(g, vecs.data(), 2);
    // B→C must NOT be pruned: A is closer to B but d(A,C)=16 > 9
    EXPECT_EQ(g.ids[1*2+1], 2u);
}

// 3 collinear points: A=(0,0), B=(0,0)+eps (approx same), C=(4,0).
// Use: A=(0,0), B=(0,0), mediant M=(1,0).
// B→C: d²=16. Neighbor M: d(B,M)²=1<16, d(M,C)²=9<16 → M provides detour.
// So B→C should be PRUNED.
TEST(GraphRefinement, PruneDetour_RemovesDetourableEdge) {
    // Point 0=B=(0,0), Point 1=M=(1,0), Point 2=C=(4,0)
    std::vector<float> vecs = {0.f, 0.f,  1.f, 0.f,  4.f, 0.f};
    CpuDeviceGraph g(3, 2);
    g.ids   = {1u, 2u,   // row 0 (B): M at d²=1, C at d²=16
               0u, 2u,   // row 1 (M): not tested
               1u, 0u};  // row 2 (C): not tested
    g.dists = {1.f, 16.f,
               1.f, 9.f,
               9.f, 16.f};
    const auto S = static_cast<knng::index_t>(-1);
    cpu_prune_detour_edges(g, vecs.data(), 2);
    // B→C (row 0, slot 1) must be pruned
    EXPECT_EQ(g.ids[0*2+1], S);
    // B→M (row 0, slot 0) must be kept (M is closer than C; no detour via C)
    EXPECT_EQ(g.ids[0*2+0], 1u);
}

TEST(GraphRefinement, PruneDetour_AllSentinels_NoChange) {
    const auto S = static_cast<knng::index_t>(-1);
    CpuDeviceGraph g(2, 2);
    g.ids   = {S, S, S, S};
    g.dists = {kSentinelDist, kSentinelDist, kSentinelDist, kSentinelDist};
    std::vector<float> vecs = {0.f, 0.f, 1.f, 0.f};
    cpu_prune_detour_edges(g, vecs.data(), 2);
    EXPECT_EQ(g.ids[0], S);
    EXPECT_EQ(g.ids[1], S);
}

// ===========================================================================
// Step 76 — Strong-component merging
// ===========================================================================

// Two isolated singletons (0, 1) with no edges. After merge, at least one
// should have a valid neighbor connecting the two.
TEST(GraphRefinement, MergeComponents_ConnectsSingletons) {
    const auto S = static_cast<knng::index_t>(-1);
    // 2 points, k=1, both singletons
    CpuDeviceGraph g(2, 1);
    g.ids   = {S, S};
    g.dists = {kSentinelDist, kSentinelDist};
    std::vector<float> vecs = {0.f, 0.f,  1.f, 0.f};
    cpu_merge_components(g, vecs.data(), 2);
    // At least one direction must now have a valid edge
    const bool any_connected = (g.ids[0] != S || g.ids[1] != S);
    EXPECT_TRUE(any_connected);
}

// Already connected graph: node 0→1, 1→0. No change expected.
TEST(GraphRefinement, MergeComponents_AlreadyConnected_NoChange) {
    CpuDeviceGraph g(2, 1);
    g.ids   = {1u, 0u};
    g.dists = {1.f, 1.f};
    const auto before_ids = g.ids;
    std::vector<float> vecs = {0.f, 0.f,  1.f, 0.f};
    cpu_merge_components(g, vecs.data(), 2);
    EXPECT_EQ(g.ids, before_ids);
}

// 4 nodes: two pairs (0-1 connected, 2-3 connected) but no edges between pairs.
// After merge, nodes from different pairs should be bridged.
TEST(GraphRefinement, MergeComponents_TwoComponentsBridged) {
    const auto S = static_cast<knng::index_t>(-1);
    // k=2: each node can hold 2 neighbors
    CpuDeviceGraph g(4, 2);
    g.ids   = {1u, S,   0u, S,   3u, S,   2u, S};
    g.dists = {1.f, kSentinelDist, 1.f, kSentinelDist,
               1.f, kSentinelDist, 1.f, kSentinelDist};
    // Vectors: 0=(0,0),1=(1,0) close; 2=(10,0),3=(11,0) close but far from 0/1
    std::vector<float> vecs = {0.f,0.f, 1.f,0.f, 10.f,0.f, 11.f,0.f};
    cpu_merge_components(g, vecs.data(), 2);
    // Nodes 0 or 1 should now have an edge to node 2 or 3 (or vice versa)
    bool bridge_found = false;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t r = 0; r < 2; ++r) {
            const auto id = g.ids[i * 2 + r];
            if (id == S) continue;
            // Bridge: node in {0,1} connects to node in {2,3}
            const bool from_left = (i <= 1);
            const bool to_right  = (id >= 2u);
            const bool from_right = (i >= 2);
            const bool to_left   = (id <= 1u);
            if ((from_left && to_right) || (from_right && to_left)) {
                bridge_found = true;
            }
        }
    }
    EXPECT_TRUE(bridge_found);
}

// ===========================================================================
// Step 77 — Ablation driver (cpu_refine_graph)
// ===========================================================================

// Full pipeline: sorted, reverse edges added, detour pruning applied.
// Result must be a valid graph: all valid ids < n, all valid dists >= 0.
TEST(GraphRefinement, RefineGraph_AllSteps_ProducesValidGraph) {
    // 5 points on a line: x = 0,1,2,3,4
    const std::size_t n = 5, k = 3;
    std::vector<float> vecs(n * 2);
    for (std::size_t i = 0; i < n; ++i) { vecs[i*2] = static_cast<float>(i); vecs[i*2+1] = 0.f; }

    // Build a graph where each point has k random neighbors
    CpuDeviceGraph g(n, k);
    // Manually set: each point has 3 neighbors (unsorted) with distances
    g.ids   = {1u, 3u, 4u,   // row 0
               0u, 2u, 4u,   // row 1
               1u, 0u, 3u,   // row 2
               2u, 4u, 0u,   // row 3
               3u, 0u, 1u};  // row 4
    // Distances: L2² on 1D
    g.dists = {1.f, 9.f, 16.f,
               1.f, 1.f, 9.f,
               1.f, 4.f, 1.f,
               1.f, 1.f, 9.f,
               1.f, 16.f, 9.f};

    GraphRefinementConfig cfg;
    cfg.enforce_out_degree = true;
    cfg.rank_reorder       = true;
    cfg.add_reverse_edges  = true;
    cfg.prune_detour       = true;
    cfg.merge_components   = true;
    cpu_refine_graph(g, cfg, vecs.data(), 2);

    const auto S = static_cast<knng::index_t>(-1);
    for (std::size_t i = 0; i < n * k; ++i) {
        const auto id = g.ids[i];
        if (id == S) continue;
        EXPECT_LT(static_cast<std::size_t>(id), n);
        EXPECT_GE(g.dists[i], 0.f);
    }
}

// Ablation: skip pruning and merging — graph should still be sorted.
TEST(GraphRefinement, RefineGraph_SkipPruneAndMerge_StillSorted) {
    CpuDeviceGraph g(2, 3);
    g.ids   = {1u, 0u, 0u,   // row 0 (unsorted distances)
               0u, 0u, 0u};  // row 1
    g.dists = {5.f, 2.f, 1.f,
               1.f, 1.f, 1.f};

    GraphRefinementConfig cfg;
    cfg.enforce_out_degree = true;
    cfg.rank_reorder       = false;
    cfg.add_reverse_edges  = false;
    cfg.prune_detour       = false;
    cfg.merge_components   = false;
    cpu_refine_graph(g, cfg);

    // Row 0 must be sorted ascending by distance
    EXPECT_LE(g.dists[0], g.dists[1]);
    EXPECT_LE(g.dists[1], g.dists[2]);
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
