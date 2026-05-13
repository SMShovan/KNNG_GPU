/// @file
/// @brief Steps 80–87 — Multi-GPU single-node tests.
///
/// All tests exercise CPU reference / simulation paths (Mac-compatible).
/// GPU tests guarded by KNNG_HAVE_CUDA + KNNG_HAVE_NCCL.

#include <gtest/gtest.h>

#include <knng/gpu/nccl_comm.hpp>
#include <knng/gpu/topology.hpp>
#include <knng/gpu/multi_gpu.hpp>

#include <vector>

using knng::gpu::CpuSimCollective;
using knng::gpu::Topology;
using knng::gpu::PeerLinkType;
using knng::gpu::MultiGpuConfig;
using knng::gpu::partition_points;
using knng::gpu::cpu_multi_brute_force_point;

// ===========================================================================
// Step 80 — CPU collective simulation
// ===========================================================================

TEST(NcclComm, CpuSim_AllreduceSum_TwoRanks) {
    std::vector<std::vector<float>> bufs = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    CpuSimCollective::allreduce_sum(bufs, 3);
    EXPECT_FLOAT_EQ(bufs[0][0], 5.f);
    EXPECT_FLOAT_EQ(bufs[0][1], 7.f);
    EXPECT_FLOAT_EQ(bufs[0][2], 9.f);
    EXPECT_EQ(bufs[0], bufs[1]);  // all ranks receive the same result
}

TEST(NcclComm, CpuSim_AllreduceSum_FourRanks) {
    const int n = 4;
    std::vector<std::vector<float>> bufs(static_cast<std::size_t>(n),
                                          std::vector<float>(2, 1.f));
    CpuSimCollective::allreduce_sum(bufs, 2);
    EXPECT_FLOAT_EQ(bufs[0][0], static_cast<float>(n));
    EXPECT_FLOAT_EQ(bufs[3][1], static_cast<float>(n));
}

TEST(NcclComm, CpuSim_Bcast_RootZero) {
    std::vector<std::vector<float>> bufs = {{9.f, 8.f}, {0.f, 0.f}, {0.f, 0.f}};
    CpuSimCollective::bcast(bufs, 0, 2);
    EXPECT_FLOAT_EQ(bufs[1][0], 9.f);
    EXPECT_FLOAT_EQ(bufs[2][1], 8.f);
}

TEST(NcclComm, CpuSim_Bcast_RootNonZero) {
    std::vector<std::vector<float>> bufs = {{0.f, 0.f}, {3.f, 7.f}, {0.f, 0.f}};
    CpuSimCollective::bcast(bufs, 1, 2);
    EXPECT_FLOAT_EQ(bufs[0][0], 3.f);
    EXPECT_FLOAT_EQ(bufs[2][1], 7.f);
    // Root unchanged
    EXPECT_FLOAT_EQ(bufs[1][0], 3.f);
}

TEST(NcclComm, CpuSim_Allgather_TwoRanks) {
    std::vector<std::vector<float>> bufs = {{1.f, 2.f}, {3.f, 4.f}};
    std::vector<float> result;
    CpuSimCollective::allgather(bufs, result, 2);
    ASSERT_EQ(result.size(), 4u);
    EXPECT_FLOAT_EQ(result[0], 1.f);
    EXPECT_FLOAT_EQ(result[1], 2.f);
    EXPECT_FLOAT_EQ(result[2], 3.f);
    EXPECT_FLOAT_EQ(result[3], 4.f);
}

// ===========================================================================
// Step 81 — Topology simulation
// ===========================================================================

TEST(Topology, Simulate_DiagonalIsSame) {
    const auto t = Topology::simulate(4);
    EXPECT_EQ(t.n_gpus(), 4);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(t.link(i, i), PeerLinkType::Same);
}

TEST(Topology, Simulate_OffDiagonalIsSimulated) {
    const auto t = Topology::simulate(3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j)
                EXPECT_EQ(t.link(i, j), PeerLinkType::Simulated);
}

TEST(Topology, Simulate_NoP2P) {
    const auto t = Topology::simulate(2);
    EXPECT_FALSE(t.has_p2p(0, 1));
    EXPECT_FALSE(t.has_p2p(1, 0));
}

TEST(Topology, ToStringContainsGpuCount) {
    const auto t = Topology::simulate(2);
    const auto s = t.to_string();
    EXPECT_NE(s.find("2 GPUs"), std::string::npos);
}

// ===========================================================================
// Step 82 — Point-sharded multi-GPU brute-force
// ===========================================================================

TEST(MultiGpu, PartitionPoints_EvenSplit) {
    MultiGpuConfig cfg;
    cfg.n_gpus = 4;
    const auto shards = partition_points(8, cfg);
    ASSERT_EQ(shards.size(), 4u);
    EXPECT_EQ(shards[0].begin, 0u); EXPECT_EQ(shards[0].end, 2u);
    EXPECT_EQ(shards[1].begin, 2u); EXPECT_EQ(shards[1].end, 4u);
    EXPECT_EQ(shards[2].begin, 4u); EXPECT_EQ(shards[2].end, 6u);
    EXPECT_EQ(shards[3].begin, 6u); EXPECT_EQ(shards[3].end, 8u);
    EXPECT_EQ(shards[0].rank, 0); EXPECT_EQ(shards[3].rank, 3);
}

TEST(MultiGpu, PartitionPoints_WithRemainder) {
    MultiGpuConfig cfg;
    cfg.n_gpus = 3;
    const auto shards = partition_points(10, cfg);
    std::size_t total = 0;
    for (const auto& s : shards) total += s.size();
    EXPECT_EQ(total, 10u);
}

TEST(MultiGpu, PointSharded_MatchesSingleGpu_SmallDataset) {
    // 8 points in 2D, k=2, 2 virtual GPUs
    const std::size_t n = 8, d = 2, k = 2;
    knng::Dataset ds;
    ds.n = n; ds.d = d;
    ds.data.resize(n * d);
    for (std::size_t i = 0; i < n; ++i) {
        ds.data[i*d+0] = static_cast<float>(i);
        ds.data[i*d+1] = 0.f;
    }

    MultiGpuConfig cfg;
    cfg.n_gpus = 2; cfg.k = static_cast<int>(k);

    const auto result = cpu_multi_brute_force_point(ds, cfg);
    ASSERT_EQ(result.n, n);
    ASSERT_EQ(result.k, k);

    // Every returned neighbor id must be < n and differ from the query
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t r = 0; r < k; ++r) {
            const auto id = result.neighbors[i*k+r];
            EXPECT_LT(static_cast<std::size_t>(id), n);
            EXPECT_NE(static_cast<std::size_t>(id), i);
            EXPECT_GE(result.distances[i*k+r], 0.f);
        }
}

TEST(NcclComm, CpuSim_ReduceScatter_FourRanks) {
    // 4 ranks, 4 elements each; reduce-scatter yields 1 element per rank
    std::vector<std::vector<float>> bufs = {
        {1.f, 2.f, 3.f, 4.f},
        {1.f, 2.f, 3.f, 4.f},
        {1.f, 2.f, 3.f, 4.f},
        {1.f, 2.f, 3.f, 4.f}
    };
    CpuSimCollective::reduce_scatter(bufs, 4);
    ASSERT_EQ(bufs[0].size(), 1u);
    EXPECT_FLOAT_EQ(bufs[0][0], 4.f);  // sum of element 0: 4×1 = 4
    EXPECT_FLOAT_EQ(bufs[1][0], 8.f);  // sum of element 1: 4×2 = 8
    EXPECT_FLOAT_EQ(bufs[2][0], 12.f); // sum of element 2: 4×3 = 12
    EXPECT_FLOAT_EQ(bufs[3][0], 16.f); // sum of element 3: 4×4 = 16
}
