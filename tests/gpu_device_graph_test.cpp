/// @file
/// @brief Step 62 — DeviceGraph and CpuDeviceGraph tests.

#include <gtest/gtest.h>
#include <knng/gpu/device_graph.hpp>
#include <knng/gpu/brute_force.hpp>   // cpu_brute_force_knn
#include <knng/core/dataset.hpp>

#include <vector>

using knng::gpu::CpuDeviceGraph;
using knng::gpu::DeviceGraph;
using knng::gpu::kSentinelDist;

// ---------------------------------------------------------------------------
// CpuDeviceGraph construction
// ---------------------------------------------------------------------------

TEST(DeviceGraph, DefaultConstruct) {
    CpuDeviceGraph g(4, 3);
    EXPECT_EQ(g.n, 4u);
    EXPECT_EQ(g.k, 3u);
    EXPECT_EQ(g.ids.size(), 12u);
    EXPECT_EQ(g.dist(0,0), kSentinelDist);
    EXPECT_EQ(g.id(0,0), static_cast<knng::index_t>(-1));
    EXPECT_EQ(g.flag(0,0), 0u);  // old
}

TEST(DeviceGraph, FromKnng) {
    knng::Knng knn(3, 2);
    knn.neighbors = {1, 2,  0, 2,  0, 1};
    knn.distances = {1.f, 4.f,  1.f, 9.f,  4.f, 9.f};
    CpuDeviceGraph g = CpuDeviceGraph::from_knng(knn);
    EXPECT_EQ(g.n, 3u);
    EXPECT_EQ(g.k, 2u);
    EXPECT_EQ(g.id(0,0), 1u);
    EXPECT_EQ(g.dist(0,1), 4.f);
    EXPECT_EQ(g.flag(1,0), 1u);   // all marked new
}

TEST(DeviceGraph, ToKnng) {
    knng::Knng knn(2, 2);
    knn.neighbors = {1, 0,  0, 1};
    knn.distances = {1.f, 2.f,  1.f, 2.f};
    CpuDeviceGraph g = CpuDeviceGraph::from_knng(knn);
    knng::Knng back = g.to_knng();
    EXPECT_EQ(back.neighbors, knn.neighbors);
    EXPECT_EQ(back.distances, knn.distances);
}

TEST(DeviceGraph, TryInsertBetter) {
    CpuDeviceGraph g(1, 2);
    // Insert initial entries manually.
    g.ids  [0] = 10; g.dists[0] = 5.f; g.flags[0] = 0u;
    g.ids  [1] = 20; g.dists[1] = 9.f; g.flags[1] = 0u;
    // Insert a better candidate (replaces worst=9).
    bool changed = g.try_insert(0, 30, 3.f);
    EXPECT_TRUE(changed);
    EXPECT_FLOAT_EQ(g.worst_dist(0), 5.f);
    EXPECT_EQ(g.flag(0, 1), 1u);  // newly inserted → is_new
}

TEST(DeviceGraph, TryInsertWorst) {
    CpuDeviceGraph g(1, 2);
    g.ids[0] = 10; g.dists[0] = 1.f;
    g.ids[1] = 20; g.dists[1] = 2.f;
    bool changed = g.try_insert(0, 99, 10.f);
    EXPECT_FALSE(changed);
}

TEST(DeviceGraph, TryInsertDuplicate) {
    CpuDeviceGraph g(1, 2);
    g.ids[0] = 5; g.dists[0] = 1.f;
    g.ids[1] = 10; g.dists[1] = 2.f;
    bool changed = g.try_insert(0, 5, 0.5f);  // already present
    EXPECT_FALSE(changed);
}

TEST(DeviceGraph, MarkOld) {
    CpuDeviceGraph g(1, 3);
    g.flags[0] = 1; g.flags[1] = 1; g.flags[2] = 0;
    g.mark_old(0);
    EXPECT_EQ(g.flag(0,0), 0u);
    EXPECT_EQ(g.flag(0,1), 0u);
    EXPECT_EQ(g.flag(0,2), 0u);
}

// ---------------------------------------------------------------------------
// GPU DeviceGraph round-trip (always works via CPU stub)
// ---------------------------------------------------------------------------

TEST(DeviceGraph, DeviceRoundTrip) {
    knng::Knng knn(3, 2);
    knn.neighbors = {1,2, 0,2, 0,1};
    knn.distances = {1.f,4.f, 1.f,9.f, 4.f,9.f};

    DeviceGraph dg = DeviceGraph::from_knng(knn);
    EXPECT_EQ(dg.n, 3u);
    EXPECT_EQ(dg.k, 2u);

    knng::Knng back = dg.to_knng();
    EXPECT_EQ(back.neighbors, knn.neighbors);
    EXPECT_EQ(back.distances, knn.distances);
}

TEST(DeviceGraph, DeviceUploadDownload) {
    CpuDeviceGraph cpu(2, 2);
    cpu.ids   = {5u, 6u, 7u, 8u};
    cpu.dists = {1.f, 2.f, 3.f, 4.f};
    cpu.flags = {1u, 0u, 0u, 1u};

    DeviceGraph dg(cpu);
    CpuDeviceGraph back = dg.to_cpu();

    EXPECT_EQ(back.ids,   cpu.ids);
    EXPECT_EQ(back.dists, cpu.dists);
    EXPECT_EQ(back.flags, cpu.flags);
}
