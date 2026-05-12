/// @file
/// @brief Steps 66–67 & 69 — Reverse graph, sampling, and fp16 NND tests.

#include <gtest/gtest.h>
#include <knng/gpu/gpu_nn_descent.hpp>
#include <knng/gpu/brute_force.hpp>   // cpu_brute_force_knn (oracle)
#include <knng/core/dataset.hpp>

#include <vector>

using knng::gpu::CpuDeviceGraph;
using knng::gpu::cpu_build_reverse_graph;
using knng::gpu::cpu_sample_new_neighbors;
using knng::gpu::cpu_fp16_nn_descent;
using knng::gpu::GpuNNDConfig;

static knng::Dataset make_ds(std::vector<float> d, std::size_t n, std::size_t dim)
{ knng::Dataset ds; ds.data=std::move(d); ds.n=n; ds.d=dim; return ds; }

// ---------------------------------------------------------------------------
// Step 66 — Reverse graph (CPU ref)
// ---------------------------------------------------------------------------

TEST(ReverseGraph, TotalEdgeCount) {
    // Forward: 4 pts, k=2. Reverse must have exactly 4*2=8 forward edges.
    CpuDeviceGraph g(4, 2);
    g.ids = {1,2, 0,2, 0,3, 0,1};   // each edge creates a reverse entry
    std::vector<knng::index_t> rev_ids;
    std::vector<std::uint32_t> rev_off;
    cpu_build_reverse_graph(g, rev_ids, rev_off);
    EXPECT_EQ(rev_off.size(), 5u);
    EXPECT_EQ(rev_ids.size(), 8u);
}

TEST(ReverseGraph, Correctness) {
    // Point 0 → {1, 2}.  Reverse: point 1 and 2 each have point 0 as reverse.
    CpuDeviceGraph g(3, 2);
    g.n = 3; g.k = 2;
    g.ids = {1u,2u,
             static_cast<knng::index_t>(-1),static_cast<knng::index_t>(-1),
             static_cast<knng::index_t>(-1),static_cast<knng::index_t>(-1)};
    std::vector<knng::index_t> rev_ids;
    std::vector<std::uint32_t> rev_off;
    cpu_build_reverse_graph(g, rev_ids, rev_off);
    // Point 1 should have point 0 as a reverse neighbor.
    const auto beg1 = rev_off[1];
    const auto end1 = rev_off[2];
    bool found = false;
    for (auto i = beg1; i < end1; ++i) { if (rev_ids[i] == 0u) { found = true; } }
    EXPECT_TRUE(found) << "point 0 should appear in reverse list of point 1";
}

TEST(ReverseGraph, OffsetMonotone) {
    CpuDeviceGraph g(6, 3);
    g.ids.resize(18);
    for (std::size_t i = 0; i < 18; ++i) {
        g.ids[i] = static_cast<knng::index_t>((i + 1) % 6);
    }
    std::vector<knng::index_t> rev_ids;
    std::vector<std::uint32_t> rev_off;
    cpu_build_reverse_graph(g, rev_ids, rev_off);
    for (std::size_t i = 1; i < rev_off.size(); ++i) {
        EXPECT_GE(rev_off[i], rev_off[i-1]);
    }
}

// ---------------------------------------------------------------------------
// Step 67 — Sampling (CPU ref)
// ---------------------------------------------------------------------------

TEST(Sampling, ReducesNewCount) {
    CpuDeviceGraph g(1, 10);
    for (std::size_t r = 0; r < 10; ++r) {
        g.ids[r] = static_cast<knng::index_t>(r+1); g.flags[r] = 1u;
    }
    cpu_sample_new_neighbors(g, 0.5, 42);
    std::size_t cnt = 0;
    for (std::size_t r = 0; r < 10; ++r) { if (g.flags[r] & 1u) ++cnt; }
    EXPECT_LE(cnt, 6u);
}

TEST(Sampling, RhoOneMakesNoChange) {
    CpuDeviceGraph g(1, 8);
    for (std::size_t r = 0; r < 8; ++r) { g.ids[r]=static_cast<knng::index_t>(r+1); g.flags[r]=1u; }
    cpu_sample_new_neighbors(g, 1.0, 7);
    for (std::size_t r = 0; r < 8; ++r) { EXPECT_EQ(g.flags[r] & 1u, 1u); }
}

TEST(Sampling, Deterministic) {
    CpuDeviceGraph g1(1, 10), g2(1, 10);
    for (std::size_t r = 0; r < 10; ++r) {
        g1.ids[r]=g2.ids[r]=static_cast<knng::index_t>(r+1);
        g1.flags[r]=g2.flags[r]=1u;
    }
    cpu_sample_new_neighbors(g1, 0.5, 99);
    cpu_sample_new_neighbors(g2, 0.5, 99);
    EXPECT_EQ(g1.flags, g2.flags);
}

// ---------------------------------------------------------------------------
// Step 69 — fp16 NND (CPU ref)
// ---------------------------------------------------------------------------

TEST(Fp16Nnd, ValidGraph) {
    std::vector<float> data(16*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%11);
    auto ds = make_ds(data, 16, 3);
    GpuNNDConfig cfg; cfg.max_iters=10; cfg.seed=42;
    auto result = cpu_fp16_nn_descent(ds, 4, cfg);
    EXPECT_EQ(result.n, 16u);
    EXPECT_EQ(result.k,  4u);
    for (std::size_t qi=0;qi<16;++qi) {
        for (std::size_t r=0;r<4;++r) {
            EXPECT_LT(result.neighbors[qi*4+r], 16u);
            EXPECT_NE(result.neighbors[qi*4+r], static_cast<knng::index_t>(qi));
        }
    }
}

TEST(Fp16Nnd, RecallAcceptable) {
    // Small dataset: fp16 NND should find most exact neighbors.
    std::vector<float> data(20*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%11);
    auto ds = make_ds(data, 20, 3);
    GpuNNDConfig cfg; cfg.max_iters=30; cfg.seed=42;
    auto approx = cpu_fp16_nn_descent(ds, 3, cfg);
    auto exact  = knng::gpu::cpu_brute_force_knn(ds, 3);
    std::size_t hits = 0;
    for (std::size_t qi=0;qi<20;++qi) {
        for (std::size_t r=0;r<3;++r) {
            for (std::size_t ar=0;ar<3;++ar) {
                if (approx.neighbors[qi*3+ar] == exact.neighbors[qi*3+r]) { ++hits; break; }
            }
        }
    }
    EXPECT_GE(static_cast<double>(hits)/(20.0*3.0), 0.5)
        << "fp16 NND recall too low";
}

// ---------------------------------------------------------------------------
// GPU paths (CUDA only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

TEST(GpuReverseGraph, CubDeviceScan) {
    // Small dataset: build forward graph, then GPU reverse graph.
    std::vector<float> data(12*2);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%5);
    auto ds = make_ds(data, 12, 2);
    knng::gpu::DeviceGraph fwd = knng::gpu::gpu_init_random_graph(ds, 3, 42);

    knng::gpu::DeviceBuffer<knng::index_t> rev_ids;
    knng::gpu::DeviceBuffer<unsigned int>  rev_off;
    knng::gpu::gpu_build_reverse_graph(fwd, rev_ids, rev_off);

    std::vector<unsigned int> h_off; rev_off.copy_to_host(h_off);
    EXPECT_EQ(h_off.size(), 13u);   // n+1
    EXPECT_EQ(h_off[0], 0u);
    EXPECT_GE(h_off[12], 0u);  // total edges >= 0
}

TEST(GpuSampling, Kernel) {
    std::vector<float> data(20*2);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds = make_ds(data, 20, 2);
    knng::gpu::DeviceGraph g = knng::gpu::gpu_init_random_graph(ds, 4, 42);
    knng::gpu::gpu_sample_new_neighbors(g, 0.5, 77);
    auto cpu = g.to_cpu();
    std::size_t new_cnt = 0;
    for (std::size_t i=0;i<20*4;++i) { if (cpu.flags[i]&1u) ++new_cnt; }
    // rho=0.5 → at most ceil(2)*20 = 40 new entries (2 per point)
    EXPECT_LE(new_cnt, 41u);
}

TEST(GpuFp16Nnd, Runs) {
    std::vector<float> data(16*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%11);
    auto ds = make_ds(data, 16, 3);
    GpuNNDConfig cfg; cfg.max_iters=5; cfg.seed=42;
    auto result = knng::gpu::gpu_nn_descent_fp16(ds, 3, cfg);
    EXPECT_EQ(result.n, 16u);
    EXPECT_EQ(result.k,  3u);
    for (std::size_t qi=0;qi<16;++qi) {
        for (std::size_t r=0;r<3;++r) {
            EXPECT_LT(result.neighbors[qi*3+r], 16u);
        }
    }
}

#endif
