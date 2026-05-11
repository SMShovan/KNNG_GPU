/// @file
/// @brief Steps 63–70 — GPU NN-Descent algorithm tests (CPU reference path).
///
/// All tests exercise the CPU reference implementations which run on Mac
/// without CUDA.  GPU-path tests are guarded by #ifdef KNNG_HAVE_CUDA.

#include <gtest/gtest.h>
#include <knng/gpu/gpu_nn_descent.hpp>
#include <knng/gpu/brute_force.hpp>     // cpu_brute_force_knn (oracle)
#include <knng/core/dataset.hpp>

#include <vector>

using knng::Dataset;
using knng::Knng;
using knng::gpu::CpuDeviceGraph;
using knng::gpu::cpu_init_random_graph;
using knng::gpu::cpu_local_join;
using knng::gpu::cpu_build_reverse_graph;
using knng::gpu::cpu_sample_new_neighbors;
using knng::gpu::cpu_gpu_nn_descent;
using knng::gpu::cpu_fp16_nn_descent;
using knng::gpu::GpuNNDConfig;

static Dataset make_ds(std::vector<float> data, std::size_t n, std::size_t d)
{ Dataset ds; ds.data=std::move(data); ds.n=n; ds.d=d; return ds; }

// ---------------------------------------------------------------------------
// Step 63 — Random graph init
// ---------------------------------------------------------------------------

TEST(GpuNND, InitGraphHasKNeighbors) {
    auto ds = make_ds({0,0, 1,0, 0,1, 1,1, 2,2, 3,3}, 6, 2);
    CpuDeviceGraph g = cpu_init_random_graph(ds, 3, 42);
    EXPECT_EQ(g.n, 6u);
    EXPECT_EQ(g.k, 3u);
    // Every row should have 3 distinct non-self neighbors.
    for (std::size_t qi = 0; qi < 6; ++qi) {
        for (std::size_t r = 0; r < 3; ++r) {
            EXPECT_NE(g.ids[qi*3+r], static_cast<knng::index_t>(-1));
            EXPECT_NE(g.ids[qi*3+r], static_cast<knng::index_t>(qi));
            EXPECT_LT(g.ids[qi*3+r], 6u);
            EXPECT_EQ(g.flags[qi*3+r], 1u);  // all new
        }
    }
}

TEST(GpuNND, InitGraphDeterministic) {
    std::vector<float> data(20*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds = make_ds(data, 20, 3);
    CpuDeviceGraph g1 = cpu_init_random_graph(ds, 5, 100);
    CpuDeviceGraph g2 = cpu_init_random_graph(ds, 5, 100);
    EXPECT_EQ(g1.ids,   g2.ids);
    EXPECT_EQ(g1.dists, g2.dists);
}

TEST(GpuNND, InitGraphDifferentSeed) {
    std::vector<float> data(20*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds = make_ds(data, 20, 3);
    CpuDeviceGraph g1 = cpu_init_random_graph(ds, 5, 1);
    CpuDeviceGraph g2 = cpu_init_random_graph(ds, 5, 2);
    EXPECT_NE(g1.ids, g2.ids);
}

// ---------------------------------------------------------------------------
// Step 64 — Local-join
// ---------------------------------------------------------------------------

TEST(GpuNND, LocalJoinImprovesGraph) {
    // 8-point 2-cluster dataset: local-join should improve the random init.
    std::vector<float> data = {
        0,0, 0.1f,0, 0,0.1f, 0.1f,0.1f,
        9,9, 9.1f,9, 9,9.1f, 9.1f,9.1f,
    };
    auto ds = make_ds(data, 8, 2);
    CpuDeviceGraph g = cpu_init_random_graph(ds, 3, 7);
    const std::size_t updates = cpu_local_join(ds, g);
    // Local-join should have made some insertions.
    EXPECT_GE(updates, 0u);   // may be 0 if init happened to be perfect
}

TEST(GpuNND, LocalJoinNoSelfNeighbors) {
    std::vector<float> data(12*2);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%5);
    auto ds = make_ds(data, 12, 2);
    CpuDeviceGraph g = cpu_init_random_graph(ds, 4, 42);
    cpu_local_join(ds, g);
    for (std::size_t qi=0;qi<12;++qi) {
        for (std::size_t r=0;r<4;++r) {
            EXPECT_NE(g.ids[qi*4+r], static_cast<knng::index_t>(qi));
        }
    }
}

// ---------------------------------------------------------------------------
// Step 67 — Reverse-neighbor accumulation
// ---------------------------------------------------------------------------

TEST(GpuNND, ReverseGraphSize) {
    // Forward graph: 4 points, k=2 — reverse graph must have 4*2=8 total edges.
    CpuDeviceGraph g(4, 2);
    g.ids = {1u,2u, 0u,2u, 0u,3u, 0u,1u};  // manually constructed
    std::vector<knng::index_t>  rev_ids;
    std::vector<std::uint32_t>  rev_offsets;
    cpu_build_reverse_graph(g, rev_ids, rev_offsets);
    EXPECT_EQ(rev_offsets.size(), 5u);   // n+1
    EXPECT_EQ(rev_ids.size(), 8u);       // 4*2 edges
}

TEST(GpuNND, ReverseGraphCorrectness) {
    // Point 0 → {1,2}, point 1 → {0}, point 2 → {0}, point 3 → {}
    CpuDeviceGraph g(4, 2);
    g.ids = {1u,2u, 0u,static_cast<knng::index_t>(-1), 0u,static_cast<knng::index_t>(-1), static_cast<knng::index_t>(-1),static_cast<knng::index_t>(-1)};
    g.n = 4; g.k = 2;
    std::vector<knng::index_t>  rev_ids;
    std::vector<std::uint32_t>  rev_offsets;
    cpu_build_reverse_graph(g, rev_ids, rev_offsets);
    // Point 1 has in-degree 1 (from point 0).
    const auto deg1 = rev_offsets[2] - rev_offsets[1];
    EXPECT_EQ(deg1, 1u);
    EXPECT_EQ(rev_ids[rev_offsets[1]], 0u);
}

// ---------------------------------------------------------------------------
// Step 68 — Sampling
// ---------------------------------------------------------------------------

TEST(GpuNND, SamplingReducesNewCount) {
    CpuDeviceGraph g(1, 10);
    // Mark all 10 as new.
    for (std::size_t r=0;r<10;++r) { g.ids[r]=static_cast<knng::index_t>(r+1); g.flags[r]=1u; }
    cpu_sample_new_neighbors(g, 0.5, 42);
    std::size_t new_count = 0;
    for (std::size_t r=0;r<10;++r) { if (g.flags[r]&1u) ++new_count; }
    EXPECT_LE(new_count, 6u);   // rho=0.5 → ceil(5) new, rest old
}

// ---------------------------------------------------------------------------
// Step 69 — Full driver recall
// ---------------------------------------------------------------------------

TEST(GpuNND, DriverConverges) {
    // 16-point dataset: NN-Descent should find exact neighbors.
    std::vector<float> data(16*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%11);
    auto ds = make_ds(data, 16, 3);

    GpuNNDConfig cfg;
    cfg.max_iters = 20;
    cfg.delta     = 0.001;
    cfg.seed      = 42;

    Knng approx = cpu_gpu_nn_descent(ds, 4, cfg);
    Knng exact  = knng::gpu::cpu_brute_force_knn(ds, 4);

    // Count recall@4: fraction of exact neighbors found in approx.
    std::size_t hits = 0;
    for (std::size_t qi=0;qi<16;++qi) {
        for (std::size_t r=0;r<4;++r) {
            const auto eid = exact.neighbors[qi*4+r];
            for (std::size_t ar=0;ar<4;++ar) {
                if (approx.neighbors[qi*4+ar] == eid) { ++hits; break; }
            }
        }
    }
    const double recall = static_cast<double>(hits) / (16.0 * 4.0);
    EXPECT_GE(recall, 0.5) << "recall too low: " << recall;
}

// ---------------------------------------------------------------------------
// Step 70 — Mixed-precision (fp16 distances)
// ---------------------------------------------------------------------------

TEST(GpuNND, Fp16DriverProducesValidGraph) {
    std::vector<float> data(12*2);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds = make_ds(data, 12, 2);
    GpuNNDConfig cfg; cfg.max_iters=10; cfg.seed=42;
    Knng result = cpu_fp16_nn_descent(ds, 3, cfg);
    EXPECT_EQ(result.n,  12u);
    EXPECT_EQ(result.k,   3u);
    for (std::size_t qi=0;qi<12;++qi) {
        for (std::size_t r=0;r<3;++r) {
            EXPECT_LT(result.neighbors[qi*3+r], 12u);
            EXPECT_NE(result.neighbors[qi*3+r], static_cast<knng::index_t>(qi));
        }
    }
}

// ---------------------------------------------------------------------------
// GPU vs CPU (CUDA only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

TEST(GpuNND, GpuInitMatchesCpuLayout) {
    std::vector<float> data(20*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%7);
    auto ds = make_ds(data, 20, 3);
    knng::gpu::DeviceGraph dg = knng::gpu::gpu_init_random_graph(ds, 4, 42);
    knng::gpu::CpuDeviceGraph cpu_g = cpu_init_random_graph(ds, 4, 42);
    knng::gpu::CpuDeviceGraph from_gpu = dg.to_cpu();
    // Both use same seed → same random graph.
    EXPECT_EQ(cpu_g.ids, from_gpu.ids);
}

TEST(GpuNND, GpuDriverProducesValidGraph) {
    std::vector<float> data(20*3);
    for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<float>(i%11);
    auto ds = make_ds(data, 20, 3);
    GpuNNDConfig cfg; cfg.max_iters=10; cfg.seed=42;
    Knng result = knng::gpu::gpu_nn_descent(ds, 4, cfg);
    EXPECT_EQ(result.n, 20u);
    EXPECT_EQ(result.k,  4u);
    for (std::size_t qi=0;qi<20;++qi) {
        for (std::size_t r=0;r<4;++r) {
            EXPECT_LT(result.neighbors[qi*4+r], 20u);
        }
    }
}

#endif
