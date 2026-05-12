/// @file
/// @brief Step 61 — DeviceNeighborList AoS struct tests.
///
/// Verifies the struct size, alignment, bit-flag helpers, and AoS↔SoA
/// conversion utilities.

#include <gtest/gtest.h>
#include <knng/gpu/neighbor_list.cuh>

#include <cstddef>

using knng::gpu::DeviceNeighborList;
using knng::gpu::aos_to_soa;
using knng::gpu::soa_to_aos;

// ---------------------------------------------------------------------------
// Struct layout
// ---------------------------------------------------------------------------

TEST(NeighborList, DefaultKSize256) {
    // K_MAX=16 (default): ids[16](64B)+dists[16](64B)+flags(4B)=132B
    // alignas(128) → next multiple of 128 ≥ 132 → sizeof = 256.
    EXPECT_EQ(sizeof(DeviceNeighborList<16>), 256u);
}

TEST(NeighborList, Alignment128) {
    // alignas(128) → alignof must be exactly 128.
    EXPECT_EQ(alignof(DeviceNeighborList<16>), 128u);
}

TEST(NeighborList, SizeScalesWithK) {
    // K=8:  ids[8](32)+dists[8](32)+flags(4)=68  → sizeof=128
    // K=32: ids[32](128)+dists[32](128)+flags(4)=260 → sizeof=384
    EXPECT_EQ(sizeof(DeviceNeighborList<8>),  128u);
    EXPECT_EQ(sizeof(DeviceNeighborList<32>), 384u);
}

// ---------------------------------------------------------------------------
// Flag bit manipulation
// ---------------------------------------------------------------------------

TEST(NeighborList, SetNewAndOld) {
    DeviceNeighborList<32> nl{};
    nl.flags = 0;
    nl.set_new(0);
    EXPECT_TRUE(nl.is_new(0));
    EXPECT_FALSE(nl.is_new(1));
    nl.set_new(5);
    EXPECT_TRUE(nl.is_new(5));
    nl.set_old(0);
    EXPECT_FALSE(nl.is_new(0));
    EXPECT_TRUE(nl.is_new(5));
}

TEST(NeighborList, AllBitsIndependent) {
    DeviceNeighborList<32> nl{};
    nl.flags = 0;
    for (unsigned int b = 0; b < 32; ++b) { nl.set_new(b); }
    EXPECT_EQ(nl.flags, 0xFFFFFFFFu);
    for (unsigned int b = 0; b < 32; ++b) {
        nl.set_old(b);
        EXPECT_EQ(nl.flags, (0xFFFFFFFFu >> (b + 1)) << (b + 1))
            << "after clearing bit " << b;
    }
    EXPECT_EQ(nl.flags, 0u);
}

// ---------------------------------------------------------------------------
// AoS ↔ SoA conversion
// ---------------------------------------------------------------------------

TEST(NeighborList, AosToSoa) {
    DeviceNeighborList<4> nl{};
    nl.ids[0] = 10; nl.dists[0] = 1.f; nl.set_new(0);
    nl.ids[1] = 20; nl.dists[1] = 2.f; nl.set_old(1);
    nl.ids[2] = 30; nl.dists[2] = 3.f; nl.set_new(2);
    nl.ids[3] = 40; nl.dists[3] = 4.f; nl.set_old(3);

    knng::index_t  ids  [4] = {};
    float          dists[4] = {};
    std::uint32_t  flags[4] = {};
    aos_to_soa(nl, 0, 4, ids, dists, flags);

    EXPECT_EQ(ids[0], 10u); EXPECT_FLOAT_EQ(dists[0], 1.f); EXPECT_EQ(flags[0], 1u);
    EXPECT_EQ(ids[1], 20u); EXPECT_FLOAT_EQ(dists[1], 2.f); EXPECT_EQ(flags[1], 0u);
    EXPECT_EQ(ids[2], 30u); EXPECT_FLOAT_EQ(dists[2], 3.f); EXPECT_EQ(flags[2], 1u);
    EXPECT_EQ(ids[3], 40u); EXPECT_FLOAT_EQ(dists[3], 4.f); EXPECT_EQ(flags[3], 0u);
}

TEST(NeighborList, SoaToAos) {
    knng::index_t ids  [4] = {5, 6, 7, 8};
    float         dists[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    std::uint32_t flags[4] = {1, 0, 1, 0};

    DeviceNeighborList<4> nl{};
    soa_to_aos(0, 4, ids, dists, flags, nl);

    EXPECT_EQ(nl.ids[0], 5u); EXPECT_FLOAT_EQ(nl.dists[0], 0.5f); EXPECT_TRUE(nl.is_new(0));
    EXPECT_EQ(nl.ids[1], 6u); EXPECT_FLOAT_EQ(nl.dists[1], 1.5f); EXPECT_FALSE(nl.is_new(1));
    EXPECT_EQ(nl.ids[2], 7u); EXPECT_FLOAT_EQ(nl.dists[2], 2.5f); EXPECT_TRUE(nl.is_new(2));
    EXPECT_EQ(nl.ids[3], 8u); EXPECT_FLOAT_EQ(nl.dists[3], 3.5f); EXPECT_FALSE(nl.is_new(3));
}

TEST(NeighborList, RoundTrip) {
    DeviceNeighborList<4> src{};
    src.ids[0]=1; src.ids[1]=2; src.ids[2]=3; src.ids[3]=4;
    src.dists[0]=0.1f; src.dists[1]=0.2f; src.dists[2]=0.3f; src.dists[3]=0.4f;
    src.set_new(0); src.set_new(2);   // 0,2 new; 1,3 old

    knng::index_t  ids[4]{}; float dists[4]{}; std::uint32_t flags[4]{};
    aos_to_soa(src, 0, 4, ids, dists, flags);

    DeviceNeighborList<4> dst{};
    soa_to_aos(0, 4, ids, dists, flags, dst);

    for (unsigned int r = 0; r < 4; ++r) {
        EXPECT_EQ(dst.ids[r], src.ids[r]);
        EXPECT_FLOAT_EQ(dst.dists[r], src.dists[r]);
        EXPECT_EQ(dst.is_new(r), src.is_new(r));
    }
}
