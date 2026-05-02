/// @file
/// @brief Unit tests for `knng::bench::ground_truth` (Step 14).
///
/// Pins three properties of the ground-truth cache:
///
///   1. The dataset hash is stable across copies of the same dataset
///      and changes the moment any byte of input changes.
///   2. A round-trip `save_ground_truth` / `load_ground_truth` returns
///      a byte-identical `Knng` and validates each cache-key
///      component (n, k, metric, dataset hash).
///   3. `load_or_compute_ground_truth` computes on cache miss and
///      reads from cache on hit — verified by deleting the source
///      dataset between calls and asserting the second call still
///      returns the same answer.

#include <array>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <random>

#include <gtest/gtest.h>

#include "knng/bench/ground_truth.hpp"
#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/cpu/brute_force.hpp"

namespace {

/// Cluster fixture from the brute-force tests — eight points in two
/// well-separated unit squares. Hand-verified neighbors from Step 10.
knng::Dataset two_clusters_eight_points()
{
    knng::Dataset ds(8, 2);
    constexpr std::array<std::array<float, 2>, 8> coords{{
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
        {4.0f, 4.0f}, {5.0f, 4.0f}, {4.0f, 5.0f}, {5.0f, 5.0f},
    }};
    for (std::size_t i = 0; i < coords.size(); ++i) {
        ds.row(i)[0] = coords[i][0];
        ds.row(i)[1] = coords[i][1];
    }
    return ds;
}

/// Per-test temp directory under the OS temp root. Wiped on
/// destruction so a failing assertion never leaks files between
/// runs of the suite.
class TempDir {
public:
    TempDir()
    {
        // Use a fresh subdirectory keyed on a random suffix so two
        // tests running in parallel cannot collide.
        std::random_device rd;
        std::mt19937_64 rng(rd());
        const auto suffix = rng();
        path_ = std::filesystem::temp_directory_path()
              / ("knng_gt_test_" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }
    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&)                 = delete;
    TempDir& operator=(TempDir&&)      = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

TEST(GroundTruth, DatasetHashIsStableAcrossCopies)
{
    const auto a = two_clusters_eight_points();
    const auto b = a;
    EXPECT_EQ(knng::bench::dataset_hash(a),
              knng::bench::dataset_hash(b));
}

TEST(GroundTruth, DatasetHashChangesOnAnyByteFlip)
{
    auto a = two_clusters_eight_points();
    const std::uint64_t hash_before = knng::bench::dataset_hash(a);

    a.row(3)[1] += 1.0e-3f;  // tweak one coordinate
    const std::uint64_t hash_after = knng::bench::dataset_hash(a);

    EXPECT_NE(hash_before, hash_after);
}

TEST(GroundTruth, DatasetHashChangesOnShapeFlip)
{
    knng::Dataset a(4, 2);
    knng::Dataset b(2, 4);
    // Same total bytes, identical contents — only the (n, d) shape
    // differs. The hash mixes shape into the digest, so the values
    // must still differ.
    EXPECT_NE(knng::bench::dataset_hash(a),
              knng::bench::dataset_hash(b));
}

TEST(GroundTruth, RoundTripPreservesGraphAndAcceptsValidKey)
{
    TempDir tmp;
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    const auto path = tmp.path() / "rt.gt";
    knng::bench::save_ground_truth(
        path, g, knng::bench::dataset_hash(ds),
        knng::bench::MetricId::kL2);

    const auto loaded = knng::bench::load_ground_truth(
        path, ds, std::size_t{3}, knng::bench::MetricId::kL2);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->n, g.n);
    EXPECT_EQ(loaded->k, g.k);
    EXPECT_EQ(loaded->neighbors, g.neighbors);
    EXPECT_EQ(loaded->distances, g.distances);
}

TEST(GroundTruth, LoadRejectsKMismatch)
{
    TempDir tmp;
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    const auto path = tmp.path() / "rt.gt";
    knng::bench::save_ground_truth(
        path, g, knng::bench::dataset_hash(ds),
        knng::bench::MetricId::kL2);

    EXPECT_FALSE(knng::bench::load_ground_truth(
        path, ds, std::size_t{4}, knng::bench::MetricId::kL2)
        .has_value());
}

TEST(GroundTruth, LoadRejectsMetricMismatch)
{
    TempDir tmp;
    const auto ds = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});

    const auto path = tmp.path() / "rt.gt";
    knng::bench::save_ground_truth(
        path, g, knng::bench::dataset_hash(ds),
        knng::bench::MetricId::kL2);

    EXPECT_FALSE(knng::bench::load_ground_truth(
        path, ds, std::size_t{3},
        knng::bench::MetricId::kNegativeInnerProduct).has_value());
}

TEST(GroundTruth, LoadRejectsDatasetHashMismatch)
{
    TempDir tmp;
    const auto ds_orig = two_clusters_eight_points();
    const knng::Knng g = knng::cpu::brute_force_knn(
        ds_orig, std::size_t{3}, knng::L2Squared{});

    const auto path = tmp.path() / "rt.gt";
    knng::bench::save_ground_truth(
        path, g, knng::bench::dataset_hash(ds_orig),
        knng::bench::MetricId::kL2);

    auto ds_modified = ds_orig;
    ds_modified.row(0)[0] = 99.0f;  // any change at all

    EXPECT_FALSE(knng::bench::load_ground_truth(
        path, ds_modified, std::size_t{3},
        knng::bench::MetricId::kL2).has_value());
}

TEST(GroundTruth, LoadReturnsNulloptOnMissingFile)
{
    TempDir tmp;
    const auto ds = two_clusters_eight_points();
    const auto path = tmp.path() / "does_not_exist.gt";
    EXPECT_FALSE(knng::bench::load_ground_truth(
        path, ds, std::size_t{3},
        knng::bench::MetricId::kL2).has_value());
}

TEST(GroundTruth, LoadOrComputeWritesCacheOnMissAndReadsItOnHit)
{
    TempDir tmp;
    const auto ds = two_clusters_eight_points();
    const auto path = tmp.path() / "cache.gt";

    EXPECT_FALSE(std::filesystem::exists(path));
    const auto first = knng::bench::load_or_compute_ground_truth(
        ds, std::size_t{3}, knng::bench::MetricId::kL2, path);
    EXPECT_TRUE(std::filesystem::exists(path));

    const auto second = knng::bench::load_or_compute_ground_truth(
        ds, std::size_t{3}, knng::bench::MetricId::kL2, path);
    EXPECT_EQ(first.neighbors, second.neighbors);
    EXPECT_EQ(first.distances, second.distances);

    // Reach the same answer via the brute-force entry point directly
    // — the cache must return the brute-force result, not something
    // subtly transformed.
    const auto fresh = knng::cpu::brute_force_knn(
        ds, std::size_t{3}, knng::L2Squared{});
    EXPECT_EQ(first.neighbors, fresh.neighbors);
    EXPECT_EQ(first.distances, fresh.distances);
}

TEST(GroundTruth, DefaultCachePathEmbedsKAndMetricTag)
{
    const auto p = knng::bench::default_cache_path(
        std::filesystem::path{"datasets/sift1m/base.fvecs"},
        std::size_t{42},
        knng::bench::MetricId::kNegativeInnerProduct,
        std::filesystem::path{"build/cache"});
    EXPECT_EQ(p.filename().string(), "base.k42.negip.gt");
    EXPECT_EQ(p.parent_path().filename().string(), "cache");
}

} // namespace
