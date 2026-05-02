#pragma once

/// @file
/// @brief Ground-truth KNN cache shared across every benchmark.
///
/// Most measurements the project will care about — recall@k from
/// Step 15 onwards, the regression suite from Phase 13 — need an
/// *exact* nearest-neighbor graph to compare an approximate builder
/// against. Recomputing brute-force ground truth on every benchmark
/// run is wasteful (a single SIFT1M brute-force takes minutes on a
/// laptop) and would couple every micro-bench to the wall time of the
/// thing it is trying to measure.
///
/// The ground-truth cache solves that by computing brute-force KNN
/// once per `(dataset, k, metric)` triple and persisting the result
/// to a small file in a documented binary format. A subsequent run
/// hits the cache, skips the recompute, and returns the same `Knng`.
///
/// ## Cache key
///
/// The triple `(dataset, k, metric)` is reduced to a single 64-bit
/// **dataset hash** plus the two scalar parameters. The hash is a
/// 64-bit FNV-1a digest over `(n, d, raw float bytes)` — see
/// `dataset_hash` in this header. FNV-1a is not cryptographic; it is
/// simply a stable, fast, no-dependency way to detect "the dataset
/// changed" between runs. Two distinct datasets that happen to
/// collide are astronomically unlikely at any realistic n×d, and a
/// collision merely reads stale ground truth — it is not a security
/// boundary.
///
/// ## On-disk format (version 1)
///
/// Fixed-width 64-byte header, then payload. All multi-byte integers
/// and floats are little-endian (matching `tools/build_knng.cpp`'s
/// `.knng` format from Step 13).
///
/// ```
/// offset  size  field
/// ------  ----  -----
///   0      8    magic               = "KNNGRDTR" (ASCII, no NUL)
///   8      4    format_version      = 1 (uint32)
///  12      4    index_byte_width    = 4 (uint32; matches knng::index_t)
///  16      4    distance_byte_width = 4 (uint32; matches float)
///  20      4    metric_id           = 0 (l2) | 1 (negative_inner_product)
///  24      8    dataset_hash        (uint64; FNV-1a over n,d,bytes)
///  32      8    n                   (uint64; rows in the source dataset)
///  40      8    k                   (uint64; neighbors per row)
///  48     16    reserved            (zero-filled, brings header to 64 B)
///  64    n*k*4  neighbors           (uint32 row-major)
///  ...   n*k*4  distances           (float32 row-major, same layout)
/// ```
///
/// A reader that finds an unexpected magic, version, byte width, or
/// (n, k, dataset_hash, metric_id) returns "cache miss" and the
/// caller falls back to recomputing.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>

#include "knng/core/dataset.hpp"
#include "knng/core/graph.hpp"

namespace knng::bench {

/// Wire-level metric identifier — matches the `metric_id` field of
/// the `.knng` and `.gt` formats. Spelled as an enum (not a free
/// `uint32_t`) so callers cannot pass an arbitrary number.
enum class MetricId : std::uint32_t {
    kL2                   = 0,  ///< `knng::L2Squared`
    kNegativeInnerProduct = 1,  ///< `knng::NegativeInnerProduct`
};

/// Stable 64-bit FNV-1a digest over a dataset's logical content.
///
/// Hashes `n`, `d`, then the raw float bytes of `ds.data` in their
/// in-memory little-endian representation. The resulting value is
/// the cache key's dataset component — two `Dataset`s with the same
/// `(n, d, data)` always hash to the same value, two different
/// datasets almost never collide.
///
/// Not cryptographic; not endian-portable across hosts of different
/// endianness (the project targets little-endian only — see
/// `src/io/fvecs.cpp` for the same caveat).
[[nodiscard]] std::uint64_t dataset_hash(const Dataset& ds) noexcept;

/// Try to load a cached ground-truth graph from `path` and validate
/// that it matches `(ds, k, metric)`. Returns `std::nullopt` on
/// every failure mode (missing file, bad magic, version mismatch,
/// cache key mismatch, truncated payload). Never throws; a corrupt
/// or stale cache should be silently overwritten by the caller.
[[nodiscard]] std::optional<Knng>
load_ground_truth(const std::filesystem::path& path,
                  const Dataset& ds,
                  std::size_t k,
                  MetricId metric) noexcept;

/// Serialize `g` to `path` in the documented binary format. Throws
/// `std::runtime_error` on I/O failure. The intermediate file is
/// written under `path + ".tmp"` and renamed into place so a crash
/// mid-write cannot leave a partial cache file the next run would
/// confuse with a valid one.
void save_ground_truth(const std::filesystem::path& path,
                       const Knng& g,
                       std::uint64_t dataset_hash_value,
                       MetricId metric);

/// Convenience: compute or load the exact ground-truth KNN for
/// `(ds, k, metric)`. Looks for an existing cache at `cache_path`,
/// validates it against the cache key, returns it on hit. On miss,
/// computes brute-force, writes the cache, and returns the freshly
/// computed graph.
///
/// The brute-force compute uses `knng::cpu::brute_force_knn` —
/// algorithmically identical to what every Phase 1 test pins
/// against, so any caller comparing against this ground truth is
/// comparing against the same reference the unit tests use.
[[nodiscard]] Knng load_or_compute_ground_truth(
    const Dataset& ds,
    std::size_t k,
    MetricId metric,
    const std::filesystem::path& cache_path);

/// Build the conventional cache filename for `(dataset_path, k, metric)`.
///
/// The filename embeds the dataset stem, `k`, and the metric tag so
/// that a single `cache_dir` can hold the ground truth for many
/// `(dataset, k, metric)` triples without filename collisions.
/// The dataset hash is *not* in the filename — it lives inside the
/// file so a stale-but-similarly-named cache can be detected on
/// load rather than masking the staleness.
[[nodiscard]] std::filesystem::path
default_cache_path(const std::filesystem::path& dataset_path,
                   std::size_t k,
                   MetricId metric,
                   const std::filesystem::path& cache_dir);

} // namespace knng::bench
