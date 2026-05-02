/// @file
/// @brief Implementation of the brute-force ground-truth cache.
///
/// The on-disk binary format is documented at the top of
/// `include/knng/bench/ground_truth.hpp`. This translation unit owns
/// three concerns:
///
///   1. The 64-bit FNV-1a digest used as the cache's dataset key.
///   2. Read / write of the documented header + payload, including
///      atomic-rename-on-write so a crash mid-write cannot leave a
///      partially-populated cache file behind.
///   3. The convenience `load_or_compute_ground_truth` entry point
///      that callers actually use; everything else is an
///      implementation detail exposed only for testing.

#include "knng/bench/ground_truth.hpp"

#include <array>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "knng/core/distance.hpp"
#include "knng/cpu/brute_force.hpp"

namespace knng::bench {

namespace {

constexpr std::array<char, 8> kMagic = {'K','N','N','G','R','D','T','R'};
constexpr std::uint32_t       kFormatVersion       = 1;
constexpr std::uint32_t       kIndexByteWidth      = sizeof(index_t);
constexpr std::uint32_t       kDistanceByteWidth   = sizeof(float);
constexpr std::size_t         kReservedBytes       = 16;

/// 64-bit FNV-1a — chosen for its stability, simplicity, and zero
/// dependencies. The constants are the official 64-bit offset
/// basis (`14695981039346656037ull`) and prime (`1099511628211ull`).
struct Fnv1a64 {
    std::uint64_t state = 14695981039346656037ULL;

    void update(const void* bytes, std::size_t n) noexcept
    {
        const auto* p = static_cast<const std::uint8_t*>(bytes);
        for (std::size_t i = 0; i < n; ++i) {
            state ^= static_cast<std::uint64_t>(p[i]);
            state *= 1099511628211ULL;
        }
    }
};

/// Sanitize a metric tag for inclusion in a filename. Returns one of
/// the short, stable strings `l2` / `negip` (rather than the verbose
/// enum-name spelling) so that `default_cache_path` produces concise,
/// shell-friendly paths.
[[nodiscard]] const char* metric_filename_tag(MetricId metric) noexcept
{
    switch (metric) {
        case MetricId::kL2:                   return "l2";
        case MetricId::kNegativeInnerProduct: return "negip";
    }
    return "unknown";
}

/// Brute-force KNN dispatched on the runtime metric. Kept private:
/// public callers use `load_or_compute_ground_truth`, which always
/// goes through this path on a cache miss.
[[nodiscard]] Knng compute_ground_truth(const Dataset& ds,
                                        std::size_t k,
                                        MetricId metric)
{
    switch (metric) {
        case MetricId::kL2:
            return cpu::brute_force_knn(ds, k, L2Squared{});
        case MetricId::kNegativeInnerProduct:
            return cpu::brute_force_knn(ds, k, NegativeInnerProduct{});
    }
    throw std::invalid_argument(
        "knng::bench::compute_ground_truth: unknown MetricId");
}

} // namespace

std::uint64_t dataset_hash(const Dataset& ds) noexcept
{
    Fnv1a64 h;
    const std::uint64_t n = ds.n;
    const std::uint64_t d = ds.d;
    h.update(&n, sizeof(n));
    h.update(&d, sizeof(d));
    if (!ds.data.empty()) {
        h.update(ds.data.data(), ds.data.size() * sizeof(float));
    }
    return h.state;
}

std::optional<Knng>
load_ground_truth(const std::filesystem::path& path,
                  const Dataset& ds,
                  std::size_t k,
                  MetricId metric) noexcept
{
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        return std::nullopt;
    }

    std::array<char, 8>   magic{};
    std::uint32_t         format_version       = 0;
    std::uint32_t         index_width          = 0;
    std::uint32_t         distance_width       = 0;
    std::uint32_t         metric_id_on_disk    = 0;
    std::uint64_t         hash_on_disk         = 0;
    std::uint64_t         n_on_disk            = 0;
    std::uint64_t         k_on_disk            = 0;
    std::array<char, kReservedBytes> reserved{};

    is.read(magic.data(), magic.size());
    is.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    is.read(reinterpret_cast<char*>(&index_width),    sizeof(index_width));
    is.read(reinterpret_cast<char*>(&distance_width), sizeof(distance_width));
    is.read(reinterpret_cast<char*>(&metric_id_on_disk),
            sizeof(metric_id_on_disk));
    is.read(reinterpret_cast<char*>(&hash_on_disk), sizeof(hash_on_disk));
    is.read(reinterpret_cast<char*>(&n_on_disk),    sizeof(n_on_disk));
    is.read(reinterpret_cast<char*>(&k_on_disk),    sizeof(k_on_disk));
    is.read(reserved.data(), reserved.size());
    if (!is) {
        return std::nullopt;
    }

    if (magic != kMagic)                                return std::nullopt;
    if (format_version != kFormatVersion)               return std::nullopt;
    if (index_width    != kIndexByteWidth)              return std::nullopt;
    if (distance_width != kDistanceByteWidth)           return std::nullopt;
    if (metric_id_on_disk != static_cast<std::uint32_t>(metric))
                                                        return std::nullopt;
    if (hash_on_disk != dataset_hash(ds))               return std::nullopt;
    if (n_on_disk    != ds.n)                           return std::nullopt;
    if (k_on_disk    != static_cast<std::uint64_t>(k))  return std::nullopt;

    Knng g(ds.n, k);
    is.read(reinterpret_cast<char*>(g.neighbors.data()),
            static_cast<std::streamsize>(g.neighbors.size() * sizeof(index_t)));
    is.read(reinterpret_cast<char*>(g.distances.data()),
            static_cast<std::streamsize>(g.distances.size() * sizeof(float)));
    if (!is) {
        return std::nullopt;
    }
    // Reject trailing garbage — a healthy cache file ends exactly at
    // the documented payload length.
    is.peek();
    if (!is.eof()) {
        return std::nullopt;
    }
    return g;
}

void save_ground_truth(const std::filesystem::path& path,
                       const Knng& g,
                       std::uint64_t dataset_hash_value,
                       MetricId metric)
{
    std::filesystem::path tmp = path;
    tmp += ".tmp";

    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) {
            throw std::runtime_error(
                "knng::bench::save_ground_truth: could not open '"
                + tmp.string() + "' for writing");
        }

        const std::uint32_t metric_id_on_disk =
            static_cast<std::uint32_t>(metric);
        const std::uint64_t n = g.n;
        const std::uint64_t k = g.k;
        constexpr std::array<char, kReservedBytes> reserved{};

        os.write(kMagic.data(), kMagic.size());
        os.write(reinterpret_cast<const char*>(&kFormatVersion),
                 sizeof(kFormatVersion));
        os.write(reinterpret_cast<const char*>(&kIndexByteWidth),
                 sizeof(kIndexByteWidth));
        os.write(reinterpret_cast<const char*>(&kDistanceByteWidth),
                 sizeof(kDistanceByteWidth));
        os.write(reinterpret_cast<const char*>(&metric_id_on_disk),
                 sizeof(metric_id_on_disk));
        os.write(reinterpret_cast<const char*>(&dataset_hash_value),
                 sizeof(dataset_hash_value));
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(reinterpret_cast<const char*>(&k), sizeof(k));
        os.write(reserved.data(), reserved.size());

        os.write(reinterpret_cast<const char*>(g.neighbors.data()),
                 static_cast<std::streamsize>(
                     g.neighbors.size() * sizeof(index_t)));
        os.write(reinterpret_cast<const char*>(g.distances.data()),
                 static_cast<std::streamsize>(
                     g.distances.size() * sizeof(float)));

        if (!os) {
            throw std::runtime_error(
                "knng::bench::save_ground_truth: write to '"
                + tmp.string() + "' failed");
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        // Fallback: rename can fail across filesystems. Copy + delete
        // is slow but always works. We keep the temp file on a copy
        // failure so the caller can investigate.
        std::filesystem::copy_file(
            tmp, path, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            throw std::runtime_error(
                "knng::bench::save_ground_truth: rename and copy "
                "from '" + tmp.string() + "' to '" + path.string()
                + "' both failed: " + ec.message());
        }
        std::error_code rm_ec;
        std::filesystem::remove(tmp, rm_ec);
        // A leaked tmp file is harmless; ignore rm_ec.
    }
}

Knng load_or_compute_ground_truth(const Dataset& ds,
                                  std::size_t k,
                                  MetricId metric,
                                  const std::filesystem::path& cache_path)
{
    if (auto cached = load_ground_truth(cache_path, ds, k, metric)) {
        return *std::move(cached);
    }

    Knng g = compute_ground_truth(ds, k, metric);

    if (!cache_path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cache_path.parent_path(), ec);
        // Directory-creation failures are non-fatal: the save call
        // below will surface a clearer error if the directory truly
        // cannot be made.
    }
    save_ground_truth(cache_path, g, dataset_hash(ds), metric);
    return g;
}

std::filesystem::path
default_cache_path(const std::filesystem::path& dataset_path,
                   std::size_t k,
                   MetricId metric,
                   const std::filesystem::path& cache_dir)
{
    const std::string stem = dataset_path.stem().string();
    const std::string filename =
        stem + ".k" + std::to_string(k) + "."
        + metric_filename_tag(metric) + ".gt";
    return cache_dir / filename;
}

} // namespace knng::bench
