/// @file
/// @brief `build_knng` — end-to-end command-line KNNG builder.
///
/// Loads a dataset, runs a CPU brute-force build, writes the resulting
/// `Knng` to a binary file. This is the first commit in the project
/// where a user can do something useful without writing a single line
/// of C++:
///
/// ```sh
/// ./build/bin/build_knng --dataset datasets/siftsmall/siftsmall_base.fvecs --k 10 --metric l2 --algorithm brute_force --output siftsmall_k10.knng
/// ```
///
/// ## Output binary format (version 1)
///
/// All multi-byte integers and floats are little-endian. The header
/// is fixed-width so a reader can `read()` the first 64 bytes
/// unconditionally before deciding what to do with the payload.
///
/// ```
/// offset  size  field
/// ------  ----  -----
///   0      8    magic               = "KNNGRAPH" (ASCII, no NUL)
///   8      4    format_version      = 1 (uint32)
///  12      4    index_byte_width    = 4 (uint32; matches knng::index_t)
///  16      4    distance_byte_width = 4 (uint32; matches float)
///  20      4    metric_id           = 0 (l2) | 1 (negative_inner_product)
///  24      4    algorithm_id        = 0 (brute_force)
///  28      8    n                   (uint64; number of rows)
///  36      8    k                   (uint64; neighbors per row)
///  44     20    reserved            (zero-filled, brings header to 64 B)
///  64    n*k*4  neighbors           (uint32 row-major)
///  ...   n*k*4  distances           (float32 row-major, same layout)
/// ```
///
/// A loader for this format will land in Phase 2 alongside the
/// recall harness; until then, the format is documented here and the
/// `metric_id` / `algorithm_id` enumeration is the source of truth.
///
/// Argument parsing is hand-rolled (no third-party dependency).
/// Long-option-only, `--key value` syntax. `--help` and a missing
/// required flag both print a usage message and exit non-zero.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/io/fvecs.hpp"

namespace {

constexpr std::string_view kUsage =
    "Usage: build_knng --dataset PATH --k N [options]\n"
    "\n"
    "Required:\n"
    "  --dataset PATH        Path to a .fvecs dataset (SIFT/GIST format).\n"
    "  --k N                 Neighbors per point. Must satisfy 1 <= k <= n-1.\n"
    "\n"
    "Optional:\n"
    "  --metric M            l2 (default) | inner_product\n"
    "  --algorithm A         brute_force (default; only choice in Phase 1)\n"
    "  --output PATH         Output .knng path (default: <dataset>.knng)\n"
    "  --help                Print this message and exit\n";

/// Parsed-and-validated CLI arguments.
struct Args {
    std::filesystem::path dataset;
    std::filesystem::path output;
    std::size_t           k = 0;
    std::string           metric    = "l2";
    std::string           algorithm = "brute_force";
};

/// Print `kUsage` to stderr and return the requested exit code.
int usage_and_exit(int code)
{
    std::cerr << kUsage;
    return code;
}

/// Parse a non-negative-integer argument value. Throws on overflow,
/// trailing garbage, or leading sign characters — every caller wants
/// strict validation, not the loose `std::stoull` behaviour.
std::size_t parse_size(const std::string& s, const char* flag)
{
    if (s.empty()) {
        throw std::invalid_argument(std::string{"empty value for "} + flag);
    }
    std::size_t pos = 0;
    const unsigned long long parsed = std::stoull(s, &pos, 10);
    if (pos != s.size()) {
        throw std::invalid_argument(
            std::string{"trailing garbage in value for "} + flag + ": '" + s + "'");
    }
    return static_cast<std::size_t>(parsed);
}

/// Parse argv into `Args`. Returns `std::nullopt` when `--help` was
/// requested (caller should print usage and exit 0). Throws
/// `std::invalid_argument` on a malformed or missing flag.
std::optional<Args> parse_args(int argc, char** argv)
{
    Args args;
    std::map<std::string, std::string> values;

    int i = 1;
    while (i < argc) {
        const std::string_view a = argv[i];
        if (a == "--help" || a == "-h") {
            return std::nullopt;
        }
        if (a.substr(0, 2) != "--") {
            throw std::invalid_argument(
                "unexpected positional argument: '" + std::string{a} + "'");
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument(
                "flag '" + std::string{a} + "' requires a value");
        }
        values[std::string{a.substr(2)}] = argv[i + 1];
        i += 2;
    }

    auto take = [&](const std::string& key, bool required) -> std::optional<std::string> {
        auto it = values.find(key);
        if (it == values.end()) {
            if (required) {
                throw std::invalid_argument("missing required flag --" + key);
            }
            return std::nullopt;
        }
        std::string v = std::move(it->second);
        values.erase(it);
        return v;
    };

    args.dataset = *take("dataset", /*required=*/true);
    args.k       = parse_size(*take("k", /*required=*/true), "--k");
    if (auto m = take("metric", false))    args.metric    = *m;
    if (auto a = take("algorithm", false)) args.algorithm = *a;
    if (auto o = take("output", false))    args.output    = *o;

    if (!values.empty()) {
        throw std::invalid_argument("unknown flag --" + values.begin()->first);
    }
    if (args.k == 0) {
        throw std::invalid_argument("--k must be > 0");
    }
    if (args.output.empty()) {
        args.output = args.dataset;
        args.output += ".knng";
    }
    if (args.metric != "l2" && args.metric != "inner_product") {
        throw std::invalid_argument(
            "--metric must be 'l2' or 'inner_product', got '" + args.metric + "'");
    }
    if (args.algorithm != "brute_force") {
        throw std::invalid_argument(
            "--algorithm must be 'brute_force' in Phase 1, got '"
            + args.algorithm + "'");
    }
    return args;
}

/// Encode a metric string as the wire `metric_id`. Kept as a free
/// function so `write_knng` does not have to know about strings.
std::uint32_t metric_id(const std::string& metric)
{
    if (metric == "l2")              return 0;
    if (metric == "inner_product")   return 1;
    throw std::invalid_argument("metric_id: unknown metric '" + metric + "'");
}

constexpr std::uint32_t kAlgorithmBruteForce = 0;
constexpr std::uint32_t kFormatVersion       = 1;
constexpr std::array<char, 8> kMagic = {'K','N','N','G','R','A','P','H'};

/// Serialise `g` to `path` in the binary format documented at the
/// top of this file.
void write_knng(const std::filesystem::path& path,
                const knng::Knng& g,
                std::uint32_t metric,
                std::uint32_t algorithm)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("could not open output for writing: '"
                                 + path.string() + "'");
    }

    const std::uint32_t index_width    = sizeof(knng::index_t);
    const std::uint32_t distance_width = sizeof(float);
    const std::uint64_t n              = g.n;
    const std::uint64_t k              = g.k;

    // Header — 64 bytes total.
    os.write(kMagic.data(), kMagic.size());
    os.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));
    os.write(reinterpret_cast<const char*>(&index_width),    sizeof(index_width));
    os.write(reinterpret_cast<const char*>(&distance_width), sizeof(distance_width));
    os.write(reinterpret_cast<const char*>(&metric),         sizeof(metric));
    os.write(reinterpret_cast<const char*>(&algorithm),      sizeof(algorithm));
    os.write(reinterpret_cast<const char*>(&n),              sizeof(n));
    os.write(reinterpret_cast<const char*>(&k),              sizeof(k));
    constexpr std::array<char, 20> reserved{};
    os.write(reserved.data(), reserved.size());

    // Payload.
    os.write(reinterpret_cast<const char*>(g.neighbors.data()),
             static_cast<std::streamsize>(g.neighbors.size() * sizeof(knng::index_t)));
    os.write(reinterpret_cast<const char*>(g.distances.data()),
             static_cast<std::streamsize>(g.distances.size() * sizeof(float)));

    if (!os) {
        throw std::runtime_error("write to '" + path.string() + "' failed");
    }
}

/// Run the chosen algorithm and return the resulting graph. Kept as
/// a separate function so `main` is the boring orchestration layer
/// and the dispatch lives in one place that future steps can extend.
knng::Knng build(const knng::Dataset& ds, std::size_t k, const std::string& metric)
{
    if (metric == "l2") {
        return knng::cpu::brute_force_knn(ds, k, knng::L2Squared{});
    }
    return knng::cpu::brute_force_knn(ds, k, knng::NegativeInnerProduct{});
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    try {
        auto parsed = parse_args(argc, argv);
        if (!parsed) {
            std::cout << kUsage;
            return 0;
        }
        args = *std::move(parsed);
    } catch (const std::exception& e) {
        std::cerr << "build_knng: " << e.what() << "\n\n";
        return usage_and_exit(2);
    }

    try {
        std::cerr << "build_knng: loading " << args.dataset << '\n';
        const knng::Dataset ds = knng::io::load_fvecs(args.dataset);
        std::cerr << "build_knng: loaded n=" << ds.n
                  << " d=" << ds.d << '\n';

        std::cerr << "build_knng: building k=" << args.k
                  << " metric=" << args.metric
                  << " algorithm=" << args.algorithm << '\n';
        const knng::Knng g = build(ds, args.k, args.metric);

        std::cerr << "build_knng: writing " << args.output << '\n';
        write_knng(args.output, g, metric_id(args.metric), kAlgorithmBruteForce);

        std::cerr << "build_knng: done\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build_knng: " << e.what() << '\n';
        return 1;
    }
}
