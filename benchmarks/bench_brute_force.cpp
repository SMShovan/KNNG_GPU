/// @file
/// @brief Time `knng::cpu::brute_force_knn` on synthetic and real inputs.
///
/// This is the project's first benchmark, the skeleton every later
/// micro/end-to-end benchmark will follow:
///
///   1. Build a `Dataset` (synthetic random floats by default; an
///      on-disk `.fvecs` file when `KNNG_BENCH_FVECS` is set in the
///      environment).
///   2. Time `brute_force_knn` over the dataset.
///   3. Report wall-clock time per iteration plus throughput in
///      "distance computations per second" so cross-(n, d) comparisons
///      are apples-to-apples.
///
/// Recall numbers and distance-count counters land in Step 14 / Step 15
/// once the recall harness exists. For Step 12 the goal is the
/// pipeline: bench target builds, runs, and produces JSON output that
/// future tooling can parse.
///
/// Example invocations:
///   ./build/bin/bench_brute_force
///   ./build/bin/bench_brute_force --benchmark_format=json
///   KNNG_BENCH_FVECS=datasets/siftsmall/siftsmall_base.fvecs ./build/bin/bench_brute_force --benchmark_filter=Fvecs

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>

#include <benchmark/benchmark.h>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/io/fvecs.hpp"

namespace {

/// Deterministic synthetic dataset: every coordinate drawn from
/// `Uniform[-1, 1]` with a fixed seed so two benchmark runs measure
/// the same arithmetic. The seed is intentionally hard-coded here —
/// Step 16 will introduce the project-wide `XorShift64` wrapper and
/// `--seed` CLI plumbing; until then, a literal `42` is the
/// least-surprising choice.
knng::Dataset make_synthetic(std::size_t n, std::size_t d, std::uint64_t seed)
{
    knng::Dataset ds(n, d);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < ds.data.size(); ++i) {
        ds.data[i] = dist(rng);
    }
    return ds;
}

/// Annotate `state` with the standard counters every brute-force
/// benchmark in this project will report. `n*(n-1)` is the number of
/// distance computations per `brute_force_knn` call (every query
/// against every reference, excluding self), so dividing by wall
/// time gives a metric-independent throughput.
void annotate(benchmark::State& state,
              std::size_t n, std::size_t d, std::size_t k)
{
    const std::int64_t per_iter = static_cast<std::int64_t>(n)
                                * static_cast<std::int64_t>(n - 1);
    state.SetItemsProcessed(state.iterations() * per_iter);
    state.counters["n"] = static_cast<double>(n);
    state.counters["d"] = static_cast<double>(d);
    state.counters["k"] = static_cast<double>(k);
}

void BM_BruteForceL2_Synthetic(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t d = static_cast<std::size_t>(state.range(1));
    constexpr std::size_t k = 10;
    constexpr std::uint64_t seed = 42;

    const knng::Dataset ds = make_synthetic(n, d, seed);

    for (auto _ : state) {
        knng::Knng g = knng::cpu::brute_force_knn(
            ds, k, knng::L2Squared{});
        benchmark::DoNotOptimize(g);
        benchmark::ClobberMemory();
    }

    annotate(state, n, d, k);
}
BENCHMARK(BM_BruteForceL2_Synthetic)
    ->ArgsProduct({{256, 512, 1024}, {32, 128}})
    ->Unit(benchmark::kMillisecond);

void BM_BruteForceL2_Fvecs(benchmark::State& state)
{
    const char* path = std::getenv("KNNG_BENCH_FVECS");
    if (path == nullptr) {
        state.SkipWithError(
            "Set KNNG_BENCH_FVECS=path/to/file.fvecs to run this benchmark "
            "(see tools/download_sift.sh for SIFT-small).");
        return;
    }

    knng::Dataset ds;
    try {
        ds = knng::io::load_fvecs(std::string{path});
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    constexpr std::size_t k = 10;
    for (auto _ : state) {
        knng::Knng g = knng::cpu::brute_force_knn(
            ds, k, knng::L2Squared{});
        benchmark::DoNotOptimize(g);
        benchmark::ClobberMemory();
    }

    annotate(state, ds.n, ds.d, k);
}
BENCHMARK(BM_BruteForceL2_Fvecs)
    ->Unit(benchmark::kMillisecond);

} // namespace
