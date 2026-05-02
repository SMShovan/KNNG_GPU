/// @file
/// @brief Time `knng::cpu::brute_force_knn` on synthetic and real inputs.
///
/// Step 16 wires the bench's JSON output up to the project's
/// standard counter set so every later benchmark — CPU SIMD,
/// OpenMP-parallel, GPU brute-force, NN-Descent, distributed —
/// emits the same fields end-to-end:
///
///   * `recall_at_k`              — quality of the approximate graph.
///                                  For brute-force this is exactly
///                                  1.0 by construction (the graph
///                                  *is* its own ground truth), so
///                                  the counter is wired through the
///                                  Step-15 implementation rather
///                                  than hard-coded — that way a
///                                  refactor that breaks recall on
///                                  brute-force shows up here too.
///   * `peak_memory_mb`           — `getrusage(RUSAGE_SELF).ru_maxrss`
///                                  normalised to MB.
///   * `n_distance_computations`  — `n*(n-1)` for intra-set KNN; later
///                                  approximate builders set whatever
///                                  count they actually do.
///
/// `tools/plot_bench.py` (committed alongside) ingests the JSON
/// produced by `--benchmark_format=json` and produces matplotlib
/// plots without taking a build-time dependency.
///
/// Example invocations:
///   ./build/bin/bench_brute_force
///   ./build/bin/bench_brute_force --benchmark_format=json
///   KNNG_BENCH_FVECS=datasets/siftsmall/siftsmall_base.fvecs ./build/bin/bench_brute_force --benchmark_filter=Fvecs

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <benchmark/benchmark.h>

#include "knng/bench/ground_truth.hpp"
#include "knng/bench/recall.hpp"
#include "knng/bench/runtime_counters.hpp"
#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/io/fvecs.hpp"
#include "knng/random.hpp"

namespace {

/// Deterministic synthetic dataset: every coordinate drawn from
/// `Uniform[-1, 1]` via the project-wide `knng::random::XorShift64`
/// PRNG. Same seed ⇒ bit-identical dataset across runs across
/// platforms, which is the property every recall regression in
/// Phase 3+ depends on.
knng::Dataset make_synthetic(std::size_t n, std::size_t d, std::uint64_t seed)
{
    knng::Dataset ds(n, d);
    knng::random::XorShift64 rng{seed};
    for (std::size_t i = 0; i < ds.data.size(); ++i) {
        // Map [0, 1) → [-1, 1) — same shape as the previous
        // std::mt19937_64 path but bit-identical to the GPU port
        // we will write in Phase 9.
        ds.data[i] = rng.next_float01() * 2.0f - 1.0f;
    }
    return ds;
}

/// Annotate `state` with the project-standard counter set. Every
/// bench TU shares this signature so `tools/plot_bench.py` can
/// ingest a heterogeneous mix of bench JSON files without per-file
/// adapters.
///
/// `recall` is the recall@k against the supplied ground truth; for
/// brute-force builders this is 1.0 by construction, but routing
/// it through `knng::bench::recall_at_k` rather than hard-coding the
/// constant means the day a refactor breaks recall on brute-force,
/// the value here drops below 1.0 and CI catches it.
void annotate(benchmark::State& state,
              std::size_t n, std::size_t d, std::size_t k,
              double recall)
{
    const std::int64_t per_iter = knng::bench::brute_force_distance_count(n);
    state.SetItemsProcessed(state.iterations() * per_iter);
    state.counters["n"] = static_cast<double>(n);
    state.counters["d"] = static_cast<double>(d);
    state.counters["k"] = static_cast<double>(k);
    state.counters["recall_at_k"] =
        benchmark::Counter(recall, benchmark::Counter::kAvgThreads);
    state.counters["peak_memory_mb"] = knng::bench::peak_memory_mb();
    state.counters["n_distance_computations"] =
        benchmark::Counter(static_cast<double>(per_iter),
                           benchmark::Counter::kAvgThreads);
}

void BM_BruteForceL2_Synthetic(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t d = static_cast<std::size_t>(state.range(1));
    constexpr std::size_t k = 10;
    constexpr std::uint64_t seed = 42;

    const knng::Dataset ds = make_synthetic(n, d, seed);

    // Ground truth from a single brute-force pass before the timed
    // loop; recall is computed against the last graph the loop
    // produced. Brute-force is its own ground truth — the value is
    // 1.0 by construction — but we go through the recall path so
    // any future regression in recall_at_k surfaces here too.
    const knng::Knng truth = knng::cpu::brute_force_knn(
        ds, k, knng::L2Squared{});
    knng::Knng last;

    for (auto _ : state) {
        last = knng::cpu::brute_force_knn(ds, k, knng::L2Squared{});
        benchmark::DoNotOptimize(last);
        benchmark::ClobberMemory();
    }

    const double recall = knng::bench::recall_at_k(last, truth);
    annotate(state, n, d, k, recall);
}
BENCHMARK(BM_BruteForceL2_Synthetic)
    ->ArgsProduct({{256, 512, 1024}, {32, 128}})
    ->Unit(benchmark::kMillisecond);

/// Step-19 variant: same arithmetic, precomputed-norms identity.
/// Recall must stay at 1.0 (it is a pure algebraic rewrite); wall
/// time should drop relative to `BM_BruteForceL2_Synthetic` on
/// long-`d` runs where the dot product dominates.
void BM_BruteForceL2Norms_Synthetic(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const std::size_t d = static_cast<std::size_t>(state.range(1));
    constexpr std::size_t k = 10;
    constexpr std::uint64_t seed = 42;

    const knng::Dataset ds = make_synthetic(n, d, seed);

    const knng::Knng truth = knng::cpu::brute_force_knn(
        ds, k, knng::L2Squared{});
    knng::Knng last;

    for (auto _ : state) {
        last = knng::cpu::brute_force_knn_l2_with_norms(ds, k);
        benchmark::DoNotOptimize(last);
        benchmark::ClobberMemory();
    }

    const double recall = knng::bench::recall_at_k(last, truth);
    annotate(state, n, d, k, recall);
}
BENCHMARK(BM_BruteForceL2Norms_Synthetic)
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

    // Ground truth via the cache. The cache key includes the dataset
    // hash, so a flag flip that swaps `--metric` (when the bench
    // grows that knob) cleanly invalidates the cache without us
    // having to reason about it here.
    const auto cache_dir = std::filesystem::path{"build"} / "ground_truth";
    const auto cache_path = knng::bench::default_cache_path(
        std::filesystem::path{path}, k, knng::bench::MetricId::kL2,
        cache_dir);
    knng::Knng truth;
    try {
        truth = knng::bench::load_or_compute_ground_truth(
            ds, k, knng::bench::MetricId::kL2, cache_path);
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    knng::Knng last;
    for (auto _ : state) {
        last = knng::cpu::brute_force_knn(ds, k, knng::L2Squared{});
        benchmark::DoNotOptimize(last);
        benchmark::ClobberMemory();
    }

    const double recall = knng::bench::recall_at_k(last, truth);
    annotate(state, ds.n, ds.d, k, recall);
}
BENCHMARK(BM_BruteForceL2_Fvecs)
    ->Unit(benchmark::kMillisecond);

} // namespace
