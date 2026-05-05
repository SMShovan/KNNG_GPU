/// @file
/// @brief Time `knng::cpu::nn_descent` across config and thread sweeps.
///
/// Mirrors `bench_brute_force.cpp`'s shape:
///   * Synthetic dataset with deterministic XorShift64 seed.
///   * Recall computed against brute-force ground truth (Step 15).
///   * Project-standard counters: `recall_at_k`, `peak_memory_mb`,
///     `n_distance_computations`, plus NN-Descent specific fields
///     `iters`, `rho`, `use_reverse`, `threads`.
///
/// Phase-5 Step 37's writeup will ingest the JSON this binary
/// emits.

#include <cstddef>
#include <cstdint>
#include <string>

#include <benchmark/benchmark.h>

#include "knng/bench/ground_truth.hpp"
#include "knng/bench/recall.hpp"
#include "knng/bench/runtime_counters.hpp"
#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/cpu/brute_force.hpp"
#include "knng/cpu/nn_descent.hpp"
#include "knng/random.hpp"

namespace {

knng::Dataset make_synthetic(std::size_t n, std::size_t d, std::uint64_t seed)
{
    knng::Dataset ds(n, d);
    knng::random::XorShift64 rng{seed};
    for (std::size_t i = 0; i < ds.data.size(); ++i) {
        ds.data[i] = rng.next_float01() * 2.0f - 1.0f;
    }
    return ds;
}

void annotate(benchmark::State& state, const knng::cpu::NnDescentConfig& cfg,
               std::size_t n, std::size_t d, std::size_t k,
               double recall, std::size_t iters_run)
{
    state.counters["n"] = static_cast<double>(n);
    state.counters["d"] = static_cast<double>(d);
    state.counters["k"] = static_cast<double>(k);
    state.counters["rho"] = cfg.rho;
    state.counters["use_reverse"] = cfg.use_reverse ? 1.0 : 0.0;
    state.counters["threads"] = static_cast<double>(cfg.num_threads);
    state.counters["iterations"] = static_cast<double>(iters_run);
    state.counters["recall_at_k"] =
        benchmark::Counter(recall, benchmark::Counter::kAvgThreads);
    state.counters["peak_memory_mb"] = knng::bench::peak_memory_mb();
    // Approximate per-iteration work is `n * k^2`, so cumulative
    // distance computations across the run are `iters_run * n * k^2`.
    // (Approximate because reverse / sampling change the constant.)
    state.counters["n_distance_computations"] =
        benchmark::Counter(
            static_cast<double>(iters_run)
                * static_cast<double>(n)
                * static_cast<double>(k) * static_cast<double>(k),
            benchmark::Counter::kAvgThreads);
}

void BM_NnDescent_Synthetic(benchmark::State& state)
{
    const std::size_t n     = static_cast<std::size_t>(state.range(0));
    const std::size_t d     = static_cast<std::size_t>(state.range(1));
    const std::size_t k     = static_cast<std::size_t>(state.range(2));
    const int         t     = static_cast<int>(state.range(3));
    const int         use_r = static_cast<int>(state.range(4));
    const double      rho   = static_cast<double>(state.range(5)) / 100.0;
    constexpr std::uint64_t seed = 42;

    const knng::Dataset ds = make_synthetic(n, d, seed);
    const knng::Knng truth = knng::cpu::brute_force_knn(
        ds, k, knng::L2Squared{});

    knng::cpu::NnDescentConfig cfg{};
    cfg.max_iters    = 32;
    // delta = 0 forces full convergence so recall is the
    // algorithm's *fixed-point* quality, not "wherever the
    // 0.001 threshold happened to land." The bench harness
    // is the right place to use the strict convergence; a
    // production caller would use the default 0.001.
    cfg.delta        = 0.0;
    cfg.seed         = seed;
    cfg.use_reverse  = use_r != 0;
    cfg.rho          = rho;
    cfg.num_threads  = t;

    knng::Knng last;
    std::vector<knng::cpu::NnDescentIterationLog> log;

    for (auto _ : state) {
        log.clear();
        last = knng::cpu::nn_descent_with_log(
            ds, k, cfg, log, knng::L2Squared{});
        benchmark::DoNotOptimize(last);
        benchmark::ClobberMemory();
    }

    const double recall = knng::bench::recall_at_k(last, truth);
    annotate(state, cfg, n, d, k, recall, log.size());
}
// (n, d, k, threads, use_reverse, rho * 100). Encoded as int64
// args because Google Benchmark args are int64.
BENCHMARK(BM_NnDescent_Synthetic)
    ->ArgsProduct({{1024}, {128}, {10},
                   {1, 4},          // threads
                   {0, 1},          // use_reverse
                   {30, 50, 100}})  // rho × 100
    ->Unit(benchmark::kMillisecond);

} // namespace
