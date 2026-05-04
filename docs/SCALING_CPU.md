# CPU scaling writeup (Step 29)

This is the headline artefact for Phases 3–4: a single page that
summarises every CPU brute-force builder the project shipped from
Step 10 (canonical) through Step 28 (hand-vectorised SIMD), with
both *strong-scaling* numbers (wall time vs thread count at fixed
`n`) and the *serial-optimisation ladder* numbers (single-threaded
wall time across every algorithmic variant). It pins the structure
later writeups (`docs/PERF_SINGLE_GPU.md`,
`docs/MULTI_GPU.md`, `docs/DISTRIBUTED_GPU.md`) will inherit.

## Setup

- **Host:** Apple M-series SoC (arm64), AppleClang 21,
  Homebrew libomp 21, Apple Accelerate framework.
- **Workload:** synthetic dataset, `n = 1024 points`,
  `d = 128 dims`, `k = 10`, deterministic
  `knng::random::XorShift64{seed = 42}`.
- **Compiler:** `-O3` Release, `-Werror`, `-march=` default
  (no `-march=native` — see "Open questions" below).
- **Bench:** `bench_brute_force --benchmark_min_time=1x`
  emitting JSON. Every counter
  (`recall_at_k`, `peak_memory_mb`, `n_distance_computations`)
  comes from Step 16's harness.

## Single-threaded ladder (Phase 3 + Phase 4)

| Step | Builder                                | Wall time | Speedup | recall@k |
|------|----------------------------------------|----------:|--------:|---------:|
| 10   | `brute_force_knn`                      |  71.26 ms | 1.00×   | 1.0000 |
| 19   | `_l2_with_norms` (precomputed norms)   |  66.41 ms | 1.07×   | 1.0000 |
| 22   | `_l2_partial_sort`                     |  62.99 ms | 1.13×   | 1.0000 |
| 28   | `_l2_simd` (NEON `vfmaq_f32`)          |  23.46 ms | **3.04×** | 1.0000 |
| 21   | `_l2_blas` (Apple Accelerate sgemm)    |   4.00 ms | **17.82×**| 1.0000 |

The two non-trivial wins compound as you would expect from the
underlying optimisation: hand-vectorising the dot product
(Step 28) trades the autovectoriser's three-instruction loop body
for a one-instruction FMA chain — ~3×. Switching to a tuned BLAS
(Step 21) trades the entire dot-product loop for a
register-tile-and-prefetch-aware GEMM kernel that Apple has been
hand-tuning for a decade — another ~6× on top.

## Strong-scaling sweep at n=1024, d=128

Three parallel implementations, all sharing the Step-19
norms-precompute identity in their inner loop. Wall times in ms;
parenthesised values are speedup vs the same builder at 1 thread.

| Threads | OMP `_omp` | OMP `_omp_scratch` | `std::thread` `_threaded` |
|--------:|-----------:|-------------------:|--------------------------:|
| 1       | 67.66 ms (1.00×) | 67.79 ms (1.00×) | 67.04 ms (1.00×) |
| 2       | 33.84 ms (2.00×) | 33.53 ms (2.02×) | 33.72 ms (1.99×) |
| 4       | 17.47 ms (3.87×) | 17.25 ms (3.93×) | 17.27 ms (3.88×) |
| 8       |  9.83 ms (6.88×) | 10.00 ms (6.78×) |  9.17 ms (7.31×) |

The three implementations land within 5% of each other at every
thread count — exactly the right outcome. OpenMP and
`std::thread` are interchangeable here; the `kHasOpenmpBuiltin`
flag is the only externally visible distinction. The
`_omp_scratch` variant (Step 25, per-thread heap with cache-line
padding) tracks the plain OMP path because at n=1024 the
per-iteration `TopK` allocation is small relative to the work;
the win compounds at larger `n`.

## Take-aways

1. **Two big wins, three small ones.** The ladder above tells the
   project's serial CPU optimisation story in one paragraph: most
   algorithmic refinements (precomputed norms, partial_sort,
   tile blocking) move the needle <15%; *hand vectorisation* and
   *delegating to a tuned library* each contribute >3×, with BLAS
   alone responsible for ~6× over SIMD. The right place to spend
   future CPU effort is "make it easier for BLAS to do its job"
   (e.g. larger tiles, better dataset layout) rather than tuning
   the hand-written kernels further.

2. **Strong scaling is near-linear up to the SoC's performance
   cores.** Apple M-series has 4 P-cores + 4 E-cores; the table
   above shows ~7× at 8 threads (E-cores at ~70% throughput
   relative to P-cores). On a homogeneous server CPU (EPYC,
   Xeon) the same code should land 7.5–8× — the difference is
   the SoC's core asymmetry, not the algorithm.

3. **OpenMP and `std::thread` produce identical numbers.** The
   `std::thread` variant uses an atomic-counter work queue; the
   OpenMP variant uses `schedule(static)`. Both are limited by
   memory bandwidth, not coordination overhead, so the choice is
   pure source-line economics: OpenMP is ~30 lines, `std::thread`
   is ~50, and both run at the same speed.

4. **Recall is invariant across the ladder.** Every cell in
   every table reports `recall@k = 1.0`. Step 15's recall harness
   is doing its job — every algorithmic refactor ships with a
   bit-for-bit matching invariant against the canonical builder.
   No "speedup at the cost of accuracy" trades anywhere in
   Phases 3–4.

5. **The biggest unrealised CPU win is `omp` × `simd` × `blas`
   stacking.** None of the three can stack with the others *as
   shipped*: the OMP variant inherits the scalar dot product;
   the SIMD variant is single-threaded; the BLAS variant is
   single-threaded but Accelerate is internally multi-threaded
   when the input is large enough. Phase 5's NN-Descent will
   need OMP × SIMD; the multi-GPU phases will need to compose
   parallelism in a similar way. The composability question is
   not solved here — the ladder shows what the *individual*
   contributions look like and what the upper bound on a stacked
   build would be.

## Methodology

```sh
cmake -S . -B build -DKNNG_BUILD_BENCHMARKS=ON \
    -DKNNG_ENABLE_BLAS=ON -DKNNG_ENABLE_OPENMP=ON
cmake --build build --target bench_brute_force --parallel

./build/bin/bench_brute_force \
    --benchmark_filter='L2(_|Norms_|Simd_|Blas_|PartialSort_|Omp_|OmpScratch_|Threaded_)Synthetic/1024/128' \
    --benchmark_min_time=1x \
    --benchmark_format=json > phase4_perf.json
```

The aggregator script flattens the JSON to the tables shown above:

```python
import json
with open('phase4_perf.json') as f:
    d = json.load(f)
for r in d['benchmarks']:
    if r.get('run_type') != 'iteration':
        continue
    rt = r['real_time']
    th = (r.get('counters', {}) or {}).get('threads')
    rec = (r.get('counters', {}) or {}).get('recall_at_k') \
          or r.get('recall_at_k')
    th_s = f' t={int(th)}' if th is not None else ''
    print(f"{r['run_name']:60s} {rt:8.2f} ms{th_s}   recall={rec}")
```

## Open questions, deferred to a later pass

- **`-march=native` rerun.** The numbers above are AppleClang
  with default codegen flags; AppleClang on M-series enables
  NEON unconditionally but does not enable Apple-specific tuning
  pragmas. A rerun with `-march=apple-m1` (or `=native`) is a
  natural follow-up; the SIMD path is the most likely to move.
- **Linux + GCC + OpenBLAS.** Every number is from a single host;
  the project's Linux CI runner with GCC + OpenBLAS will produce
  a different absolute scale (BLAS jump should still be
  dominant; SIMD path may swing differently because GCC's
  autovectoriser is less aggressive than AppleClang's, so the
  hand-written path's relative win could be larger).
- **Weak scaling.** The strong-scaling table fixes `n=1024` and
  varies threads. A weak-scaling table (vary `n`, scale threads
  with `n`) is the natural companion artefact, especially for
  the BLAS path — Accelerate is internally multi-threaded above
  some `n`-threshold and the apparent single-threaded number
  stops being meaningful. Deferred to the day a Linux runner
  with `numactl --interleave=all` and a quiescent machine is
  available.
- **NUMA-aware first-touch wiring.** Step 26's
  `knng::cpu::first_touch` exists but the bench harness does
  not call it on its synthetic dataset. The right comparison
  is "BLAS path with first_touch off vs on at SIFT1M scale" on
  a multi-socket Linux host. Deferred until that host exists in
  CI.
- **Composability.** No row of either table runs OMP + SIMD or
  OMP + BLAS together. The composability of Phase-3's
  serial-optimisation primitives with Phase-4's
  parallelism wrappers is the natural Phase-4 follow-up; the
  stacking question is more complex than "just add a
  `#pragma omp parallel for` to the SIMD path" because the
  per-thread heap state from Step 25 needs to coexist with the
  per-thread SIMD state.

## Reproducing this artefact

```sh
# From the repo root, on a quiescent machine:
cmake -S . -B build -DKNNG_BUILD_BENCHMARKS=ON \
    -DKNNG_ENABLE_BLAS=ON -DKNNG_ENABLE_OPENMP=ON
cmake --build build --parallel
ctest --test-dir build               # 127/127 expected
./build/bin/bench_brute_force \
    --benchmark_filter='L2.*1024.*128' \
    --benchmark_min_time=1x \
    --benchmark_format=json > phase4.json
python3 docs/aggregate_phase4.py phase4.json   # flattens to the tables
```

The qualitative shape — single-threaded ladder topped by BLAS,
near-linear strong scaling up to 8 threads, recall invariant
everywhere — is what this writeup commits to. Absolute numbers
will vary across hosts; the *order* of the rows in each table
should not.
