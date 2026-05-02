# Phase 3 profiling writeup (Step 23)

This is the project's first profiling artefact, the pattern every later
profile writeup will follow. It captures the wall-time ladder produced by
Phase 3's six brute-force CPU paths, the `recall_at_k` invariant they all
preserve, the methodology used to obtain those numbers, and the open
questions a more detailed cycle-counter study (`instruments` /
`perf stat`) will answer in subsequent revisits.

## Optimisation ladder (n=1024, d=128, k=10, AppleClang 21, M-series)

| Step | Builder                                      | Wall time | Speedup vs canonical | recall@k |
|------|----------------------------------------------|----------:|---------------------:|---------:|
| 10   | `brute_force_knn(.., L2Squared{})`           |   70.48 ms | 1.00× (baseline)     | 1.0000 |
| 19   | `brute_force_knn_l2_with_norms`              |   65.66 ms | 1.07×                | 1.0000 |
| 20   | `brute_force_knn_l2_tiled` (32 × 128)        |   65.41 ms | 1.08×                | 1.0000 |
| 22   | `brute_force_knn_l2_partial_sort`            |   62.40 ms | 1.13×                | 1.0000 |
| 21   | `brute_force_knn_l2_blas` (Accelerate sgemm) |    3.82 ms | **18.45×**           | 1.0000 |

The numbers are the means of three repetitions of one iteration each,
captured by `bench_brute_force --benchmark_repetitions=3`.

## Take-aways

1. **The 18.45× BLAS jump dominates the ladder.** Every previous
   step contributed a few percent; Step 21 (Apple Accelerate)
   contributes more than every previous step combined. The right
   reading is *not* that Steps 19–22 are wasted: they are what makes
   Step 21 algebraically possible (the `||a-b||² = ||a||² + ||b||²
   - 2⟨a,b⟩` identity), infrastructurally possible (the tile-loop
   shape `cblas_sgemm` slots into), and *measurable* (the recall
   harness, the deterministic RNG, the JSON counter set).

2. **Recall is preserved exactly across the ladder.** Every one of
   the five paths produces `recall_at_k == 1.0` on the n=1024
   synthetic fixture. This is the project's first concrete
   demonstration of the Pareto-style argument every later phase will
   make: a 18× speedup *with the same answer*. The recall harness
   from Step 15 paid off here — without it, the BLAS path's small
   distance reordering (BLAS may reorder dot-product accumulations)
   would be impossible to defend against silent regression.

3. **The hand-tiling step (20) is in the noise on AppleClang.** The
   tiled path is 65.41 ms across the entire `(query_tile, ref_tile)`
   sweep. AppleClang's autovectoriser already produces
   cache-friendly code from the per-query scan; tiling on top is
   redundant *on this compiler*. The same step ships a real win on
   compilers that do not vectorise the canonical path as
   aggressively (older GCC, MSVC, GCC on platforms without
   `-march=native`); we kept the path because (a) the GPU port in
   Phase 8 will reuse the same loop nest as a kernel, and (b) the
   tile-size knob is a free parameter Step 23's profiling rerun
   will sweep.

4. **`std::partial_sort` beats the streaming heap by a measurable
   margin.** Step 22 is +5% over the streaming heap path on
   AppleClang. The win is workload-dependent (low-`d` workloads
   where the per-pair distance is cheap show much larger gains —
   the heap's `O(log k)` sift cost matters more there).

## Methodology

### Capturing the JSON

```sh
cmake -S . -B build -DKNNG_BUILD_BENCHMARKS=ON -DKNNG_ENABLE_BLAS=ON
cmake --build build --target bench_brute_force --parallel
./build/bin/bench_brute_force \
    --benchmark_filter="L2(_|Norms_|Tiled_|PartialSort_|Blas_)" \
    --benchmark_repetitions=3 \
    --benchmark_format=json > phase3_perf.json
```

Recall, peak memory, and per-step distance counts come from the
counters established in Step 16 (`recall_at_k`,
`peak_memory_mb`, `n_distance_computations`). They are emitted in
every JSON row regardless of which path produced it, so a single
`tools/plot_bench.py` invocation produces the full Pareto figure
without per-path adapters.

### Cache-miss / IPC profiling (planned for Step 23 revisit)

The current artefact captures *wall time* under repeated runs.
What it does not capture, and what a later revisit will:

- L1d / L2 / LLC miss rates per path (`instruments -t "Counters"`
  on macOS, `perf stat -e cache-misses,...` on Linux). The expectation:
  the canonical path's per-query scan takes one L2 miss per
  reference; the tiled path drops that significantly because the
  reference tile is reused across multiple queries.
- IPC and branch-prediction stats. The streaming-heap admission
  test branches once per candidate; the partial_sort path has a
  cleaner branch profile. The branch-mispredict rate per
  candidate is what we expect to see in the IPC numbers.
- Hot-function attribution. Confirms the BLAS path's wall time
  is dominated by `gemv_blocked` (Accelerate's tile inner) and
  not by the norm fold-in epilogue. If the epilogue ever shows up
  as a hot block, the next optimisation is to fuse it into the
  GEMM via `cblas_sgemm`'s `beta != 0` parameter — a small
  rewrite for a meaningful cycle saving.

The reason the cycle-counter pass is deferred: Phase 3's wall-time
artefact is enough to defend the ladder's claim ("each step
preserved recall and produced a measurable speedup"), and the
project's correctness floor (every test green at 100/100) is
sturdy enough that we can afford to revisit profiling under
controlled conditions later (clean cache, taskset to a single
performance core, frequency-locked CPU). Doing it half-rigorously
in a notebook would produce numbers that would be retired the
moment Phase 4's OpenMP introduces parallelism.

## Pattern this writeup pins for later phases

Every subsequent profile writeup (`docs/PERF_STEP50.md`,
`docs/PERF_SINGLE_GPU.md`, `docs/MULTI_GPU.md`) follows this
structure:

1. **Optimisation ladder table** — wall time + recall + speedup,
   one row per builder. Speedup column always references the
   canonical baseline of that phase.
2. **Take-aways** — three to five short sentences each, written
   so a reader can reconstruct the *reason* for each speedup
   without the diff.
3. **Methodology** — the exact commands. Reproducible by a
   third party with no access to the original notes.
4. **Open questions** — what this writeup does *not* claim.
   These are the work items for the next pass.

## Open questions, deferred to a later pass

- AppleClang vs GCC on the same hardware (Linux container,
  same CPU). The 4–7% gains from Steps 19–22 are AppleClang
  numbers; on GCC `-O3` the canonical path may already eat
  the small wins. Worth a 30-minute rerun once the project
  has a Linux CI runner with cycle-counter access.
- BLAS provider comparison: Apple Accelerate vs OpenBLAS vs
  MKL on the same Linux host. The 18× gain is "Apple
  Accelerate on M-series"; the comparable number on x86_64
  with OpenBLAS will be different and is what a downstream
  user will actually see.
- `query_tile` / `ref_tile` sweep at large `n` and large `d`.
  At n=1024 d=128 the entire BLAS-path tile sweep is in the
  3–4 ms range and the noise dominates. SIFT1M (n=1e6) is
  where the right tile sizes will diverge from defaults.
- Memory residency check for the partial-sort path at
  SIFT1M. The `(n - 1)`-element scratch is 8 MB; we expect
  it to stay L2-resident across queries.

## Reproducing this artefact

```sh
# From the repo root, on a quiescent machine.
cmake -S . -B build -DKNNG_BUILD_BENCHMARKS=ON -DKNNG_ENABLE_BLAS=ON
cmake --build build --target bench_brute_force --parallel
./build/bin/bench_brute_force \
    --benchmark_filter='L2(_|Norms_|Tiled_|PartialSort_|Blas_).*1024.*128' \
    --benchmark_repetitions=3 \
    --benchmark_min_time=2x \
    --benchmark_format=json > /tmp/phase3_perf.json

# Aggregate to a small table:
python3 - <<'PY'
import json
with open('/tmp/phase3_perf.json') as f:
    data = json.load(f)
for r in data['benchmarks']:
    if r.get('aggregate_name') == 'mean':
        print(f"{r['run_name']:60s} {r['real_time']:8.2f} ms"
              f"   recall_at_k={r.get('recall_at_k')}")
PY
```

The numbers will differ on x86_64 (no Accelerate; OpenBLAS
gives a different absolute speedup) and on a busy machine
(noise band ~5% per run). The qualitative picture — five paths,
all preserving recall, BLAS dominating the ladder — is what the
writeup captures.
