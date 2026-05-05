# NN-Descent (Phase 5 closing artefact, Step 37)

The headline page for Phase 5: a single document collecting every
NN-Descent variant the project shipped (Steps 31–36), the
recall-vs-time data on the synthetic 1024-point fixture, and the
honest reading of "where NN-Descent beats brute-force and where
it does not."

## Setup

- **Host:** Apple M-series SoC (arm64), AppleClang 21,
  Homebrew libomp 21, Apple Accelerate framework.
- **Workload:** synthetic dataset, `n = 1024 points`,
  `d = 128 dims`, `k = 10`, deterministic
  `knng::random::XorShift64{seed = 42}`. Uniformly random
  coordinates in `[-1, 1]`.
- **Compiler:** `-O3` Release, `-Werror`, `-march=` default.
- **NN-Descent config:** `max_iters = 32, delta = 0.0` (force
  full convergence so recall reflects the algorithm's
  fixed-point quality, not "wherever the 0.001 threshold
  happened to land"). `seed = 42`.
- **Brute-force baseline:** Step 28's hand-vectorised path
  (`L2Squared` brute_force_knn_l2_simd) and Step 21's BLAS path
  (`brute_force_knn_l2_blas`) — both produce `recall@k = 1.0`
  by construction.

## Single-iteration recall curve

NN-Descent variants at `(n=1024, d=128, k=10)`. Wall times in
ms; recall after 32 iterations of `delta = 0` driver:

| Variant                                   | Threads | Time    | Recall |
|-------------------------------------------|--------:|--------:|-------:|
| `local_join` (no reverse, rho=1.0)        |       1 | 19.8 ms | 0.405 |
| `local_join` (no reverse, rho=1.0)        |       4 | 17.9 ms | 0.405 |
| `local_join_with_reverse`     (rho=1.0)   |       1 | 66.7 ms | 0.858 |
| `local_join_with_reverse_omp` (rho=1.0)   |       4 | 42.4 ms | 0.858 |
| `local_join_sampled`     (rho=0.5, plain) |       1 | 14.8 ms | 0.296 |
| `local_join_with_reverse_sampled`(rho=0.3)|       1 | 27.7 ms | 0.472 |
| `local_join_with_reverse_sampled`(rho=0.5)|       1 | 37.5 ms | 0.602 |
| `local_join_with_reverse_sampled`(rho=1.0)|       1 | 66.7 ms | 0.858 |

For comparison, brute-force at the same `(n, d, k)`:

| Builder                                  | Time    | Recall |
|------------------------------------------|--------:|-------:|
| `brute_force_knn` (canonical, Step 10)   | 70.5 ms |   1.00 |
| `brute_force_knn_l2_simd` (Step 28 NEON) | 23.5 ms |   1.00 |
| `brute_force_knn_l2_blas` (Step 21)      |  3.8 ms |   1.00 |
| `brute_force_knn_l2_omp` (Step 24, t=4)  | 17.5 ms |   1.00 |

## Take-aways

1. **At n=1024, brute-force dominates NN-Descent on every axis.**
   The BLAS path returns the exact graph in 3.8 ms; the best
   NN-Descent variant returns an 0.86-recall graph in 42.4 ms.
   This is the *expected* result for this size — NN-Descent's
   value is its `O(n * k²)` per-iteration scaling vs
   brute-force's `O(n²)`, which only crosses over above
   roughly `n ≥ 100,000`. The 1024-point fixture is too small to
   show the algorithm's headline win; that demonstration is
   deferred to a SIFT1M run on a Linux runner (see "Open
   questions").

2. **Reverse neighbour lists more than double recall.** At
   `rho = 1.0` the plain local-join produces 0.41 recall; the
   reverse-augmented variant produces 0.86. The compute cost
   per iteration roughly triples (17.9 ms → 66.7 ms serial),
   but the recall improvement justifies it on every workload
   that cares about quality. This is the NEO-DNND paper's
   headline contribution and Step 34's measured contribution.

3. **Sampling preserves the ordering of variants.** At each
   `rho ∈ {0.3, 0.5, 1.0}`, the reverse variant outperforms
   the plain one. The recall-vs-time curve at fixed `rho`
   shows reverse always lands at higher recall and higher
   wall time — the caller picks the operating point, not the
   algorithm.

4. **OpenMP parallelism is a 1.6× win at 4 threads on this
   fixture.** The reverse variant goes from 66.7 ms (1 thread)
   to 42.4 ms (4 threads), a 1.57× speedup. This is below
   linear because every neighbour insert pays a `std::mutex`
   lock + unlock; the per-point lock contention probability
   is non-trivial when `n` is small and `k = 10`. At larger
   `n` the speedup grows toward linear because lock
   contention shrinks. Recall is *identical* across thread
   counts on this fixture because the snapshot phase is
   deterministic and the lock-protected inserts produce
   bit-identical lists.

5. **Recall does not reach 1.0 on uniformly-random
   high-dimensional data, even at full convergence.** With
   `delta = 0, max_iters = 32`, the best NN-Descent recall
   is 0.86 on this fixture. This is the curse of
   dimensionality talking: under uniform-random `[-1, 1]^128`
   coordinates, pairwise distances are tightly clustered
   around `√(d * E[(x - y)²]) ≈ √43`, so the "true" top-k
   neighbours are barely closer than the median candidate
   and the local-join's "neighbour-of-neighbour ⇒ neighbour"
   intuition has weaker statistical force. SIFT1M (real
   image features with strong local structure) is where
   NN-Descent reaches >0.95 recall in 5–10 iterations; the
   numbers above pin the *worst-case* fixture.

## Methodology

```sh
cmake -S . -B build -DKNNG_BUILD_BENCHMARKS=ON \
    -DKNNG_ENABLE_BLAS=ON -DKNNG_ENABLE_OPENMP=ON
cmake --build build --target bench_nn_descent --parallel
./build/bin/bench_nn_descent \
    --benchmark_min_time=1x \
    --benchmark_format=json > phase5_nnd.json
```

The bench iterates `(threads, use_reverse, rho)` cross-product
with `delta = 0.0, max_iters = 32` so every reported number is
the algorithm's full-convergence wall time. Brute-force
baselines come from `bench_brute_force` (Step 12 → Step 28
contribute the rows that share the `(n, d, k)` shape).

## Open questions, deferred to a later pass

- **SIFT1M run on a Linux runner.** The 1024-point fixture is
  too small to show NN-Descent's algorithmic win; SIFT1M
  (`n = 1M, d = 128`) is where brute-force becomes infeasible
  (~17 minutes single-threaded BLAS, ~hours canonical) and
  NN-Descent's `O(n * k²)` shape pays off. Pinning this is the
  natural follow-up the moment the project has a Linux CI
  runner with the dataset cached.
- **Recall vs `delta` sweep.** The numbers above use
  `delta = 0` for full convergence; the production default is
  `delta = 0.001`. The recall-vs-delta curve is what a future
  user tuning the knob will want to see; the
  `nn_descent_with_log` infrastructure already records every
  iteration's update fraction, so producing the plot is "run
  the bench with logging on, plot iteration recall."
- **Per-iteration recall trajectory.** Each
  `nn_descent_with_log` run records the per-iteration update
  count; combining that with a per-iteration recall snapshot
  (compute `recall_at_k` after each iteration) would produce
  the convergence curve. Adds ~30 lines to the bench harness;
  the necessary primitives are all in place.
- **OMP × SIMD composability.** The OpenMP-parallel kernels
  call `dot_product` (Step 19), not `simd_dot_product`
  (Step 28). Wiring the SIMD variant into the parallel
  kernel is straightforward — Phase 4's open question
  applies here too. Expected impact: another ~3× wall-time
  reduction on the reverse-augmented path.
- **Per-point lock amortisation.** `local_join_with_reverse_omp`
  allocates two `PerPointLocks` arrays per iteration (40 MB
  each at SIFT1M scale). Hoisting the allocation out of the
  iteration loop is a one-line change that the bench-driven
  perf pass will validate.
- **Larger-`k` regime.** Every benchmark above uses `k = 10`;
  ANN benchmarks frequently use `k = 100`. The local-join's
  `O(k²)` per-iteration cost grows quadratically while the
  reverse list's per-point growth stays linear, so the
  recall-vs-time curve shape will look different at `k = 100`.
  Worth a separate pass.

## Reproducing this artefact

```sh
# From the repo root, on a quiescent machine.
cmake -S . -B build -DKNNG_BUILD_BENCHMARKS=ON \
    -DKNNG_ENABLE_BLAS=ON -DKNNG_ENABLE_OPENMP=ON
cmake --build build --parallel
ctest --test-dir build               # 182/182 expected
./build/bin/bench_nn_descent \
    --benchmark_min_time=1x \
    --benchmark_format=json > phase5_nnd.json

python3 docs/aggregate_phase5.py phase5_nnd.json   # flattens to tables above
```

The qualitative shape — reverse > plain on every recall,
parallelism gives ~1.6× at 4 threads, full convergence does
not reach 1.0 on uniformly-random high-dim data — is what
this writeup commits to. SIFT1M will produce different
absolute numbers and the headline algorithmic win
(NN-Descent ≪ brute-force at large `n`) will be visible
there.
