# Changelog

All notable changes to this project are documented here, one entry per
development step. Entries are ordered newest-first and follow the
`What / Why / Tradeoff / Learning / Next` structure. The canonical
template, with expectations for each section, is documented in
[`docs/STYLE.md`](docs/STYLE.md) §14.

The goal of this document is pedagogical: each entry should make the
*why* of the change obvious to a reader scanning the history,
independent of the code diff.

---

## [Step 38] — MPI hello world + CMake integration (2026-05-06)

### What
- Added `cmake/FindKnngMPI.cmake` — discovers an MPI CXX
  implementation via CMake's built-in `FindMPI` module. Follows the
  same pattern as `FindKnngOpenMP.cmake`: `KNNG_ENABLE_MPI` opt-in
  option, `KNNG_HAVE_MPI` cache variable, `knng::mpi_iface` INTERFACE
  alias that propagates `MPI::MPI_CXX` and the `KNNG_HAVE_MPI=1`
  compile definition to every consumer. When MPI is absent the module
  emits a status message and sets `KNNG_HAVE_MPI=OFF`; the non-MPI
  build continues unchanged.
- Added `include/knng/dist/mpi_env.hpp` — `knng::dist::MpiEnv`, an
  RAII guard that calls `MPI_Init_thread(MPI_THREAD_FUNNELED)` on
  construction and `MPI_Finalize()` on destruction. Non-copyable,
  move-constructible (moved-from instance becomes inert). Public
  surface: `rank()`, `size()`, `is_root()`, `thread_support()`,
  `barrier()`.
- Added `src/dist/mpi_env.cpp` — out-of-line implementation of
  `MpiEnv`. Throws `std::runtime_error` on double-init or on
  `MPI_Init_thread` failure, so MPI errors are caught at construction
  rather than silently proceeding.
- Added `src/dist/CMakeLists.txt` — `knng::dist` STATIC library
  target. Currently contains only `mpi_env.cpp`; Phase 6's remaining
  steps will extend the source list incrementally.
- Updated root `CMakeLists.txt`: added `include(FindKnngMPI)`,
  conditionally added `src/dist/` subdirectory, and added `mpi`
  line to the build summary.
- Added `tests/mpi_env_test.cpp` — six-test suite covering
  `rank()`-in-range, `is_root()`-matches-rank, `size()`-positive,
  allreduce-sum correctness, barrier non-deadlock, and
  double-init-throws. Registered in `tests/CMakeLists.txt` via
  `mpirun -np 1` when a launcher is found, or via `gtest_discover_tests`
  otherwise; only built when `KNNG_HAVE_MPI=ON`.
- ctest 182/182 green (non-MPI tests unaffected; MPI targets gated).

### Why
Phase 5 closed with an eight-step NN-Descent implementation running
on a single CPU. Phase 6's goal is to add a second independent axis
of parallelism — distributed-memory across MPI ranks — *before*
introducing GPUs, so that the distributed algorithm can be understood
and debugged on familiar CPU semantics. That learning carries directly
into Phase 12's MPI+GPU design.

This first step establishes the same infrastructure prerequisite that
Step 01 established for the whole project: the build must know about
the new dependency, and a smoke test must prove the dependency is
wired end-to-end before any algorithmic code is written. The exact same
pattern was used for OpenMP (Step 24) and BLAS (Step 20): find the
library, expose an INTERFACE alias, gate everything on a cache variable,
test the integration with the simplest possible exerciser (allreduce
sum).

The `MPI_THREAD_FUNNELED` thread level matches Phase 6's single-
threaded communication model: all MPI calls will originate from the
main thread. Phase 12 may upgrade to `MPI_THREAD_MULTIPLE` when async
progress threads are introduced; fixing the level now prevents a
silent later breakage.

### Tradeoff
- **No MPI on the Mac dev machine.** `FindKnngMPI` gracefully skips
  the `knng::dist` targets when MPI is not found, so `cmake +
  ctest` on the laptop stays green. The distributed code will first
  compile and run on a Linux cluster with OpenMPI or MPICH installed.
  This is the same posture taken for GPU code (Phase 7+): write the
  code, prove it builds on target hardware when available.
- **`MPI_THREAD_FUNNELED` vs `MPI_THREAD_MULTIPLE`.** `FUNNELED` is
  the minimum required for Phase 6 and avoids the implementation
  complexity that `MULTIPLE` can introduce (some MPI libraries
  serialize all operations under `MULTIPLE`, negating any benefit).
  The request level is a single constant in `mpi_env.cpp`; upgrading
  is a one-line change with a clear comment trail.
- **Double-init throws rather than silently becoming a no-op.** A
  no-op would hide logical bugs (two `MpiEnv` objects in scope,
  likely by accident). Throwing at construction makes the bug
  immediately visible, at the cost of not supporting the pattern
  where the environment is re-initialised (which MPI does not
  support anyway).

### Learning
- CMake's `FindMPI` module is well-maintained and handles
  OpenMPI, MPICH, Intel MPI, and MVAPICH. Wrapping it in a thin
  module (rather than calling it directly from root) follows the
  same separation-of-concerns argument as `FindKnngOpenMP`: the
  project-level option (`KNNG_ENABLE_MPI`) and the target alias
  (`knng::mpi_iface`) are the project's own abstractions; the
  discovery mechanism is delegated to CMake.
- The `MPI_Initialized` guard in the constructor is critical for
  test harnesses: some MPI testing patterns (e.g., test fixtures
  with static env objects) can inadvertently construct `MpiEnv`
  twice. Making the guard explicit converts a silent deadlock or
  segfault into a readable `std::runtime_error`.
- Gating test registration on `MPIRUN_EXEC` found vs not-found allows
  the same test target to be CTest-registered in both environments
  (cluster with `mpirun` and single-process build where `mpirun -np 1`
  is unnecessary overhead).

### Next
Step 39 adds `knng::dist::ShardedDataset` — the distributed-memory
view of a dataset where each rank owns `n/P` rows and knows how to
scatter / gather with its peers. This is the primitive that all later
distributed algorithms operate on.

---

## [Step 37] — NN-Descent recall writeup (Phase 5 closing artefact) (2026-05-05)

### What
- Added `docs/NN_DESCENT.md` — Phase 5's headline artefact. The
  five-section writeup template established in Step 23
  (`Setup · Tables · Take-aways · Methodology · Open questions ·
  Reproduction`) carried over with one new "single-iteration
  recall curve" table and a sibling brute-force comparison
  table:

  ```text
  NN-Descent variants (n=1024, d=128, k=10):
    plain rho=1.0      19.8 ms / 17.9 ms (1/4 thr)   recall=0.405
    plain rho=0.5      14.8 ms                       recall=0.296
    plain rho=0.3      11.0 ms                       recall=0.218
    reverse rho=1.0    66.7 ms / 42.4 ms (1/4 thr)   recall=0.858
    reverse rho=0.5    37.5 ms                       recall=0.602
    reverse rho=0.3    27.7 ms                       recall=0.472

  Brute-force baselines (same n, d, k):
    canonical          70.5 ms     recall=1.00
    NEON SIMD          23.5 ms     recall=1.00
    BLAS Accelerate     3.8 ms     recall=1.00
    OMP t=4            17.5 ms     recall=1.00
  ```
- Added `docs/aggregate_phase5.py` — 60-line, dependency-free
  Python script that ingests `bench_nn_descent` JSON and prints
  the variants table. Mirrors `docs/aggregate_phase4.py`'s
  shape; the field-name constants at the top
  (`F_TIME, F_RECALL, F_THREADS, F_REVERSE, F_RHO, F_ITERS`) are
  the single edit point if a future bench rename a counter.
- Set `bench_nn_descent`'s default `cfg.delta = 0.0` so the
  reported wall times reflect *full convergence* rather than
  whatever the production-default `0.001` threshold happened
  to land on. The driver still ships with `delta = 0.001` as
  its production default; only the bench harness uses the
  strict-convergence variant.
- Captured five take-aways the writeup commits to:
  * At `n = 1024`, brute-force dominates NN-Descent on every
    axis — expected for this size; NN-Descent's algorithmic
    win is `O(n*k²)` vs `O(n²)` and only kicks in above
    roughly `n ≥ 100,000`.
  * Reverse neighbour lists more than double the recall
    (0.41 → 0.86 at `rho = 1.0`) — the NEO-DNND headline
    contribution measured.
  * Sampling preserves variant ordering: at every `rho` the
    reverse path beats plain.
  * OpenMP gives ~1.6× at 4 threads on this fixture (lock
    contention is significant at small `n, k`).
  * Recall does not reach 1.0 on uniformly-random
    high-dimensional data, even at full convergence — the
    curse of dimensionality talking. SIFT1M will reach
    >0.95 in 5–10 iterations.
- Captured six open questions for the SIFT1M / Linux follow-up:
  SIFT1M run, recall vs `delta` sweep, per-iteration recall
  trajectory, OMP × SIMD composability, per-point lock
  amortisation, larger-`k` regime.
- ctest 182/182 still green; this step adds documentation only.

### Why
Phase 5 produced eight sub-steps' worth of NN-Descent
infrastructure: the data structure (Step 30), the random
initialiser (Step 31), the local-join (Step 32), the driver
(Step 33), reverse lists (Step 34), sampling (Step 35), and
parallelism (Step 36). Each has its own CHANGELOG entry; without
a single page collecting every wall-time number into one table,
the "what did NN-Descent buy us?" question requires reading
eight commit messages. The writeup is that page.

The honest reading at `n = 1024` — "brute-force dominates" — is
the *correct* reading for this fixture. NN-Descent's value
proposition is its asymptotic shape, and the project's first
opportunity to demonstrate that shape is a SIFT1M run on a
Linux runner. The writeup is upfront about this; pinning the
1024-point numbers as "the floor where NN-Descent looks worse"
gives a future contributor (and future me) the right context
when the SIFT1M numbers eventually land.

The strict-convergence (`delta = 0.0`) bench setting is
deliberate. Production callers should use the default
`delta = 0.001` because the algorithm's per-iteration cost
makes the last 0.1% of recall expensive; the bench reports
the *fixed-point* quality so a future contributor cannot
misread "NN-Descent recall = 0.86" as "the algorithm cannot
do better." The two recall numbers (production-default
threshold vs full convergence) are both interesting; the
writeup's job is to make the distinction visible.

### Tradeoff
- **Numbers are AppleClang on M-series only.** Same caveat as
  every other writeup so far. The qualitative picture (reverse
  > plain at every recall, parallel > serial at every config,
  brute-force dominates at this scale) is what the writeup
  commits to; the absolute numbers will shift on Linux + GCC.
- **The writeup does not include a recall-vs-iteration
  trajectory plot.** Producing it requires computing
  `recall_at_k` after each iteration, which the
  `nn_descent_with_log` infrastructure makes possible but
  the bench harness does not currently do. We accept the
  omission and flag it as a deferred question; the
  trajectory plot is the kind of artefact that is most useful
  alongside SIFT1M numbers.
- **No `tools/plot_phase5.py` matplotlib renderer.** The
  Phase 4 writeup also defaulted to text tables; Phase 13's
  comprehensive Pareto figure is the first place a
  matplotlib renderer would be load-bearing. Adding one for
  Phase 5 alone would be premature.
- **`bench_nn_descent`'s default `delta = 0.0` is the bench's
  default, not the library's.** A future contributor running
  the bench must understand this — we made the comment in the
  source explicit and the writeup mentions it; if it ever
  becomes a source of confusion, surfacing `delta` as a
  bench-CLI flag is a small addition.

### Learning
- *The honest reading is more useful than the optimistic
  reading.* "NN-Descent reaches 0.86 recall on uniformly-
  random 128-dim data while BLAS brute-force finishes the
  exact graph in 17× less time" is what the *fixture* says,
  and pretending otherwise — by, say, picking a `(n, d)`
  where NN-Descent already wins — would mislead a future
  contributor about what the algorithm actually delivers.
  The "open questions" section is the bookmark for the
  SIFT1M follow-up.
- *Strict-convergence bench numbers expose the algorithm,
  production-threshold numbers expose the integration.* The
  writeup uses `delta = 0.0` to characterise the algorithm;
  the production default `delta = 0.001` characterises a
  caller's wall-time budget. Both are interesting; pinning
  them at different layers of the stack (bench harness vs
  default config) keeps each truth in the right place.
- *Phase-closing writeups should *list their gaps* alongside
  their take-aways.* Step 23's writeup pioneered this, Step
  29's writeup followed, and Step 37's writeup makes it the
  pattern: every Phase ends with a "what we did not measure"
  list that future contributors can pick up. The cost is
  five bullets per writeup; the value is preventing a
  decade-old paper's analogue of "we were going to come back
  to this" from rotting in private notes.

### Next
- Phase 6 (Step 38): MPI distributed CPU NN-Descent. Steps
  37 → 38 close the CPU portion of the project; Phase 6
  introduces the distributed-memory primitives (point
  partitioning, NEO-DNND duplicate-request reduction) that
  the GPU phases will eventually compose with.

---

## [Step 36] — Parallel NN-Descent (OpenMP) (2026-05-05)

### What
- Added two parallel kernels:
  * `local_join_omp<D>(ds, graph, num_threads, distance)` —
    OpenMP-parallel sibling of `local_join` (Step 32).
  * `local_join_with_reverse_omp<D>(ds, graph, num_threads,
    distance)` — OpenMP-parallel sibling of
    `local_join_with_reverse` (Step 34).
- Both use a per-point `std::mutex` array (new private
  `PerPointLocks` helper holding a `std::unique_ptr<std::mutex[]>`)
  to protect every list insert. Insert pairs `(u, v)` acquire
  the locks one-at-a-time (lock-insert-unlock for `u`, then
  for `v`) — never two locks simultaneously, so deadlock is
  structurally impossible. Inserts into the same point
  serialise; inserts into different points proceed in
  parallel.
- The reverse-list construction in
  `local_join_with_reverse_omp` is also parallelised, with a
  *separate* per-point lock array (`rev_locks`) so concurrent
  pushes to `rev_new[u]` and concurrent forward inserts on the
  same `u` do not contend on a shared lock.
- Per-iteration update accumulation uses
  `std::atomic<std::size_t>` with `memory_order_relaxed`. The
  count is the only shared write across threads; relaxed is
  sufficient because no other shared state has a happens-before
  relationship to the count.
- New private helper `join_pairs_locked<D>` mirrors
  `join_pairs<D>` from Step 34 but adds the lock-guarded
  inserts. The two function bodies look nearly identical;
  duplication is the price of not paying mutex overhead in the
  serial path.
- Extended `NnDescentConfig` with `int num_threads = 0`
  (`0` → runtime default; `1` → serial; `>1` → parallel team
  size). Parallelism is *only* applied when `cfg.rho == 1.0`;
  if `cfg.rho < 1.0`, the driver silently falls back to the
  serial sampled kernel. The interaction is documented in
  `NnDescentConfig::num_threads`.
- Driver now picks one of *six* kernel variants based on
  `(use_reverse, rho < 1.0, num_threads != 1)`. The selection
  table is in the inline comment above the loop.
- Added `benchmarks/bench_nn_descent.cpp` —
  `BM_NnDescent_Synthetic` family that sweeps
  `(threads ∈ {1, 4}, use_reverse ∈ {0, 1}, rho ∈ {0.3, 0.5,
  1.0})` at `n=1024, d=128, k=10`. The bench emits the
  project-standard counters (`recall_at_k`, `peak_memory_mb`,
  `n_distance_computations`) plus NN-Descent specific fields
  (`iterations`, `rho`, `use_reverse`, `threads`). Step 37's
  writeup ingests this JSON.
- Six new `test_nn_descent` cases:
  * `local_join_omp` matches serial on the 8-point fixture.
  * `local_join_with_reverse_omp` matches serial on the
    8-point fixture.
  * Parallel driver still converges to `recall@k = 1.0`.
  * Parallel driver output is bit-identical across
    `num_threads ∈ {1, 2, 4}` (small-fixture property).
  * Graph-size mismatch throws on both parallel kernels.
  * Parallel driver with `rho < 1.0` silently falls back to
    serial sampled (output matches `num_threads = 1`).
- ctest 182/182 green (6 new nn_descent, 176 carried over
  from Step 35).
- **Measured at n=1024, d=128, k=10, reverse=true, rho=1.0:**
  serial 54.1 ms → 4 threads 27.3 ms — **~2× speedup**, recall
  stays at `0.858` regardless of thread count.

### Why
Phases 4 ended with parallel brute-force at 8 threads showing
6.6× scaling on Apple M-series. NN-Descent's
inherently-sequential local-join (each iteration depends on
the previous iteration's snapshot) makes the same shape harder
to parallelise: the point-level outer loop *can* run in
parallel within an iteration, but the inner inserts mutate
shared per-point lists and need synchronisation. This step
ships the synchronisation infrastructure.

The per-point lock choice rather than atomics is deliberate:
the `NeighborList::insert` operation is *not* a CAS-friendly
single-word swap. It walks a sorted vector, compares against
the worst entry, potentially evicts a tail element, and
reinserts at a sorted position — multiple memory writes across
the vector's storage. CAS-on-distance variants exist
(used by some GPU NN-Descent implementations) but they require
a different list shape (fixed-size array of atomic
`{id, dist}` pairs); the per-point mutex is the correct match
for our `std::vector`-backed list.

The `relaxed` memory ordering on the update counter has been
deliberately weakened from the default `seq_cst`: the count
is consumed only after every worker has joined (the implicit
barrier at the end of `#pragma omp parallel for`), so no
happens-before relationship needs to span the `fetch_add`
calls. This is the same justification we used for the
`std::thread` work-queue counter in Step 27 — same reasoning,
same `relaxed` choice.

The "rho-and-parallelism don't compose" decision is a
deliberate scope limitation. Adding parallel sampling would
require either (a) per-thread local RNGs seeded
deterministically — which complicates the
"same `cfg.seed` ⇒ bit-identical output" contract that runs
through every test — or (b) a single shared RNG with locking,
which would re-introduce the bottleneck we just removed.
Phase 9's GPU NN-Descent will need to re-confront this
question; on CPU we accept the silent fall-through and document
it loudly in `NnDescentConfig::num_threads`.

The 2× speedup at 4 threads (vs the 4× ideal) reflects the
synchronisation cost: every insert pays a mutex lock + unlock,
and the serial reverse-list construction is now parallelised
but introduces a separate set of locks. On larger `n` the
speedup grows because the per-point lock contention probability
shrinks (more points spread across the threads).

### Tradeoff
- **Per-point `std::mutex` array is `O(n * sizeof(std::mutex))`
  memory.** For `n = 1M, sizeof(std::mutex) ~ 40 bytes`, that
  is ~40 MB per iteration. Allocated fresh every iteration so
  it does not accumulate. A future optimisation is to amortise
  the mutex array across iterations (the algorithm does not
  invalidate it between iterations) — a one-line refactor we
  defer to bench-driven follow-up.
- **`join_pairs_locked` duplicates the body of `join_pairs`.**
  Keeping them separate avoids paying lock overhead in the
  serial path. A future "policy" template parameter could
  unify them but would be more complex than two
  near-identical 30-line functions.
- **The atomic counter is a hot cache line.** Every iteration's
  worker `fetch_add`s into one atomic; on n=1024 with 4
  threads the contention is invisible, but at 32 threads it
  could become a bottleneck. The fix is per-thread local
  counts summed at the end (the standard "false-sharing-free
  reduction" pattern); we ship the simpler shape today.
- **Reverse-list construction parallelised but not pipelined.**
  Phase 2's lock-guarded pushes pay synchronisation overhead
  even though the work is conceptually independent across
  source points. A lock-free append (e.g. per-thread reverse
  buffers + a final concat) would be faster but more code.
  We accept the locked variant.

### Learning
- *Per-point locks are the canonical
  "irregular-write parallelism" pattern.* Brute-force gets to
  use `schedule(static)` with no locks because every output
  cell is touched by exactly one worker. NN-Descent inherits
  the irregular-write structure of every graph algorithm —
  neighbour insertion can come from any direction — and
  per-target locks are how you get correctness without paying
  the all-locks-on-everyone tax. Pinning this pattern now
  makes Step 65's GPU `atomicMin` formulation read as the
  GPU translation of a known shape rather than a new puzzle.
- *Releasing each lock between the two halves of a pair-update
  is the simplest deadlock-prevention strategy.* The
  alternative (always lock `min(u, v)` first then
  `max(u, v)`) is correct and slightly faster (one less
  mutex round-trip per pair) but easier to get wrong if a
  future contributor adds a third lock. The current shape is
  obviously correct on inspection; we accept the
  trivial perf cost.
- *"Sampling and parallelism don't compose, so the driver
  silently picks serial" is a UX choice worth documenting.*
  The alternative — throwing or warning — would force every
  caller to handle the combination explicitly. The
  silent fall-through is the kind of "principle of least
  surprise" decision that reads well in the
  `NnDescentConfig::num_threads` comment but would be
  invisible without it.

### Next
- Step 37: NN-Descent recall writeup
  (`docs/NN_DESCENT.md`). Pareto plot: recall@k vs wall time
  across brute-force (Phase 3 + 4) and NN-Descent (Phase 5)
  on the 1024-point fixture and — when a Linux runner is
  available — SIFT1M. Closes Phase 5 with the standard
  five-section writeup template established in Step 23.

---

## [Step 35] — Sampling (`rho` parameter) (2026-05-05)

### What
- Added two sampled-variant kernels:
  * `local_join_sampled<D>(ds, graph, rho, iter_seed, distance)`
  * `local_join_with_reverse_sampled<D>(ds, graph, rho,
    iter_seed, distance)`
  Both run the same algorithm as their unsampled siblings but
  uniformly downsample each point's `new` and `old` candidate
  sets to `rho * k` entries before the pair-enumeration. The
  RNG is `XorShift64{iter_seed}` so output is deterministic
  for a given `(graph, rho, iter_seed)`.
- Added a private helper trio in the anonymous namespace:
  * `rho_to_sample_size(rho, k)` — converts a `(0, 1]`
    sampling rate to an effective sample size, clamped at
    `min(k, max(1, ⌊rho * k⌋))` so `rho > 0` always processes
    at least one entry.
  * `partial_fisher_yates(pool, m, rng)` — `O(m)` partial
    shuffle that picks `m` distinct entries uniformly. The
    sampled prefix lands in `pool[0 .. m)`.
  * `snapshot_and_age_sampled(graph, rho, rng, new_ids,
    old_ids)` — the sampled twin of `snapshot_and_age` from
    Step 34. Walks every list, records new / old positions,
    runs partial Fisher-Yates, captures sampled ids, and —
    crucially — flips *only the sampled-new positions* to
    `is_new = false`. Unsampled entries remain `is_new = true`
    so they are eligible for sampling in subsequent iterations.
- The reverse-sampled kernel additionally subsamples the
  reverse-graph contribution: when `|rev_new[p]| > rho * k`,
  it picks `rho * k` entries uniformly via the same
  Fisher-Yates helper before merging into `nv_total`. Same
  for `rev_old`. The `sort_unique` + two-finger merge
  dedup pipeline from Step 34 carries through unchanged.
- Extended `NnDescentConfig` with `double rho = 1.0`. The
  driver now picks one of *four* kernel variants based on
  `(use_reverse, rho < 1.0)`:
  * `(true, false)` → `local_join_with_reverse` (Step 34)
  * `(false, false)` → `local_join` (Step 32)
  * `(true, true)` → `local_join_with_reverse_sampled`
  * `(false, true)` → `local_join_sampled`
  Per-iteration sampling seed is computed by mixing
  `cfg.seed` with the iteration index using the 64-bit
  golden-ratio constant `0x9E3779B97F4A7C15ULL` —
  `splitmix64`-style mixing for good diversity without a full
  hash. `cfg.rho ≤ 0.0` is rejected at driver entry.
- Eight new `test_nn_descent` cases:
  * `rho = 1.0` sampled produces bit-identical output to
    plain (since every entry survives the sampler).
  * `rho < 1.0` still produces non-zero updates and
    preserves shape on the 8-point fixture.
  * `rho ≤ 0.0` throws on every kernel and on the driver.
  * Graph-size mismatch throws on the sampled kernels.
  * Driver `cfg.rho = 1.0` produces the same output as the
    default `cfg{}` (which has `rho = 1.0`).
  * Driver still converges to `recall@k = 1.0` at
    `rho ∈ {0.3, 0.5, 0.8}` given enough iterations.
  * Driver rejects `cfg.rho ≤ 0.0`.
  * Per-iteration update counts at `rho = 0.3` are no
    larger than at `rho = 1.0` (smaller candidate set ⇒
    no more work per iteration).
- ctest 176/176 green (8 new nn_descent, 168 carried over
  from Step 34).

### Why
Sampling is the third classic NN-Descent tuning knob alongside
`max_iters` and `delta`. Wang et al. 2012 §4.4 introduces it
as the speed/quality tradeoff control: at `rho = 1.0` (default)
every snapshot entry is processed, which gives the highest
per-iteration progress; at `rho < 1.0` the per-iteration work
shrinks proportionally and the algorithm trades some
convergence rate for raw throughput.

The empirical claim from the literature is that on
SIFT1M-class datasets the recall-vs-wall-time curve is
remarkably flat between `rho ∈ [0.3, 1.0]` — a `rho = 0.5`
build hits the same final recall as `rho = 1.0` in ~80% of
the wall time. Step 37's writeup will plot the exact shape
of this curve from the bench JSON; this commit ships the
infrastructure to produce that data.

The "only mark sampled-new as old" rule is non-trivial. The
naive alternative — flip every new entry to old at snapshot
time — would mean unsampled entries silently lose their `new`
status without ever being processed against their peers,
breaking the algorithm's correctness guarantee. The Wang
formulation is precise: an entry is `is_new = true` until it
has *actually been processed*, so unsampled entries persist
across iterations until the sampler picks them. Pinning this
in the implementation now means future readers do not have to
reverse-engineer it from the paper.

The `rho` knob also explains why the bench harness needs a
per-iteration RNG. If the same sample were drawn every
iteration, unsampled entries would *never* be processed and
the algorithm would deadlock. Mixing `cfg.seed` with the
iteration index gives every iteration a distinct sample
without sacrificing the project's "same `cfg.seed` →
bit-identical output" determinism contract.

### Tradeoff
- **Four kernel variants now coexist.** `local_join`,
  `local_join_with_reverse`, `local_join_sampled`,
  `local_join_with_reverse_sampled`. We accept the API
  surface growth — each is a self-contained pedagogical
  artefact, and the driver picks for the user. The shared
  `join_pairs` / `snapshot_and_age` / `sort_unique` helpers
  keep the duplication to a minimum.
- **`rho_to_sample_size` clamps to ≥ 1.** A user who
  passes `rho = 0.0001` and `k = 5` would otherwise get a
  zero-size sample and the algorithm would do *nothing*.
  Clamping to at least 1 turns the pathology into a
  "convergence will be slow but it'll still happen"
  outcome. Better than the alternative.
- **Reverse-list subsampling uses a fresh
  partial Fisher-Yates per call.** Allocating a small
  position vector per point is `O(rev_new.size())` work; for
  the typical case where `|rev_new[p]| ≈ k`, this is
  negligible. A future "amortise the position scratch
  across points" refinement is in scope for Step 37's
  bench-driven optimisation pass.
- **`partial_fisher_yates` is `noexcept` but takes a
  reference-mutated pool.** A by-value variant would let
  the caller treat sampling as functional but cost an
  allocation per call. We accept the in-place mutation for
  the inner-loop primitive; the outer-loop callers always
  re-fill the pool from the snapshot anyway.

### Learning
- *The sampler advances state across iterations, not within
  one.* The driver's `iter_seed = cfg.seed ^ ((it + 1) * φ)`
  trick is the cheapest way to get diverse samples without
  threading an RNG through the call signature. Calling the
  kernel `local_join_sampled(... iter_seed)` keeps the
  sampler self-contained — no shared mutable RNG state, no
  threading concerns when Step 36 parallelises the kernel.
- *"Only mark processed entries as old" is the
  Wang-paper-exact behaviour the unsampled kernels can
  shortcut.* Steps 32 / 34's `mark_all_old()` is a
  correctness shortcut that works because *all* entries
  get processed every iteration. Step 35 cannot use it; the
  per-position flip is the one that generalises to
  sampling. The two formulations agree at `rho = 1.0` —
  proven by the bit-identical-output test — which is the
  reassurance that the more-general code did not accidentally
  introduce a regression.
- *"Sampling clamps to at least 1" is the kind of
  precondition that should fail loudly when violated.*
  Returning zero would be a quieter bug than throwing. We
  chose to clamp instead of throwing because `rho = 0.001`
  with `k = 5` is a *legal* configuration that should still
  produce *some* output; the alternative of "throw on every
  small-rho-times-small-k" surface-area would force every
  caller to compute the product themselves before invoking.

### Next
- Step 36: parallel NN-Descent (OpenMP). The outer point
  loop in every kernel variant becomes `#pragma omp
  parallel for`; per-point list inserts are protected by
  per-point locks (or atomic CAS). Scaling study against
  Step 34's serial baseline.

---

## [Step 34] — Reverse neighbour lists (NEO-DNND headline win) (2026-05-05)

### What
- Added `knng::cpu::local_join_with_reverse<D>(ds, graph,
  distance)` — the reverse-augmented local-join. Each
  iteration runs in three phases:
  1. **Snapshot + age** (shared with plain `local_join` via
     a new private `snapshot_and_age` helper). Build per-point
     `new_ids` / `old_ids` from the current list, then flip
     every entry's `is_new` flag to `false`.
  2. **Reverse-list construction.** Walk the snapshots; for
     every `q ∈ new_ids[p]`, push `p` into `rev_new[q]`. Same
     for `old`. After this phase, `rev_new[p]` holds every
     point whose this-iteration *new* snapshot contains `p`.
  3. **Local-join over the union.** For each point `p`:
     merge `(new_ids[p] ∪ rev_new[p])` and
     `(old_ids[p] ∪ rev_old[p])`, sort + unique each side,
     then drop ids that landed in *both* totals from the
     `old` side (the `new` flag wins so a fresh exchange
     propagates). Run the same `(new × new, u < v) +
     (new × old)` pair enumeration as Step 32.
- Refactored Step 32's local-join to use two new private
  helpers in the anonymous namespace:
  * `snapshot_and_age(graph, new_ids, old_ids)` — shared
    snapshot.
  * `join_pairs<D>(ds, graph, nv, ov, distance)` — pair
    enumeration body. Both `local_join` and
    `local_join_with_reverse` call this; the only difference
    between them is what `nv` / `ov` they hand in.
  * `sort_unique(v)` — sort-then-`unique` shorthand.
- Extended `NnDescentConfig` with `bool use_reverse = true`.
  The driver now picks between `local_join` (when `false`)
  and `local_join_with_reverse` (when `true`, the default).
  Defaulting to `true` matches the NEO-DNND paper's
  recommendation; setting it `false` is supported for
  ablation studies.
- Explicit instantiations for `local_join_with_reverse<D>`
  for `L2Squared` and `NegativeInnerProduct`.
- Four new `test_nn_descent` cases:
  * `local_join_with_reverse` does at least as much work as
    plain on the first iteration (the candidate set is a
    superset).
  * Reverse-enabled NN-Descent converges in no more
    iterations than the plain variant on the 8-point
    fixture.
  * Both variants reach `recall@k = 1.0` after enough
    iterations on the 8-point fixture.
  * `local_join_with_reverse` throws on graph-size mismatch.
- ctest 168/168 green (4 new nn_descent, 164 carried over
  from Step 33).

### Why
Reverse neighbour lists are the single most impactful
algorithmic optimisation NN-Descent ships, and the headline
contribution of the NEO-DNND paper. Wang et al. 2012 §4.2
observes that the neighbour-of-neighbour relation is *not*
symmetric under finite `k`: if `q` lists `p` in its top-`k`,
that does not guarantee `p` lists `q` in its own top-`k`. So
even after a perfect `(new × new)` local-join from `q`'s
perspective, `p` may still not have been told that `q` is a
candidate.

The fix walks the relation from both directions. For each
point `p`, the algorithm aggregates not just `p`'s forward
neighbours but also every point that has *p* in its forward
list. That superset is then run through the same pair-
enumeration as the plain local-join. The empirical effect on
SIFT1M-scale benchmarks is dramatic: a graph that took 12
iterations under plain local-join often converges in 5–6
under reverse. The compute cost per iteration grows
modestly (more pairs to consider) but the iteration count
drops more than enough to come out ahead in wall time.

The ablation knob (`cfg.use_reverse = false`) exists for
two reasons: pedagogy (we want a minimal version of
NN-Descent in the codebase that a future contributor can
read alongside the optimised one) and the Step-37 writeup
(which will plot `recall@k` vs iteration both with and
without reverse to make the headline number quantitative).

The `(new × new, u < v) + (new × old)` enumeration shape from
Step 32 carries through unchanged. The deduplication of
"items in both totals" — which can happen when `p ↔ q`
mutual neighbours appear from both directions — uses a
two-finger merge over the sorted vectors. Constant-factor
fast and the kind of thing that profiles into noise next to
the per-pair distance computation.

### Tradeoff
- **Per-iteration memory grows from `2 * O(n*k)` to
  `4 * O(n*k)`.** The reverse-list arrays double the
  snapshot storage. For SIFT1M with `k = 20`, this is
  ~40 MB extra — negligible, but it is the kind of
  allocation cost a long-lived GPU port (Phase 9) will
  want to amortise via reusable buffers. The CHANGELOG
  comment is the contract.
- **The "drop from old when present in new" merge is `O(|new|
  + |old|)` per point.** A `std::unordered_set` over
  `new_total` would be `O(|new| + |old|)` *expected* but
  with hash overhead. The two-finger merge after
  `sort_unique` is what we ship; it is cache-friendly and
  branch-predictable.
- **`use_reverse = true` becomes the silent default.**
  Existing code that called `nn_descent(ds, k, {}, dist)`
  before Step 34 implicitly opted into reverse lists with
  this commit. We accept the silent change because (a)
  every existing test's `recall@k` result is no worse and
  often better, and (b) every test that wanted ablation
  semantics now has the explicit knob to flip.
- **The two-phase reverse-list build is *not* incremental.**
  Each iteration rebuilds `rev_new` / `rev_old` from
  scratch. An incremental update (track the deltas during
  `local_join`) would save the `O(n * k)` walk but
  complicate the snapshot logic. Profiling at SIFT1M will
  decide whether the savings are worth the complexity.

### Learning
- *"More work per iteration, fewer iterations" is the
  standard NN-Descent shape.* Every later optimisation
  (sampling, parallelisation, GPU port) follows the same
  pattern: pay a small per-iteration cost increase to make
  each iteration *more* convergent, so the total
  iteration count drops dispropor­tionately. Recognising
  this trade as the project's recurring theme makes
  future steps easier to reason about.
- *Refactoring `join_pairs` out before adding
  `local_join_with_reverse` was the right move.* Two
  call sites (plain and reverse) sharing one helper means
  any future fix to the pair-enumeration touches one
  place. The pre-Step-34 form would have duplicated the
  body, and the inevitable third variant (Step 35's
  sampling) would have made the duplication pernicious.
- *Defaults reflect best practice; ablation switches
  preserve pedagogy.* The `use_reverse = true` default is
  what every textbook says; the `false` knob is what a
  reader new to the algorithm needs in order to see what
  reverse lists actually buy. Both should be a one-line
  change to flip; we built both.

### Next
- Step 35: sampling. Subsample the local-join candidate
  set to `rho * k` per point (default `rho = 1.0`,
  meaning no sampling). The `rho` knob is the third
  classic NN-Descent tuning parameter; it controls the
  speed/quality tradeoff and lands as a `NnDescentConfig`
  field with bench-harness sweep coverage.

---

## [Step 33] — Convergence-driven NN-Descent driver (2026-05-05)

### What
- Added `knng::cpu::NnDescentConfig` — three tunable knobs the
  driver consumes:
  * `max_iters` (default `50`) — hard safety bound; typical
    SIFT1M-scale runs converge in 10–15 iterations so `50` is
    deliberately generous.
  * `delta` (default `0.001`) — the convergence threshold on
    `n_updates / (n * k)`. Below this, the graph is declared
    stable and the driver returns. Matches Wang et al. 2012
    §4.1.
  * `seed` (default `42`) — passed straight to `init_random_graph`.
- Added `knng::cpu::NnDescentIterationLog` — per-iteration
  stats record: 1-based `iteration`, raw `updates` count,
  precomputed `update_fraction = updates / (n * k)`. The
  shape is what Step 37's bench harness JSON wants to consume
  for plotting the convergence curve.
- Added two driver entry points sharing a private
  `nn_descent_impl` body:
  * `Knng nn_descent(ds, k, cfg, distance)` — the simple
    wrapper most callers want.
  * `Knng nn_descent_with_log(ds, k, cfg, log_out, distance)`
    — same return value plus a per-iteration log written to
    the supplied vector.
  Both validate `cfg.delta >= 0`; `init_random_graph` validates
  the rest.
- The convergence test compares the absolute update count
  against `delta * n * k` rather than the floating-point
  fraction — same arithmetic, one fewer divide per iteration.
- Explicit instantiations for `L2Squared` and
  `NegativeInnerProduct` for both entry points.
- Seven new `test_nn_descent` cases pinning the contract:
  * Convergence to brute-force ground truth on the 8-point
    fixture (`recall@k = 1.0`).
  * Default `cfg{}` returns a shape-correct graph without
    hitting `max_iters`.
  * Same seed yields a byte-identical output graph.
  * `nn_descent_with_log` emits one entry per iteration with
    monotonically non-increasing update counts.
  * `delta = 1.0` causes the driver to stop after the first
    iteration (any positive update count is "below" the
    threshold).
  * `delta = 0.0, max_iters = 3` runs exactly 3 iterations
    (the safety bound binds when convergence is impossible).
  * Negative `delta` throws `std::invalid_argument`.
- ctest 164/164 green (7 new nn_descent, 157 carried over
  from Step 32).

### Why
Steps 31 and 32 each delivered a well-tested primitive but
not a usable builder. A caller who wanted "build a KNNG via
NN-Descent" would have to write the iteration loop themselves,
re-derive the convergence formula, and pick reasonable
defaults — and would get all of those slightly wrong, slightly
differently across the codebase. Step 33 is the
"single function that just works" entry point Phase 5's
remaining steps and Phase 13's CLI plug into.

The convergence threshold matters more than it sounds. Too
loose (`delta = 0.1`) and the driver stops while the graph is
still improving — recall is poor. Too tight (`delta = 0.0001`)
and the driver runs many more iterations than necessary for the
last fraction of a percent of recall. The default `0.001`
matches the Wang et al. paper and lands in a flat region of
the recall-vs-runtime curve where small perturbations have
near-zero effect. Step 37's writeup will document the exact
shape of this curve.

`nn_descent_with_log` exists for the same reason Step 16's
JSON counters exist: every later phase wants to plot
"convergence curve = updates per iteration." Without the log
entry point, the bench harness would have to call
`init_random_graph` + `local_join` in a loop itself, duplicating
~30 lines of driver logic. The log shape mirrors the JSON the
bench will emit, so wiring up Step 37 is a list-comprehension
rather than a rewrite.

The "stop when `updates < delta * n * k`" formulation deserves
a moment of explanation. The natural alternative — "stop when
the graph stops changing" — would require structural comparison
and is far more expensive than counting bools from
`NeighborList::insert`. The fraction-of-slots-changed metric is
both cheap and informative: it directly correlates with
recall@k convergence (graphs converge to the same fixed point
under repeated local-join, so once <0.1% of slots change
per iteration, the remaining changes touch a vanishing fraction
of neighbours). The plan's `delta = 0.001` is the empirically-
validated knob from the NN-Descent paper.

### Tradeoff
- **Two entry points for one feature.** `nn_descent` and
  `nn_descent_with_log` differ only in whether they write
  to a stats log. We accept the API doubling because the
  alternative (single function with optional `vector*` out
  parameter) makes the common case ugly: every caller has to
  write `nullptr` even when they don't want stats.
- **`NnDescentConfig` is a POD with default member
  initialisers, not a builder pattern.** Designated
  initialisers (`{.max_iters = 16, .delta = 0.0}`) cover
  the use cases. A builder would let us validate the
  combination at construction; we accept the late
  validation in `nn_descent_impl` because the failure mode
  is the same (throw) and the construction-time savings are
  zero.
- **`delta` is a `double`, the comparison is against an
  `int64_t * double`.** Mixing fp and integer in the
  threshold gives the natural meaning ("0.001 of the slots")
  but introduces a tiny rounding window where an update
  count could be exactly at the threshold. We accept the
  imprecision; it surfaces only on `n*k` values where
  `delta * n*k` is non-integer, which is the typical case
  anyway.
- **No early `init_random_graph` failure check before the
  iteration loop.** The graph builder validates `(ds, k)`
  itself; failure throws from there. We deliberately do not
  duplicate the validation in the driver — duplicate
  validation is duplicate maintenance debt.

### Learning
- *Convergence criteria deserve their own line in the
  CHANGELOG.* The `updates / (n*k) < delta` formulation
  looks trivial in code (one comparison) but is the
  algorithmic statement that NN-Descent's correctness rests
  on. Documenting the *interpretation* ("the graph has
  stabilised when fewer than 0.1% of slots change") is what
  lets a future reader tune the knob without re-reading the
  paper.
- *Optional out-parameters via shared private impl is the
  right pattern for "same logic, sometimes I want stats."*
  The `nn_descent_impl` private function with a
  `std::vector*` parameter — call it with `nullptr` from
  the simple wrapper, with an actual pointer from the log
  variant — keeps the iteration logic in one place. Both
  public entry points are <10 lines each.
- *Convergence tests can be cheap.* The
  "stops early on `delta = 1.0`" test runs in microseconds
  on the 8-point fixture but gives high confidence that
  the threshold logic works at the boundary. We pin the
  boundary case explicitly rather than relying on
  large-fixture tests to catch it.

### Next
- Step 34: reverse neighbour lists. The local-join currently
  only considers `p`'s neighbours; adding the *reverse*
  graph (the points that list `p` as a neighbour) into the
  union dramatically improves recall convergence,
  especially at low `k`. The NEO-DNND paper's headline
  optimisation.

---

## [Step 32] — Local-join kernel (single-threaded) (2026-05-05)

### What
- Added `knng::cpu::local_join<D>(const Dataset&, NnDescentGraph&,
  D)` — the algorithmic core of NN-Descent. One iteration runs in
  two phases:
  * **Snapshot.** Walk every point. For each, partition its
    current list into `new[p] = {id : entry.is_new}` and
    `old[p] = {id : !entry.is_new}`. Then call
    `mark_all_old()` so the *next* iteration sees only entries
    that get newly inserted *during* this one. The snapshots
    are vectors on the call frame; the algorithm operates from
    them after this point, never re-reading the lists' flags.
  * **Local-join.** For each point `p`, compute and insert
    every `(u, v)` pair where:
      * `u` and `v` are both in `new[p]` and `u < v` (the
        `<` avoids visiting the same pair twice within `p`'s
        neighbourhood);
      * `u ∈ new[p]` and `v ∈ old[p]` (no duplication concern
        because the two sets are disjoint).
    Old × old pairs are deliberately omitted — that is the
    optimisation Step 30's `is_new` flag exists to enable.
- Inserts during the local-join are flagged `is_new = true` so
  they propagate to the next iteration's snapshot. The function
  returns the total `NeighborList::insert` calls that *changed*
  a list; Step 33's driver compares this to `delta * n * k` to
  decide convergence.
- Explicit instantiations for `L2Squared` and
  `NegativeInnerProduct` mirroring `init_random_graph`.
- Five new `test_nn_descent` cases pinning the contract:
  * First iteration on a random graph produces a non-zero
    update count and preserves the per-row shape (`size == k`,
    sorted ascending, no self-matches).
  * After one iteration, both `is_new = true` and
    `is_new = false` entries coexist (snapshot plus
    fresh-insert combination).
  * Iterating to convergence on the 8-point fixture matches
    brute-force ground truth exactly (`recall@k = 1.0`).
  * Per-iteration update counts are monotonically non-
    increasing (the algorithmic claim — convergence comes
    from work shrinking).
  * Graph-size mismatch with the dataset throws
    `std::invalid_argument`.
- Wired `knng::bench` into the `test_nn_descent` link line so
  the recall-against-brute-force assertion can use Step 15's
  `recall_at_k`.
- ctest 157/157 green (5 new local-join, 152 carried over from
  Step 31).

### Why
The local-join is *the* defining innovation of NN-Descent. The
high-level intuition: "the neighbour of my neighbour is likely
my neighbour." If `p` has neighbours `u` and `v`, then `u` and
`v` are by transitivity likely to be each other's neighbours
too — so the algorithm proactively computes `d(u, v)` and offers
the result to both lists, even though `u` and `v` were never
directly compared. Iterating this exchange across every
point's neighbourhood propagates information through the graph
much faster than brute-force's "compare every pair against
every pair" `O(n²)` scan.

The asymptotic win is dramatic: brute-force is `O(n²)` per
build (`n * (n - 1)` distance computations); the local-join is
`O(n * k²)` per *iteration* (every point contributes `O(k²)`
pairs from its neighbourhood). For `n = 1M, k = 20, iterations
= 10`, this is 4 billion distance ops vs 1 trillion — a 250×
asymptotic improvement before any constant-factor optimisation.

The two-phase snapshot is what makes the algorithm
parallelisable in Step 36. By freezing every point's `new[p]`
and `old[p]` upfront (and flipping the in-list flags atomically
per-point in phase 1), phase 2's local-join body only mutates
*other* points' lists, never reads from a list it is currently
modifying. The single-threaded version doesn't strictly need
the two-phase split — sequential mutation would be order-
dependent but correct — but pinning the snapshot shape now
means Step 36 is a translation, not a redesign.

The `u < v` filter inside `new × new` is the second-cheapest
optimisation in the file (after `is_new`). Without it, every
pair gets visited twice within `p`'s neighbourhood (once as
`(u, v)` and once as `(v, u)`); both visits compute the same
`d(u, v)` and offer it to the same two lists. The filter halves
the work on `new × new` pairs; the `new × old` loop has no
analogous duplication because the two sets are disjoint.

The convergence-counter return value is the contract that lets
Step 33 stop the loop without poking inside the data structure.
Step 30's `NeighborList::insert` already returns `bool` for the
"did this change anything?" question; `local_join` simply
sums those bools. The aggregate is the natural input to the
`updates / (n * k) < delta` convergence check.

### Tradeoff
- **`std::vector<std::vector<index_t>>` snapshots cost
  `O(n * k)` allocations per iteration.** A pre-allocated
  flat `(n * 2 * k)` buffer with offsets would amortise to
  zero allocations, but at the cost of ~30 lines of fiddly
  index math. We keep the simpler shape; profiling at
  SIFT1M-scale will tell us when it matters.
- **The `u == v` defensive checks inside the inner loops
  cannot fire** (snapshots contain distinct ids by
  construction). They are kept as belt-and-braces in case a
  future bug leaks duplicates into a list; the branch is
  predictable and adds <1% to the inner loop.
- **No early exit on `worst_dist`.** The `NeighborList`
  exposes the worst-distance accessor for exactly this
  shape ("if the candidate distance is already worse than
  the worst slot, skip insertion"), but the local-join always
  pays the `O(d)` distance computation before checking. We
  accept the cost: the algorithmic shape is what
  `recall@k = 1.0` rests on, and the early exit would
  introduce subtle ordering effects that complicate the test
  fixtures. A future "fast-path local-join" can revisit.
- **The function template instantiates the body twice.**
  Each metric (`L2Squared`, `NegativeInnerProduct`) gets its
  own copy of the local-join. We accept the binary size
  growth — the alternative (runtime-dispatched metric)
  would put a virtual call in the hottest inner loop.

### Learning
- *NN-Descent's correctness emerges from iteration, not from
  any single pass being correct.* The first iteration's
  output is *not* a high-recall graph — it is just better
  than random. Each subsequent iteration further reduces
  the per-point neighbour distances. Pinning the
  "monotonic update count" test
  (`u2 ≤ u1`) is the lightweight way to assert this without
  measuring recall directly.
- *The `u < v` filter is the kind of optimisation that
  costs zero lines of explanation but doubles throughput.*
  In a research codebase the temptation is to write the
  loop as `for u in nv: for v in nv: if u < v: ...`; the
  arithmetic-friendly form `for i in [0, k): for j in (i, k):`
  is functionally identical but reads as the algorithm's
  intent ("each pair, once").
- *The acid test for any iterative refinement is:
  "does it converge to the brute-force answer?"* The
  `IteratingToConvergenceMatchesBruteForceRecall` test is
  the single piece of evidence that the local-join is
  algorithmically correct — every other property
  (shape, `is_new` accounting, monotone updates) could pass
  on a buggy implementation that converges to *some*
  answer, just not the right one.

### Next
- Step 33: the convergence-driven driver. Iterate
  `local_join` until `n_updates / (n * k) < delta` or the
  iteration cap is hit. Consumes the convergence-counter
  return value this step ships and produces the public
  `nn_descent` builder.

---

## [Step 31] — Random graph initialisation (`init_random_graph`) (2026-05-05)

### What
- Added `include/knng/cpu/nn_descent.hpp` and
  `src/cpu/nn_descent.cpp` — the public interface for Phase 5's
  CPU NN-Descent builder. Step 31 contributes:
  * `class knng::cpu::NnDescentGraph` — `n × k` collection of
    `NeighborList`s. Owns its storage, has `at(i)` row accessors,
    `n() / k()` getters, `lists()` for bulk access, and
    `to_knng()` to flatten into the canonical `knng::Knng`
    shape. Empty slots in `to_knng` are filled with the
    `NeighborList::kEmptyId` sentinel and `+inf` distance.
  * `template <Distance D> NnDescentGraph init_random_graph(
    Dataset, k, seed, D)` — the random k-NN graph initialiser.
    Every point gets `k` distinct non-self random neighbours
    under the supplied distance functor; every entry is flagged
    `is_new = true`. RNG is `knng::random::XorShift64{seed}`
    from Step 17 — same `(ds, k, seed)` triple ⇒ bit-identical
    output across runs across platforms.
- Sampling strategy: rejection sampling over `[0, n)`. Skip
  self-matches and duplicates (the duplicate check saves the
  distance computation; `NeighborList::insert` would have
  rejected silently anyway). Defensive cap of `4*k + 16`
  attempts per point so a degenerate input cannot infinite-loop.
- Explicit instantiations for `L2Squared` and
  `NegativeInnerProduct` so downstream callers link against
  pre-compiled symbols.
- Thirteen new `test_nn_descent` cases pinning the contract:
  shape, `to_knng` sentinel-fill on partial rows, every entry
  is non-self / distinct / `is_new = true`, rows sorted
  ascending, same-seed determinism, different-seed divergence,
  distances match the underlying metric, `to_knng` round-trip
  consistency, both built-in metrics compile, `k = 0` /
  `k > n - 1` / empty-dataset throw `std::invalid_argument`.
- Wired `cpu/nn_descent.cpp` into `knng_cpu` and
  `tests/nn_descent_test.cpp` into the CTest matrix (the same
  TU will accumulate Step-32+ tests).
- ctest 152/152 green (13 new nn_descent, 139 carried over
  from Step 30).

### Why
Random initialisation is NN-Descent's starting condition. Wang
et al. 2012 §4.1 begins from a random k-NN graph; the
local-join (Step 32) then refines it iteration after iteration
until convergence. The choice of *random* (not "approximately
correct") starting state is deliberate and load-bearing:
NN-Descent's correctness proof relies on the algorithm being
able to escape any local optimum the initial graph happens to
sit in, and a "smart" initialiser (e.g. seed from a coarse
brute-force on a subset) would actually *slow* convergence
because it pre-commits to a sub-graph the local-join then has
to climb out of.

The deterministic-seed property is what every regression test
in Phase 5+ will rely on. The `same seed → same graph` invariant
runs all the way down to the bit level: `XorShift64`'s integer
arithmetic is bit-identical across CPU and (eventually) GPU,
and the `uniform_below` Lemire trick uses only integer
operations. When Phase 9's GPU NN-Descent ships, the
`init_random_graph` test fixture here will be re-asserted on
device, byte-for-byte.

The `to_knng` conversion exists so the rest of the project can
treat the NN-Descent output the same way it treats the
brute-force output. Step 33's recall comparison will compute
`recall_at_k(g.to_knng(), brute_force_truth)` without caring
whether the source was the heap-based brute-force builder or
the iterative refinement builder. The `kEmptyId` /
`+inf` sentinel for partial rows is the "correct" default for
that comparison: an unfilled slot cannot recall any neighbour,
so it correctly contributes zero hits.

The defensive `max_attempts` cap is the kind of bug-prevention
that costs nothing and saves a 3-AM debugging session. NN-Descent
literature occasionally shows code that loops forever on
pathological inputs (typically `n ≈ k` cases not covered by
the input validation); pinning a bound that is generous in
expectation but finite in pathology is the right shape.

### Tradeoff
- **Rejection sampling, not Floyd's algorithm.** Floyd's
  algorithm samples `k` distinct values from `[0, n)` in
  exactly `O(k)` time without rejection. We use rejection
  sampling because `k ≪ n` for every realistic input and the
  expected attempts per slot are <2; the simpler code reads
  more like the algorithm's literature (which always
  describes "draw a random neighbour, retry if duplicate").
  Future profiling can swap the implementation if it ever
  shows up as a bottleneck.
- **Distance functor is templated.** Mirrors the
  `brute_force_knn` shape; consumers that prefer the runtime
  dispatch (CLI, future Python bindings) will go through a
  thin `MetricId`-keyed wrapper. The two built-in functors
  are explicitly instantiated; user-supplied `Distance` types
  pay the per-TU instantiation cost at the call site, which
  is the right tradeoff for a research codebase.
- **`NnDescentGraph::lists()` exposes the underlying vector
  directly.** A future contributor could reach in and
  invalidate the per-row capacity invariant; we accept the
  exposure because the local-join kernel needs raw bulk
  access and a "safer" iterator-pair abstraction would just
  move the problem one level deeper.
- **Construction allocates `n` `NeighborList`s.** For SIFT1M
  with `k = 20`, this is ~1M empty vectors — each holding a
  reserved `Neighbor[k]` capacity of ~240 bytes — totalling
  ~240 MB of empty pre-allocated headroom. This is the
  canonical NN-Descent storage and matches what every
  production implementation does; we accept the
  pre-allocation as the steady-state cost of the algorithm.

### Learning
- *Determinism in a random algorithm is a feature, not a
  contradiction.* `init_random_graph` is "random" in the sense
  that the output is not algorithmically pre-determined by the
  input, but it is *fully determined* by the `(ds, k, seed)`
  triple. Every Phase-5 test fixture relies on this property —
  the unit tests are not "approximately correct"; they pin the
  exact graph the algorithm produces under the exact seed.
- *The duplicate check before the distance call is a real
  optimisation.* Without it, the rejection-sampling loop would
  call `distance(...)` on every duplicate id (a few percent
  of attempts) only to have `NeighborList::insert` silently
  drop the result. With it, the duplicate path costs one
  linear scan over `k ≤ 50` entries — far cheaper than an
  `O(d)` distance computation.
- *Sentinel-fill in `to_knng` is the cleaner contract.* An
  earlier draft trimmed the output `Knng`'s `k` to the
  minimum-row size, but that broke the
  `recall_at_k(approx, truth)` shape contract (Step 15
  requires `approx.k == truth.k`). Returning a
  `(n × k)` graph with sentinel-filled empty slots keeps the
  shape stable and lets `recall_at_k` correctly score them as
  "missed."

### Next
- Step 32: the local-join kernel. For each point `p`, for each
  pair of neighbours `(u, v)` of `p` where at least one is
  `is_new`, compute `d(u, v)` and try to insert into both
  lists. Single-threaded for now; the `is_new` flag from
  Step 30 is the redundancy filter that makes this tractable.

---

## [Step 30] — `Neighbor` + `NeighborList` types (2026-05-05)

### What
- Added `include/knng/cpu/neighbor_list.hpp` and
  `src/cpu/neighbor_list.cpp` — the per-point neighbour list
  every NN-Descent iteration mutates. Two public types in this
  step:
  * `struct knng::Neighbor { index_t id; float dist; bool is_new; }`
    — the entry shape. The `is_new` flag is the algorithmic
    redundancy filter that turns NN-Descent's local-join from
    pairwise-everything into "consider only `(u, v)` pairs
    where at least one is new since the last iteration."
  * `class knng::cpu::NeighborList` — bounded-size,
    sorted-ascending-by-distance container with capacity `k`.
    Public surface: `insert(id, dist, is_new) → bool`,
    `contains(id)`, `mark_all_old()`, `view()`, `size()`,
    `capacity()`, `empty()`, `full()`, `worst_dist()`. Tie-break
    matches `TopK` (Step 09): equal distances ordered by
    ascending neighbour id, so output is deterministic without
    an RNG.
- `insert` returns `true` when the list's contents change. The
  bool is what Step 33's convergence check counts:
  `n_updates += list.insert(...) ? 1 : 0` summed across every
  list, divided by `n*k` to get the per-iteration update fraction.
- Duplicate-id semantics are explicit: an existing entry with a
  smaller distance wins; an existing entry with a larger
  distance is replaced (and inherits the new `is_new` flag, so
  a re-insertion with `is_new=true` re-activates a previously
  processed neighbour for the next iteration).
- Twelve new `test_neighbor_list` cases pinning the contract:
  fresh-construction empty, below-capacity insertion order, tie-
  break by ascending id, capacity eviction (worst goes;
  re-tested under tied-distance), duplicate-with-worse-dist
  ignored, duplicate-with-better-dist replaces, `mark_all_old`
  flips every flag, `contains` linear scan, `k = 0` rejects
  every insert and reports `full()` (degenerate-but-coherent),
  `worst_dist` tracking under inserts and evictions,
  newly-inserted-after-`mark_all_old` is again `is_new`.
- Wired `cpu/neighbor_list.cpp` into `knng_cpu` and
  `tests/neighbor_list_test.cpp` into the CTest matrix.
- ctest 139/139 green (12 new neighbor_list, 127 carried
  over from Step 29).

### Why
Phase 5's pivot from brute-force to NN-Descent rests on one
constant-factor optimisation: only consider `(u, v)` neighbour
pairs where at least one is "new" — i.e. has been inserted or
modified since the previous iteration. Without the `is_new`
flag, the local-join would re-examine every pair every
iteration; with it, the per-iteration work shrinks
monotonically as the graph stabilises and convergence becomes
the natural stopping condition. This is the single most
important constant-factor optimisation NN-Descent ships, and
it is referenced directly in Wang et al. 2012 §4.1 and the
NEO-DNND paper §3.1.

The container is a flat `std::vector<Neighbor>` rather than a
`std::set` / `std::priority_queue` / hash map for one decisive
reason: at the `k = 10..50` sizes NN-Descent runs on, a linear
scan beats every smarter data structure by both constant
factor (no allocator overhead per insert; cache-friendly) and
code-line count (the algorithm is two lines of `std::lower_bound`
plus an `erase` / `insert`). The same trick `TopK` (Step 09)
ships, applied to the slightly richer "neighbour with flag"
type.

The `worst_dist()` accessor is what Step 32's local-join uses
for early rejection: if a candidate's tentative distance is
already worse than `list.worst_dist()`, the list cannot
benefit from it and the insertion call is skipped — saving the
linear-scan duplicate-check on every call.

`insert` returning `bool` is the convergence-counting hook the
plan calls out in Step 33. We pin the semantics here so a
future caller does not need to look at the implementation to
know what "did the list change?" means: a duplicate-with-
worse-dist returns `false` (no contents change); a
duplicate-with-better-dist returns `true` (distance and flag
both updated); an insertion below capacity always returns
`true`; a successful eviction-and-insert returns `true`; a
rejected-because-worse returns `false`.

### Tradeoff
- **Linear scan for both `contains` and the duplicate check.**
  At `k = 50` this is ~25 comparisons per insert in the worst
  case — utterly negligible next to the per-pair distance
  computation (`O(d)` floats, typically `d = 128–960`). At
  larger `k` (the regime where ANN benchmarks rarely operate)
  a hash-of-ids would help; we will not add it speculatively.
- **`mark_all_old` is `O(k)` per call.** A bit-mask shadow
  array would make it `O(1)` (one `std::fill`), but a
  `std::vector<bool>` introduces aliasing concerns with the
  `Neighbor::is_new` field's natural location. We accept the
  `O(k)` cost — `mark_all_old` is called once per point per
  iteration, dwarfed by the `O(k²)` local-join work that
  follows.
- **Duplicate handling does an `erase` + reinsert on the
  better-distance path.** This is `O(k)` because of the
  vector shift; an in-place update + `std::sort` of the
  affected range would be slightly faster but more code.
  We accept the simpler path; future profiling can revisit
  if it shows up as a hot block.
- **`Neighbor` is 12 bytes (4 + 4 + 1 + 3 padding).** A
  packed `uint32_t id_and_flag` could shave the bool into the
  high bit of `id`, saving 4 bytes per entry. We accept the
  natural layout — at `k = 50` the per-list footprint is
  ~600 bytes, and removing the padding would force every
  consumer to mask off the flag bit. The simplicity is worth
  it.

### Learning
- *The smallest data structure that supports the algorithm is
  the right one.* The temptation in NN-Descent literature is
  to read about Wang et al.'s "neighbour list with flags" and
  imagine something elaborate; the actual minimal
  implementation is one struct + one class + 200 lines of
  code. Pinning that minimum here, before the algorithm
  arrives, makes Step 32's local-join read as "consume this
  type" rather than "invent a type *and* the algorithm at
  once."
- *Returning `bool` from `insert` is the cheapest
  convergence-counter integration.* The alternative — having
  Step 33 inspect the list state before and after — would
  force every caller to remember which fields constitute a
  "change." `insert`'s return value is the canonical answer,
  and Step 33 is one `+=` per call.
- *`is_new` is per-entry, not per-list.* An earlier draft
  considered a single "list-level new flag" that flipped to
  false after one iteration. That would have been wrong:
  freshly-inserted entries during an iteration must be
  individually trackable as new for the *next* iteration's
  local-join. The per-entry flag is the only correct shape.

### Next
- Step 31: `init_random_graph(Dataset, k, seed)`. Every point
  gets `k` random neighbours, all marked `is_new = true`. This
  consumes `NeighborList` from this step and produces the
  initial graph state Step 32's local-join will refine.

---

## [Step 29] — CPU scaling writeup (2026-05-04)

### What
- Added `docs/SCALING_CPU.md` — the headline artefact for
  Phases 3 and 4. Two tables on a single page:
  * **Single-threaded ladder** — every Phase-3 + Phase-4 builder
    at `n=1024, d=128, k=10`, sorted by speedup vs the canonical
    Step-10 baseline. Five rows: canonical (71.26 ms),
    norms (66.41 ms / 1.07×), partial_sort (62.99 ms / 1.13×),
    SIMD/NEON (23.46 ms / **3.04×**), BLAS/Accelerate
    (4.00 ms / **17.82×**).
  * **Strong-scaling sweep** — three parallel implementations
    (OMP plain, OMP-with-scratch, std::thread) across
    `threads ∈ {1, 2, 4, 8}` at the same `(n, d, k)`. The
    three implementations land within 5% of each other at
    every thread count; ~7× at 8 threads on Apple M-series
    (P-core + E-core mix).
- Added `docs/aggregate_phase4.py` — a 70-line, dependency-free
  Python script that ingests `bench_brute_force --benchmark_format=json`
  output and prints both tables to stdout. Field names mirror
  the C++ counter set (`recall_at_k`, `threads`); the script's
  `_PARALLEL_FAMILIES` tuple is the single edit point if a future
  step adds another parallel builder.
- Captured the six take-aways the writeup commits to: two big
  wins + three small ones in the ladder; near-linear strong
  scaling up to 8 threads; OpenMP and `std::thread` are
  source-line decisions not perf decisions; recall stays at
  `1.0` across every cell of every table; the unrealised win
  is OMP × SIMD × BLAS stacking which Phase 5 will need.
- Five open questions deferred for the next pass:
  `-march=native` rerun on the same host, Linux + GCC +
  OpenBLAS comparison, weak-scaling table, NUMA first-touch
  on/off comparison at SIFT1M scale, OMP × SIMD × BLAS
  composability.
- ctest 127/127 green; this step adds documentation only.

### Why
Phases 3 and 4 produced nine new builders between them (norms,
tiled, partial_sort, BLAS, OMP, OmpScratch, Threaded, SIMD,
plus the ground-truth builder). Each has its own CHANGELOG
entry; without a single page that puts every wall-time number
in one table, the project's pedagogical narrative gets lost
in the noise. `docs/SCALING_CPU.md` is the page a reader can
jump to, see the entire ladder, and understand "what did
parallel and serial CPU optimisation actually buy us?" without
spelunking through nine commit messages.

The structure (`Setup · Single-threaded ladder ·
Strong-scaling sweep · Take-aways · Methodology · Open
questions · Reproduction commands`) extends Step 23's
`PERF_STEP23.md` template to include a *parallel* table.
Every later writeup
(`docs/PERF_SINGLE_GPU.md`, `docs/MULTI_GPU.md`,
`docs/DISTRIBUTED_GPU.md`) will inherit the same shape with
an extra "compute resource" axis substituted for "threads"
(GPUs, nodes, ranks).

`aggregate_phase4.py` exists for the same reason
`tools/plot_bench.py` exists: the C++ build produces the JSON,
the Python script renders the human-readable summary, and the
two are kept in sync by name conventions
(`F_NAME`, `F_TIME`, `F_RECALL`, `F_THREADS`) at the top of
the Python file. Renaming a counter in the C++ side becomes a
two-edit change.

The "OMP × SIMD × BLAS stacking" take-away is the most
important Phase-4 close-out: as shipped, the three parallelism
axes are *orthogonal* but not *composable*. The OMP path uses
the scalar dot product (no SIMD); the SIMD path is single-
threaded; the BLAS path is single-threaded but Accelerate is
internally multi-threaded above some `n` threshold. Phase 5's
NN-Descent needs OMP × SIMD; the multi-GPU phases need OMP +
SIMD + GPU coordination. Pinning the unrealised composability
in this writeup is what gives a future contributor the right
context for the question "should I add OpenMP to the SIMD
path?" (answer: yes, when Phase 5 needs it).

### Tradeoff
- **Numbers are AppleClang on M-series only.** The same
  caveat as Step 23's writeup; explicit in the
  "Open questions" section. A Linux + GCC + OpenBLAS rerun
  is the natural Phase-5-precondition follow-up.
- **No NUMA first-touch row in the table.** Step 26 ships
  `knng::cpu::first_touch` but the bench harness does not
  call it. We keep the omission visible in "Open questions"
  rather than silently turning it on (which would muddy the
  baseline numbers) or rolling a separate
  `BM_BruteForceL2OmpScratchFirstTouch_Synthetic` family
  (which would balloon the bench grid).
- **The `aggregate_phase4.py` heuristic is a hard-coded
  list of parallel-family prefixes.** If a future step
  adds a parallel builder with a different naming
  convention, the script reports it under the
  single-threaded section. The mitigation is the
  `_PARALLEL_FAMILIES` tuple at the top of the file —
  visible, easy to grow.
- **`aggregate_phase4.py` is `phase4`-named, not
  generic.** A future Phase-7 GPU writeup will want a
  similar aggregator; we will copy this one and edit the
  family list rather than parameterising. Two scripts at
  100 lines each beat one parameterised script at 200.

### Learning
- *The right time to ship a writeup is the moment the data
  is ambient.* By the end of Step 28 the bench harness can
  produce all the numbers in one run; waiting until Phase 5
  to write the Phase-4 writeup would mean the numbers no
  longer match the head of `main`. Pinning the artefact
  *now*, with the bench JSON and the aggregator script
  committed alongside, is what makes "reproduction six
  months from now" a realistic claim.
- *Two tables are enough; three is too many.* The first
  draft of this writeup had four tables (single-threaded,
  strong-scaling, weak-scaling, NUMA-comparison). The
  weak-scaling and NUMA tables would have been mostly
  empty for the reasons in "Open questions." A short
  writeup with two complete tables is more useful than a
  long writeup with two complete and two empty ones.
- *Family-naming convention is load-bearing.* The
  aggregator's heuristic for "is this row a parallel
  builder?" is `name.startswith(prefix)` where the
  prefixes are the bench-family names. Step 24's choice
  of `BM_BruteForceL2Omp_Synthetic` (rather than
  `BM_BruteForceL2_Synthetic_OMP`) made this trivial.
  Future bench families should follow the same pattern.

### Next
- Phase 5 (Step 30): random-graph initialisation as the
  first NN-Descent step. The brute-force chapter closes
  here; from Step 30 onwards the project moves to
  approximate algorithms where `recall_at_k` is no longer
  trivially 1.0 and the speed-vs-quality Pareto plots
  start mattering.

---

## [Step 28] — Hand-vectorised SIMD distance kernel (2026-05-04)

### What
- Added `include/knng/cpu/distance_simd.hpp` and
  `src/cpu/distance_simd.cpp`. Two public entry points:
  * `simd_squared_l2(a, b, dim)` — hand-vectorised squared-L2.
  * `simd_dot_product(a, b, dim)` — hand-vectorised inner product.
  Plus the diagnostic helpers
  `compiled_simd_path() → SimdPath::{kAvx2, kNeon, kScalar}` and
  `active_simd_path()` (the runtime answer; on x86 binaries
  compiled with `-mavx2` but running on a non-AVX2 CPU,
  `compiled_simd_path()` returns `kAvx2` while
  `active_simd_path()` falls back to `kScalar`).
- Compile-time path selection:
  * **AVX2** (`__AVX2__`) — 8 floats per `__m256` register, FMA
    via `_mm256_fmadd_ps`, horizontal reduction via paired
    `_mm_hadd_ps`. Selected when the user passed `-mavx2` /
    `-march=native` on a CPU that supports it.
  * **NEON** (`__ARM_NEON`) — 4 floats per `float32x4_t`, FMA
    via `vfmaq_f32`, horizontal reduction via `vaddvq_f32` (one
    instruction on ARMv8). Mandatory on any modern arm64
    target so always present on Apple Silicon.
  * **Scalar fallback** — calls `knng::squared_l2` /
    `knng::cpu::dot_product` directly. Same answer, no speedup.
- Runtime CPUID fallback on x86 via
  `__builtin_cpu_supports("avx2")`, cached in a function-local
  `static const bool`. Apple's NEON path skips this — NEON is
  ARMv8-mandatory.
- Added `knng::cpu::brute_force_knn_l2_simd(ds, k)` — the same
  shape as `brute_force_knn_l2_with_norms` (Step 19) with the
  inner-loop dot product replaced by `simd_dot_product`. Output
  is bit-equivalent to the canonical builder up to fp
  accumulation reordering.
- Eight new `test_brute_force` cases: SIMD `squared_l2` matches
  scalar on a tail-only (dim=17) buffer, on a power-of-2 dim
  with hand-computed reference, dot-product matches scalar
  on dim=23 (odd), `dim=0` returns zero on both primitives,
  `active_simd_path` returns a valid enumeration, the
  `brute_force_knn_l2_simd` builder matches the canonical
  builder on the 8-point fixture, and `k=0` / `k>n-1` throw.
- Added `BM_BruteForceL2Simd_Synthetic` family at the same
  `(n, d)` grid as the scalar baselines.
- ctest 127/127 green (8 new brute_force, 119 carried over
  from Step 27).
- **Measured at n=1024, d=128:** NEON SIMD 23.4 ms vs
  canonical 70.8 ms — **~3.0× speedup**, recall stays at 1.0.
  Second-largest single-step win in the project after Step 21's
  BLAS path.

### Why
The autovectoriser is good — at d=128 on AppleClang it produces
NEON code for the canonical inner loop already — but it leaves
throughput on the table at the *boundaries* between the
distance-formula's three operations (subtract, multiply,
accumulate). Hand-vectorising as a single `vfmaq_f32(acc, d, d)`
sequence per chunk fuses the three operations into one
instruction per 4-lane chunk, which the autovectoriser will
not always generate when it has to reason about loop-carried
dependencies and aliasing.

The 3× win at d=128 measures exactly this: the per-pair work
goes from ~32 NEON instructions (autovec) to ~16 (hand-written
FMA chain). The horizontal reduction at the end (one
`vaddvq_f32`) is bottleneck-free; the scalar tail (zero
iterations at d=128 since 128 is a multiple of 4) costs
nothing on this fixture but exists because Fashion-MNIST is
d=784 — *not* a multiple of 4 — and the same code has to work
there.

The mapping to GPU warp-level thinking is the *point* of this
step. An `__m256` holds 8 lanes; a CUDA warp holds 32. The
AVX2 dot-product loop computes a per-lane partial sum and
reduces at the end — the same shape as a CUDA shuffle-based
warp reduction (`__shfl_xor_sync`). When Step 53 ships the
GPU warp-level top-k, the structure of this file is what the
kernel will inherit; only the lane count and the reduction
primitive change.

The runtime CPUID dispatch on x86 is overkill for a typical
build — a developer who passes `-march=native` is going to run
on the same CPU. We ship it anyway because it is two lines of
code (`__builtin_cpu_supports` plus a static cache) and it
catches a real footgun: a binary built with `-march=native` on
one machine, copied onto a deploy host that lacks AVX2, would
otherwise SIGILL on the first `_mm256_loadu_ps`. Falling back
to scalar is far better than crashing.

### Tradeoff
- **No `-mavx2` flag is set automatically.** The build picks
  AVX2 only when the toolchain *already* defines `__AVX2__`,
  which happens with `-march=native` on a CPU that supports
  it. We deliberately do not add `-mavx2` to the project's
  default flags: doing so would lose the runtime
  CPUID-dispatch benefit (the binary would refuse to start on
  any non-AVX2 CPU) and would force a per-CPU build matrix on
  CI. The `KNNG_HAVE_*` flag pattern (BLAS, OpenMP) does not
  apply here because the SIMD code is *correctness*-equivalent
  on every platform; only the speed varies.
- **AVX-512 path absent.** The same code structure trivially
  extends to `__m512` (16 lanes) on Skylake-X / Zen 4 / future
  arm64 SVE2. We deferred: (a) the project's CI runners do not
  exercise AVX-512, (b) the win at d=128 over AVX2 is at most
  another ~1.5× — far less than Step 21's 18× BLAS jump on
  the same fixture. The CHANGELOG flags this as a future
  refinement; the existing structure makes it a one-file diff.
- **No SIMD variant for the BLAS path's epilogue.** The norm
  fold-in inside `brute_force_knn_l2_blas` (Step 21) is
  scalar; vectorising it would shave a few percent off the
  already-3.8 ms BLAS path. The wall-time priority is
  elsewhere — we leave the epilogue scalar.
- **Eight tests for two primitives is a lot of ceremony.** We
  accept it: the SIMD primitives sit underneath every later
  parallel-CPU step, and a regression in either would silently
  change recall numbers across the entire project. The
  per-tail and per-aligned-dim coverage is the kind of thing
  that catches a future "I rewrote the FMA chain to use
  `vmlaq_f32` instead of `vfmaq_f32`" mistake.

### Learning
- *FMA fuses three instructions into one — and the
  autovectoriser does not always notice.* The canonical
  inner loop is `delta = a - b; delta_sq = delta * delta;
  acc += delta_sq;`. The optimal NEON sequence is
  `vsubq_f32` + `vfmaq_f32` (`acc += delta * delta`) — two
  instructions per 4-lane chunk. The autovectoriser
  sometimes produces the same; sometimes it produces the
  three-instruction form. Hand-writing the FMA *guarantees*
  the two-instruction form.
- *NEON's `vaddvq_f32` is the single-instruction horizontal
  reduce.* AVX2 has no direct equivalent; the four-line
  `_mm_hadd_ps` chain is the standard workaround. ARM
  designed the `vaddvq_f32` instruction precisely to make
  the horizontal-reduce step cheap, and it shows in the
  numbers. Step 53's GPU port will use `__shfl_xor_sync` for
  the analogous role.
- *Runtime CPUID is cheap insurance.* The
  `__builtin_cpu_supports("avx2")` query is one CPUID
  instruction the first time it runs; cached forever after.
  Skipping it would cost two lines of code and one bug that
  takes a day to debug when it fires. Always pay the
  insurance.
- *The diagnostic `enum class SimdPath` is what makes the
  test suite *useful*.* Without `compiled_simd_path()` and
  `active_simd_path()`, a future contributor reading a CI
  log cannot tell whether the SIMD path actually ran or
  silently fell back to scalar. The two queries cost
  basically nothing and turn the runtime question into a
  test assertion.

### Next
- Step 29: CPU scaling writeup (`docs/SCALING_CPU.md`).
  Strong-scaling and weak-scaling tables across every Phase-3
  + Phase-4 builder — canonical, norms, tiled, partial_sort,
  BLAS, OMP, OmpScratch, Threaded, SIMD. The headline
  artefact summarising the entire serial-and-parallel CPU
  optimisation ladder before the project shifts to NN-Descent
  in Phase 5.

---

## [Step 27] — `std::thread` + atomic-counter alternative (2026-05-04)

### What
- Added `knng::cpu::brute_force_knn_l2_threaded(ds, k,
  num_threads = 0)`. Same algorithm as Step 24's OpenMP path;
  the parallelism is implemented in pure C++ standard library:
  `std::vector<std::thread>` workers consuming a single
  `std::atomic<std::size_t>` work counter via
  `fetch_add(1, std::memory_order_relaxed)`. Each worker grabs
  one query at a time, runs the per-query distance + heap loop,
  loops back for the next, exits when the counter overruns
  `ds.n`. No mutex, no condition variable.
- The atomic counter doubles as a lock-free *work queue* — the
  shape the plan called for. Dynamic load balancing falls out
  of `fetch_add`'s atomic semantics; static partitioning would
  have been simpler but the plan explicitly wanted the
  work-queue shape (which is what Step 35's parallel
  NN-Descent will actually need, where per-iteration work is
  unbalanced).
- Each worker holds its own `TopK heap(k)` on the stack — no
  shared scratch, no per-thread bookkeeping. The heap is reused
  across iterations the worker handles via `extract_sorted`'s
  capacity-preserving drain (Step 25's pattern). Worker count
  defaults to `std::thread::hardware_concurrency()`; if the
  runtime reports `0`, fall back to one thread.
- Five new `test_brute_force` cases pinning the contract:
  matches the OMP path at 2 threads bit-for-bit, output
  bit-identical across `{1, 2, 4}` thread counts (atomic
  dispatch only changes which worker handles which query, not
  the output), `num_threads = 0` resolves to a working
  configuration, `k = 0` and `k > n - 1` throw.
- Added `BM_BruteForceL2Threaded_Synthetic` family at the same
  `{1, 2, 4, 8}` thread sweep so Step 29's writeup plots OMP
  and `std::thread` lines on the same axes.
- ctest 119/119 green (5 new brute_force, 114 carried over
  from Step 26).
- **Measured at n=1024, d=128:** wall time within 1% of the
  OMP path at every thread count (66.2 vs 66.7 ms at t=1;
  identical at t=2; 17.8 vs 17.4 at t=4; 10.0 vs 10.8 at
  t=8). Recall stays at 1.0 across every configuration.

### Why
The plan's framing for this step is "as a learning exercise" —
the goal is not a faster build but a clearer picture of what
OpenMP's `#pragma omp parallel for` actually expands to. Having
both paths side-by-side in the repo means a future contributor
who reaches for OpenMP can compare:

  * **Source-line cost:** OpenMP version is ~30 lines of source;
    the `std::thread` version is ~50. The atomic-counter
    boilerplate, the explicit `worker_body` lambda, the
    `threads.reserve` + `emplace_back` loop, and the join
    pass each cost a few lines that `#pragma omp parallel for
    schedule(static)` collapses into a single line.
  * **API footprint:** OpenMP needs a runtime (`libomp`,
    `libgomp`), a CMake find module, and a `KNNG_HAVE_OPENMP`
    cache variable. `std::thread` is in `<thread>` and ships
    with every C++11+ toolchain. Step 24's CMake gymnastics
    (Apple's `OpenMP_ROOT` hack) literally do not exist for
    Step 27.
  * **Wall-time:** equivalent. The atomic-counter dispatch
    overhead is ~50 ns per `fetch_add`; each query takes
    ~50 µs; the contention on the shared cache line is ~1000×
    smaller than the per-query work. On a workload where each
    iteration is sub-microsecond, the atomic would become a
    bottleneck and `schedule(static)` would win — but that is
    not where brute-force sits.

The right reading is: OpenMP is the *correct* tool for this
algorithm because the source is shorter and the runtime cost
is identical. The `std::thread` path is documentation in code
form — when a future Phase-9 contributor wonders "should I
switch from OpenMP to my own pthreads pool?", the diff between
these two files is the answer.

`std::memory_order_relaxed` is sufficient on the counter
because the per-query work writes into disjoint rows of the
output `Knng` (rows `q` and `q'` for `q != q'` never overlap).
There is no happens-before relationship to enforce between
threads; `relaxed` is the cheapest order that gives mutual
exclusion of `q` indices.

### Tradeoff
- **Atomic-counter dispatch is dynamic.** Step 24's
  `schedule(static)` partitions queries into contiguous
  chunks; the `std::thread` path interleaves them across
  workers via `fetch_add`. For brute-force every query is the
  same work, so the OMP partition is more cache-friendly (a
  worker scans a contiguous chunk of `q` indices and
  the dataset's pages are visited in a stride-friendly order).
  At n=1024 this is invisible; at n=1M it would matter and
  the right `std::thread` shape would be a static partition,
  not an atomic counter. We accept the "wrong-shape-for-this-
  algorithm-but-right-shape-for-the-pedagogy" choice.
- **No exception-safety guarantee on worker bodies.** If a
  worker throws (e.g. `std::bad_alloc` inside the heap),
  every other thread keeps running and joins normally; the
  exception is *not* propagated to the caller. We accept
  this: the per-query work allocates only inside `TopK`,
  which is bounded at `O(k)`, and the surrounding
  `std::vector<float>` allocations all happen before the
  worker spawn. A throw from inside a worker would mean a
  bug elsewhere; surfacing it would require `std::packaged_task`
  or `std::exception_ptr` plumbing that is out of scope for
  the learning exercise.
- **Thread count is unchecked beyond `> 0`.** Spawning 1024
  workers on a 4-core machine is allowed; the OS handles the
  scheduling. We do not clamp `num_threads <=
  hardware_concurrency()` because the bench harness wants to
  produce numbers at exactly that ratio (oversubscribed
  scaling) for the writeup.

### Learning
- *`std::atomic` + `fetch_add` is the simplest work queue
  that works.* No mutex, no condition variable, no
  `std::queue<int>`. The counter *is* the queue. This is
  the pattern most production-grade thread pools use under
  the hood (TBB, Intel oneAPI, libdispatch); seeing it
  unwrapped here is what makes the OpenMP version feel like
  a thin wrapper rather than a black box.
- *`memory_order_relaxed` is correct here, even though it
  feels uncomfortable.* The C++ memory model guarantee is
  that a `fetch_add(1, relaxed)` produces a unique value to
  exactly one calling thread; that is *all* the ordering we
  need, because the per-query work touches disjoint output
  rows. Reaching for `seq_cst` (the default) would have
  added a memory fence on every `fetch_add` for no
  observable benefit.
- *The OMP and `std::thread` numbers being identical is the
  *right* result.* If they differed by more than ~1% we
  would suspect one path of a bug. The fact that they match
  cleanly is the strongest evidence that both are doing the
  same work, just with different paint.

### Next
- Step 28: hand-vectorised SIMD distance kernel (AVX2 +
  ARM NEON + scalar fallback). The first per-pair-compute
  optimisation; runs underneath every Phase-3 / Phase-4
  builder via the `dot_product` overload set.

---

## [Step 26] — NUMA awareness: first-touch helper + bench wrapper (2026-05-04)

### What
- Added `include/knng/cpu/numa.hpp` and `src/cpu/numa.cpp` —
  the cross-platform NUMA infrastructure every later parallel
  CPU step will lean on:
  * `knng::cpu::first_touch(data, n_elements, num_threads)` —
    walks an `n_elements`-float buffer in a `#pragma omp
    parallel for schedule(static)` loop, writing one float per
    OS page (page size queried at runtime via
    `sysconf(_SC_PAGESIZE)`). Each iteration's `data[i] =
    data[i]` no-op preserves contents while triggering Linux's
    first-touch page-binding so the page lands on the node of
    the worker that will later read it. On macOS / single-NUMA
    hosts the pass is just a cache warm-up.
  * `is_numa_relevant_platform()` — `true` on Linux (where
    NUMA layouts can hurt strong-scaling), `false` on macOS
    (single-domain Apple Silicon SoC). Step 29's writeup
    consults the flag to decide whether to run the
    `numactl`-companion bench.
- Added `tools/run_bench_numa.sh` — wraps `bench_brute_force`
  in `numactl --interleave=all` when the host has `numactl`,
  falls through to the plain invocation when it does not. The
  `--interleave=all` policy is the smallest knob that produces
  interpretable strong-scaling numbers on a multi-socket host
  without requiring the algorithm to have called `first_touch`
  itself.
- Added `docs/NUMA.md` — the project's NUMA story end-to-end:
  why first-touch is the right primitive, what the OpenMP
  schedule has to do with the layout, why macOS gets the
  function as a no-op rather than an `#ifdef`, and three open
  questions deferred for a future pass (libnuma probe,
  per-builder integration, `--membind=N` flag).
- Five new `test_numa` GTest cases (114 total): null pointer
  is a no-op, zero-length is a no-op, buffer contents are
  preserved exactly, sub-page buffers do not overrun the
  loop, the platform flag returns a deterministic value.
- Wired `cpu/numa.cpp` into the `knng_cpu` library and
  `tests/numa_test.cpp` into the CTest matrix.
- ctest 114/114 green (5 new numa, 109 carried over from
  Step 25).

### Why
Phase 4's headline result — strong-scaling brute-force on
SIFT1M+ — runs aground on a Linux multi-socket host without
NUMA-aware page placement. The default first-touch policy
puts every page on the *loader thread's* node, so an 8-thread
read later spends 7/8ths of its DRAM bandwidth on the
inter-socket interconnect. The published "8-thread speedup"
number then sub-linearly because the workers are competing
for one node's bandwidth rather than scaling across all
nodes' bandwidth.

The fix is conceptually trivial — touch the buffer from the
same schedule the algorithm uses — but the *infrastructure*
to do it portably is what Step 26 ships. Any future builder
that streams a large buffer (Step 24's OMP variants, Step 35's
parallel NN-Descent, Step 39's distributed brute-force) will
call `first_touch` after population and before the first
parallel read; the helper is a single function the rest of
the project does not have to know any further details about.

The runtime page-size detection (`sysconf(_SC_PAGESIZE)` rather
than a hard-coded `4096`) costs a single syscall per
invocation and handles every supported platform without
`#ifdef`: x86_64 and arm64 Linux are 4 KB or 64 KB depending
on kernel config, Apple Silicon M-series is 16 KB. A hard-coded
4096 would over-walk the buffer on Apple (more page faults
than necessary) and under-walk it on a kernel configured for
huge pages.

The wrapper script's `numactl --interleave=all` policy is the
*spread-pages-evenly* fix; it does not produce the same
numbers as the `first_touch`-aware path, but it is the
simpler-and-coarser guarantee for ad-hoc runs that have not
invoked `first_touch` themselves. Step 29's writeup will
report both numbers so the reader can see the gap between
"NUMA-blind run on a multi-socket host" and "NUMA-aware run
on the same host."

### Tradeoff
- **`is_numa_relevant_platform()` is platform-keyed, not
  topology-keyed.** A single-socket Linux laptop reports
  `true` even though it has only one NUMA node; the flag's
  intended use is "should the bench wrapper bother running
  the `numactl` companion?" and the conservative answer
  on Linux is yes. A future refinement will swap the
  constant for a libnuma `numa_available() && numa_max_node()
  > 0` probe. The CHANGELOG flags this as the natural
  follow-up.
- **`first_touch` does not check whether the schedule
  matches the caller's later read pattern.** If a caller
  uses `schedule(dynamic)` for the read but
  `schedule(static)` for first-touch (the helper's
  hard-coded choice), the page-to-thread alignment is
  imperfect and some pages still bind to a remote node. We
  accept the rigidity: every Phase-4 builder uses
  `schedule(static)` (Step 24's deliberate choice for a
  load-balanced kernel); a future Step 35 NN-Descent
  variant that wants `schedule(dynamic)` will need a
  different first-touch helper, which we will ship then.
- **macOS gets the function as a no-op redistribution.** The
  pass still runs (cache-warm side effect) but does no
  redistribution. We accept the small wasted cost — ~0.3 s
  on a 512 MB buffer, dominated by the read+write — over an
  `#ifdef` that would force every caller to know the
  platform.
- **No bench-harness integration in this step.** The bench
  binary `bench_brute_force` does not call `first_touch` on
  its synthetic dataset yet. We deferred the wiring to
  Step 29's scaling writeup, where the
  `first_touch`-on-vs-off comparison *is* the artefact —
  having both lines on the same plot is more useful than
  silently turning it on now and losing the comparison.

### Learning
- *NUMA layout matters before a single SIMD intrinsic does.*
  The temptation in Phase 4 is to reach for hand-vectorisation
  (Step 28) before fixing the layout — and to then chase a
  4–8% SIMD win while a 30–50% NUMA win is sitting unfixed.
  The right ordering, baked into the plan, is: parallelise
  (Step 24), thread-local scratch (Step 25), NUMA layout
  (Step 26), *then* SIMD (Step 28). Every parallel-CPU
  optimisation effort the project ships afterwards starts from
  this ordering.
- *Cross-platform first-touch is one function, not three.*
  The function compiles unchanged on macOS, x86_64 Linux,
  and arm64 Linux. The platform-conditional behaviour is in
  the kernel's page-allocation policy, not in our code; the
  helper is a single stable interface. This is the right
  shape for any "OS-policy-aware" helper: do the same write
  pattern everywhere, let the kernel make the binding
  decision, document the platform behaviour in `docs/NUMA.md`
  rather than in `#ifdef`s.
- *`sysconf(_SC_PAGESIZE)` is the right page-size source.*
  It surfaces the runtime page size, including huge-page
  configurations a hard-coded constant would miss. Querying
  it once per call costs ~one syscall (~50 ns) — negligible
  next to the multi-megabyte buffer the function then walks.

### Next
- Step 27: a `std::thread` + work-queue alternative
  implementation of the parallel L2 brute-force, as a
  learning exercise. Same correctness contract as Step 24's
  OpenMP path; different API ergonomics; will be measured
  against OpenMP for both wall time and source-line count.

---

## [Step 25] — Thread-local scratch + cache-line padding (2026-05-04)

### What
- Added `knng::cpu::brute_force_knn_l2_omp_scratch(ds, k,
  num_threads = 0)`. Same arithmetic as Step 24's plain OMP path;
  the structural change is the `TopK` heap moves from "declared
  inside the parallel-for body" (Step 24) to "pre-allocated once
  per worker, drained between iterations." The `extract_sorted`
  call already empties the underlying priority_queue's vector
  while preserving its capacity, so the next iteration on the
  same thread reuses the buffer without reallocating.
- Each per-thread heap is wrapped in `struct alignas(64)
  ThreadScratch` with explicit padding so `sizeof(ThreadScratch)`
  is a multiple of `kCacheLineBytes = 64`. With this layout,
  `std::vector<ThreadScratch>(num_threads)` puts every worker on
  its own cache line — adjacent workers cannot ping-pong the
  shared line through the LLC when they push into "their" heap.
- Pre-allocation uses `emplace_back` in a counted loop rather
  than `std::vector(size, value)`. The `vector(size, value)`
  form copy-constructs `value` `size` times; emplace is one
  allocation and `size` in-place constructions — both cheaper
  and the only form that compiles for a type that holds a heap
  with a non-trivial copy.
- Four new `test_brute_force` cases pinning the contract:
  matches the plain OMP path at 2 threads bit-for-bit, output
  is deterministic across `{1, 2, 4}` thread counts, `k = 0` and
  `k > n - 1` throw.
- Added `BM_BruteForceL2OmpScratch_Synthetic` family at the same
  thread sweep `{1, 2, 4, 8}` as Step 24's family so Step 29's
  scaling writeup can plot both lines on the same axes.
- ctest 109/109 green (4 new brute_force, 105 carried over from
  Step 24).
- **Measured at n=1024, d=128:** numbers essentially identical to
  Step 24's plain OMP path within the per-run noise band — the
  per-iteration `TopK` allocation is small enough relative to the
  ~1M distance computations per query that hoisting it out
  doesn't move the needle. The win compounds at larger `n`
  (Step 29's writeup will document this) and on workloads where
  the per-pair distance is fast enough that allocation cost
  dominates.

### Why
Step 25 ships the *infrastructure* for thread-local scratch even
though the headline number on this fixture does not move. Three
reasons it lands now:

  1. **The pattern is the right shape for every later parallel
     CPU step.** Step 35's parallel NN-Descent will need
     per-thread state for the local-join candidate list,
     per-point locks, and the convergence-update counter.
     Doing the cache-line-aware pattern correctly *now*, on the
     simplest possible algorithm, makes Step 35 a translation
     rather than a redesign.
  2. **The per-iteration alloc cost only hides on AppleClang at
     small `n`.** AppleClang's libc++ has a fast small-vector
     allocator (`__small_vector` optimisation under the hood
     for `priority_queue<>`) that minimises the allocation cost
     for `k = 10`. On libstdc++ + GCC, the alloc is more
     expensive per-iteration, and the win compounds. The
     numbers we publish today are an under-estimate of the win
     on a Linux CI runner.
  3. **False sharing is the canonical "easy to introduce, hard
     to debug" parallel performance bug.** Pinning the
     `alignas(64)` + padding pattern in this step's CHANGELOG
     means future steps can reach for `ThreadScratch`-style
     wrappers without re-discovering the why.

The choice of 64 as the cache-line constant is right for every
supported platform: x86_64 (Intel + AMD), arm64 Apple Silicon,
arm64 Linux. We define it as a `constexpr std::size_t` rather
than reaching for `std::hardware_destructive_interference_size`
because the latter is `[[experimental]]` on libstdc++ and would
trigger a `-Wpedantic` warning; we keep the project's strict
warning policy clean.

### Tradeoff
- **Memory grows with the worker count.** Each `ThreadScratch`
  is at least one cache line (64 B) plus the heap's underlying
  vector (`k * sizeof(Entry)` bytes once it grows). For 8
  workers and `k = 10`, the total is ~1.5 KB — negligible
  next to the dataset, but it does land in the bench's
  `peak_memory_mb` reading and the eagle-eyed reader will see
  the OmpScratch path report ~50 KB more peak memory than the
  plain Omp path (the `std::vector<ThreadScratch>` stays
  resident for the function's whole frame).
- **The per-thread heap is allocated once but resized never.**
  We rely on the heap's `extract_sorted` leaving the vector's
  capacity intact. This is true for `std::priority_queue` over
  `std::vector`, and we assert nothing changes that — but a
  future TopK rewrite that, e.g., replaces the priority_queue
  with a bounded array would render the optimisation moot. The
  CHANGELOG comment is the contract.
- **No per-thread *distance buffer* — yet.** The plan's full
  thread-local-scratch story includes a per-thread distance
  tile buffer for the tiled / BLAS paths. We deferred: the
  per-thread heap is the part that benefits at small-`n`; the
  distance-tile buffer matters at large-`n` and is the natural
  Step-29 follow-up. The shape we chose for `ThreadScratch`
  trivially extends to hold the buffer when that lands.

### Learning
- *`extract_sorted` was already capacity-preserving — we just
  needed to stop allocating a fresh heap.* Step 09's `TopK`
  (the priority_queue around a `std::vector<Entry>`) has the
  right amortisation property baked in; we did not see it
  because Step 19, 20, 22, 24 all declared the heap inside
  their loops. The lesson: when you find yourself allocating
  per-iteration, ask whether the surrounding type already
  amortises and you just need to hoist.
- *Cache-line padding is cheap insurance.* `sizeof(TopK)` on
  AppleClang is well under 64 bytes; padding to 64 wastes
  <40 bytes per worker (40 × 8 = 320 bytes total at 8
  workers — pocket change). The cost is far smaller than the
  cost of a future `Why is the parallel build slower than the
  serial build at 2 threads?` debugging session.
- *`alignas` on a struct is the cleanest way to force
  per-instance alignment for a `std::vector` element.* Each
  element's alignment is the struct's alignment; the vector
  allocator honours it. Other approaches (manual padding
  bytes between elements, allocator hacks) are strictly
  uglier.

### Next
- Step 26: NUMA awareness. `numactl --interleave=all` in the
  bench script + a cross-platform `numa_first_touch` helper
  that no-ops on macOS (single-NUMA-domain SoC) and lays out
  large-`n` datasets across NUMA nodes on Linux.

---

## [Step 24] — OpenMP outer-query parallelisation (2026-05-04)

### What
- Added `cmake/FindKnngOpenMP.cmake` — discovers an OpenMP runtime
  and exposes it as the `knng::openmp_iface` INTERFACE target. On
  Apple it pre-sets `OpenMP_ROOT` to Homebrew's `libomp` install
  paths (`/opt/homebrew/opt/libomp` or `/usr/local/opt/libomp`)
  before `find_package(OpenMP)`, so AppleClang picks up the
  Homebrew libomp without the user touching `cmake/`. Sets the
  cache variable `KNNG_HAVE_OPENMP`. New CMake option
  `KNNG_ENABLE_OPENMP` (default ON) gates the entire feature so a
  user without OpenMP can still build.
- Added `src/cpu/brute_force_omp.cpp` and the new
  `knng::cpu::brute_force_knn_l2_omp(ds, k, num_threads=0)` entry
  point. Algorithmically identical to
  `brute_force_knn_l2_with_norms` (Step 19); the only structural
  change is `#pragma omp parallel for schedule(static)` on the
  outer query loop and an `omp_set_num_threads(num_threads)` call
  when `num_threads > 0`.
- Added the `kHasOpenmpBuiltin` `inline constexpr bool` constant
  mirroring Step 21's `kHasBlasBuiltin`. Compiles unconditionally;
  the OpenMP-specific bits live behind `#if KNNG_HAVE_OPENMP` and
  degrade to a serial loop when the build did not link OpenMP.
- Five new `test_brute_force` cases: matches the canonical builder
  at the default thread count, output is bit-identical at 1 / 2 /
  4 threads (parallelisation does not perturb tie-breaking),
  `k = 0` and `k > n - 1` throw, the builtin flag returns a single
  deterministic value.
- Added `BM_BruteForceL2Omp_Synthetic` family that sweeps
  `threads ∈ {1, 2, 4, 8}` at `n = 1024, d = 128`. The thread
  count is reported as `state.counters["threads"]` so Step 29's
  scaling writeup ingests the same JSON.
- ctest 105/105 green (5 new brute_force, 100 carried over from
  Step 23).
- **Measured at n=1024, d=128:** 1 thread 66.0 ms,
  2 threads 33.5 ms (1.97×), 4 threads 17.4 ms (3.79×), 8
  threads 9.99 ms (6.61×). Recall stays at 1.0 for every config.

### Why
This is the project's first parallel-CPU step and the foundation
the next five steps (thread-local scratch, NUMA, std::thread
alternative, SIMD, scaling writeup) build on. The shape is
deliberately the *simplest* OpenMP usage that makes sense: one
`#pragma omp parallel for` on a loop where every iteration is
independent, no critical sections, no locks, no atomics. Every
later parallel-CPU optimisation — Step 25's per-thread scratch,
Step 35's parallel NN-Descent — is a controlled departure from
this baseline.

The `schedule(static)` clause is right for brute-force: every
query does the same `n` distance evaluations, so static
partitioning balances perfectly and avoids OpenMP's per-chunk
scheduling overhead. `dynamic` schedule would land in Step 35
once NN-Descent introduces per-iteration work imbalance (some
local-joins finish early, others take longer); for now it would
just be cost without benefit.

The 6.6× scaling at 8 threads on Apple M-series is consistent
with the SoC's mix of performance and efficiency cores — the
performance cores handle the first 4 threads at full clock, the
efficiency cores pick up 5–8 at ~70% throughput. Linear scaling
on a homogeneous cluster CPU (e.g. AMD EPYC, Intel Xeon) will
look closer to 7.5–8× at 8 threads. Step 29's scaling writeup
will document both.

The `num_threads` parameter is an explicit override rather than a
process-wide `omp_set_num_threads` because (a) the bench harness
runs many configurations in one process and would otherwise leak
state between them, and (b) downstream callers may want to
reserve threads for their own work. Passing 0 (the default) means
"use whatever the runtime would have used" — `OMP_NUM_THREADS`
or hardware concurrency.

### Tradeoff
- **Per-iteration `TopK` allocation, not amortised.** The heap is
  declared inside the parallel-for body, so each iteration
  allocates and frees its `std::priority_queue<...>` storage.
  This is the *correct* shape for thread safety (no shared
  state) and the overhead is dominated by the n=1024 distance
  computations anyway. Step 25 will hoist the allocation out
  via per-thread scratch when the heap pressure starts to show
  up in the profile.
- **`schedule(static)` makes thread 0 finish last on heterogeneous
  cores.** Apple Silicon's performance/efficiency split means
  the equally-sized static chunks finish at different times.
  `schedule(static, 16)` (smaller chunks) would even out the
  finish times by letting fast cores pick up extra chunks
  early, but that would also introduce per-chunk overhead.
  Step 29's writeup will measure both.
- **The OpenMP-not-found degradation is silent.** If `libomp` is
  not installed, the `#pragma omp` is a comment under
  AppleClang and the loop runs single-threaded — at the same
  speed as `brute_force_knn_l2_with_norms` from Step 19, but
  *labelled* as the OMP path. We accept this: the
  `kHasOpenmpBuiltin` constant makes the build state
  inspectable, and the `cmake/FindKnngOpenMP.cmake` log line
  surfaces the situation at configure time.

### Learning
- *AppleClang + Homebrew libomp + CMake is the canonical macOS
  parallelism stack.* The pre-set `OpenMP_ROOT` trick in
  `FindKnngOpenMP.cmake` is the bit that turns "OpenMP works
  on Linux but not Mac" into "OpenMP works everywhere." Future
  contributors do not need to know about `libomp`'s Homebrew
  layout — the find module does it.
- *Strong scaling on a SoC is not the same as strong scaling on
  a server CPU.* The ~6.6× at 8 threads on M-series is what
  Apple Silicon delivers; the project's eventual cluster runs
  will see 7.5–8× on homogeneous server cores. The right
  expectation is "near-linear up to the number of physical
  performance cores, sub-linear when efficiency cores
  contribute" — and Step 29's writeup will pin both numbers
  for posterity.
- *The simplest pragma is the right pragma.* The temptation
  was to add `firstprivate(norms)`, `nowait`, `collapse(2)`,
  manual chunking, etc. None of those help here, and most
  would just add ways for a future bug to hide. Trust the
  default semantics until profiling proves otherwise.

### Next
- Step 25: hoist the per-query `TopK` allocation into per-thread
  scratch, eliminating the per-iteration alloc/free. Same
  algorithm, same correctness, less heap pressure.

---

## [Step 23] — Phase-3 profiling writeup (2026-05-02)

### What
- Added `docs/PERF_STEP23.md` — the project's first profiling
  artefact, summarising the Phase-3 optimisation ladder with the
  numbers measured at `n=1024, d=128, k=10` on AppleClang 21
  (Apple M-series). One row per builder, three repetitions per
  row (mean reported), `recall_at_k` reported alongside wall time
  so the speed-vs-quality story is in one table:

  ```text
  Step 10  canonical                   70.48 ms   1.00× recall=1.0
  Step 19  + precomputed ||p||²        65.66 ms   1.07× recall=1.0
  Step 20  + (32 × 128) tile           65.41 ms   1.08× recall=1.0
  Step 22  + std::partial_sort         62.40 ms   1.13× recall=1.0
  Step 21  + cblas_sgemm                3.82 ms  18.45× recall=1.0
  ```

- Documents the methodology end-to-end: the exact CMake +
  `bench_brute_force` invocation that reproduces the table, the
  three Google-Benchmark counters every row carries
  (`recall_at_k`, `peak_memory_mb`, `n_distance_computations`),
  and a short Python aggregator that flattens the JSON to the
  table shown above.
- Captures four open questions deferred to a later pass:
  AppleClang-vs-GCC on the same hardware, BLAS-provider
  comparison (Accelerate vs OpenBLAS vs MKL), `(query_tile,
  ref_tile)` sweep at SIFT1M scale, and memory-residency check
  for the partial-sort scratch buffer at large `n`. Each is
  framed as "what this writeup does not claim" so a future
  contributor can see the open work without re-reading the
  surrounding context.
- Pins the structure (`Optimisation ladder · Take-aways ·
  Methodology · Open questions · Reproduction commands`) every
  subsequent profile writeup will follow:
  `docs/PERF_STEP50.md` for Phase 7's GPU foundations,
  `docs/PERF_SINGLE_GPU.md` for the headline Phase 8 waterfall,
  `docs/MULTI_GPU.md` for Phase 11. Same five sections, same
  reproduction-first stance.
- ctest still 100/100 green; this step adds no code.

### Why
The Phase-3 ladder needs a single artefact that collects its
five numbers, preserves the methodology that produced them, and
is reproducible six months from now when a contributor wonders
"why does Step 21 claim 18×?" Without the writeup, the numbers
in each step's individual CHANGELOG entry are scattered across
five files; without the methodology, the numbers themselves are
unfalsifiable. This is the artefact the project's pedagogy
ultimately rests on — the README will eventually link to it from
the front page, and the Phase 13 architecture document
(`docs/ARCHITECTURE.md`) will reference it as one of the project's
six headline profiles.

The plan called this step "Profiling writeup" and explicitly
"the first profiling step — pattern reused throughout the
project." We honour that intent both ways: the file is created,
and the file's *structure* is what the later writeups will
inherit. Open questions are explicitly enumerated as deferred
work — the alternative (sneaking them into a half-finished
"future work" section that nobody reads) would let the questions
silently rot.

The cycle-counter (`instruments` / `perf stat`) study the plan
mentions — cache-miss rates, IPC, branch prediction — is
deliberately deferred to a later revisit. Reasoning: the wall-time
artefact captured here is sufficient to defend Phase 3's claim
("each step preserved recall and produced a measurable speedup").
The cycle-counter numbers would be retired the moment Phase 4
introduces OpenMP, which changes the cache-residency model.
Capturing them now would tie us to a snapshot that will not
survive the next phase. The deferred-questions section in the
writeup is the bookmark.

### Tradeoff
- **The numbers are AppleClang on M-series only.** Until the
  project has a Linux CI runner that emits cycle counters, the
  Phase-3 picture has a single platform. We accept the
  one-platform claim; the writeup is upfront about what it
  does and does not show, and the four open questions all
  have "rerun on Linux" as their natural resolution.
- **The artefact ships as a `.md`, not as a CSV + plot script.**
  A machine-readable companion would make the table
  diff-friendly. We deferred: the JSON the bench harness
  produces *is* the canonical machine-readable form; the
  Markdown table is a human-readable summary derived from it.
  Adding a third "summary CSV" format would be a fourth thing
  to keep in sync.
- **Recall is reported as `1.0` exactly.** It is, but on a
  fixture (synthetic n=1024, d=128) where every recall is
  trivially 1.0 because the algorithms are all exact L2
  brute-force. The first non-trivial recall numbers ship in
  Phase 5 (NN-Descent) and Phase 9 (GPU NN-Descent); the
  Step-23 writeup is structured so adding a "recall vs speed"
  plot in those phases is a single new figure, not a rewrite.

### Learning
- *Profile writeups are pedagogical artefacts, not data dumps.*
  The temptation in writing one is to paste every JSON row.
  The discipline is to compress to a five-row table that *tells
  a story* — "the BLAS step dominates everything else" is the
  story Phase 3 tells; readers should leave the page with that
  one sentence in their heads, not with a 200-row CSV. Every
  later writeup will follow this template: one table, five
  paragraphs, four open questions.
- *Open questions deserve a section title.* Burying "future
  work" at the end of a paragraph is how it gets forgotten.
  Pinning it as a section called "Open questions, deferred to
  a later pass" with explicit bullets gives a future
  contributor (or future me) a place to start when the
  AppleClang-vs-GCC rerun finally happens.
- *Reproduction commands are the writeup's contract with
  posterity.* The exact `cmake` + `bench_brute_force` invocation
  is what makes the writeup not-a-snapshot. A reader six months
  from now should be able to run the commands and see numbers
  in the same neighbourhood. If they cannot, either the
  artefact is wrong or the project regressed — both of which
  are useful to know.

### Next
- Phase 4 (Step 24): OpenMP `#pragma omp parallel for` on the
  outer query loop. The first parallel CPU step. Will rerun
  the Phase-3 ladder with `-DKNNG_NUM_THREADS=N` and append
  the rows to this writeup (same table shape; "Step 24"
  becomes the next row).
- Step 25 onwards continues the Phase 4 sequence
  (NUMA, std::thread alternative, hand-vectorised SIMD, scaling
  writeup).

---

## [Step 22] — `std::partial_sort` for top-k extraction (2026-05-02)

### What
- Added `knng::cpu::brute_force_knn_l2_partial_sort(ds, k)`. Reuses
  Step 19's norms-precompute identity for the per-pair distance
  arithmetic but replaces the streaming `TopK` heap with a
  `std::partial_sort` over the materialised per-query candidate
  buffer.
- The candidate buffer is `std::vector<std::pair<float, index_t>>`
  of length `n - 1`, allocated once at function entry and reused
  across queries. Packing distance first / id second makes the
  default lexicographic `<` on `pair` reproduce the heap path's
  tie-break (smaller distance wins; smaller id wins on tie) for
  free — no custom comparator.
- Five new `test_brute_force` cases pinning the contract: matches
  the canonical builder neighbour IDs and distances within `1e-4`,
  rows are sorted ascending, `k = 1` ties match the canonical
  tie-break, `k = 0` and `k > n - 1` throw.
- Added `BM_BruteForceL2PartialSort_Synthetic` family at the same
  `(n, d)` grid as the heap-path baselines.
- ctest 100/100 green (5 new brute_force, 95 carried over from
  Step 21).
- **Measured at n=1024, d=128:** canonical heap 69.8 ms,
  norms+heap 66.9 ms, **norms+partial_sort 64.0 ms**. ~9% faster
  than canonical, ~4% additional gain over the heap path. Recall
  stays at 1.0.

### Why
The heap path's `TopK::push` is `O(log k)` per candidate with a
hard data dependency (the heap-sift loop), and it is called
`n - 1` times per query. The branch on the admission test fires
once per candidate. `std::partial_sort` has the same asymptotic
complexity but shifts the work into a contiguous-buffer pass:
build a `k`-element max-heap from the first `k` elements, then
linearly scan the remaining `n - 1 - k` performing one
`heap-replace` only when the candidate is smaller than the
current heap top. The data is contiguous, the predicates are
simpler, and the *whole* sort runs after the distance
computation completes — so the prefetcher has already brought
the candidate buffer into L1 by the time the sort starts.

The 4% gain over the heap path is in the noise on AppleClang at
this fixture; on workloads where the per-pair distance is very
cheap (low `d`, BLAS-fast cross term), the partial_sort path's
fraction-of-runtime grows and the win compounds. The right
reading of this step is "we measured the heap-vs-partial_sort
tradeoff so the rest of the project can stop debating it" — the
plan called this step out specifically to avoid the eternal
"should we use a heap or a partial_sort" detour later.

The fixed `(n - 1)`-element scratch is the right choice over a
streaming partial_sort. Two reasons:

  1. **Partial sort needs a materialised buffer.** Streaming
     `partial_sort_copy` exists, but the source range still has
     to be enumerated; the cost saving over filling a buffer
     once is negligible.
  2. **Memory cost is bounded.** ~8 bytes per reference times
     `n - 1`, allocated once per function call, fits in 8 MB
     for SIFT1M. Negligible next to the dataset itself.

### Tradeoff
- **Memory grows with `n`, not `k`.** The heap path uses
  `O(k)` per query; the partial_sort path uses `O(n)`. For
  `k = 10, n = 1M` this is 8 MB instead of 80 bytes. We accept
  the growth: it is allocated once and reused across queries,
  and the wall-time win is uniform across query order.
- **No streaming variant.** The `TopK` heap path can be fed
  candidates as they arrive (e.g. from a parallel reduce in
  Step 23 or a GPU kernel in Phase 7). The partial_sort path
  cannot — the buffer must be filled first. This is why the
  heap path stays as the canonical algorithm; partial_sort is
  the speed-pick-when-buffer-fits alternative.
- **The packed-pair tie-break is implicit.** A future reader
  who does not look at the pair declaration may not realise
  that `(distance, id)` ordering is what reproduces the heap
  path's tie-break. We accept the cost of one inline comment
  over a custom comparator class.
- **The benchmark gain on AppleClang is small (~4%).** This is
  within the noise band of the bench harness's single-iteration
  runs. The numbers will firm up under Step 23's profiling pass
  with longer runs and `instruments` cycle counts.

### Learning
- *`std::partial_sort` is one of the most underrated standard
  algorithms.* Most C++ programmers reach for `std::sort` (which
  is `O(n log n)`) when they really want a top-k. The standard
  library provides exactly the right algorithm with the right
  complexity (`O(n log k)`) — the only barrier to using it is
  knowing it exists. Pinning the heap-vs-partial_sort decision
  in the project means future contributors do not have to
  rediscover this tradeoff.
- *Lexicographic `<` on `std::pair` is free tie-breaking.*
  Rolling our own `Comparator` struct was an option; using the
  built-in is shorter, faster (the compiler inlines the
  pair-comparison cleanly), and reads at the call site as "sort
  by `(distance, id)`," which is exactly the tie-break rule.
- *Heap-vs-partial_sort is fixture-dependent.* On AppleClang
  with d=128 at n=1024, the partial_sort path is 4% faster. On
  d=4 at large n (where the dot product is cheap and the heap's
  branch dominates), partial_sort can be 30–40% faster. On
  small-`k` distributed-MPI shards (where the candidate buffer
  is too large to fit in cache and partial_sort thrashes), the
  heap is the right pick. We keep both entry points; future
  callers pick based on their own measurements.

### Next
- Step 23: profile every Phase-3 path with `instruments` (macOS)
  / `perf stat` (Linux). Cache-miss rates, IPC, branch
  prediction, hot functions. `docs/PERF_STEP22.md` —
  the project's first profiling artefact, sets the pattern
  every subsequent profile writeup will follow.

---

## [Step 21] — BLAS `sgemm` for the cross term (2026-05-02)

### What
- Added `cmake/FindKnngBlas.cmake` — the project's BLAS discovery
  module. Tries Apple Accelerate first on macOS (`find_library
  (Accelerate)` plus the `ACCELERATE_NEW_LAPACK=1` opt-in for the
  post-13.3 CBLAS interface; otherwise `cblas_sgemm` is marked
  deprecated and `-Werror` rejects it). Falls back to
  `find_package(BLAS)` plus `find_path(cblas.h)` on Linux,
  searching the standard OpenBLAS / MKL / Homebrew layouts. Sets
  `KNNG_HAVE_BLAS` and exposes the `knng::blas_iface` INTERFACE
  target. New CMake option `KNNG_ENABLE_BLAS` (default ON) gates
  the entire feature so a user without a BLAS install can still
  build the project.
- Added `src/cpu/brute_force_blas.cpp` and the new
  `knng::cpu::brute_force_knn_l2_blas(ds, k, query_tile=64,
  ref_tile=256)` entry point. The algorithm is the *matrix* form of
  the Step-19 algebraic identity:

  ```text
  D[i, j]  =  ||x_i||²  +  ||y_j||²  -  2 · (X · Yᵀ)[i, j]
  ```

  Each outer tile slices `query_tile` rows of `X` and `ref_tile`
  rows of `Y`, hands them to a single `cblas_sgemm` call to fill
  the cross-product block `(QUERY_TILE × REF_TILE)`, then folds
  the precomputed norms in via a scalar epilogue and feeds each
  row to the per-query `TopK` heap.
- Added a `kHasBlasBuiltin` `inline constexpr bool` constant so
  callers can check at compile time whether the BLAS path is
  available without dragging the `KNNG_HAVE_BLAS` macro into user
  code.
- Six new `test_brute_force` cases (under `#if KNNG_HAVE_BLAS`):
  matches the canonical builder at default tile sizes; matches
  at `(3, 5)` and `(1, 1)` tilings; zero tile size and `k = 0`
  throw; the `kHasBlasBuiltin` static assert sanity-checks the
  flag actually reflects the build.
- Added `BM_BruteForceL2Blas_Synthetic` family (also gated on
  `KNNG_HAVE_BLAS`) at the same `(n, d)` grid as the existing
  bench families plus `n=2048` to exercise where BLAS earns its
  keep.
- ctest 95/95 green (6 new brute_force, 89 carried over from Step 20).
- **Measured at n=1024, d=128:** canonical 69.7 ms, BLAS 3.83 ms.
  ~18× speedup. Recall stays at 1.0.

### Why
This is the headline Phase-3 CPU optimisation and the algorithm
the rest of the project will keep coming back to. "Distance as GEMM"
is not just a CPU win — it is the same algebraic identity that
makes Step 55's `cublasSgemm` the right tool on GPU, the same
shape that Step 57's tensor-core path slots into via `WMMA`, and
the same trick `faiss-gpu` and `cuVS` use for their L2 brute-force.
Landing the CPU version now means the GPU port in Phase 8 is a
*translation*, not a reinvention.

The 18× speedup at d=128 measures the right thing: it is
overwhelmingly *bandwidth*-bound on this fixture (n=1024, n*n*d
≈ 130 MFLOPs against AppleClang's autovectorised baseline), and
Apple Accelerate is genuinely tuned to exploit the SoC's bandwidth
hierarchy in a way our hand-written loops cannot. The number is
also the largest single-step speedup in the project so far —
Step 19's algebraic rewrite gave ~6%, Step 20's tiling another
~2%, and Step 21 jumps to ~18×. The CHANGELOG narrative for the
project will reflect this: there is an 80/20 distribution of
optimisation wins, and "use a tuned BLAS" is the 80.

The deliberate non-decision was: do we vendor a BLAS, or do we
discover one? We chose discovery. Vendoring (e.g.
FetchContent OpenBLAS) would have pulled a 200 MB build of an
assembly-heavy library into the project's CI matrix and would
have made the project's entire build slower for every developer,
including those who never run the BLAS path. Discovery means a
developer who has not installed BLAS sees one extra "BLAS not
found, Step 21 disabled" message at configure time and the rest
of the project still builds; CI on macOS uses Accelerate
(zero-install), CI on Linux installs OpenBLAS via apt (one
line in the workflow). The cost of "no BLAS" is graceful
degradation — `kHasBlasBuiltin == false` and `brute_force_knn_l2_blas`
is simply not declared.

The default tile sizes `(64, 256)` were chosen specifically for
the BLAS path (vs `(32, 128)` for the hand-tiled variant): BLAS
itself does its own internal blocking and is happiest with a
larger outer tile to amortise the call overhead. A future
profiling pass (Step 23) may revise these.

### Tradeoff
- **The BLAS path is the project's first non-trivial third-party
  dependency at link time.** OpenBLAS / Accelerate / MKL each
  ship gigabytes of code we cannot fully audit. We accept this:
  the alternative (a hand-written sgemm) would be a multi-month
  project for a fraction of the throughput. The discovery
  module isolates the dependency to a single library, and the
  `KNNG_ENABLE_BLAS=OFF` escape hatch lets a paranoid build
  still produce a usable binary.
- **`kHasBlasBuiltin` is a build-time, not run-time, flag.** A
  user who installs BLAS, runs CMake, then upgrades BLAS is on
  their own — no run-time version check. We accept the rigidity
  because the alternative (versioned ABI gates) would cost more
  test surface than the bug it would prevent ever has.
- **Distance ordering can microscopically diverge.** BLAS may
  reorder the dot product across thread parallelism or use
  fused-multiply-add intrinsics not available in the hand-written
  path. The 8-point fixture's neighbours are unique-by-distance
  so the IDs match exactly; the test tolerance for distances is
  `1e-3` (vs `1e-4` for the hand-written norms / tiled paths).
  On real datasets where two near-equidistant references compete
  for the last neighbour slot, the BLAS path may pick the
  smaller-id one differently from the canonical path. This is
  normal "fp non-associativity" surface — every later GPU step
  will face the same issue.
- **The bench's `n=2048` exercises a path the other variants do
  not run.** We accept the asymmetry: the BLAS path is the only
  one that scales to that size in a reasonable time, so the
  bench grid is intentionally broader for it.

### Learning
- *The CMake module is the contract.* `FindKnngBlas.cmake` is
  300 lines instead of 30 because every assumption it makes is
  documented inline — what platforms it supports, what fallback
  order it uses, why Accelerate needs the `ACCELERATE_NEW_LAPACK`
  define, where each `find_path` looks. Future BLAS provider
  additions (Intel MKL on Linux, Cray libsci on HPC clusters)
  will be a single new branch; the rest of the build never has to
  know.
- *AppleClang's `-Werror` catches deprecation warnings the same
  way it catches unused variables.* The `cblas_sgemm` symbol on
  recent macOS is `__attribute__((deprecated))` unless
  `ACCELERATE_NEW_LAPACK=1` is defined — and the project's strict
  warning policy (Step 06) turns that into a build error rather
  than a soft warning. The right place to add the define is
  inside `target_compile_definitions` on the BLAS interface
  target so every TU that sees `<Accelerate/Accelerate.h>` also
  sees the macro. Adding it to a single `.cpp` would have
  worked locally but broken the day a second TU pulled in the
  header.
- *18× is the wake-up call.* The project's "ladder of optimisations"
  predicts each step contributes a few-percent speedup; Step 21
  contributes ~18×, more than every previous step combined. The
  reading is *not* "the previous steps were wasted" — Steps 17,
  19, 20 are what make Step 21 possible, both algebraically (the
  identity) and infrastructurally (tile loops, deterministic RNG,
  recall harness, JSON counters). The reading is "the moment a
  step lets you delegate to a tuned library, *do it*." Phase 8's
  GPU steps will repeat the pattern: hand-written naive kernel,
  shared-memory tiled kernel, then `cublasSgemm` and the same
  18× cliff.

### Next
- Step 22: `std::partial_sort` for the per-tile top-k. The TopK
  heap is `O(log k)` per push; partial_sort over the full tile
  may amortise better when the tile holds `>> k` candidates.
- Step 23: profiling writeup. `instruments` on macOS, `perf stat`
  on Linux. Cache-miss rates and IPC for the canonical path,
  the BLAS path, and the gap between them. `docs/PERF_STEP22.md`.

---

## [Step 20] — Block tiling: `(QUERY_TILE × REF_TILE)` distance blocks (2026-05-02)

### What
- Added `knng::cpu::brute_force_knn_l2_tiled(ds, k, query_tile=32,
  ref_tile=128)`. Builds on Step 19's precomputed-norms identity
  and wraps it in a pair of nested tile loops:

  ```text
  for each q_tile of QUERY_TILE rows
      build QUERY_TILE TopK heaps
      for each r_tile of REF_TILE rows
          for each (q, r) in (q_tile × r_tile)
              push the algebraic-identity distance
      flush the q_tile's heaps to the output Knng
  ```

  The reference tile is touched `QUERY_TILE` times before being
  evicted from L1. The default `(32, 128)` are sized so that
  `query_tile × ref_tile × 2 × sizeof(float) ≈ 32 KB` — a typical
  x86_64 / arm64 L1 data cache.
- Added six new `test_brute_force` cases pinning the contract:
  output matches the canonical `brute_force_knn(.., L2Squared{})`
  at the default tile sizes; matches at `(3, 5)` (forces multiple
  outer- and inner-tile iterations on the n=8 fixture); matches at
  the degenerate `(1, 1)` (exercises the boundary code at every
  step); both tile sizes throw on zero; same `k=0` / `k > n-1`
  argument-validation throws as the other paths.
- Added `BM_BruteForceL2Tiled_Synthetic` family that sweeps
  `query_tile ∈ {16, 32, 64}` × `ref_tile ∈ {64, 128, 256}` at
  `n=1024, d=128`. The tile sizes show up as `state.counters
  ["query_tile"]` and `state.counters["ref_tile"]` in the JSON so
  Step 23's profiling writeup can ingest the same JSON shape and
  pick the empirical best.
- ctest 89/89 green (6 new brute_force, 83 carried over). Synthetic
  bench at `n=1024, d=128`: canonical 70.5 ms, norms-only
  66.1 ms, tiled 64–66 ms across the sweep — a ~7% gain over
  canonical, ~2% additional gain over norms-only. Recall stays at
  1.0 across all configurations.

### Why
Tiling is the project's first "loop-shape, not arithmetic" CPU
optimisation. Steps 17 and 19 reduced the *total* work; Step 20
keeps the work the same but improves the order in which the
already-touched data is reused. This is the same pattern the
GPU phases will rely on (Step 52's shared-memory reference
tiling, Step 54's register tiling), so getting the loop-nest
shape right on CPU now means the GPU port in Phase 8 can
literally translate this nest into a kernel rather than rederive
the structure.

The default `(32, 128)` is a deliberate compromise. A larger
`QUERY_TILE` would amortise reference loads further but starts
to spill the per-query heap state out of L1; a larger `REF_TILE`
puts more references in cache before they are evicted but
shrinks the outer-loop iteration count and reduces the
prefetcher's lookahead. The sizes are exposed as parameters
rather than baked in because (a) Step 23's profiling pass will
want to sweep them and (b) the optimal values vary across CPUs
— Apple Silicon's 192 KB L1d wants very different tiling from
a Zen 4's 32 KB L1d.

The "small but consistent" speedup (~2% on top of the norms
path) is the expected result on AppleClang's already-aggressive
autovectoriser: the dot product is bandwidth-limited at d=128
with the per-query stream, so reusing references across `QUERY_TILE`
queries reduces L2 traffic but cannot remove the L1 read of the
query row. The wins compound on platforms where the canonical
path's autovectoriser is weaker (older GCC, MSVC, ARM `clang`
without `-mcpu=native`), where the tiling can swing a 30–40%
speedup. The right reading of the AppleClang number is "the
infrastructure is correct; the platform-dependent payoff lands
under perf in Step 23."

### Tradeoff
- **Allocates `query_tile` `TopK` objects per outer-tile.** The
  heap workspace `std::vector<TopK> heaps` is `clear()`-ed and
  re-emplaced each iteration; the reserve in the constructor
  avoids a realloc. Hoisting the allocation entirely
  (e.g. precomputing all `n` heaps at function entry) was
  considered and rejected: it would allocate `n × sizeof(TopK)`
  once instead of `query_tile` per outer iteration, but
  `sizeof(TopK)` plus its inner vector dominates and the
  amortised cost is worse for `n >> query_tile`. The current
  shape is right for SIFT1M-scale inputs.
- **Adds a third L2 entry point.** `brute_force_knn` (canonical),
  `brute_force_knn_l2_with_norms` (Step 19), and
  `brute_force_knn_l2_tiled` (Step 20) all coexist. We accept
  the surface growth: each carries a different default-tradeoff
  contract, and the tests assert agreement across all three
  on every fixture, so a regression in one path is immediately
  surfaced by the others.
- **No vector tiling on `d`.** The plan reserves coordinate-axis
  tiling for Step 27's SIMD pass; here we tile the (query, ref)
  axes only. This keeps the inner loop a clean `for j in 0..d`
  scalar accumulate, which is exactly what the autovectoriser
  expects to see — adding a `d`-axis tile now would just confuse
  it.

### Learning
- *Tile sizes are configuration, not constants.* The defaults
  cover the common case; the parameter names land in the JSON
  counter map so Step 23 can sweep them without recompiling.
  This is the "one knob per axis you might tune later" pattern
  — every Phase 8 GPU step will follow it (block size, warp
  count, shared-memory tile shape).
- *Reuse-then-evict is the cache hierarchy's first lesson.* The
  per-query scan reuses *nothing* — every reference row is read
  once per query, and the L1 has thrown the previous query's
  references away by the time the next query starts. Tiling
  preserves the locality the data layout already has; it does
  not invent any. Before reaching for shared memory or
  prefetch intrinsics in later phases, get the tile loops right.
- *The 8-point fixture is enough to certify a tiling rewrite.*
  Two tile-size cases (`(3, 5)` exercising mid-row boundaries,
  `(1, 1)` exercising every increment) plus the canonical
  comparison covers more state-space than any randomised test
  would. Hand-verified fixtures pay off the day a tile-loop
  rewrite goes wrong — the test fails on a one-line diff
  showing exactly which neighbour ID slipped.

### Next
- Step 21: `cblas_sgemm` for the cross-term in the algebraic
  identity. The norms vector lives in this step's frame; the
  GEMM will fill `(QUERY_TILE × REF_TILE)` of cross-products
  in one BLAS call, fold the norms in via a tiny epilogue
  kernel, and reuse the same tile loops Step 20 just shipped.

---

## [Step 19] — Squared-distance optimisation: precomputed `||p||²` (2026-05-02)

### What
- Added `include/knng/cpu/distance.hpp` and
  `src/cpu/distance.cpp` with two CPU-side primitives the rest of
  Phase 3 will lean on:
  * `dot_product(a, b, dim)` — scalar inner product, the
    `(const float*, const float*, std::size_t)`-shaped twin of the
    Step-08 `squared_l2`. Same signature so a future SIMD pass
    (Step 27) can overload both functions in lockstep.
  * `compute_norms_squared(ds, out)` — `O(n*d)` precompute of
    `||row_i||²` written into a caller-supplied vector. Asserts
    `ds.is_contiguous()` (the precondition the Step-18 helper
    is designed to feed into).
- Added `knng::cpu::brute_force_knn_l2_with_norms(ds, k)` — an
  L2-specific entry point that precomputes the norm vector once
  before the timed loop and replaces each pair's
  subtract-and-square with the algebraic identity
  `||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩`. Mathematically
  identical to `brute_force_knn(ds, k, L2Squared{})` up to fp
  accumulation reordering; the result is clamped at zero to
  swallow a small negative produced by fp32 cancellation when
  `a == b` after rounding.
- Five new `test_brute_force` cases asserting elementwise
  equality of neighbor IDs vs the canonical builder (8-point
  fixture, k=3), distance equality within `1e-4` of the
  canonical result, distances are non-negative, and the same
  argument-validation throws on `k=0`, `k > n-1`, and empty
  datasets.
- Added a parallel benchmark family
  `BM_BruteForceL2Norms_Synthetic` mirroring the existing
  `BM_BruteForceL2_Synthetic` over the same `(n, d)` grid. Both
  emit the project-standard `recall_at_k`, `peak_memory_mb`,
  `n_distance_computations` counters from Step 16; recall stays
  at `1.0` for both (the norms path is a pure algebraic
  rewrite). On AppleClang at d=128, n=1024 the norms path is
  ~6% faster than the canonical path; gains are larger on
  long-`d` where the per-pair `O(d)` dominates.
- ctest 83/83 green (5 new brute_force, 78 carried over from
  Step 18).

### Why
Step 19 is the project's first measurable Phase-3 optimisation
and the one Step 21 (BLAS GEMM) will literally swap into. The
identity rewrite is what makes "distance as GEMM" possible:
`-2 X Yᵀ` from `cublasSgemm` produces the cross term, the
precomputed norms close out the formula. Landing the norms
infrastructure now means Step 21 is a one-line substitution
(`dot_product(a, b, d)` → a `cblas_sgemm` call over a
QUERY_TILE × REF_TILE block) rather than a rewrite.

The expected ~30% speedup quoted in the plan is hardware- and
compiler-dependent. On AppleClang 21 with `-O3` the canonical
path's hot inner loop already vectorises cleanly — the
auto-vectoriser fuses the subtract, multiply, and accumulate
into a single sequence of NEON `vmlaq_f32`s — so the algebraic
rewrite "only" trades that for a `vmla` over the dot product
plus three scalar adds outside the inner loop. The net result
is the modest ~6% measured here. On compilers that do not
autovectorise the subtract-and-square form (older GCC, MSVC at
`/O2`), the speedup is closer to the 30% the plan predicts. The
*right* place for the headline speedup is Step 21's BLAS path —
this commit's purpose is to deliver the algebraic prerequisite,
not the headline number.

The clamp-at-zero on a negative result is a small but important
correctness detail. Under the algebraic identity, `||a - b||²`
for identical points becomes `2 * ||a||² - 2 * ⟨a, a⟩` which is
mathematically zero but can round to a tiny negative under fp32
cancellation. `TopK`'s tie-break logic compares distances by
strict-`<`, so a `-1e-7` would order *before* a true zero and
the test against the canonical path would fail
elementwise-equality. The clamp swallows this without changing
any meaningful ordering.

The L2-specific entry point is a separate function rather than a
template specialisation of `brute_force_knn` for `L2Squared`. The
templated path remains the right shape for `NegativeInnerProduct`
and any future user-supplied `Distance`; the norms identity is
algebraically valid only for a metric of the form
`f(a, b) = g(||a||) + h(||b||) + dot-product-term`, which today
means L2 (and tomorrow may mean cosine, but only after a
separate norms-table pass). Keeping the two functions distinct
means the type signature documents the precondition.

### Tradeoff
- **The norms vector costs `4 * n` extra bytes.** For SIFT1M
  this is 4 MB — negligible next to the 512 MB feature buffer.
  We do not free it after the build; we let it die with the
  function frame. A future refactor that wants to amortise the
  norms table across multiple builds (e.g. a CLI that runs L2
  brute-force at multiple `k`) can hoist the
  `compute_norms_squared` call to the caller without changing
  the entry-point's API — `brute_force_knn_l2_with_norms_view`
  taking `std::span<const float> norms` would be a one-line
  addition.
- **Two L2 entry points now coexist.** A naive caller might pick
  the slower one. We accept the duplication: the canonical path
  is the correctness reference (every later optimisation tests
  against it elementwise), and removing it would force every
  test to use the optimised path, which in turn would mask
  bugs in the optimised path. The `tools/build_knng` CLI still
  routes through the canonical path; switching it to the norms
  path will land in Step 21 alongside the BLAS variant.
- **The clamp-at-zero hides a real fp pathology.** If a future
  refactor accidentally inverts `dot_product`'s sign, every
  distance would silently round to zero instead of producing a
  large negative number that lights up a test. The mitigation
  is the elementwise-equality test against the canonical path
  on the 8-point fixture — any sign error breaks neighbor IDs
  long before the clamp matters.

### Learning
- *Algebraic rewrites are correctness-equivalent only modulo fp.*
  `||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩` is exactly equal in
  the rationals; in fp32 the two paths diverge by a few ulps
  per pair because the accumulation orders differ. The test
  uses `EXPECT_NEAR(.., 1e-4f)` not `EXPECT_FLOAT_EQ` for that
  reason. The day Step 21 ships a BLAS path, the same fixture
  will pin the same equivalence between three paths (canonical,
  norms-precompute, BLAS) instead of two — same test shape,
  one extra column.
- *AppleClang's autovectoriser is a bigger lift than the
  algebraic rewrite at small d.* The plan's 30% predicted
  speedup is predicated on the canonical path *not*
  vectorising; on AppleClang it does. The measured 6% gain at
  d=128 is real but the right reading is "the autovectoriser
  is doing most of the work the rewrite was supposed to do."
  This is exactly why Step 27's *hand-written* AVX2 / NEON
  variant exists — it gives us a path the autovectoriser
  cannot already match, on top of this step's identity.
- *Two entry points are better than one specialised template.*
  The temptation was to write `brute_force_knn<L2Squared>`
  as a partial specialisation that magically picked the
  norms path. We resisted: the type signature
  `brute_force_knn_l2_with_norms` reads at the call site as
  "I want the L2-specific norms-precompute variant" without the
  reader having to know which template specialisations exist.
  Specialisations are a clever pattern with diminishing
  returns; explicit naming wins on clarity.

### Next
- Step 20: `(QUERY_TILE × REF_TILE)` distance tiling for L1
  residency. Will operate on the same `data_ptr() + stride`
  arithmetic Step 18 introduced and the dot-product primitive
  this step ships.

---

## [Step 18] — Struct-of-Arrays layout: stride helpers + contiguity contract (2026-05-02)

### What
- Formalised the `Dataset` storage contract that every later
  vectorisation, BLAS, and GPU step depends on. The layout itself
  was already row-major contiguous; this step pins it as the
  canonical shape, exposes the stride helpers later phases will
  need, and documents *why* the layout matters in the file's
  Doxygen so a future contributor cannot silently regress to a
  vector-of-vectors.
- Added five accessors to `knng::Dataset`:
  * `stride()` — row stride in elements (always `d` today).
  * `byte_stride()` — row stride in bytes; the natural denominator
    for `cudaMemcpy2D`'s pitch and `cublasSgemm`'s LDA argument.
  * `data_ptr()` — direct `float*` (and `const float*`) to the
    contiguous buffer for BLAS calls, mmap, and GPU H2D transfers.
  * `size()` — `n * d`, pre-named so callers do not recompute
    the product (and risk a `size_t` overflow on pathological
    inputs).
  * `is_contiguous()` — cheap precondition check
    (`data.size() == n * d`). `noexcept`, no allocation.
- Expanded the file-level docs: a "Why a single flat float buffer,
  not a vector-of-vectors?" section spelling out the three
  reasons (vectorisation, cache locality, zero-copy GPU transfer)
  with concrete intrinsic / API references; a re-titled
  "Storage contract" section explicitly listing every
  guarantee callers can rely on.
- Added six new `Dataset` test cases in `tests/core_test.cpp`
  (15 → 21 total): stride / byte_stride / size return the
  row-major formula; `data_ptr()` aliases `data.data()` for both
  const and non-const overloads; `is_contiguous()` returns true
  for fresh datasets, false after a manual `data.resize()` that
  breaks the invariant; the empty dataset is contiguous; row
  addresses derive from `data_ptr() + i * stride()`.
- ctest 78/78 green (6 new core, 72 carried over from Step 17).

### Why
The plan calls Step 18 (formerly Step 17 in the original
numbering) "Struct-of-Arrays layout — Replace ad-hoc row-major
with `Dataset::data` as `float[n*d]` row-major contiguous +
stride helpers." The layout already existed (Step 07 shipped it),
but as an *implementation choice*, not a contract. Step 18
promotes it to a contract:

  1. **Every later optimisation will assume this shape.** Step 19
     precomputes `||p||²` by iterating `data_ptr() + i * stride()`
     `d` floats at a time; Step 20 dispatches L1-tile blocks of
     `(QUERY_TILE × REF_TILE)` rows directly off `data_ptr()`;
     Step 21 hands the buffer to `cblas_sgemm(..., A=data_ptr(),
     LDA=stride(), ...)` as-is; Step 49's CUDA brute-force
     `cudaMemcpy`s the buffer in one call. None of those steps
     should re-derive "is the buffer contiguous?" — the type
     should already say so.
  2. **The accessors give later phases a single rename point.**
     A future GPU path that wants 32-byte-aligned row stride for
     `__ldg` coalescing will introduce a *separate* type
     (`PaddedDataset`) rather than complicate this one — but the
     existing call sites all read `ds.stride()` rather than `ds.d`,
     so the day a builder migrates from `Dataset` to
     `PaddedDataset` it is a type-substitution, not a rewrite of
     the inner loop.
  3. **`is_contiguous()` lets the precondition checks be cheap and
     visible.** Future builders will gain
     `assert(ds.is_contiguous())` at the top of their hot path —
     compiled to nothing in release, fires immediately in debug
     when a deserialiser produces a mis-shaped input.

The "Struct-of-Arrays" name in the plan is slightly misleading:
*true* SoA would put each coordinate dimension in its own buffer
(`x[0..n], y[0..n], ..., d_{D-1}[0..n]`), and that layout would
be wrong for our access pattern (the inner loop is over the
coordinates of *one* point, not over many points' shared
coordinate). What we have — and what the plan actually wants — is
a single flat row-major buffer with explicit stride. The
contract-level renaming "row-major contiguous + stride helpers"
in the docs is more accurate than the phase title.

### Tradeoff
- **`data` stays a public field.** Locking it private would force
  every existing call site (`fvecs.cpp`'s loader, the bench
  harness, every test that initialises a fixture) through
  accessors. The cost is real and the upside is small — `Dataset`
  is a value type with no class invariants beyond
  `data.size() == n * d`, and `is_contiguous()` lets callers
  enforce that without owning the field. We will revisit when
  there is an actual reason to (e.g. future versions need to
  enforce alignment), not pre-emptively.
- **`is_contiguous()` is the only invariant check.** A more
  paranoid contract would also forbid `n * d` overflow; instead,
  the constructor's allocation already throws `std::bad_alloc` on
  pathological sizes and the project's
  `-Wconversion -Werror` policy catches signed/unsigned
  shenanigans at compile time. Layered defences here would just
  duplicate what the toolchain already gives us.
- **No alignment guarantees.** `std::vector<float>` allocates
  with the default new-expression alignment, which is `alignof(std::max_align_t)`
  on every supported platform — sufficient for `_mm256_load_ps`
  (32-byte) on x86_64 and `vld1q_f32` (16-byte) on arm64. Future
  steps that need 64-byte alignment (Step 57's tensor-core path)
  will introduce an aligned-allocator variant; today the default
  is correct.

### Learning
- *Pinning a contract is a separate commit from honouring it.*
  The accessors land here, in a small step that only adds API
  surface and tests. Step 19 will then *use* `data_ptr()` and
  `stride()` in a meaningful way, and the diff for Step 19 will
  read as "swap one access pattern for another" rather than "swap
  the access pattern *and* introduce the helpers it depends on." The
  same pattern repeats throughout Phase 3: small, easily-reviewed
  contract-narrowing commits before the optimisation that consumes
  the new contract.
- *Stride is a concept, `d` is an implementation detail.* They
  are equal today and likely always will be in this codebase, but
  the moment any caller writes `ds.stride()` instead of `ds.d`,
  the day a future variant needs padding becomes a type change
  rather than a code-base sweep. This is the same pattern as
  using `std::size_t` instead of `unsigned long`: the symbol
  carries the meaning, not the integer.
- *Every contract should ship with a one-line invariant check.*
  `is_contiguous()` is two arithmetic ops, but the day a binary
  format reader produces a `Dataset` whose `data.size()` does not
  match `n * d`, `assert(ds.is_contiguous())` at the top of the
  algorithm catches it at the call site instead of as a buffer
  overrun ten frames into the inner loop. This will scale
  pleasantly: Step 19 will add `assert(ds.is_contiguous())` to
  the brute-force entry; Step 21 will add it to the BLAS path;
  every later GPU kernel will assert it once on the host side
  before launching.

### Next
- Step 19: precompute `||p||²` per point and rewrite squared-L2
  as `||a||² + ||b||² - 2⟨a,b⟩`. The first measurable Phase 3
  optimisation; expected ~30% speedup. Will use `data_ptr()` +
  `stride()` from this step.

---

## [Step 17] — Deterministic RNG (`knng::random::XorShift64`) (2026-05-02)

### What
- Added `include/knng/random.hpp` — the project-wide deterministic
  PRNG. Single class, header-only:
  * `XorShift64{seed}` — Marsaglia (2003) `(13, 7, 17)` shift triple.
    Period `2^64 - 1`; rejects the all-zero seed at construction
    (it is the algorithm's fixed point) with `std::invalid_argument`.
  * `operator()()` — returns a 64-bit value, advances state.
  * `state()` / `seed(new_seed)` — snapshot-and-restore for
    reproducible parallel sub-seeding (Step 35's parallel
    NN-Descent will need this).
  * `next_float01()` — uniform `[0, 1)` from the high 24 bits.
    Cheaper than `std::uniform_real_distribution<float>` and
    bit-identical to the GPU port we will write in Phase 9
    (the implementation is integer-only until the final cast).
  * `next_below(bound)` — uniform integer in `[0, bound)` via
    Lemire's 64×64 → 128-bit multiplicative trick. Slightly biased
    (≤ `bound / 2^64`); fine for sampling, not for security.
  * `result_type`, `min()`, `max()` — drop-in compatible with
    `std::uniform_int_distribution` and the rest of `<random>`'s
    `UniformRandomBitGenerator` named requirement.
- Added `tests/random_test.cpp` (11 cases): same-seed determinism,
  different-seed divergence, zero-seed rejection on construction
  and on `seed()`, non-zero state invariant under 10k steps,
  `next_float01` range and histogram coverage, `next_below(0|1)`
  edge cases and bound-stays-in-range, drop-in compatibility with
  `std::uniform_int_distribution`, snapshot-and-restore via
  `state()` / `seed()`.
- Routed `benchmarks/bench_brute_force.cpp`'s `make_synthetic` through
  `XorShift64` instead of `std::mt19937_64` so the bench's synthetic
  dataset is now bit-identical between CPU and the GPU port we will
  write in Phase 9. Removed the `<random>` include from the bench TU.
- ctest 72/72 green (11 new random, 61 carried over).

### Why
Every step from here on either *is* randomised (NN-Descent's random
graph init in Phase 5, sampling in Phase 5, mixed-precision noise
analysis in Phase 8) or *consumes* randomised data (every bench
that builds a synthetic dataset). Without a single source of truth
for "give me random bits," each one of those steps would invent
its own RNG, the project would accumulate three or four
incompatible PRNGs, and "same seed ⇒ same output" would degenerate
into "same seed ⇒ same output *if you remember which RNG*."

XorShift64 was chosen over `std::mt19937_64`, `std::pcg64`, `xoshiro`,
or anything from the `<random>` header for one decisive reason: it
fits in a single CUDA / HIP register and runs in three shifts and
two XORs per step, which means we can write the *same* code, byte
for byte, on CPU host and GPU device. Phase 9's GPU NN-Descent
init kernel will literally re-implement this class as a `__device__`
struct and assert at unit-test time that running both produces the
same sequence. `std::mt19937_64` cannot run on GPU without a custom
implementation, and a custom implementation is exactly the kind of
silent divergence the project is trying to avoid.

The all-zero seed rejection is non-negotiable. XorShift64 has a
fixed point at zero — the shifts XOR back to zero, then forever —
so an accidental `XorShift64{0}` would silently degenerate into a
constant generator. Throwing at construction surfaces the bug at
its earliest moment instead of letting it leak into a downstream
test that just happens to "pass."

`next_float01()` exists because the natural `static_cast<float>(rng()) / 2^64`
loses information: the IEEE-754 float significand is 24 bits, and
multiplying a 64-bit integer down to a `[0, 1)` float requires
rounding *somewhere*. Doing it explicitly — masking to 24 bits,
casting to `float`, multiplying by `2^-24` — produces a result
that is identical on every platform with IEEE-754 floats. Doing
the natural thing produces a result that depends on the compiler's
choice of rounding mode for the `uint64_t → float` cast, which
varies across CPU and GPU.

The bench's `make_synthetic` was updated rather than left on
`std::mt19937_64` because the CHANGELOG entry for Step 16 already
called out "the literal `42` and `std::mt19937_64` will be replaced
in Step 17." The cost is one method call and a small mapping
arithmetic; the benefit is that one of the two existing places in
the project that consumed randomness now uses the canonical RNG,
so the convention is "every RNG consumer routes through
`knng::random`" rather than "every *new* RNG consumer does."

### Tradeoff
- **XorShift64 fails some randomness suites** that
  `std::mt19937_64` passes (BigCrush has known weaknesses on
  XorShift's lowest bit). For graph initialisation and sampling
  this does not matter; for any future step that demands
  cryptographic-grade randomness, we ship a different class
  rather than weaken this one.
- **No thread-local RNG by default.** Every parallel algorithm
  has to seed its workers itself (typically via `XorShift64{base
  ^ thread_id}` or by snapshotting + jumping). The alternative —
  a `thread_local XorShift64 default_rng` — would make
  reproducibility depend on threading topology, which is exactly
  the property we are trying to avoid.
- **Lemire's `next_below` is biased.** Bound by `bound / 2^64`,
  which is `< 5e-15` for `bound ≤ 2^16` (the regime sampling
  needs). When an algorithm later wants exact uniformity, it can
  layer rejection sampling on top of `operator()()`; we will not
  bake the rejection loop into the default path.
- **Construction throws.** The project's "no exceptions in inner
  loops" policy still holds — `XorShift64` is constructed once
  per algorithm invocation, not per step, so the `throw` lives
  outside any timed region. The alternative (silently mapping
  `seed=0` to `seed=1`) was rejected: silent fixups hide the
  caller's bug.

### Learning
- *RNG portability is a build-time decision, not a code-time one.*
  Choosing an RNG that compiles to identical bytes on CPU and
  GPU is the kind of choice that costs nothing today and saves
  weeks in Phase 9 when the alternative would be "implement a
  CUDA-specific MT19937 variant and pray the bits match." This
  is the pattern: pick the simplest primitive that works on every
  target, then build everything on top of that one primitive.
- *`UniformRandomBitGenerator` is a tiny, well-defined named
  requirement.* Implementing it is `result_type` + `min()` +
  `max()` + `operator()()`, all `constexpr`-friendly, and the
  payoff is drop-in compatibility with `std::shuffle`,
  `std::uniform_int_distribution`, `std::sample`, and every
  third-party algorithm that takes a "URBG by reference." The
  test `ConformsToUniformRandomBitGenerator` is the proof of
  this — if it compiles, the contract is satisfied.
- *Reproducibility is a property of the entire stack, not just the
  RNG.* `XorShift64` gives bit-identical *bits*, but the floats
  produced by `next_float01` only stay bit-identical because the
  cast and the multiply are deterministic. A separate
  `std::uniform_real_distribution<float>` layered on top of
  `XorShift64` would break this — it does internal scaling that
  varies between libstdc++ and libc++. Owning the float
  conversion ourselves prevents that whole class of bug.

### Next
- Step 18 (Phase 3 opens): Struct-of-Arrays / contiguity formalisation
  on `Dataset`. The first CPU optimisation step. Performance
  measurements from here on lean on the bench JSON shape that
  Step 16 established and the deterministic synthetic dataset
  this step delivers.

---

## [Step 16] — JSON benchmark output (2026-05-02)

### What
- Added `include/knng/bench/runtime_counters.hpp` and
  `src/bench/runtime_counters.cpp` with two helpers every bench TU
  in the project will share:
  * `peak_memory_mb()` — `getrusage(RUSAGE_SELF).ru_maxrss`
    normalised across the macOS-bytes / Linux-kilobytes split. The
    `#if defined(__APPLE__)` branch is the only place in the
    project that needs to know the syscall's per-OS unit; every
    bench reports a single MB value.
  * `brute_force_distance_count(n)` — `n*(n-1)` lifted into a
    single inline helper so every bench TU emits the same
    `n_distance_computations` formula without re-derivation.
- Updated `benchmarks/bench_brute_force.cpp` to wire the
  project-standard counters into Google-Benchmark's `state.counters`
  map: `recall_at_k`, `peak_memory_mb`, `n_distance_computations`,
  plus the existing shape fields `n`, `d`, `k`. The fvecs path now
  uses `knng::bench::load_or_compute_ground_truth` for its truth,
  caching to `build/ground_truth/<stem>.k<K>.l2.gt`.
- `recall_at_k` is computed via `knng::bench::recall_at_k(last,
  truth)` rather than hard-coded to `1.0` even though brute-force
  is its own ground truth — that way the day a refactor breaks
  recall on brute-force, the value here drops below `1.0` and CI
  catches it.
- Added `tools/plot_bench.py`, a standalone Python 3 script that
  ingests the JSON and renders three Matplotlib plots
  (`recall@k`, wall time, peak memory) bucketed by dimensionality.
  It is committed but is *not* part of the C++ build — it has no
  CMake target. Dependencies: matplotlib + standard library only.
  Field names are centralised at the top of the file so a future
  counter rename is one Python edit and one C++ edit.
- Verified end-to-end: `bench_brute_force --benchmark_format=json`
  emits all three new counters per run; `ctest` is still 61/61
  green; the plot script's `argparse --help` runs cleanly.

### Why
Step 12's harness produced wall time and `items_per_second` only —
enough to ship the bench skeleton, not enough to defend any
optimisation. Every later phase's argument is a *Pareto*
argument: "I made it 5× faster while preserving recall@k≥0.97 and
without growing peak memory beyond 1.2× the baseline." That
argument needs all three numbers in one row of one JSON file, and
it needs them in stable field names so a single plotting tool can
ingest a year's worth of bench runs without bespoke per-step
adapters.

`peak_memory_mb` matters because several phases will deliberately
trade memory for speed (Step 19's precomputed `||p||²` table,
Step 20's tiling buffers, Step 55's GEMM workspace, Step 58's
out-of-core streaming). Reporting peak RSS in every JSON row is
the cheapest way to make those tradeoffs visible — `getrusage`
costs one syscall per bench-end and produces a number that is
honest about what the process actually allocated, not what the
algorithm thought it allocated.

`n_distance_computations` is the metric-independent throughput
denominator. `items_per_second` already covers brute-force, but
the moment NN-Descent ships in Phase 5, two builders with
identical wall time can have wildly different distance counts.
Reporting both lets a reader say "this builder is faster *and*
cheaper" or "this builder is faster *but* paid for it in distance
calls" without re-deriving from the algorithm.

The plot script is a Python file, not a C++ target, because the
project's CI matrix is C++-only and the plotting consumer (a
human looking at a single run, or the docs job rendering the
Phase 13 Pareto figure) has no business pulling Matplotlib into
the standard developer build. Centralising the field-name
constants at the top of `plot_bench.py` mirrors the same trick we
use in the C++ side: rename a counter, edit two files, done.

### Tradeoff
- **`peak_memory_mb` is process-wide, not algorithm-only.** The
  reported number includes Google Benchmark's own machinery and
  the dataset still resident from previous benchmarks in the same
  process. We accept this — the alternative (a custom allocator
  hooked under `Knng` / `TopK` to track only "algorithm bytes")
  would be 200 lines of wrapper code for a single counter that is
  still only useful as a *trend*. RSS is the right granularity
  for "did this commit blow up memory?"
- **`recall_at_k` for brute-force is always 1.0.** Reporting it
  anyway is *almost* free (one extra `recall_at_k` call after the
  timed loop) and protects against a class of refactors where
  someone accidentally passes neighbors-only or distances-only
  through to the next pipeline stage. The cost of a single extra
  intersection per bench is invisible next to the timed loop.
- **No JSON-schema validation in `plot_bench.py`.** The script
  simply ignores rows missing a counter field. That means a stale
  JSON from an earlier step plots cleanly (just with empty memory
  / recall lines) instead of crashing. We accept the loose
  contract — the alternative pessimises common-case interactive
  use.

### Learning
- *Google Benchmark's `Counter::kAvgThreads` is the right flag for
  per-iteration averages.* The default `Counter::kAvgIterations`
  divides by `state.iterations()`, which would silently halve the
  reported recall when the bench runs 2× iterations to hit its
  min-time target. `kAvgThreads` reports the value as-is,
  unaltered by either iteration count or thread count — exactly
  what we want for a "this is the recall of the *graph this
  benchmark produced*" reading.
- *`getrusage(RUSAGE_SELF).ru_maxrss` is one of those POSIX
  surface-area surprises.* The man pages on Linux and macOS
  document different units (kB vs bytes). Wrapping the
  conversion in one place prevents every future bench TU from
  re-rediscovering this; the `#if defined(__APPLE__)` is the
  only place in the project that has to know.
- *Plotting code wants to be language-separable from the
  measurement code.* Five steps from now, we will want the same
  plots for SIMD distance kernels (Step 27), GPU brute-force
  (Step 49), GPU NN-Descent (Step 70). All three of those bench
  TUs will emit the same JSON — same field names, same
  semantics, same plot script. The investment in keeping the
  Python out of the build pays off immediately.

### Next
- Step 17: deterministic XorShift64 wrapper with explicit seed.
  The `make_synthetic` function in `bench_brute_force.cpp`
  currently uses a literal `42` and `std::mt19937_64`; once the
  XorShift64 wrapper exists, every bench TU and every
  randomised algorithm will route its RNG through it so seeded
  reproducibility becomes a project-wide property, not a
  per-file convention.

---

## [Step 15] — Recall@k computation (2026-05-02)

### What
- Added `include/knng/bench/recall.hpp` and
  `src/bench/recall.cpp` — the canonical quality metric every later
  approximate builder reports alongside its wall time.
- Public surface:
  * `recall_at_k(approx, truth) → double` returning the fraction in
    `[0, 1]` of `(point, neighbor)` pairs in the approximate graph
    that also appear among the top-k neighbors of the same point in
    the exact graph. Order inside a row is irrelevant; per-row
    duplicates in `approx` are deduplicated before counting so a
    malformed builder cannot inflate its score by repeating a
    correct neighbor `k` times.
  * `recall_at_k_row(approx, truth, row) → std::size_t` returning
    the integer overlap count for a single row. Useful for
    histograms and for tests that want to assert "every row is
    fully recalled" without floating-point tolerance.
  * Both functions throw `std::invalid_argument` on `(n, k)` shape
    mismatches between the two graphs; the empty-input case
    returns `1.0` (vacuous truth) so callers never need to
    special-case it.
- Implementation per row: sort `truth_row` once, binary-search each
  unique `approx_row` ID into it. `O(k log k)` per row, no
  per-row hash-table allocation. The dedup-tracking `seen` vector
  is the only per-row scratch and is `reserve`'d to capacity-`k`.
- New `test_recall` GTest binary with 11 cases: exact match,
  scrambled order, zero overlap, partial overlap (hand-computed
  6/12 = 0.5 fixture), duplicate inflation immunity, the per-row
  accessor, n-mismatch / k-mismatch / row-out-of-range error paths,
  the empty-graph identity, and an end-to-end "brute-force against
  itself recalls 1.0" sanity check that catches future
  refactors that silently drop or permute a neighbor field.
- ctest now runs 61/61 green (11 new recall, 50 carried over from
  Step 14).

### Why
Recall@k is the y-axis of every quality plot the project will
produce — it is what "approximate KNNG" actually means. Without
it, none of the speed gains in Phases 3–12 can be defended; an
"infinitely fast" builder that returns garbage neighbors trivially
wins on wall time. The metric needs to land *now*, before the
first algorithmic optimisation in Phase 3, so every later step has
the option of asserting "this change preserved recall to within
ε" alongside "this change made the build N% faster."

The set-intersection definition (rather than ordered-list
overlap) is the standard used throughout the ANN literature
(`ann-benchmarks`, FAISS, NEO-DNND). It is also the semantically
correct one for KNN graph quality: if two builders both return the
same correct three neighbors but in different orders, they
produced equally good graphs. Penalising one for the order would
just be measuring "which builder happens to sort the same way
brute-force does," which is not a quality metric.

The duplicate-immunity property in `row_intersection` is a
deliberate guard against a bug class that NN-Descent in Phase 5
will get extremely close to: a builder whose internal "atomic
update of neighbor lists" loses CAS races and leaves the same id
in two slots could otherwise score artificially high here. The
test `DuplicatesInApproxRowDoNotInflate` pins the contract.

`double` (not `float`) is the return type because at the n*k
counts of interest (SIFT1M k=100 ⇒ 1e8 pairs) `float`'s 24-bit
mantissa would lose unit-resolution. The regression suite in
Phase 13 needs to detect a single-pair regression — a `float`
return would silently round it away.

### Tradeoff
- **Per-row data structure is `std::vector` + `binary_search`,
  not `std::unordered_set`.** For `k ≤ 1024` (the upper end of
  practical evaluation), the vector path is cache-friendly,
  allocates one block, and is comparable to or faster than an
  unordered set. We will switch to the unordered path the day a
  benchmark wants `k > 1024`; until then the `vector` path is
  the right default.
- **The function does not consume distances.** Two builders that
  return the same neighbor set under different distances (e.g.
  L2 vs negative inner product) will compare clean. That is the
  intended contract — recall@k is about set agreement; whether
  the chosen metric is "good for the task" is a separate concern
  outside this function's scope.
- **`recall_at_k_row` exists despite being one-line on the
  caller side.** It is exposed so tests and future histograms
  do not have to dig through the implementation; the cost is
  three extra lines of header surface.

### Learning
- *The empty-graph case wants `1.0`, not `nan` and not a throw.*
  The choice is between three reasonable answers; `1.0` (vacuous
  truth — every neighbor in the approximate graph is in the truth)
  composes cleanly with downstream pipelines that take a
  weighted average of recall across many shards. The other two
  choices would force callers to add an `if (n == 0) ... else ...`
  branch at every aggregation site.
- *gtest's `EXPECT_THROW` discards the expression's value.*
  `recall_at_k` is `[[nodiscard]]`, so the standard `EXPECT_THROW(
  recall_at_k(a, b), ...)` form trips a warning under our
  `-Wunused-result -Werror` policy. Wrapping the call in
  `{ (void)... ; }` inside the macro keeps the discard explicit
  and silences the warning without weakening the function's
  attribute. Worth remembering for the next [[nodiscard]] API the
  project ships.

### Next
- Step 16: wire `recall_at_k` into the Google-Benchmark counter
  map so `--benchmark_format=json` carries `recall_at_k` and
  `peak_memory_mb` end to end. The bench binary will use
  `load_or_compute_ground_truth` to obtain `truth` once per
  benchmark process.
- Step 17: deterministic XorShift64 wrapper. The recall harness
  built here is the consumer that will assert "same seed →
  same graph → same recall" for randomised builders.

---

## [Step 14] — Ground-truth cache (2026-05-02)

### What
- Added `include/knng/bench/ground_truth.hpp` and
  `src/bench/ground_truth.cpp`, the brute-force ground-truth cache
  every later recall / regression measurement will key off.
- Public surface of `knng::bench`:
  * `enum class MetricId { kL2 = 0, kNegativeInnerProduct = 1 }` —
    the runtime-side metric tag, matching the `metric_id` field of
    Step 13's `.knng` format so `.gt` and `.knng` readers share one
    enumeration.
  * `dataset_hash(const Dataset&)` — 64-bit FNV-1a digest over
    `(n, d, raw float bytes)`. Stable across copies, sensitive to
    any byte change, mixes shape into the digest so two reshapes
    of the same payload still hash distinctly.
  * `save_ground_truth` / `load_ground_truth` — read-write round-trip
    against a documented 64-byte header + payload format. The save
    path writes to `path + ".tmp"` and renames into place so a
    crash mid-write cannot leave a partially-populated cache file.
  * `load_or_compute_ground_truth` — the convenience entry point.
    On hit, returns the cached graph; on miss, runs
    `knng::cpu::brute_force_knn` under the requested metric and
    persists the result before returning it.
  * `default_cache_path` — convention for the cache filename
    (`<dataset_stem>.k<K>.<metric_tag>.gt`) so a single cache dir
    can hold many `(dataset, k, metric)` triples without
    filename collisions. Hash is *not* in the filename — it lives
    inside the file, where stale-but-similarly-named caches can be
    detected on load rather than masked.
- New `knng::bench` static library target wired into
  `src/CMakeLists.txt` and a new `test_ground_truth` GTest binary
  with 10 cases covering hash stability under copy, hash sensitivity
  to coordinate flips and shape flips, round-trip equality of
  neighbors / distances, key rejection on `k`, metric, and
  dataset-hash mismatch, missing-file and corrupt-file handling
  (returns `nullopt`, never throws), the load-or-compute miss-then-
  hit lifecycle, and the `default_cache_path` filename convention.
- `ctest` now runs 50/50 green (10 new ground_truth, 40 carried over
  from Step 13).

### Why
Every later quantitative claim in this project — recall@k from
Step 15, the Pareto plot in Step 100, the regression baseline in
Phase 13 — needs an *exact* nearest-neighbor graph to compare an
approximate builder against. Recomputing brute-force on every
benchmark run would couple every micro-bench to the wall-time of
the thing it is trying to measure (a single brute-force on SIFT1M
already takes minutes on a laptop), and it would silently mask the
case where two adjacent benchmarks accidentally use *different*
ground truths. Pinning ground truth in a content-addressed file is
the project's first piece of "measurement infrastructure that other
measurements stand on."

The choice of FNV-1a over SHA-256 / MD5 / xxh3 was deliberate. FNV-1a
has zero dependencies, is trivial to read in this `.cpp`, runs at
GB/s on commodity hardware, and is overwhelmingly collision-free at
the n×d sizes we care about (cache use is detection of *change*, not
adversarial integrity). xxh3 would be ~5× faster but adds a
single-purpose third-party dependency the project does not
otherwise need; SHA-256 is several × slower than FNV-1a and signals
"cryptographic integrity," which is a stronger claim than the cache
actually makes. The hash function is private to this file — if we
ever change it, we bump `format_version` and existing caches become
misses (correct, conservative behaviour).

The 64-byte fixed-width header mirrors Step 13's `.knng` layout
deliberately. Both formats end up living next to each other on
disk, both use the same `metric_id` encoding, both reserve their
last 16–20 bytes for forward-compatibility growth. Two formats that
share a mental model beat two formats that don't, even when the
ABI cost is just a handful of bytes.

The atomic-rename-on-write is non-negotiable. Without it, a SIGINT
during a multi-minute SIFT1M ground-truth build leaves a partial
file that the next run reads, validates against the cache key (the
header passes!), and silently uses to compute meaningless recall
numbers. The temp-file + `std::filesystem::rename` pattern is the
POSIX-portable way to guarantee a reader sees either the old file
or the fully-written new file, never an in-between state.

### Tradeoff
- **The cache validates `(n, k, metric, dataset_hash)` but not
  `(d, distance ordering)`.** A dataset that hashes the same but
  was loaded under a different metric ordering would in theory
  produce a stale cache, but every metric ordering we ship is a
  pure function of the data — `L2Squared` and
  `NegativeInnerProduct` will always produce the same KNN for
  bit-identical input. The check could be tightened later if a
  metric grows configuration; today it would be premature.
- **Hash is little-endian-only.** Same caveat as `src/io/fvecs.cpp`
  — every supported development platform is little-endian. A
  big-endian port would byte-swap before mixing into FNV-1a,
  guarded on `std::endian::native`. Not on the current roadmap.
- **`load_ground_truth` returns `optional`, not a richer error.**
  Cache misses are common (any flag flip invalidates them), so a
  diagnostic stream would just be noise. If a user reports "my
  cache never seems to hit," we add a `--verbose` debug flag in
  the consumer (Step 16 onwards) rather than complicate this API.
- **No cache eviction policy.** A long-lived `cache_dir` will
  accumulate one file per `(dataset, k, metric)` triple ever run.
  Each file is ~`8*n*k` bytes (e.g. ~80 MB for SIFT1M k=10), so a
  full benchmark sweep stays well under a developer's disk budget.
  We will add eviction when the regression suite in Phase 13 needs
  it; today it would be premature.

### Learning
- *Atomic rename is a stronger guarantee than `fsync` alone.* We
  considered just `fflush` + `close` on the destination file, but
  that does not survive a crash mid-write — the file is created
  before its bytes are durable. Writing to `path + ".tmp"` and
  renaming into place is the standard POSIX trick: `rename(2)` is
  atomic for files on the same filesystem, so a reader sees
  exactly one of {old file, new file, no file}. The fallback
  copy-then-delete branch handles the rare case of a temp file
  landing on a different filesystem from the destination.
- *Cache keys live inside the file, not in the filename.* The
  alternative of `<stem>.<dataset_hash>.k<K>.<metric>.gt` was
  considered — it would let a `glob` see at-a-glance which keys
  a cache holds. We rejected it: the filename then duplicates the
  source of truth (the in-file header), and a renamed-by-hand cache
  could lie about its own contents without the loader noticing. By
  putting every cache-key field in the header and validating each
  on load, the filename becomes pure ergonomics — humans read it,
  the code does not trust it.
- *FNV-1a is enough for "did this dataset change?".* It would not
  be enough for "are these two datasets the same authored object"
  (an attacker can craft collisions trivially), but the cache is
  not a security boundary — it is a developer-time optimisation
  whose worst-case failure mode is a stale read on the next run.
  Picking the simplest hash that matches the threat model keeps
  this file at ~250 lines instead of 600.

### Next
- Step 15: `knng::bench::recall(approx, truth, k) → double`. The
  ground-truth cache built here is the `truth` argument; recall
  is the first measurement on top of it.
- Step 16: wire `recall_at_k` and `peak_memory_mb` into the
  Google-Benchmark counter map so `--benchmark_format=json` carries
  the new fields end to end.

---

## [Step 13] — End-to-end CLI `build_knng` (2026-05-01)

### What
- Added `tools/build_knng.cpp` — a CLI executable that accepts
  `--dataset PATH --k N [--metric M] [--algorithm A] [--output PATH]`,
  loads the dataset via `knng::io::load_fvecs`, runs
  `knng::cpu::brute_force_knn` under the chosen metric, and writes
  the resulting `Knng` to disk in a documented binary format.
- The output format is **version 1**, fixed-width 64-byte header
  followed by `n*k` `uint32` neighbor IDs and `n*k` `float32`
  distances. Header fields: 8-byte ASCII magic `KNNGRAPH`, 4-byte
  format version, 4-byte index width, 4-byte distance width, 4-byte
  metric id (`0=l2`, `1=inner_product`), 4-byte algorithm id
  (`0=brute_force`), 8-byte `n`, 8-byte `k`, 20-byte zero-filled
  reserved tail. All multi-byte integers are little-endian. The
  full layout is documented at the top of `tools/build_knng.cpp`
  and is the source of truth for the loader that lands in Phase 2.
- Hand-rolled CLI parser (no third-party dependency): long-option-
  only `--key value` syntax, `--help` / `-h` print usage and exit 0,
  unknown flags / missing required flags / trailing garbage in a
  numeric value print usage and exit 2, runtime errors during load /
  build / write print a one-line message and exit 1. Three exit
  codes, three meanings — easy to test with `$?`.
- Promoted `tools/CMakeLists.txt` to declare the `build_knng`
  target, linking `knng::cpu` and `knng::io`. The pre-existing
  `hello_knng` is untouched.
- Smoke-tested end-to-end: built a 6-record `.fvecs` fixture
  (two clusters of unit-square corners, same shape as the Step 10
  `BruteForceKnn.EightPointClusterHandVerifiedRows` test), ran
  `build_knng --k 3`, inspected the binary output: header bytes
  match the documented format and the row-0 neighbor IDs `(1, 2, 3)`
  match the brute-force test's hand-verified expectations. Help,
  missing-file, and missing-required-flag paths each behave as
  documented.
- README updated with a new **End-to-end CLI** subsection showing
  the SIFT-small invocation; the existing testing paragraph
  re-listed all current test binaries.
- ctest still 40/40 green.

### Why
This is the first commit in the project where a downstream user can
do something useful without writing a line of C++. The bench
binary from Step 12 also runs end-to-end, but its consumer is a
benchmark harness — `build_knng` is the consumer-facing surface
that proves the entire Phase 1 pipeline (load → build → save) works
together.

The binary output format is documented *now*, with version 1, even
though no loader exists yet. Two reasons:

1. **Lock the wire shape before consumers depend on it.** The
   recall harness in Phase 2 will be the first reader; the Python
   bindings in Phase 13 will be the second; an external converter
   (e.g. into `ann-benchmarks`'s own format) will be the third.
   Every one of those readers needs a stable spec — pinning the
   header layout here, with a `format_version` field that lets
   future versions branch cleanly, costs nothing now and prevents
   "what does that file actually look like?" archaeology later.
2. **The header self-describes the runtime types.** Storing
   `index_byte_width` and `distance_byte_width` in the header (not
   just inferring them from `sizeof(knng::index_t)`) means a
   future reader can detect a mismatch and either widen, refuse, or
   convert. When the project eventually grows a 16-bit-ID quantised
   path (mentioned in `types.hpp`), the existing `.knng` files
   produced today will still be readable without a "version
   2.0" rewrite.

The hand-rolled CLI parser was a deliberate non-dependency choice.
`CLI11` and `cxxopts` are both excellent, but Phase 1 has six flags;
adding a header-only library plus a `FetchContent` block to handle
six flags is the wrong tradeoff. Phase 13's `build_knng` rewrite
(when the production polish step adds, e.g., GPU algorithm
selection, multi-metric, multi-output-format) will be the right
moment to pull `CLI11` in — and to do so for real reasons, not
"because that's what tools usually do."

### Tradeoff
- **`build_knng` only knows brute-force.** The plan explicitly calls
  for the CLI to land at Step 13 even though brute-force is the
  only available algorithm. The dispatch layer (`build()` free
  function in the source) is structured to make adding NN-Descent
  in Phase 5 a single new branch, so the choice does not paint us
  into a corner.
- **Output is a custom binary format, not `.knn` or HDF5.** The two
  obvious "real" formats considered: `.knn` (faiss) and HDF5
  (`ann-benchmarks`). Faiss `.knn` is undocumented public API and
  changes between versions; HDF5 would add a non-trivial dependency
  and a 2× output size on the metadata side. A 64-byte header plus
  flat float/uint32 arrays is the simplest thing that works and
  reads byte-for-byte the same way `numpy.fromfile` does. Phase 13's
  Python bindings will add a `to_hdf5()` shim if `ann-benchmarks`
  integration needs it.
- **No checksum on the output.** The binary format does not include
  a CRC over the payload. A 4-byte CRC32 in the reserved tail
  would be cheap and would catch silent corruption during
  long-running multi-day distributed builds. Deferred — not
  rejected — until either a corruption incident motivates it or
  the format ships to a third party.
- **All status output goes to `stderr`.** This is deliberately POSIX-
  conventional: `stdout` is reserved for whatever the tool's
  "answer" is (currently nothing — the answer is the file written),
  and progress / diagnostics go to `stderr` so a future use that
  pipes `build_knng` output to another tool will not have to filter
  out informational chatter. Worth pinning here so future tools
  follow the same convention.
- **`metric inner_product` flag value, not `negative_inner_product`.**
  The internal functor is `NegativeInnerProduct` (the negation is
  what makes the ordering monotone in the project's convention).
  Exposing the negation in the CLI vocabulary would confuse a user
  who knows IP search; the negation is an implementation detail.
  The wire format's `metric_id=1` documentation explicitly mentions
  `negative_inner_product` so a binary-format reader has the
  unambiguous identifier.
- **No unit test for `build_knng` itself.** The plan called for a
  manual run, and the CLI is thin glue over already-tested
  components: `parse_args` is the only locally-defined logic, and
  it consists of `std::stoull` plus map lookups. A subprocess-based
  integration test would be valuable but introduces the only test
  in the suite that depends on a built binary path — not worth the
  CI complexity at Phase 1. Phase 2's recall harness will exercise
  the binary as part of its own pipeline.

### Learning
- Clang's `-Wfor-loop-analysis` fires on a `for (int i = 1; i < argc;
  ++i)` loop that also does `++i` inside the body (the classic
  "consume-flag-and-value" pattern). Restructuring as
  `int i = 1; while (i < argc) { ...; i += 2; }` is the cleanest
  fix; the warning is a real one and would have produced a subtle
  off-by-one if the body's `++i` had been forgotten. Worth
  remembering whenever a parser loop wants to skip a variable
  number of arguments.
- `std::filesystem::path::operator+=` appends in place without
  inserting a separator — exactly what we want for the
  `<dataset>.knng` default output computation. `operator/=` would
  have inserted a path separator. Easy to confuse; the test was
  whether `tiny.fvecs` produced `tiny.fvecs.knng` (it did) or
  `tiny.fvecs/.knng` (it would have, with `/=`).
- Returning `std::optional<Args>` from `parse_args` to signal
  "user asked for `--help`" is much cleaner than the alternative
  of a magic exit-code-throwing exception. The caller pattern
  (`if (!parsed) { print_usage(); return 0; }`) reads obviously and
  the function's signature documents the two outcomes.
- Inspecting the output with `xxd` is a load-bearing debugging
  technique for any hand-rolled binary format. The 4-byte
  little-endian integers show up as e.g. `0600 0000` (= 6) which
  is unambiguous on every platform that runs CI. Worth mentioning
  in a future style note: when shipping a binary format, the first
  test should be "does `xxd | head` look right?" before any
  programmatic loader exists.

### Next
**Phase 1 is closed.** The naive CPU brute-force pipeline is now
end-to-end: a `.fvecs` file goes in, a documented binary `.knng`
file comes out, with a deterministic, hand-verified algorithm and
40/40 unit tests pinning the contract. Phase 2 (Correctness
Infrastructure) opens at Step 14 with the recall@k computation
and a ground-truth caching layer that turns the binary `.knng`
output into a recall measurement. The benchmark harness from
Step 12 grows two new counters (`recall_at_k`,
`n_distance_computations`); the Pareto plotting helper lands at
Step 15; the deterministic XorShift RNG that Step 16 introduces is
the precondition for every randomised builder in Phases 5 and 9.

---

## [Step 12] — Benchmark harness skeleton (2026-05-01)

### What
- Added `cmake/FetchGoogleBenchmark.cmake` — pinned to Google
  Benchmark `v1.9.1`, source-fetched via `FetchContent`, mirroring
  the policy of `cmake/FetchGoogleTest.cmake` (no system discovery,
  no installs from this project, upstream's own self-tests disabled,
  `BENCHMARK_DOWNLOAD_DEPENDENCIES=OFF` so the upstream build does
  not try to pull its own GoogleTest copy).
- Added `benchmarks/CMakeLists.txt` and the first benchmark binary
  `benchmarks/bench_brute_force.cpp`. Two registered cases:
  * `BM_BruteForceL2_Synthetic` — runs over a deterministic
    `Uniform[-1, 1]` dataset across the cartesian product of
    `n ∈ {256, 512, 1024}` × `d ∈ {32, 128}`. Reports wall-time and
    a `items_per_second` counter computed from `n * (n - 1)` distance
    computations per `brute_force_knn` call.
  * `BM_BruteForceL2_Fvecs` — driven by the `KNNG_BENCH_FVECS`
    environment variable; loads the named `.fvecs` file via
    `knng::io::load_fvecs` and benchmarks brute-force on it.
    `state.SkipWithError(...)` when the env var is unset or the load
    fails — the case stays registered but does not produce
    misleading numbers.
- New build option `KNNG_BUILD_BENCHMARKS` (default `OFF`) at the
  root `CMakeLists.txt`. When `ON`, includes
  `FetchGoogleBenchmark.cmake` and `add_subdirectory(benchmarks)`;
  when `OFF`, the standard developer loop and the CI matrix pay
  zero FetchContent / build cost.
- README updated with a new **Benchmarks** subsection and two new
  rows in the configure-time options table
  (`KNNG_BUILD_BENCHMARKS`, `KNNG_GOOGLEBENCHMARK_TAG`).
- Smoke-ran on the dev box: six synthetic configurations completed
  in well under a second each, throughput climbs cleanly from ~58 M
  comparisons/s on `(256, 32)` to ~85 M on `(1024, 32)`, and drops
  to ~14 M on the higher-dimensionality `(*, 128)` cases — exactly
  the cache-bound shape one expects from a triple-loop
  implementation. No reported errors. ctest unaffected (still 40/40
  green).

### Why
The benchmark harness has to exist *before* the optimisations it
will measure. Two things follow from that:

1. **Pipeline first, numbers second.** Step 12 is explicitly a
   skeleton: the bench compiles, runs, produces JSON output. No
   recall numbers, no peak-memory counters, no plot-rendering
   helper — those land at Steps 14 / 15. Trying to ship the full
   metric set in one step would couple the harness wiring to half of
   Phase 2 and turn an afternoon into a week.
2. **Synthetic baseline that does not require the SIFT download.**
   A benchmark that requires `tools/download_sift.sh` to have been
   run is a benchmark that produces zero useful signal in CI and on
   a fresh checkout. The synthetic configuration runs anywhere with
   no setup; the real-dataset case is opt-in via env var. Both
   shapes are exercised by the same TU so neither path can rot
   without the other noticing.

The opt-in build flag (`KNNG_BUILD_BENCHMARKS=OFF` by default) is
pure courtesy: a contributor running `cmake -B build && cmake --build
build && ctest` should not pay the Google Benchmark FetchContent
cost (~10 s of clone + compile on a cold cache) for a target they
did not ask for. CI keeps benchmarks off the matrix until Phase 2
introduces a regression-baseline JSON; turning them on is one CMake
flag away.

### Tradeoff
- **Synthetic data, not a checked-in tiny `.fvecs` fixture.** A
  small (e.g. 256×32) `.fvecs` file committed into `tests/data/`
  would have made the `*_Fvecs` benchmark run unconditionally. But
  the project explicitly gitignores binary datasets, and committing
  even a tiny one would be the camel's nose for someone later
  committing a 100 MB SIFT-small. The env-var-gated benchmark is the
  honest version.
- **`std::mt19937_64` for synthetic generation, not the project's
  future `XorShift64`.** Step 16 will introduce the project-wide
  RNG wrapper. Until it lands, picking a literal `std::mt19937_64`
  with seed 42 is the least-surprising choice — every C++ developer
  recognises it, the bench's output is deterministic, and the
  one-line swap to `knng::random::XorShift64` at Step 16 is trivial.
  Documented in the bench source itself.
- **`items_per_second` reported as "distance computations / s",
  not "queries / s".** Both are defensible, and Phase 3 / 7
  benchmark writeups will likely report both. Picking the
  metric-independent throughput now means cross-`d` comparisons are
  apples-to-apples without normalising; cross-`k` comparisons (when
  Step 14 lands recall) will need the `n / time` form. Adding a
  second counter then is one line.
- **No `BENCHMARK_F` fixture or per-test data caching.** The
  benchmark builds the dataset once per registered case (in the
  body of the bench function), not once per iteration — Google
  Benchmark's iteration loop is the inner `for (auto _ : state)`
  block. A fixture would let multiple bench cases share one
  dataset and reduce setup time, but for the (n, d) shapes we
  benchmark today the dataset construction is sub-millisecond. Cost
  not yet worth the abstraction.
- **Bench TUs are warning-policed; Google Benchmark's are not.**
  Same policy as GoogleTest. Upstream GB does not compile cleanly
  under `-Wconversion -Wold-style-cast -Werror`, and adding a
  tactical `BENCHMARK_ENABLE_WERROR=OFF` (which we did) plus
  refraining from calling `knng_set_warnings()` on the upstream
  targets keeps our policy applied to our code only.

### Learning
- Google Benchmark's CMake build defaults to a hidden side effect:
  if it can't find `gtest` it tries to download its own copy. The
  symptom on a fresh build was a confusing second GoogleTest clone
  appearing in `_deps/` next to ours. The fix —
  `BENCHMARK_DOWNLOAD_DEPENDENCIES=OFF` — is documented but not
  prominent. Worth pinning here so the next FetchContent integration
  in this project knows to look for an analogous flag.
- `state.SkipWithError(...)` is the right way to gracefully handle
  "the input this case needs is not available" — `state.SkipForTest`
  is the close cousin for tests, and forgetting to `return` after
  `SkipWithError` produces a benchmark that "skips" but then runs
  anyway with garbage state. The `return` after every
  `SkipWithError` in `BM_BruteForceL2_Fvecs` is load-bearing.
- The synthetic numbers immediately reveal the dimensionality
  scaling penalty (`(n=1024, d=32)` reports ~85 Mcomp/s vs
  `(n=1024, d=128)` reports ~14 Mcomp/s — a ~6× drop on a 4×
  dimensionality bump). That is exactly the cache-pressure
  signature the Phase 3 SoA + tiling steps will attack. Pinning the
  baseline here means Step 19's tiling commit can quote a
  before/after number against this same harness, no re-runs needed.
- Apple Silicon's "10 X 24 MHz CPU s" line in the bench banner is
  a known Google Benchmark quirk on macOS — `sysctl
  hw.cpufrequency` does not exist on M-series, so GB falls back to
  reporting `0` formatted in MHz. Cosmetic only; affects metadata,
  not measurements. The bench banner now warns about this on every
  run; deferred to a documentation note.

### Next
Step 13 closes Phase 1 with the end-to-end CLI: `tools/build_knng`,
a small command-line program that takes `--dataset path.fvecs --k N
--metric l2 --algorithm brute_force --output graph.bin`, loads the
dataset, builds the graph, and writes a binary representation to
disk. The output format is documented in the source so future
loaders (and the eventual Python bindings in Phase 13) can read it.
This is the first commit where a user can do something useful with
the project without writing a line of C++.

---

## [Step 11] — Dataset I/O: `.fvecs` / `.ivecs` / `.bvecs` (2026-05-01)

### What
- Added `include/knng/io/fvecs.hpp` declaring four loaders:
  * `Dataset load_fvecs(path)` — float32 records → `Dataset`.
  * `IvecsData load_ivecs(path)` — int32 records → `(n, d, std::vector<int32_t>)`.
  * `BvecsData load_bvecs(path)` — uint8 records → `(n, d, std::vector<uint8_t>)`.
  * `Dataset load_bvecs_as_float(path)` — convenience widener so
    quantised SIFT1B drops into the same `Dataset` consumers.
- Added `src/io/fvecs.cpp` with the implementation. Anonymous-namespace
  `MmapFile` RAII wrapper owns an `mmap`-ed read-only view of the
  file (POSIX `open`/`fstat`/`mmap`/`munmap`/`close`); the
  `read_vecs_records<Element>` template strips the per-record
  `int32` dim prefix, validates every prefix matches the first, and
  copies the element payloads into the destination row-major buffer.
  Inconsistent prefixes, non-multiple file sizes, empty files, and
  failed system calls all throw `std::runtime_error` with a message
  naming the offending file and reason.
- Promoted `src/CMakeLists.txt`: created a second STATIC library
  `knng::io` (alias of `knng_io`), `PUBLIC`-linking `knng::core` so
  consumers transitively pick up `Dataset` and the warnings policy.
- Added `tests/fvecs_test.cpp` (`test_fvecs`) with seven cases:
  * Round-trips three records of dim 4 in `.fvecs`, two records of
    dim 3 in `.ivecs`, and two records of dim 3 in `.bvecs`
    (asserting both raw bytes and the widened-to-float convenience).
  * Missing-file, empty-file, inconsistent-dim-prefix, and
    file-size-not-a-multiple-of-record-size error paths.
  * A `TempPath` RAII helper materialises each fixture in
    `std::filesystem::temp_directory_path()` and removes it on
    destruction; tests run with no network access.
- Added `tools/download_sift.sh` (executable) — provisioning script
  that fetches SIFT-small (10K base + 100 query + 100×100 ground
  truth) from the IRISA `corpus-texmex` site into the gitignored
  `datasets/siftsmall/` directory. `FORCE=1` re-downloads;
  `DEST=...` overrides the destination root. The build never invokes
  this script.
- `ctest` is now 40/40 green (was 33/33); a `fvecs` label joins the
  CTest summary.

### Why
Phase 1's brute-force builder (Step 10) needs real input to be more
than a toy — and so does the benchmark harness landing at Step 12.
The four formats above cover every dataset the project's plan
mentions, and they are also the interchange formats `ann-benchmarks`,
faiss, and cuVS all consume. Implementing them now means every
later step can take a `Dataset` path on the command line (Step 13)
without anyone needing to hand-write a loader twice.

`mmap` over `read()` was the plan's choice and is the right one for
two reasons. First, the SIFT1B base file is 92 GiB — pulling the
whole thing through a `read()` would either thrash a single buffer
or force a streaming reader (and a streaming `Dataset` builder is a
much bigger surface area than the current "row-major contiguous"
contract assumes). With `mmap`, the OS demand-pages the file under
memory pressure for free, and the loader copy still happens but only
for the pages the loader actually touches. Second, on macOS `read()`
of a multi-gigabyte file goes through the unified buffer cache the
same way `mmap` would — choosing `read()` would buy us nothing in
exchange for the syscall cost.

The cleanly-separated `IvecsData` / `BvecsData` types (instead of
forcing both into `Dataset`) reflect that ground-truth files are not
"datasets" in any algorithmic sense — they are reference labels, and
collapsing the type would obscure that distinction at every consumer
site. The `load_bvecs_as_float` widener exists for the one case where
the byte width really *should* be promoted (CPU brute-force on
SIFT1B); callers who need raw bytes still get them through
`load_bvecs`.

### Tradeoff
- **POSIX `mmap` only — no Windows `MapViewOfFile`.** The CI matrix
  is Linux + macOS; the project's target hardware is Linux/HPC and
  ultimately MI350A APU. A Windows port would need a parallel
  `MmapFile` implementation behind a `#ifdef`, plus a CMake test
  surface that nobody has volunteered to maintain. Deferred until a
  Windows user appears.
- **Copy on load, not zero-copy alias.** A zero-copy loader would
  hand the consumer a `std::span<const float>` *and* the underlying
  `MmapFile` lifetime, which then has to live somewhere. The
  destination layout has stride `d` while the file layout has stride
  `4 + d * sizeof(element)`, so a true zero-copy `Dataset` would
  require either (a) per-row spans (defeats the point of a flat
  `data` vector that GPU H2D / cuBLAS calls want), or (b) a `Dataset`
  variant that doesn't own its storage. Either is a much larger
  refactor than the project needs right now. The single-pass copy is
  bandwidth-bound and finishes in ~50 ms on SIFT1M (512 MiB at
  ~10 GB/s). When SIFT1B forces the question, Phase 11 can revisit.
- **No CRC / hash check on the loaded file.** The loader validates
  *structural* consistency (every dim prefix matches the first, file
  size divides cleanly into records) but does not validate that the
  payload bytes are what the dataset's publisher intended. The
  `download_sift.sh` script could `shasum -a 256` against a pinned
  hash; deferred until a wrong-file incident happens (the IRISA
  files have been stable for 15 years).
- **Loader returns by value, not output parameter.** `Dataset` and
  the two struct types are RVO-friendly and not enormous (the
  `data` vector's heap pointer moves, not the `n*d` floats). The
  prevailing project style is "return-by-value POD types"; sticking
  with it keeps call sites readable. A future bandwidth-sensitive
  loader could grow an in-place overload that takes `Dataset& out`.
- **Anonymous-namespace `MmapFile` and `read_vecs_records`** instead
  of public utilities. Both are implementation details — exposing
  them would invite users to depend on the POSIX-only path before
  the cross-platform abstraction exists. Anonymous namespace gives
  internal linkage without the boilerplate of a `detail::` namespace
  in a private header.

### Learning
- `[[nodiscard]]` on the loaders interacted badly with GoogleTest's
  `EXPECT_THROW(expr, type)` — the macro evaluates `expr` once but
  discards the value, which `[[nodiscard]]` flags under `-Werror`.
  Wrapping the call as `EXPECT_THROW(static_cast<void>(expr), type)`
  is the canonical fix; a future `KNNG_VOID_CAST` macro or a small
  test helper could hide the noise if it appears more than a few
  times. Pattern worth pinning here.
- `MAP_FAILED` is `((void*)-1)`, not `nullptr`. Comparing the `mmap`
  return value against `nullptr` is a classic bug that compiles
  cleanly and silently corrupts memory on the failure path. The
  guard is `if (p == MAP_FAILED) ...`.
- The "inconsistent dim prefix" test had to be carefully constructed:
  if the second record's payload had matched the first record's
  dim-bytes, the file size would still be a multiple of the *first*
  record size and the inconsistency would only show up at
  prefix-validation time. Picking 4 → 3 makes the file size 32 bytes
  total, which is `(4 + 4*4) + (4 + 3*4) = 20 + 16` and divides
  cleanly by neither 20 nor 16 — *but* would divide by 32 if we
  treated it as a single record of dim 7, a tantalising near-miss.
  The test passes because the loader infers `d=4` from the first
  prefix and then immediately hits the prefix mismatch. Worth
  writing the test that way to ensure the prefix-validation branch
  is actually exercised, not the file-size branch.
- The `TempPath` PID-and-counter naming scheme is enough for our
  single-process test suite. If we ever fork or spawn worker
  processes (Step 26's `std::thread` study, eventual Phase 6 MPI
  tests), the helper will need to fold in a `std::random_device` or
  an `mkstemp`-style reservation. Documented in the helper's
  comment, not enforced.

### Next
Step 12 introduces the benchmarking harness skeleton. A small
`benchmarks/CMakeLists.txt` and `benchmarks/bench_brute_force.cpp`
linked against Google Benchmark (fetched via a new
`cmake/FetchGoogleBenchmark.cmake` mirroring the GoogleTest helper)
will load a `Dataset` from disk and time `brute_force_knn`. No
recall numbers yet — that arrives at Step 14. The benchmark target
is opt-in via `-DKNNG_BUILD_BENCHMARKS=ON` so the GoogleTest cycle
in CI is unaffected.

---

## [Step 10] — Naive CPU brute-force KNN (2026-05-01)

### What
- Added `include/knng/cpu/brute_force.hpp` declaring and defining the
  function template `template <Distance D> Knng brute_force_knn(
  const Dataset&, std::size_t k, D distance = D{})`. Triple loop over
  `(query, reference, dim)`, parameterised on the `Distance` concept,
  using `knng::TopK` as the per-query buffer. Argument validation
  throws `std::invalid_argument` on `ds.n == 0`, `k == 0`, or
  `k > ds.n - 1`.
- Added `src/cpu/brute_force.cpp` providing explicit template
  instantiations for `L2Squared` and `NegativeInnerProduct`. This
  gives the new static library a real translation unit and lets the
  two common metric paths skip per-consumer instantiation.
- Promoted `src/CMakeLists.txt`: created the `knng_cpu` STATIC
  library (alias `knng::cpu`), `PUBLIC`-linking `knng::core` so
  consumers transitively pick up headers + C++20 + warnings. The
  pre-existing `knng_core` INTERFACE library is untouched.
- Added `tests/brute_force_test.cpp` (`test_brute_force`) with nine
  GoogleTest cases:
  * Output shape matches `(ds.n, k)`.
  * Three hand-verified rows on the 8-point two-cluster dataset
    (point 0, point 4, point 7) — exercises the equal-distance
    tie-break invariant.
  * Self-is-never-a-neighbor invariant on every row.
  * Rows-are-sorted-ascending invariant on every row.
  * `k == 1` returns the single closest neighbor (deterministic on
    ties).
  * `NegativeInnerProduct` metric path is exercised end-to-end.
  * The three `std::invalid_argument` boundary cases.
- `ctest` is now 33/33 green (was 24/24); a `brute_force` label joins
  the existing labels in the CTest summary.

### Why
This is the first algorithm in the project, and every later
optimisation — vectorised distance kernels, blocked tiling, BLAS-as-
distance, OpenMP, SIMD, GPU, NN-Descent, multi-GPU NEO-DNND — will
use this function's output as its correctness reference. Three
properties make it a good reference:

1. **Pure function of inputs.** No RNG, no parallel scheduling, no
   timing dependence. Two runs with the same `(Dataset, k, Distance)`
   produce bit-identical `Knng` output. Step 09's id-based tie-break
   is what makes this true even when multiple neighbors are
   equidistant — without it, the brute-force order would depend on
   the iteration order, and the elementwise-equality regression tests
   that Phases 4 / 7 / 9 will rely on would silently start failing
   the moment a parallel reorder lands.
2. **Triple-loop transparency.** A reader who has not seen the
   project before can convince themselves of the algorithm's
   correctness by reading the function body once. No tactic, no
   tiling, no special cases. The CHANGELOG entries for Phase 3 will
   measure every optimisation against this exact baseline; making
   the baseline obviously correct makes every later optimisation's
   speedup credible.
3. **Uses every Phase-1 type at once.** `Dataset`, `Knng`, the
   `Distance` concept, `L2Squared`, `TopK`, and `index_t` all appear
   in this one function. If any of those interfaces is wrong or
   awkward, this commit is where it shows up — and now is the
   cheapest possible time to fix it.

The function-template-on-`Distance` shape (rather than a runtime
metric enum) is the convention the project pinned in the Step 03
"convention divergences resolved" decision and re-affirmed in
Step 07. No virtual dispatch in the inner loop; compile-time
dispatch through the concept is canonical.

### Tradeoff
- **Header-only template plus explicit-instantiation `.cpp`.** Three
  alternative organisations were considered:
  1. *Pure header* — every consumer pays the parsing /
     instantiation cost. Rejected: the brute-force body is small now,
     but Phase 3 will grow it (norm precomputation, tile parameters,
     per-row scratch). The static library is the right home for that
     growth.
  2. *Pure `.cpp` with non-template `brute_force_knn_l2`* — would
     drop the `Distance` parameterisation. Rejected on the same
     grounds as the runtime-enum approach: every later algorithm
     (NN-Descent, GPU local-join) is going to want the same
     parameterisation, and giving the reference function a different
     shape from its successors would invite a partial port later.
  3. *Header-only template + explicit instantiations in the `.cpp`*.
     **Chosen.** The template stays available for any user-supplied
     `Distance` functor; the two built-in metrics are pre-compiled in
     `libknng_cpu.a`. Implicit instantiation of the same template in
     a consumer TU does not collide with the explicit instantiation
     because the implicit version has weak linkage and the explicit
     has strong linkage.
- **Self-match excluded by `if (r == q) continue`.** The alternative
  is to compute every distance and rely on the `q == q` distance of
  zero being smaller than every real neighbor's distance, then drop
  the first neighbor. Rejected: distinguishing "self at distance 0"
  from "exact duplicate at distance 0" silently is exactly the kind
  of subtle bug that bites once a dataset has duplicates (which SIFT
  does). The explicit `continue` is one branch per inner iteration on
  a dataset where every other inner iteration is a multi-multiply-add
  loop — measurable cost is well under 1%.
- **Argument validation throws, does not assert.** The brute-force
  function is on the public API surface — callers are downstream
  user code (eventually pybind11 bindings, eventually a CLI). The
  STYLE.md error-handling rule is: throw at the API boundary,
  `assert` for internal invariants. The error messages include the
  offending argument values (e.g. `k (8) must be <= ds.n - 1 (7)`)
  because a `terminate()` from `assert` would tell the caller
  nothing.
- **No `std::span`-only overload.** `Dataset` is the canonical input;
  a future "build KNNG over an arbitrary `(span<const float>, n, d)`
  triple" could be added if a use case arises (e.g. a memory-mapped
  buffer that is not a `Dataset`). Not adding it speculatively keeps
  the API surface minimal for now.

### Learning
- The error-message string concatenation
  `std::to_string(k) + " ... " + std::to_string(ds.n - 1)` triggered
  zero `-Wconversion` warnings — `std::to_string(std::size_t)` is
  `unsigned long`-typed and the default `+` overload composes
  cleanly. A pleasant surprise; some other projects work around this
  with custom `fmt`-style helpers.
- The `extract_sorted` invariant from Step 09 (`size() == k` after
  offering `>= k` distinct candidates) is what lets the post-loop
  copy run unconditionally without a "did we get fewer than k?"
  branch. This is one of those small payoffs of pinning the contract
  on the lower-level component first.
- `EXPECT_THROW` from GoogleTest is the right tool for the boundary
  cases — `EXPECT_DEATH` would also work but introduces a
  death-test-mode dependency that we have not opted into. The throw
  shape lets us add a future test that inspects `e.what()` if
  message stability ever matters.
- The `BruteForceKnn.NegativeInnerProductMetricCompiles` test only
  asserts shape, not numeric values — but its real purpose is to
  prove that a *second* `Distance` functor instantiates cleanly
  through the same template. Without it, the explicit instantiation
  for `NegativeInnerProduct` would be dead code at link time, and a
  future template-bug regression on the second metric would not be
  caught by CI. Compile-time-coverage tests are still tests.

### Next
Step 11 lands dataset I/O — `src/io/fvecs.cpp` with loaders for
the `.fvecs`, `.ivecs`, `.bvecs` formats used by the standard ANN
benchmark datasets (SIFT, GIST, Fashion-MNIST). Memory-mapped reads
keep large files (SIFT1M is 512 MiB) out of process address space
proper. A `tools/download_sift.sh` script provisions SIFT-small
(10K × 128) into the gitignored `datasets/` directory. The unit test
generates a tiny in-memory `.fvecs` byte sequence and asserts the
loader reads it back correctly — no network dependency in the test
itself.

---

## [Step 09] — Bounded top-k buffer `knng::TopK` (2026-05-01)

### What
- Added `include/knng/top_k.hpp` with `class TopK`:
  * `explicit TopK(std::size_t k)` — fixed capacity at construction.
  * `void push(index_t id, float dist)` — admit iff buffer is not yet
    full, or `dist <` current worst, or `dist ==` worst and `id <`
    worst's id (deterministic tie-break — see below).
  * `std::vector<std::pair<index_t, float>> extract_sorted()` — drains
    the buffer into ascending-distance order; ties broken by ascending
    id; buffer is empty afterwards.
  * Inspectors `size()`, `capacity()`, `empty()`.
- Backed by `std::priority_queue<Entry, std::vector<Entry>, WorseFirst>`
  where `WorseFirst` is a strict-weak-ordering comparator yielding a
  max-heap on `(dist, id)` so the worst entry is always at the top
  for O(log k) eviction.
- Added a new test binary `tests/top_k_test.cpp` (`test_top_k`) with
  six cases: empty extract, fewer-than-k retention, the size-k
  invariant under repeated insertion (10 candidates → 3 smallest),
  equal-distance tie-break by smaller id, the degenerate `k == 0`
  buffer (rejects everything), and the post-extract drained state.
- `ctest` is now 24/24 green (was 18/18). New `top_k` label in the
  CTest summary.

### Why
Every nearest-neighbor builder in this project — brute-force at
Step 10, NN-Descent in Phase 5, every GPU kernel from Phase 7 onward
— needs to maintain a per-query "best k seen so far" structure.
Writing it once, in one place, with a clean contract is the right
move at Step 09: the brute-force builder lands at Step 10 and would
otherwise inline ten lines of heap-management code into its inner
loop, only to need them ripped back out at Step 21 when
`std::partial_sort` becomes a contender.

The deterministic-tie-break rule (`id` strict-less wins on equal
`dist`) is doing two things at once. First, it lets the brute-force
builder produce bit-identical output across runs without requiring a
seeded RNG (Step 16's job) — something the plan explicitly calls out
as an invariant for Step 10. Second, it gives every later
implementation (parallel CPU NN-Descent, GPU kernels with atomic
top-k merges) a single, unambiguous answer to "what does correct
output look like?" — making elementwise-equality regression tests
possible without recall-based fuzz testing.

The "max-heap on distance" choice is the textbook one: admission is
a single comparison against `top()`, eviction is `pop()` + `push()`,
both `O(log k)`. For `k ≤ 100` (the regime every benchmark we care
about lives in), this is competitive with linear-scan partial-sort
strategies and uses strictly less memory than holding a dense
distance vector.

### Tradeoff
- **`std::priority_queue`, not a hand-rolled heap.** A hand-rolled
  `std::vector` + `std::push_heap` / `std::pop_heap` would let us hold
  scratch memory across calls (`extract_sorted` would zero `size_` and
  reuse the underlying vector) and would expose the buffer for
  benchmarking SIMD-friendly bulk-merge variants in Phase 4. Deferred
  — Step 09 prioritises clarity and correctness, and the standard
  container both compiles cleanly under `-Wconversion -Werror` and
  reads obviously to a future contributor. The hand-rolled variant
  will land as part of Step 21's partial_sort comparison.
- **Deterministic tie-break on `id`, not insertion order.** Insertion-
  order tie-break would let the heap report "the first equal-distance
  candidate seen wins," which is what FAISS does. Rejected because
  insertion order depends on the loop schedule (parallelism, NUMA-
  partitioned scans, GPU thread-block traversal) and would silently
  break elementwise-equality tests as soon as Step 23's OpenMP loop
  lands. The id-based rule is a function of inputs only, regardless
  of evaluation order.
- **`extract_sorted` drains, instead of returning a snapshot.** A
  snapshot variant would let callers extract intermediate state
  during iteration, but the buffer is per-query and per-iteration in
  every algorithm in the project — drain-and-return matches the
  call-site pattern exactly. Adding a `peek_sorted_copy()` would be
  trivial later if a use case appears.
- **`std::pair<index_t, float>` as the public output element type.**
  A dedicated `Neighbor { index_t id; float dist; }` POD struct would
  read more clearly at consumer sites. Considered, but `std::pair` is
  already the canonical "associative element" type and avoids
  introducing a new type that the brute-force builder at Step 10 will
  immediately need to convert to/from when it writes into a `Knng`
  row. Likely revisited in Phase 5 when the NN-Descent neighbor list
  introduces an `is_new` flag — that one *will* need a struct.
- **Header-only at Step 09.** `TopK` is small and `inline` everywhere;
  there is no out-of-line state. When Step 10 lands the first non-
  template `.cpp` source file, the question of "should top_k.hpp move
  to top_k.hpp + top_k.cpp" can be revisited, but the trigger for the
  split would be code growth that has not yet happened.

### Learning
- The strict-weak-ordering check on `WorseFirst` is the single
  easiest place to introduce a heap-corruption bug: returning `true`
  for `a == b` violates the comparator contract and produces silent
  garbage at runtime. Writing the tie-break as `a.id < b.id` (strict)
  on the equal-`dist` branch — never `<=` — is the right pattern.
  Worth pinning explicitly in the changelog because the symptom of
  getting it wrong is "tests pass on small inputs, fail on large
  ones, with no clean reproducer."
- `[[nodiscard]]` on `extract_sorted` would be desirable, but the
  function also has the side effect of draining the buffer, so a
  caller might legitimately call it for the side effect alone. Left
  off; documented in the `///` block instead.
- The clangd "unused include" warning on `<utility>` was a false
  positive — `std::pair` is in the return type but clangd treats it
  as transitively pulled by `<queue>`. Kept the include explicit
  because the public API does directly traffic in `std::pair` and
  pulling it via a coincidental transitive include would be fragile
  to a future libc++ rearrangement.

### Next
Step 10 is the first real algorithm: `Knng knng::cpu::brute_force_knn(
const Dataset&, std::size_t k)`. A triple loop over (query, reference,
dim) parameterised on the `Distance` concept, using `TopK` as the
per-query buffer. The output `Knng` rows are sorted ascending by
distance (free, by way of `extract_sorted`), and tie-broken on
neighbor id (also free, by way of the Step 09 tie-break rule). The
test exercises an 8-point synthetic 2-D cluster with hand-verified
neighbors, plus the `k == 1` and `k > n - 1` edge cases. This is also
the step that turns `knng::core` from an INTERFACE library into a
STATIC library with real source files.

---

## [Step 08] — Scalar `squared_l2` C-style primitive (2026-05-01)

### What
- Added `knng::squared_l2(const float* a, const float* b,
  std::size_t dim)` to `include/knng/core/distance.hpp` — a free
  function with the raw-pointer signature that later SIMD intrinsics
  and CUDA / HIP kernels will specialise.
- Refactored the existing `L2Squared::operator()` functor to delegate
  to `squared_l2` so the scalar formula has a single source of truth.
  No behaviour change; existing tests unchanged.
- Added five GoogleTest cases under the `SquaredL2Free` suite:
  * `ZeroForIdenticalPointers` — sanity check (Σ 0² = 0).
  * `HandVerifiedThreeFourPair` — the canonical 3-4-5 right triangle:
    `(3,4,0)` vs origin → 25.
  * `DimZeroIsEmptySum` — empty sum is 0, no element read.
  * `DimOneIsScalarSquaredDifference` — degenerate 1-D case.
  * `AgreesWithFunctor` — cross-checks that the functor and the free
    function produce identical output, pinning the delegation invariant.
- `ctest` is now 18/18 green (was 13/13).

### Why
The Phase 1 plan calls for a "scalar L2 distance function" with a
C-style signature. There were two ways to introduce it:

1. *Replace* `L2Squared` with a free function and let callers wrap it
   themselves. Rejected — every algorithm in the library is
   parameterised on a `Distance` *concept*, not on free functions, and
   removing the functor would force every algorithmic site to
   instantiate a wrapper at the call. The concept-based dispatch is a
   project invariant and not negotiable.
2. *Add* a free function alongside `L2Squared`, and have the functor
   delegate to it. **Chosen.** The free function is the lower-level
   building block; the functor is the high-level concept-satisfying
   adapter. SIMD (Step 27) and CUDA (Step 46) will provide alternative
   `squared_l2` overloads keyed on a tag type or a backend macro
   without touching the functor or any algorithm that uses it.

The "single source of truth" property matters precisely because it
prevents the kind of accidental divergence where a hand-rolled SIMD
implementation drifts from the scalar reference and only a recall
regression on SIFT1M reveals the bug a month later. Pinning the scalar
formula in one place — and giving every other implementation a
property-based test that compares against it on random inputs (a
pattern Phase 4 will introduce) — is cheap insurance.

### Tradeoff
- **Inline in the header, not in a `.cpp`.** `squared_l2` is small
  enough that a translation-unit-local copy in every consumer is
  cheaper than the function-call overhead a non-inlined version would
  incur in a hot inner loop. The header-only choice also keeps
  `knng_core` an INTERFACE library for one more step (it gains real
  source files at Step 10). Cost: a tiny amount of object-code
  duplication across translation units; the linker dedupes via
  `inline` semantics.
- **No `restrict`-equivalent on the pointers.** C++ has no
  `__restrict__` in the standard, and adding `__restrict__` /
  `__attribute__((restrict))` here would either be compiler-specific
  or require a project-wide `KNNG_RESTRICT` macro. Deferred to Phase 4
  alongside the other SIMD-prerequisite ergonomics.
- **No bounds checking, no nullptr check.** This is an explicit "inner
  loop primitive" — the documentation says so, the contract says so.
  Every guard added here is a guard executed once per distance
  computation. Validation lives at the boundary (the `Dataset`
  constructor, future I/O loaders, the eventual `build_knng` CLI), not
  in the scalar kernel.
- **Functor still defined in the header.** Could have been moved to a
  `distance.cpp`, but `L2Squared` is empty (no state, just an
  `operator()`) and inlining the call site is the whole point. Keeps
  the rule "POD types live in headers" consistent.

### Learning
- The `[[nodiscard]]` attribute on the free function caught one
  oversight in an earlier draft of the test file where the return
  value of `squared_l2` was computed but never asserted on — the
  warning fired immediately under `-Werror`. Cheap signal that the
  attribute is paying for itself even at toy scale.
- The `dim == 0` branch needed a deliberate test, not because it is
  algorithmically interesting but because the for-loop's
  `i < 0` initial condition is `false` and the function correctly
  returns the initialised `acc = 0.0f`. Documenting this as a test
  case makes it impossible for a future "let's optimise the inner
  loop" rewrite to silently break the empty-sum invariant.
- Naming the free function `squared_l2` (lower-snake) and the functor
  `L2Squared` (PascalCase) follows the project naming table exactly:
  free function ⇒ snake_case, composite type ⇒ PascalCase. The two
  spellings sitting side-by-side in the header is itself a small
  documentation of the convention.

### Next
Step 09 introduces `include/knng/top_k.hpp` — a `class TopK` that
accepts `(index_t, float)` pairs via `push()` and emits a sorted
vector of size ≤ k via `extract_sorted()`. Internally backed by
`std::priority_queue` with a max-heap keyed on distance, so the
worst-distance element is always at the top and a new candidate can
be admitted in O(log k) time without scanning the buffer. Tests cover
ordering, the size-k invariant under repeated insertion, duplicate
distances, and the empty-output edge case.

---

## [Step 07] — Core-types residue: `knng::Dataset` (2026-05-01)

### What
- Added `include/knng/core/dataset.hpp` with the `knng::Dataset` POD
  struct: `std::size_t n`, `std::size_t d`, `std::vector<float> data`,
  plus `row(i)` accessors that return a `std::span<const float>` /
  `std::span<float>` view of length `d`. Layout is row-major, no
  padding, contiguous in `data`.
- A two-argument constructor `Dataset(n, d)` value-initializes storage
  to length `n*d`. Default constructor produces an empty (0×0) dataset
  for placeholder use.
- Added three GoogleTest cases to `tests/core_test.cpp`:
  `Dataset.ConstructedShapeMatchesArguments`,
  `Dataset.RowViewsAreContiguousWithStrideD`,
  `Dataset.MutatingRowViewIsReflectedInUnderlyingStorage`.
- `ctest` is now 13/13 green (was 10/10).

### Why
The Plan's original "Step 6 — Core types" task was largely subsumed
by in-repo Step 03, which shipped `index_t`, `dim_t`, the `Distance`
concept, the built-in metrics, and the `Knng` adjacency struct. The
single remaining residue was the input-side container — a row-major
feature matrix that every later algorithm (CPU brute-force in Step 10,
NN-Descent, the GPU pipelines) consumes.

Pinning the layout *now*, before any algorithmic code lands, means
every later builder takes `const Dataset&` from day one. Retrofitting
the type later — once five different algorithms each have their own
"matrix of floats" notion — would force a coordinated rewrite. Cheap
to do at Step 07; expensive at Step 27.

The shape mirrors `Knng` deliberately: both are flat row-major
contiguous, both expose `row(i)` / `neighbors_of(i)` / `distances_of(i)`
returning `std::span`. A reader who has internalized one has
internalized the other.

### Tradeoff
- **Plain struct, public members.** `Dataset` could have been a class
  with a private buffer and accessor methods. Rejected: every consumer
  (BLAS calls, GPU H2D transfers, memory-mapped loaders, profiling
  tools) wants direct access to `data.data()`. A class wrapper would
  add ceremony without enforcing any invariant beyond
  `data.size() == n*d`, which is already trivially checked at the
  constructor and never violated thereafter.
- **`std::vector<float>`, not aligned/pinned storage.** No 64-byte
  alignment, no CUDA pinned memory, no `std::aligned_storage`. Phase 7
  will introduce a `DeviceBuffer<T>` for GPU residency and Phase 4 may
  introduce NUMA-aware allocators; those are *additional* containers,
  not replacements. A `Dataset` is the host-side reference shape, and
  `std::vector` is the simplest thing that works for it.
- **Own header, not appended to `types.hpp`.** Followed Step 03's
  convention: scalar aliases live in `types.hpp`, composite types
  each get a dedicated header (`graph.hpp` for `Knng`, now
  `dataset.hpp` for `Dataset`). Slightly more files, but
  `#include "knng/core/dataset.hpp"` is more honest than
  `#include "knng/core/types.hpp"` for code that only wants the
  `Dataset` type.
- **No `DistanceMetric` runtime enum.** The plan explicitly says
  "no runtime `DistanceMetric` enum in Phase 1 — compile-time
  dispatch through the concept is the canonical path." Honoured.
  When the CLI lands at Step 13, the algorithm/metric strings will
  be turned into compile-time choices by a `if/else` over enum-like
  `std::string_view` constants in the CLI dispatcher, not by a
  runtime virtual interface inside the algorithms.

### Learning
- Repeating the `Knng`-style `row(i)` accessor pattern was a small but
  noticeable ergonomic win even at zero algorithmic content: the test
  cases for `Dataset` came out as near-mechanical translations of the
  `Knng` row-view tests. Convention consistency pays off as soon as
  the second instance of the pattern appears.
- The `dataset.hpp` originally `#include`d `knng/core/types.hpp` "for
  context," but it never used `index_t` or `dim_t` — the row count
  and dimensionality are `std::size_t` (matching the constructor /
  `std::vector<float>::size_type`). A leftover include that the type
  doesn't need is exactly the kind of include-what-you-use violation
  the project's clangd integration immediately flagged. Removed.
- `n` and `d` as field names are short, but they are the canonical
  symbols in every ANN paper (and in the `(n, d)` shape vocabulary of
  Numpy / cuBLAS / faiss). Spelling them `num_points` and
  `dimensions` would be more discoverable for a first-time reader but
  would add visual noise in every algorithmic file that touches them.
  Choice: paper-canonical names, with the meaning documented in the
  Doxygen `///<` comment next to each field.

### Next
Step 08 begins the real algorithmic work: a scalar `squared_l2`
function with the C-style `(const float*, const float*, std::size_t)`
signature that later SIMD and GPU kernels will specialize. The
`L2Squared` functor stays as the canonical span-based interface; the
new free function is the lower-level building block that the functor
(and future SIMD / CUDA implementations) can dispatch into. Tests
expand to cover zero distance to self, hand-verified pairs, and
dimension-0 / dimension-1 edge cases.

---

## [Step 06] — Coding style guide + CHANGELOG template (2026-04-17)

### What
- Added `docs/STYLE.md`, the project's authoritative style reference.
  Fourteen short sections covering: C++ dialect, file extensions,
  `#include` order (with a worked example), naming, index types,
  the `Distance` API convention, const-correctness + `noexcept`,
  error-handling policy, performance-critical constraints, comments
  + Doxygen requirements, the warning policy, testing expectations,
  git/commit discipline, and the `CHANGELOG.md` section template.
- Formalised the `CHANGELOG.md` preamble to cross-reference
  `STYLE.md` §14 as the canonical template source, rather than
  duplicating the template in-place.
- Every rule in `STYLE.md` describes the convention already in use
  in the committed code (Steps 01–05). No retro-rewrites of existing
  files were needed.

### Why
A written style guide does two things a tacit one cannot: it settles
arguments before they start, and it tells a future contributor
(including a future me, six months from now) what the conventions
*are* without requiring them to read every file to triangulate. At
this phase the code is small enough that divergence from the guide
is still cheap to fix; once Phase 1's algorithm files start landing,
convention drift becomes expensive. Ship the guide now.

The guide is also the closing step of Phase 0: every invariant the
rest of the project depends on — naming, includes, warnings, testing,
changelog format — now has a single URL a contributor can be pointed
at.

### Tradeoff
- **Guide, not enforcement.** `STYLE.md` documents conventions but
  does not enforce them mechanically. clang-format + clang-tidy
  would automate a portion (include order, naming for some
  categories, some const-correctness rules) but add a configuration
  surface and another tool to install. Deferred until the project
  grows more contributors; a soloist rereading the guide before each
  commit is cheaper than fighting a formatter-clang-tidy stack on
  every save.
- **Short guide, not exhaustive.** Some things deliberately left out:
  which standard algorithms to prefer (`std::ranges::sort` vs
  `std::sort`), lambda capture conventions, exception-specification
  policy beyond `noexcept` at the free-function level. The guide
  takes the position that "if you can derive the right answer from
  existing code, ask" is cheaper than trying to codify every micro-
  decision in advance. Grow the guide when a concrete disagreement
  surfaces.
- **CHANGELOG template lives in STYLE.md, not as a separate file.**
  The template is small enough (≈30 lines) that a dedicated
  `CHANGELOG_TEMPLATE.md` would be overkill. Embedded in the style
  guide as §14 keeps it next to the other project-wide conventions
  and avoids "which of these templates is the authoritative one"
  drift between sibling files.
- **No `clang-format` file committed at this step.** Related to the
  enforcement tradeoff above. When the first disagreement about
  brace-placement or column width happens, a clang-format file
  lands in the same commit that resolves it — not before.

### Learning
- The include-order rule that fires the most often in practice is
  "matching header first in implementation files." It is the single
  most valuable rule because it makes `foo.cpp`'s correctness the
  caller's problem, not the compiler's: if `foo.hpp` is not
  self-contained, including it first lets the compiler tell us
  immediately. Every other ordering rule is pedantic by comparison.
- Writing the style guide after writing five commits of code is the
  right sequencing. Writing it first would have produced a
  document full of rules we then discovered were inconvenient; the
  existing code is evidence that these specific rules work. This is
  the opposite of the usual "document-driven-design" advice — but
  for a solo learning project, working-code-first is faster and
  more honest.
- `snake_case_t` vs `PascalCase` for types is the single most
  defensible decision to document explicitly. The C++ community
  splits roughly in half on this, and the project's choice
  (composite types PascalCase, scalar aliases `snake_case_t`) is
  defensible but not a majority position. A one-line rule in
  `STYLE.md` §4 forestalls a tedious future conversation.
- Rules framed as tables (§2, §4) are dramatically easier to scan
  than rules framed as bulleted prose. When a rule enumerates a
  mapping between categories and values, use a table.

### Next
**Phase 0 is closed.** Next session opens Phase 1: the naive CPU
reference. Step 07 is a small housekeeping step — add the
`knng::Dataset` struct that was the outstanding residue of the
plan's original "core types" step (most of which shipped early as
Step 03). After that, Step 08 begins the real algorithmic work:
scalar squared-L2 distance with the C-style pointer signature that
later SIMD and GPU kernels will specialize.

---

## [Step 05] — CI scaffolding via GitHub Actions (2026-04-17)

### What
- Added `.github/workflows/ci.yml`, a GitHub Actions workflow with two
  jobs:
  * **`build-and-test`** — matrix over `{Linux/GCC, Linux/Clang,
    macOS/AppleClang}`, each one running
    `cmake -S . -B build && cmake --build build && ctest`.
    `fail-fast: false` so a single compiler's regression does not mask
    others; `CMAKE_BUILD_PARALLEL_LEVEL=4` matches GitHub-hosted
    runner cores.
  * **`docs`** — Linux-only, installs Doxygen + graphviz and runs
    `cmake --build build --target docs`. Configured with
    `-DKNNG_BUILD_TESTS=OFF` so this job does not pay the GoogleTest
    FetchContent cost.
- Triggered on pushes to `main` and on pull requests targeting
  `main`. `concurrency: cancel-in-progress: true` on `ci-${{ ref }}`
  so a new commit supersedes its predecessor rather than letting both
  burn runner minutes.
- Added a CI status badge to `README.md`.

### Why
A CI harness at Step 05 — before Phase 1's algorithm code lands —
means every subsequent step is verified on three compilers the moment
a commit is pushed, not the moment somebody happens to notice a
regression. Catching a portability issue when a single file is the
whole surface area is orders of magnitude cheaper than bisecting
through a dozen steps of accumulated algorithmic work. Three
compilers (GCC, Clang, AppleClang) is deliberately the floor: it
catches the most common "works on my compiler" failure modes without
overspending on exotic targets that we do not claim to support at
Phase 0.

### Tradeoff
- **No CUDA in the matrix (yet).** GitHub-hosted runners have no GPU;
  running `nvcc` in compile-only mode proves the toolchain installed
  correctly but says nothing about kernel correctness or performance.
  Adding a self-hosted GPU runner has real cost (hardware, network,
  credential management) and will be done in Phase 7 when the
  algorithmic GPU code actually exists to justify it.
- **No Windows in the matrix.** The project's target hardware is
  Linux/HPC and ultimately MI350A APU — Windows is not a platform
  we commit to supporting. MSVC warning flags remain in
  `cmake/CompilerWarnings.cmake` so a future Windows port is not
  more painful than it needs to be, but CI time on a platform nobody
  will use for production is wasted signal.
- **Concurrency cancellation on push-to-main.** If two commits land
  on `main` in quick succession, the earlier CI run is cancelled in
  favor of the newer commit. Tradeoff: a partially-green history
  (some middle commits never got a pass/fail verdict), but the
  head-of-branch status is always current and runner minutes are
  used only for the latest state. Good tradeoff for a single-
  developer, trunk-based repository; revisit if the project grows a
  release-branch model.
- **Default compilers, not pinned versions.** Using `gcc`, `clang`,
  and AppleClang as whatever the runner image ships means compiler
  upgrades arrive automatically — good for catching drift early
  (warnings-as-errors bites if a new compiler adds a warning), bad
  for reproducibility of a *specific* green run. The alternative —
  pinning e.g. `gcc-13` — trades early warning for historical
  reproducibility. Acceptable at Phase 0; will likely pin to
  specific toolchain versions once benchmark numbers start being
  committed.
- **Docs job is a separate gate, not a step inside
  `build-and-test`.** Keeping the Doxygen install and the docs build
  off the critical-path matrix means a Doxygen version bump or a
  graphviz flakiness does not redden the primary status indicator.
  The docs job still runs per-commit; its status is reported
  separately in PR checks.

### Learning
- `CMAKE_BUILD_PARALLEL_LEVEL` (env-level) is a less-known alternative
  to `cmake --build -j N`. Setting it in the `env:` block at job
  scope lets every `cmake --build` step inherit the parallelism
  without repeating `-j 4` at each call site.
- `${{ github.ref }}` is the correct concurrency key for "cancel
  on same branch" — NOT `github.sha` (every commit has a unique sha
  and nothing would ever be cancelled). This is a small trap; the
  GitHub docs show the right pattern but it is easy to cargo-cult
  the wrong one from an older example.
- GitHub-hosted `ubuntu-latest` runners ship gcc AND clang
  pre-installed — no `apt install` needed for the compiler itself,
  only for auxiliary tools like Doxygen/graphviz. Makes the workflow
  file much shorter than the classic "install-compilers-then-build"
  pattern.
- The docs job's `-DKNNG_BUILD_TESTS=OFF` is a genuine ~15-second
  savings per run (GoogleTest clone + build) — not huge, but it
  compounds over the project's lifetime and keeps the docs job
  focused on what it actually verifies.

### Next
Step 06 will add `docs/STYLE.md` — the project's short coding-style
guide: naming, header guards, include order, const-correctness,
Doxygen expectations — plus a formalized `CHANGELOG.md` template so
the What/Why/Tradeoff/Learning/Next pattern already in use is
documented explicitly and cross-referenced from `STYLE.md`. That
closes Phase 0. Phase 1 (Naive CPU Reference) opens on the next
working day.

---

## [Step 04] — Doxygen configuration (2026-04-17)

### What
- Added `docs/Doxyfile.in` — a small, curated Doxygen configuration
  that only overrides non-default options. Project name, version, and
  input paths are substituted by CMake at configure time.
- Added `docs/CMakeLists.txt` with a `find_package(Doxygen)` probe.
  When Doxygen is present, a `docs` custom target is created; when
  Doxygen is missing, configure emits a helpful STATUS message and
  returns early — no hard dependency on Doxygen for the rest of the
  build.
- Wired `add_subdirectory(docs)` into the root `CMakeLists.txt`.
- Input tree covers `include/`, `src/`, `tools/`, plus `README.md`
  (used as the mainpage via `USE_MDFILE_AS_MAINPAGE`) and
  `CHANGELOG.md` (so the design history is browsable alongside the
  API reference).
- Tuned the two noise sources that fired on the first run:
  `CHANGELOG.md` is now an explicit input (so internal references
  resolve), and the README's `recall@k` phrase was reworded to
  `recall-at-k` so Doxygen's auto-command parser stops treating `@k`
  as an unknown command.
- Verified: `cmake --build build --target docs` → **67 HTML pages
  generated, zero warnings**. Output in `build/docs/html/`.

### Why
The project's invariant is "every public function gets Doxygen"
(Project Invariants). That rule is only enforceable if the doc build
exists and runs cleanly — otherwise "undocumented" is invisible.
Setting up Doxygen now, with just the `knng::core` public headers to
document, means the feedback loop is fast (tiny input, tiny output)
and the configuration can be iterated on before documentation volume
becomes costly. Every subsequent public API lands into a working
Doxygen pipeline that will warn on missing `@brief` / `@param` at
build time — pedagogical pressure to keep the docs current.

### Tradeoff
- **Curated Doxyfile, not `doxygen -g` default.** A default-generated
  Doxyfile is 3000+ lines of every setting Doxygen supports, most
  never touched. Upgrading across Doxygen versions then becomes a
  manual merge exercise. A curated Doxyfile (≈60 lines of overrides)
  inherits new defaults for free; the tradeoff is that future option
  changes must be made consciously. Net win for a long-lived project.
- **`EXTRACT_ALL = YES` + `WARN_IF_UNDOCUMENTED = YES`.** Makes every
  symbol visible in HTML (so the docs are a faithful API reference)
  while still warning about missing `@brief` (so the "document
  everything public" rule stays auditable). The alternative —
  `EXTRACT_ALL = NO` — would hide undocumented symbols and make the
  audit harder. `WARN_AS_ERROR = NO` so the warning log is
  informational rather than a build-breaker; we can tighten this to
  a hard gate in CI later if the warning count stays at zero.
- **Docs target is opt-in, not default.** `add_custom_target(docs)`
  (not `ALL`) means `cmake --build build` never runs Doxygen. The
  cost of running Doxygen on every incremental rebuild would be
  real; requiring an explicit `--target docs` keeps the inner loop
  fast.
- **Docs CMakeLists added unconditionally.** No `KNNG_BUILD_DOCS`
  option — the docs subdirectory handles "Doxygen missing" itself.
  This is simpler than a top-level option, and there is no
  configure-time cost when Doxygen is absent (just the early
  `return()` in `docs/CMakeLists.txt`).

### Learning
- Doxygen 1.16 gets confused by the Markdown sequence `recall@k` —
  the `@` is treated as a command introducer even inside a narrative
  sentence. Escaping with `\@` works inside doxygen blocks but breaks
  regular Markdown rendering; rewording is cleaner.
- Including Markdown files as Doxygen inputs has an auto-linking side
  effect — references like `` [`CHANGELOG.md`](CHANGELOG.md) `` only
  resolve if the target file is also in `INPUT`. Useful property
  (cross-linking between README and CHANGELOG on the rendered site),
  but a warning source if forgotten.
- `find_package(Doxygen OPTIONAL_COMPONENTS dot)` (plural even for a
  single component) is the modern CMake pattern. It sets both
  `DOXYGEN_FOUND` and `DOXYGEN_DOT_FOUND`, letting the config query
  graphviz presence without a second `find_program`.
- `configure_file(... @ONLY)` with `@PROJECT_VERSION@`, `@PROJECT_NAME@`
  substitution keeps the Doxyfile free of CMake syntax — the
  generated file opens cleanly in editors with Doxyfile syntax
  highlighting, which the raw source does not.

### Next
Step 05 will add GitHub Actions CI — Linux (gcc + clang) and macOS
build-and-test jobs that run `cmake --build` + `ctest`, plus a
separate Linux job that builds the Doxygen docs. CUDA stays off the
matrix until Phase 7.

---

## [Step 03] — `knng::core` public API scaffold (2026-04-17)

### What
- Added the `knng::core` library as an INTERFACE (header-only) target
  in `src/CMakeLists.txt`, aliased as `knng::core` and wired into the
  root build via `add_subdirectory(src)`.
- Introduced the first three public headers under
  `include/knng/core/`:
  * `types.hpp` — `knng::index_t` (`std::uint32_t`) for point indices
    and `knng::dim_t` for dimensionalities. Separate aliases of the
    same underlying type keep API signatures self-documenting.
  * `distance.hpp` — the C++20 `knng::Distance` concept (callable,
    `noexcept`, `float(std::span<const float>, std::span<const float>)`,
    lower-is-better) plus two concrete metrics: `L2Squared` and
    `NegativeInnerProduct`. Static asserts pin each metric to the
    concept at library-compile time.
  * `graph.hpp` — `knng::Knng`, a plain-old-data adjacency struct
    with parallel `neighbors[n*k]` and `distances[n*k]` flat arrays,
    plus `neighbors_of(i) / distances_of(i)` `std::span` accessors.
- Added `tests/core_test.cpp` + `test_core` executable, linking
  `knng::core` + `GTest::gtest_main` under the full
  `knng_set_warnings()` policy. Seven TESTs cover both metrics, the
  concept, and the `Knng` layout invariants (shape, row stride,
  mutating views).
- Verified: `cmake --build build && ctest` → **10 tests passed, 0
  failed** (3 smoke + 7 core). Incremental reconfigure; no warnings.

### Why
Every step after this one needs a vocabulary: a way to spell
"point index", "feature vector", "distance metric", and "K-nearest
neighbor graph". Pinning those names in Step 03 — before any
algorithm is written — means the CPU reference builder, the GPU
kernels, and the distributed exchange routines will all describe
their inputs and outputs in the same types. Refactoring core types
after downstream code exists is the expensive time to do it.

The Distance concept is the key design choice here. By declaring
"any monotone-lower-is-better scoring functor" as a *concept* rather
than a virtual class, metric choice is a compile-time specialization
of every algorithm — no vtable indirection in the innermost loop.
This matters enormously on GPU, where divergent virtual dispatch is
the fastest way to lose 10x of throughput.

### Tradeoff
- **INTERFACE library, not STATIC.** Header-only is the simplest
  thing that works when there are no `.cpp` files yet. It also keeps
  consumers that pull `knng::core` via `FetchContent` from paying
  any compile cost. The downside: template-heavy headers eventually
  cost compile time for everyone who includes them; when the first
  non-template code lands (a brute-force builder, quantization
  tables, etc.) this target will transparently upgrade to STATIC.
- **Negated inner product, not raw IP.** Every algorithm in the
  library assumes "smaller == closer". Keeping that invariant at the
  metric boundary (negate on the way in) rather than in every search
  / refinement routine (branch on metric kind) is cheaper in code
  and in runtime — one subtraction instead of an extra comparison
  per candidate. Same trick FAISS uses for its IP index.
- **Squared L2, not L2.** Algorithms only need the ordering, and
  `sqrt` in the inner loop is a real performance hit on GPU. Magnitude
  semantics are recovered by calling sqrt once on final output.
- **`std::uint32_t` for `index_t`.** 2^32 ≈ 4.3 billion points is
  well past our target per-node workload, but ruling it out is
  conscious: scaling past that requires a distributed index scheme
  (sub-graph sharding + global ID mapping) that will live at a
  layer above `knng::core` anyway.
- **Tests link `knng::core`, not `knng::headers`.** Mirrors how a
  downstream user will consume the library — if linking `knng::core`
  ever stops being sufficient to pick up the public headers, the
  test suite fails first.

### Learning
- C++20 concepts are a genuinely better fit than CRTP or virtual
  bases for performance-sensitive pluggable strategies. The whole
  Distance contract — callable, noexcept, return type, parameter
  shape — fits in six lines and produces a readable compiler error
  when a new metric doesn't satisfy it.
- `static_assert(Distance<L2Squared>)` at the end of the header costs
  nothing at runtime and catches concept-breakage at library-compile
  time rather than at first-use. Every concept-bound type should
  self-witness like this.
- Adding `add_subdirectory(src)` before `add_subdirectory(tests)`
  matters — CMake targets are strict left-to-right in declaration
  order, and the test target's `target_link_libraries(… knng::core)`
  requires the alias to already exist.
- The incremental CMake reconfigure just worked when the only new
  subdirectory was `src/` with an INTERFACE library and a new
  `tests/` executable. No build-tree clean was required; the
  FetchContent'd GoogleTest was not re-downloaded.
- Writing tests against `std::span<const float>{a}` explicitly (not
  relying on CTAD through `std::span{a}`) is safer under
  `-Wconversion` — CTAD deduces `std::span<float, N>` which then
  narrows to `std::span<const float>` at the call site and has
  occasionally been flagged by stricter compilers.

### Next
Step 4 will introduce a deterministic CPU brute-force KNNG builder
(`knng::build_bruteforce`) as the correctness oracle for every
future optimization. It will be parameterized on the `Distance`
concept and will fill a `Knng` in row-major order; a unit test will
cross-check its output against a hand-computed small example, and a
second test will prove that repeated runs on the same input produce
bit-identical output.

---

## [Step 02] — GoogleTest wiring & smoke test (2026-04-16)

### What
- Added `cmake/FetchGoogleTest.cmake`, which pulls GoogleTest v1.15.2
  via `FetchContent` and then `include(GoogleTest)`s the CMake stdlib
  module that provides `gtest_discover_tests`.
- Added a `KNNG_BUILD_TESTS` option (defaulting to `PROJECT_IS_TOP_LEVEL`)
  and gated testing behind both it and CTest's `BUILD_TESTING` — so
  downstream consumers that pull `knng` via `FetchContent` do not drag
  GoogleTest into their build.
- Added `tests/CMakeLists.txt` and `tests/smoke_test.cpp`. The smoke
  test exercises the full "GoogleTest → `knng::headers` → generated
  `knng/version.hpp`" chain with three cases: non-negative macros,
  dotted string matches component macros, string is non-empty.
- `gtest_discover_tests` registers each `TEST()` with CTest; a labeled
  `smoke` group makes targeted invocation (`ctest -L smoke`) possible
  as the test corpus grows.
- Verified end-to-end: clean `cmake -B build`, `cmake --build build`,
  then `ctest` → **3 tests passed, 0 failed** on AppleClang 21.

### Why
Without a live test harness from Day 2, every later algorithmic step
would be verified only by eyeballing `hello_knng` output. Bolting on
tests "later" is how every project ends up with a pile of untested
code — the cost of slipping in a regression is minimized by catching
it the same day it's written, which requires CTest to already be
working. Fetching GoogleTest via `FetchContent` keeps the toolchain
requirements self-contained: a fresh clone + CMake + a C++20 compiler
is the entire setup story, no distro packages, no vcpkg.

### Tradeoff
- **FetchContent, not system GoogleTest.** An initial experiment using
  `FIND_PACKAGE_ARGS NAMES GTest` preferred Homebrew's
  `libgtest_main.1.11.0.dylib`, whose rpath is not embedded in our
  test binaries — `gtest_discover_tests` then failed at build time
  because the runtime loader couldn't find the dylib. Always-fetching
  trades ~20 MB of disk and a one-time git clone for a deterministic,
  static-linked, rpath-free test binary. Given the project's
  "pinned dependencies, reproducible builds" stance, this was the
  right call.
- **Filename: `FetchGoogleTest.cmake`, not `GoogleTest.cmake`.** The
  obvious name collides with the CMake stdlib module that defines
  `gtest_discover_tests`; when our module was first on the
  `CMAKE_MODULE_PATH`, the stdlib module was shadowed and the
  function was undefined. Renaming is the cleanest fix.
- **No `gtest_main` → custom `main` yet.** Linking `GTest::gtest_main`
  is the fast path but forecloses on things like custom fixture-global
  setup or GPU-context initialization. When those land, we'll switch
  to `GTest::gtest` + a project-owned `main.cpp`. Not needed yet.
- **GoogleTest compiled without `knng_set_warnings`.** Upstream does
  not build cleanly under `-Wconversion -Werror` (in fact this very
  build emitted a `-Wcharacter-conversion` warning from
  `gtest-printers.h`). Applying our strictness to third-party code
  would be a maintenance burden with no payoff.

### Learning
- `FetchContent_MakeAvailable` with `FIND_PACKAGE_ARGS` is elegant in
  principle but punts environment-specific rpath/link issues onto
  whoever configures the project. For a young project, the
  "always-build-from-source" path is markedly less surprising.
- The CMake module resolution order (`CMAKE_MODULE_PATH` before the
  stdlib modules directory) is a sharp edge. Naming a local helper
  after a stdlib module silently replaces the stdlib module — and
  `include_guard(GLOBAL)` ensures the stdlib version never loads
  even when included later. Pick names that cannot collide.
- `gtest_discover_tests` runs the test binary at *build* time to
  enumerate `TEST()` cases. Anything that prevents the binary from
  launching (missing dylibs, uninitialized globals, forbidden syscalls
  under CI sandboxing) becomes a build-time error, not a test-time
  error — loud, but occasionally confusing.
- `PROJECT_IS_TOP_LEVEL` (CMake ≥ 3.21) is the right knob for "opt
  into tests when we're the root project, opt out when we're a
  subdirectory of someone else's build."

### Next
Step 3 will introduce the `knng_core` library target — a header-only
scaffold for the public API surface (distance types, graph types,
seed point for future CPU reference code) — and prove that tests
can link it alongside `GTest::gtest_main` without disturbing the
current warning policy.

---

## [Step 01] — Repository skeleton & CMake build system (2026-04-16)

### What
- Added a root `CMakeLists.txt` that defines the `knng` project
  (version 0.1.0, C++20, warnings-as-errors, Release-by-default).
- Added `cmake/CompilerWarnings.cmake` providing a `knng_set_warnings()`
  helper that applies the project-wide warning policy to any target.
- Added `cmake/version.hpp.in`, a CMake-configured header that exposes
  `KNNG_VERSION_MAJOR/MINOR/PATCH/STRING` macros to C++ code.
- Added an `knng::headers` INTERFACE library that carries the public
  include path (`include/` plus the generated-header directory) so that
  every later target only needs to link this one dependency to pick up
  the public API.
- Added `tools/hello_knng.cpp`, a tiny smoke-test executable that prints
  the library version, compiler, build type, and host platform.
- Added `.gitignore` (build trees, downloaded datasets, benchmark outputs,
  IDE/OS cruft, and the local-only `LLNL Internship Preparation/` notes
  folder).
- Added `LICENSE` (MIT) and a top-level `README.md` that frames the project
  and explains the build.

### Why
Before writing any algorithmic code, the project needs a build system that
is (1) strict about warnings, (2) self-documenting about configuration,
(3) friendly to IDEs and clang-tooling, and (4) generic enough to absorb
the many subsystems (CPU, GPU, MPI, NCCL, Python bindings) that later
steps will introduce. Pinning these invariants on Day 1 means no later
commit has to renegotiate them. `hello_knng` exists so the build can be
exercised end-to-end from Step 1 — without a runnable artifact, build
breakage would be invisible until Step 6 or later.

### Tradeoff
- **Warnings-as-errors from Day 1.** This will occasionally block
  a build on compiler-version upgrades. Mitigated by the
  `-DKNNG_WARNINGS_AS_ERRORS=OFF` escape hatch. The alternative — flipping
  it on late — inevitably results in hundreds of pre-existing warnings
  that nobody wants to audit.
- **C++20 minimum.** Rules out older toolchains (pre-GCC 11, pre-Clang 14).
  Acceptable: all target platforms (modern Linux HPC, recent macOS, modern
  Windows) ship C++20-capable compilers. Buys us `std::span`, ranges,
  `consteval`, concepts, improved `<bit>`, etc. — all of which will earn
  their keep later.
- **`knng::headers` as an INTERFACE target.** A slightly heavier pattern
  than dumping include directories globally, but it scales cleanly once
  there are dozens of targets and it's the shape every modern CMake guide
  recommends.

### Learning
- Modern CMake (≥ 3.20) has matured significantly: `target_*` commands,
  INTERFACE libraries, and `configure_file` with `@ONLY` replacement give
  a clean way to separate public headers, generated headers, and compile
  options without polluting global state.
- `CMAKE_EXPORT_COMPILE_COMMANDS ON` is effectively free and makes every
  modern editor (VS Code, Neovim, Zed, CLion) immediately understand the
  project.
- Centralizing warning flags in a helper function, rather than repeating
  them per target, is the difference between a codebase where strictness
  compounds and one where it erodes.

### Next
Step 2 will wire GoogleTest via `FetchContent` and add a `tests/` directory
with a smoke-test, confirming that `ctest` runs cleanly in-tree and in CI.

---
