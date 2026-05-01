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
