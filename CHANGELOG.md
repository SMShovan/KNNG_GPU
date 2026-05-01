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
