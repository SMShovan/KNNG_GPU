# Changelog

All notable changes to this project are documented here, one entry per
development step. Entries are ordered newest-first and follow the structure:

```
## [Step NN] — Short title (YYYY-MM-DD)
### What
### Why
### Tradeoff
### Learning
### Next
```

The goal of this document is pedagogical: each entry should make the *why*
of the change obvious to a reader scanning the history, independent of the
code diff.

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
