# knng — Coding Style

This document is the authoritative reference for code conventions in
`knng`. When in doubt it takes precedence over editor defaults,
personal habits, or conventions borrowed from other projects. The
goal is a short document that settles arguments; if a rule is not
here and you cannot derive the right answer from existing code, ask.

---

## 1. C++ standard and dialect

- **C++20 everywhere.** Concepts, ranges, `std::span`, `consteval`,
  improved `<bit>` are all in scope and should be used when they
  genuinely clarify intent.
- No C++23 features. No compiler-specific extensions
  (`CMAKE_CXX_EXTENSIONS` is OFF project-wide).
- CUDA compute capability floor: 70 (Volta). This is a phase-7
  concern — ignore until then.

## 2. Files and extensions

| Kind                     | Extension |
|--------------------------|-----------|
| C++ header               | `.hpp`    |
| C++ implementation       | `.cpp`    |
| CUDA header              | `.cuh`    |
| CUDA implementation      | `.cu`     |
| CMake module             | `.cmake`  |
| Markdown (docs)          | `.md`     |

- Every header starts with `#pragma once`. No historical `#ifndef`
  guards.
- Public headers live under `include/knng/<subsystem>/`. Private
  implementation details stay in `src/<subsystem>/`.
- Public headers are always fetch-able as
  `#include "knng/<subsystem>/<name>.hpp"`. Never relative paths.
- File names are `snake_case` and match the primary type or subject
  they declare (`graph.hpp` declares `knng::Knng`; `distance.hpp`
  declares `knng::Distance` and its built-in metrics).

## 3. Include order

1. **Matching header first**, if this is an implementation file
   (e.g. `foo.cpp` starts with `#include "knng/.../foo.hpp"`).
   This self-test catches the "forgot an include in the header"
   bug immediately.
2. Then, each group on its own, separated by blank lines:
   1. C++ standard library (`<vector>`, `<span>`, …)
   2. Third-party (`<gtest/gtest.h>`, `<cuda_runtime.h>`, …)
   3. Project-local (`"knng/core/types.hpp"`, …)
3. Within a group, alphabetical.

Worked example (`tests/core_test.cpp`):

```cpp
#include <array>
#include <cstddef>
#include <span>

#include <gtest/gtest.h>

#include "knng/core/distance.hpp"
#include "knng/core/graph.hpp"
#include "knng/core/types.hpp"
```

## 4. Naming

| Category                          | Convention            | Example                             |
|-----------------------------------|-----------------------|-------------------------------------|
| Function, variable, namespace     | `snake_case`          | `brute_force_knn`, `knng::cpu`      |
| Composite type (`struct`, `class`)| `PascalCase`          | `Knng`, `Dataset`, `L2Squared`      |
| Scalar type alias                 | `snake_case_t`        | `index_t`, `dim_t`                  |
| Enumerator / constant             | `kConstantCase`       | `DistanceMetric::kL2`, `kMaxNeighbors` |
| Template parameter                | `PascalCase`          | `T`, `Distance`, `IndexT`           |
| Macro                             | `UPPER_SNAKE_CASE`    | `KNNG_VERSION_STRING`, `GPU_CHECK`  |
| File name                         | `snake_case.hpp/.cpp` | `brute_force.cpp`, `top_k.hpp`      |

The scalar-alias-as-`snake_case_t` convention matches `<cstdint>`
(`std::uint32_t`, `std::size_t`) and keeps `knng::index_t` visually
distinct from `knng::Index` — which would otherwise look like a
composite type. Composite types stay `PascalCase` so that "I'm
looking at a type-that-owns-invariants" is still instantly readable.

## 5. Index types

- `knng::index_t = std::uint32_t` — every point index, every
  neighbor ID.
- `knng::dim_t   = std::uint32_t` — every dimensionality.
- Sentinel for "no neighbor yet": `std::numeric_limits<index_t>::max()`.
- Signed/unsigned mixing under `-Wconversion` is the known cost of
  the unsigned choice; resolve at the assignment site with an
  explicit `static_cast<index_t>(...)`, not by loosening warnings.

## 6. Distance API

- The canonical metric interface is the C++20 `knng::Distance`
  concept: callable, `noexcept`, signature
  `float(std::span<const float>, std::span<const float>)`, lower-is-
  better monotone.
- Every algorithm is parameterized on a `Distance`-satisfying functor.
  No virtual dispatch in inner loops; the template bloat is the
  price of admission.
- A runtime `DistanceMetric` enum may appear later purely as a
  CLI/binding dispatcher. It is never the primary API.

## 7. Const-correctness and noexcept

- Everything that is not mutated is `const`. Pass by `const T&` for
  non-trivial types; by value for `std::span`, pointer, enum, and
  scalar types (`int`, `float`, `index_t`, …).
- Member functions that do not mutate `*this` are `const`.
- Free functions with no side effects and no throwing subcalls are
  `noexcept`. Distance functors, accessors, and coordinate-math
  helpers are typical examples.

## 8. Error handling

- **Public API boundary:** validate inputs and throw
  `std::invalid_argument` or `std::out_of_range` with a message that
  names the offending argument.
- **Internal invariants:** `assert(...)`. Asserts are the correct
  tool for "this cannot happen unless the caller lied"; exceptions
  are not control flow.
- **Kernel / BLAS / CUDA errors:** surface through the `GPU_CHECK`
  macro when it lands in Phase 7. Do not silently swallow a CUDA
  error code.

## 9. Performance-critical code

- No dynamic allocation in inner loops. Scratch buffers are owned by
  the caller and sized at setup time.
- No virtual functions in inner loops. See §6.
- Prefer `std::span` over `const std::vector<T>&` at API boundaries
  — it decouples the caller's storage from the callee's view.

## 10. Comments and Doxygen

- **Every public API gets Doxygen.** Required tags: `@brief`,
  `@param` for each parameter, `@return` when non-void,
  `@tparam` for every template parameter. Optional but encouraged:
  `@note` for non-obvious behaviour, `@throws` when applicable.
  The `knng::core` headers are the reference example.
- File-level purpose goes in a `/// @file` comment at the top of
  every public header.
- Private helpers use terse `//` comments, and only when the *why*
  is not obvious. Do not restate what the code does.
- No comment blocks explaining git history — the changelog and
  git log serve that purpose.
- Doxygen warnings (missing `@brief`, unresolved references) are
  surfaced by the `docs` target. Treat them as bugs.

## 11. Warnings

Applied by `knng_set_warnings(<target>)` on every project-owned
target:

- **GCC / Clang / AppleClang:** `-Wall -Wextra -Wpedantic -Wshadow
  -Wconversion -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
  -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
  -Wformat=2 -Wimplicit-fallthrough -Werror`.
- **MSVC:** `/W4 /WX /permissive-`.

A warning in our code is an error. Third-party code (GoogleTest,
cuBLAS headers, …) is not compiled with this policy — do not apply
`knng_set_warnings()` to targets we did not write.

## 12. Testing

- Every algorithmic change lands with a GoogleTest unit test.
  Separate test binaries per logical area: `test_smoke`, `test_core`,
  later `test_brute_force`, etc. See `tests/CMakeLists.txt`.
- Every GPU kernel, when they appear in Phase 7, has a CPU reference
  and an elementwise-equality test on a small input (within fp
  tolerance).
- `ctest --output-on-failure` must pass on a clean tree at every
  commit. CI enforces this on three compilers.

## 13. Git and commit discipline

- One step per commit. Commit subject: `[Step NN] Short title`.
  Subject line ≤ 72 characters.
- Commit body describes the change in prose; the `CHANGELOG.md`
  entry is the place for deep design notes.
- AI-assisted commits include the `Co-Authored-By:` trailer for
  Claude.
- Never amend a published commit. Never force-push `main`.

## 14. CHANGELOG template

`CHANGELOG.md` at the repo root is the primary pedagogical artifact.
Every step gets one section, newest-first:

```markdown
## [Step NN] — Short title (YYYY-MM-DD)

### What
Concrete, bullet-pointed changes. File paths, type names, test
counts. The diff answers "what?" — this section summarises it so
scanners do not have to read the diff.

### Why
The motivation. Aimed at a future reader who has the diff but not
the context. A paragraph or two; longer is fine if the reasoning
is genuinely involved.

### Tradeoff
What was given up. Every design decision has an alternative path
and a reason it was not taken. Name them.

### Learning
Non-obvious gotchas, patterns worth reusing, surprises encountered
during implementation. The pedagogical value of the changelog lives
here.

### Next
One paragraph describing what the next step will do. Serves as a
handoff so the next session can resume without re-deriving context.
```

Rules:

- Newest entry first. Separator between entries: `\n---\n`.
- Dates are ISO (`YYYY-MM-DD`) and reflect the day work was done,
  not the commit date (they will usually match).
- The `### Next` section of Step N should be consistent with the
  `### What` section of Step N+1 when that step lands — if they
  diverge, the divergence is itself worth noting in Step N+1's
  `### Why`.
- If a step pulls forward content originally planned for a later
  step, annotate it explicitly (see Steps 03 and 07 for the
  template).
