# knng

[![CI](https://github.com/SMShovan/KNNG_GPU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SMShovan/KNNG_GPU/actions/workflows/ci.yml)

A GPU-accelerated framework for building **approximate K-nearest neighbor
graphs (KNNG)** over large, high-dimensional vector datasets.

The codebase is being developed as a long-running, incremental study of how
a state-of-the-art ANN graph builder actually gets assembled — beginning with
a single-threaded CPU reference implementation, then climbing through
classical CPU optimizations, single-GPU kernels, multi-GPU scaling, and
finally a distributed multi-node implementation.

Every commit is one small, defensible step. The full changelog lives in
[`CHANGELOG.md`](CHANGELOG.md) and is the primary place to read about the
design decisions behind each step.

## Status

Very early. This repository currently contains the build scaffolding, a
`hello_knng` smoke-test executable, a GoogleTest-based `ctest` suite, and
the `knng::core` public API scaffold — scalar/index types, a `Distance`
concept with `L2Squared` + `NegativeInnerProduct` metrics, and a flat
`Knng` adjacency struct. No algorithms yet. See the changelog for the
up-to-date step count.

## Design goals

- **Correctness first.** Every algorithmic change is paired with unit tests
  and recall-at-k measurements against a known ground truth.
- **Measured optimization.** Every optimization commit states the expected
  speedup, the observed speedup, and the tradeoff (memory, complexity,
  numeric precision) in the changelog.
- **Portable across vendors.** CUDA is the primary development target, but
  GPU code is written through a thin backend shim (`include/knng/gpu/backend.hpp`)
  so that a HIP / ROCm port is mechanical rather than architectural.
- **Reproducible benchmarks.** Standard ANN benchmark datasets (SIFT1M,
  GIST1M, Fashion-MNIST) with deterministic seeds and JSON benchmark output.

## Layout

```
include/knng/        public headers (stable API surface)
src/                 implementation
tests/               GoogleTest unit + integration tests
benchmarks/          Google Benchmark programs + scripts
tools/               CLI utilities (dataset download, graph build, ...)
docs/                Doxygen config + long-form design notes
cmake/               CMake modules (warnings, toolchains, find-modules)
third_party/         vendored or FetchContent'd dependencies
datasets/            gitignored — populated by tools/download_*.sh
```

## Build

Requires CMake ≥ 3.20 and a C++20 compiler (GCC 11+, Clang 14+, or
AppleClang 14+).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/hello_knng
ctest --test-dir build --output-on-failure
cmake --build build --target docs    # optional, requires Doxygen
```

`hello_knng` prints the library version and the compile-time build
configuration; use it as a smoke test after changing the build system.
`ctest` runs the GoogleTest suite covering the public API
(`tests/core_test.cpp`), the bounded top-k buffer (`tests/top_k_test.cpp`),
the CPU brute-force builder (`tests/brute_force_test.cpp`), and the
`.fvecs` / `.ivecs` / `.bvecs` loaders (`tests/fvecs_test.cpp`).

### End-to-end CLI

`build_knng` is the user-facing tool: load a `.fvecs` dataset, run
brute-force CPU KNN, write the resulting graph to a binary file
documented at the top of `tools/build_knng.cpp`.

```sh
./tools/download_sift.sh          # one-time SIFT-small fetch
./build/bin/build_knng \
    --dataset datasets/siftsmall/siftsmall_base.fvecs \
    --k 10 \
    --metric l2 \
    --output siftsmall_k10.knng
```

### Configure-time options

| Option                        | Default                 | Effect                                                                         |
|-------------------------------|-------------------------|--------------------------------------------------------------------------------|
| `KNNG_WARNINGS_AS_ERRORS`     | `ON`                    | Treat compiler warnings as errors                                              |
| `KNNG_BUILD_TESTS`            | `ON` (top-level build)  | Build the GoogleTest suite and fetch GoogleTest if needed                      |
| `KNNG_GOOGLETEST_TAG`         | `v1.15.2`               | Git tag used by `FetchContent` for GoogleTest                                  |
| `KNNG_BUILD_BENCHMARKS`       | `OFF`                   | Build the Google-Benchmark-driven benchmark suite (fetches Google Benchmark)   |
| `KNNG_GOOGLEBENCHMARK_TAG`    | `v1.9.1`                | Git tag used by `FetchContent` for Google Benchmark                            |

More options will be added as later steps introduce CUDA, MPI, NCCL, etc.

### Benchmarks

The benchmark suite is opt-in:

```sh
cmake -B build -DKNNG_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_brute_force
./build/bin/bench_brute_force --benchmark_format=json
```

`bench_brute_force` runs `knng::cpu::brute_force_knn` over a deterministic
synthetic dataset across several `(n, d)` sizes. Set
`KNNG_BENCH_FVECS=path/to/file.fvecs` to also benchmark over a real
on-disk dataset (see `tools/download_sift.sh` for SIFT-small).

### Documentation

If [Doxygen](https://www.doxygen.nl/) is installed, the `docs` target
generates the full API reference as HTML:

```sh
cmake --build build --target docs
open build/docs/html/index.html   # macOS; use xdg-open on Linux
```

The `docs` target is only created when Doxygen is detected at configure
time — a missing Doxygen is a warning, not a configure failure. The
generated HTML is never produced by a default `cmake --build`; it must
be requested explicitly.

## Contributing

Project conventions — naming, include order, comment style, warning
policy, test expectations, the `CHANGELOG.md` template — are
documented in [`docs/STYLE.md`](docs/STYLE.md). Read it before sending
a patch.

## License

MIT — see [`LICENSE`](LICENSE).
