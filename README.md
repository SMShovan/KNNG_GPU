# knng

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

Very early. This repository currently contains only the build scaffolding
and a `hello_knng` smoke-test executable. See the changelog for the
up-to-date step count.

## Design goals

- **Correctness first.** Every algorithmic change is paired with unit tests
  and recall@k measurements against a known ground truth.
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
```

`hello_knng` prints the library version and the compile-time build
configuration; use it as a smoke test after changing the build system.

### Configure-time options

| Option                        | Default | Effect                                      |
|-------------------------------|---------|---------------------------------------------|
| `KNNG_WARNINGS_AS_ERRORS`     | `ON`    | Treat compiler warnings as errors           |

More options will be added as later steps introduce CUDA, MPI, NCCL, etc.

## License

MIT — see [`LICENSE`](LICENSE).
