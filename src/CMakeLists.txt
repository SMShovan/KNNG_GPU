# src/CMakeLists.txt
#
# Library targets:
#   * `knng::core` — INTERFACE library covering the public type
#     scaffold (types, Distance concept, Dataset, Knng, TopK). Header-
#     only by design; consumers link it to inherit include paths and
#     the C++20 requirement.
#   * `knng::cpu`  — STATIC library for CPU-side algorithms whose
#     implementations need real translation units (brute_force from
#     Step 10 onward). Transitively brings in `knng::core` so a
#     downstream consumer that links `knng::cpu` does not also need
#     to spell out `knng::core`.

add_library(knng_core INTERFACE)
target_link_libraries(knng_core INTERFACE knng::headers)
add_library(knng::core ALIAS knng_core)

set(_knng_cpu_sources
    cpu/brute_force.cpp
    cpu/brute_force_omp.cpp
    cpu/brute_force_threads.cpp
    cpu/distance.cpp
    cpu/distance_simd.cpp
    cpu/neighbor_list.cpp
    cpu/nn_descent.cpp
    cpu/numa.cpp
)
if(KNNG_HAVE_BLAS)
    list(APPEND _knng_cpu_sources cpu/brute_force_blas.cpp)
endif()
add_library(knng_cpu STATIC ${_knng_cpu_sources})
target_link_libraries(knng_cpu PUBLIC knng::core)
if(KNNG_HAVE_BLAS)
    target_link_libraries(knng_cpu PUBLIC knng::blas_iface)
endif()
if(KNNG_HAVE_OPENMP)
    target_link_libraries(knng_cpu PUBLIC knng::openmp_iface)
endif()
knng_set_warnings(knng_cpu)
add_library(knng::cpu ALIAS knng_cpu)

# `knng::io` — POSIX-mmap-backed loaders for the `.fvecs` / `.ivecs` /
# `.bvecs` benchmark file formats. Linux + macOS only (no Windows path
# yet); the project's CI matrix does not include Windows.
add_library(knng_io STATIC
    io/fvecs.cpp
)
target_link_libraries(knng_io PUBLIC knng::core)
knng_set_warnings(knng_io)
add_library(knng::io ALIAS knng_io)

# `knng::bench` — benchmark-side primitives shared between the bench
# binaries (`benchmarks/`), the regression suite (Phase 13), and any
# unit tests that need them. Currently:
#   * Step 14: ground-truth cache (compute-once-per-(dataset,k,metric))
#   * Step 15: recall@k computation
# Header-only utilities live alongside the .cpp under `include/knng/bench/`.
add_library(knng_bench STATIC
    bench/ground_truth.cpp
    bench/recall.cpp
    bench/runtime_counters.cpp
)
target_link_libraries(knng_bench PUBLIC knng::cpu)
knng_set_warnings(knng_bench)
add_library(knng::bench ALIAS knng_bench)

# `knng::gpu_ref` — CPU-reference implementations of GPU algorithms.
# Always compiled (no CUDA dependency) so that tests on Mac and CPU-only
# CI can run the correctness oracle paths.  When CUDA is enabled, the
# real GPU kernels in knng::gpu link against this library transitively.
add_library(knng_gpu_ref STATIC
    gpu/distance_cpu_ref.cpp
    gpu/brute_force_cpu_ref.cpp           # Step 48
    gpu/coalesced_distance_cpu_ref.cpp    # Step 52
    gpu/tiled_brute_force_cpu_ref.cpp    # Step 53
    gpu/warp_top_k_cpu_ref.cpp           # Step 54
    gpu/register_tiled_cpu_ref.cpp       # Step 55
    gpu/distance_gemm_cpu_ref.cpp        # Step 56
    gpu/fp16_distance_cpu_ref.cpp        # Step 57: fp16 conversion + distance reference
)
target_link_libraries(knng_gpu_ref PUBLIC knng::core)
knng_set_warnings(knng_gpu_ref)
add_library(knng::gpu_ref ALIAS knng_gpu_ref)
