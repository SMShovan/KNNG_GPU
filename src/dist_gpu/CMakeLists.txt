# src/dist_gpu/CMakeLists.txt
#
# Phase 12 (Steps 89–100): Distributed multi-node GPU algorithms.
#
# Included by the root CMakeLists.txt only when KNNG_HAVE_MPI is ON
# (same guard as src/dist).
#
# Targets:
#   knng_dist_gpu_ref  — always-compiled CPU simulation.
#                        Links knng::core + knng::mpi_iface + knng::dist.
#                        Enables tests on CPU-only / Mac machines.
#   knng_dist_gpu      — real CUDA+MPI path.
#                        Only built when KNNG_HAVE_CUDA is ON.
#                        Links knng_dist_gpu_ref + knng::gpu + CUDA::cudart.
#
# Sources grow one per step: Step 89 starts with cuda_aware_mpi.cu.
# Later steps append to _dist_gpu_sources and dist_gpu_cpu_ref.cpp.

# ── CPU reference (always compiled when MPI is present) ───────────────────
add_library(knng_dist_gpu_ref STATIC
    dist_gpu_cpu_ref.cpp
    topology_dist.cpp            # Step 90: hierarchical topology (pure MPI)
    replication_policy.cpp       # Steps 93 & 95: dedup + BW-aware policy (CPU ref)
)
target_link_libraries(knng_dist_gpu_ref
    PUBLIC
        knng::core
        knng::mpi_iface
        knng::dist
)
# Suppress CPU stubs in dist_gpu_cpu_ref.cpp when the real CUDA path
# provides the same symbols from *.cu files in knng_dist_gpu.
if(KNNG_HAVE_CUDA)
    target_compile_definitions(knng_dist_gpu_ref PRIVATE KNNG_DIST_GPU_CUDA_BUILD=1)
endif()
knng_set_warnings(knng_dist_gpu_ref)
add_library(knng::dist_gpu_ref ALIAS knng_dist_gpu_ref)

# ── Real CUDA+MPI path ────────────────────────────────────────────────────
if(KNNG_HAVE_CUDA)
    include(FindKnngCudaAwareMPI)

    set(_dist_gpu_sources
        cuda_aware_mpi.cu        # Step 89: device-buffer MPI wrappers
        brute_force_dist.cu      # Step 91: ring-based distributed BF on GPU
        nn_descent_dist.cu       # Steps 92-96: distributed NND + NEO-DNND opts
        request_dedup_gpu.cu     # Step 93: GPU-side bitset dedup + CUB prefix-sum
    )

    add_library(knng_dist_gpu STATIC ${_dist_gpu_sources})
    target_link_libraries(knng_dist_gpu
        PUBLIC
            knng_dist_gpu_ref
            knng::gpu
            CUDA::cudart
            knng::mpi_iface
    )
    if(KNNG_HAVE_NCCL)
        target_link_libraries(knng_dist_gpu PUBLIC knng::nccl_iface)
    endif()
    if(KNNG_HAVE_CUDA_AWARE_MPI)
        target_link_libraries(knng_dist_gpu PUBLIC knng::cuda_aware_mpi_iface)
    endif()
    set_target_properties(knng_dist_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES         "${CMAKE_CUDA_ARCHITECTURES}"
    )
    knng_set_warnings(knng_dist_gpu)
    add_library(knng::dist_gpu ALIAS knng_dist_gpu)
endif()
