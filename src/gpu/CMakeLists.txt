# src/gpu/CMakeLists.txt
#
# GPU library targets — only included by the parent when KNNG_HAVE_CUDA is ON.
#
# Targets defined here:
#   knng::gpu  — STATIC library accumulating all GPU translation units.
#                Links knng::core transitively; callers that add later
#                .cu files simply append to _knng_gpu_sources.

set(_knng_gpu_sources
    hello_cuda.cu
    distance_kernel.cu    # Step 47: naive pairwise L2² kernel
)

add_library(knng_gpu STATIC ${_knng_gpu_sources})

target_link_libraries(knng_gpu
    PUBLIC
        knng::core
        knng::gpu_ref       # CPU reference implementations (always compiled)
        knng::cuda_iface
        CUDA::cudart
)

set_target_properties(knng_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         "${CMAKE_CUDA_ARCHITECTURES}"
)

# Apply the project warning policy to non-CUDA translation units;
# CUDA files are exempted from -Wconversion / -Wpedantic because nvcc
# does not support all GCC/Clang diagnostic flags.
knng_set_warnings(knng_gpu)

add_library(knng::gpu ALIAS knng_gpu)
