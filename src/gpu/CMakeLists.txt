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
    distance_kernel.cu      # Step 47: naive pairwise L2² kernel
    brute_force_thread.cu   # Step 48: thread-per-query KNN kernel
    brute_force_block.cu    # Steps 49 & 50: block-per-query KNN + end-to-end
    coalesced_distance.cu   # Step 52: column-major reference layout
    tiled_brute_force.cu    # Step 53: shared-memory reference tiling
    warp_brute_force.cu     # Step 54: warp-shuffle top-k
    register_tiled_bf.cu    # Step 55: query-side register tiling
    distance_gemm.cu        # Step 56: cuBLAS GEMM distance
    fp16_distance.cu        # Step 57: fp16 storage, fp32 compute
    wmma_distance.cu        # Step 58: Tensor Core (WMMA) GEMM (CC≥70)
    streamed_brute_force.cu # Step 59: stream-pipelined BF KNN
    gpu_nn_descent.cu       # Steps 63-70: GPU NN-Descent kernels
    graph_refinement.cu     # Steps 72-77: CAGRA-style graph refinement
    nccl_comm.cu            # Step 80: NcclComm RAII wrapper (NCCL-gated)
    topology_gpu.cu         # Step 81: Topology::probe() — real CUDA P2P query
    pipeline.cu             # Step 84: GpuTriplePipeline — 3-stream H2D/compute/D2H
    brute_force_multi.cu    # Steps 82-83: GPU point-sharded + tile-sharded BF
    nn_descent_multi.cu     # Step 85: GPU multi-GPU NN-Descent + AllGather
)

add_library(knng_gpu STATIC ${_knng_gpu_sources})

target_link_libraries(knng_gpu
    PUBLIC
        knng::core
        knng::gpu_ref       # CPU reference implementations (always compiled)
        knng::cuda_iface
        CUDA::cudart
        CUDA::cublas        # Step 56: cuBLAS GEMM distance
)

# NCCL: link the real NCCL library when it was found, otherwise the
# KNNG_HAVE_NCCL guard in each .cu file keeps those paths dead.
if(KNNG_HAVE_NCCL)
    target_link_libraries(knng_gpu PUBLIC knng::nccl_iface)
endif()

set_target_properties(knng_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         "${CMAKE_CUDA_ARCHITECTURES}"
)

# Apply the project warning policy to non-CUDA translation units;
# CUDA files are exempted from -Wconversion / -Wpedantic because nvcc
# does not support all GCC/Clang diagnostic flags.
knng_set_warnings(knng_gpu)

add_library(knng::gpu ALIAS knng_gpu)
