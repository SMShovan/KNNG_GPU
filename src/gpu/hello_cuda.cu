/// @file
/// @brief Step 44 — CUDA smoke-test kernel.
///
/// Prints the block and thread ID for each launched thread.  This
/// kernel has no algorithmic content; it exists solely to prove that
/// the CUDA compiler, linker, and runtime wiring are correct end-to-end.
/// Run `build/bin/hello_cuda` on a GPU machine to see the output.

#include <cstdio>

namespace knng::gpu {

/// @brief Hello-world kernel: each thread prints its location in the grid.
__global__ void hello_kernel()
{
    printf("hello from block (%d,%d,%d) thread (%d,%d,%d)\n",
           blockIdx.x,  blockIdx.y,  blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

/// @brief Host launcher: run hello_kernel on a 2-block × 4-thread grid.
///
/// @return  cudaSuccess, or the first CUDA error encountered.
cudaError_t launch_hello()
{
    hello_kernel<<<2, 4>>>();
    return cudaDeviceSynchronize();
}

} // namespace knng::gpu
