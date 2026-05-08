#pragma once

/// @file
/// @brief HIP-portable GPU backend macro layer.
///
/// Every translation unit that calls GPU primitives (malloc, memcpy,
/// kernel launch) includes this header rather than `<cuda_runtime.h>`
/// or `<hip/hip_runtime.h>` directly.  All CUDA API names are hidden
/// behind thin macros so that the hipify step in Phase 8 is mechanical:
/// replace `KNNG_HAVE_CUDA` with `KNNG_HAVE_HIP` and regenerate — no
/// source changes needed.
///
/// When neither backend is present (Mac dev machine, CPU-only CI), all
/// macros reduce to no-ops or host-side equivalents so that the
/// remaining C++ code still compiles and the CPU reference paths in
/// unit tests remain exercisable.
///
/// ## Macro families
///
/// | Macro                          | Purpose                                |
/// |-------------------------------|----------------------------------------|
/// | `GPU_CHECK(call)`             | Assert `call` returns success; throw   |
/// |                                | `std::runtime_error` on failure.       |
/// | `GPU_MALLOC(ptr, bytes)`      | Device-side allocation.                |
/// | `GPU_FREE(ptr)`               | Device-side free.                      |
/// | `GPU_MEMCPY_H2D(dst,src,n)`   | Host → device copy of `n` bytes.       |
/// | `GPU_MEMCPY_D2H(dst,src,n)`   | Device → host copy of `n` bytes.       |
/// | `GPU_MEMCPY_D2D(dst,src,n)`   | Device → device copy of `n` bytes.     |
/// | `GPU_SYNC()`                  | Synchronise the default stream.        |
/// | `GPU_STREAM_CREATE(s)`        | Create a stream handle `s`.            |
/// | `GPU_STREAM_DESTROY(s)`       | Destroy stream `s`.                    |
/// | `GPU_LAUNCH(fn,g,b,sh,st,...)`| Launch kernel `fn<<<g,b,sh,st>>>(…)`. |
/// | `GPU_DEVICE`                  | `__device__` (empty on CPU stub).      |
/// | `GPU_GLOBAL`                  | `__global__` (empty on CPU stub).      |
/// | `GPU_HOST`                    | `__host__`   (empty on CPU stub).      |
/// | `GPU_HOST_DEVICE`             | `__host__ __device__` qualifier.       |
/// | `GPU_SHARED`                  | `__shared__` (empty on CPU stub).      |
/// | `GPU_INLINE`                  | `__forceinline__` under CUDA/HIP,      |
/// |                                | `inline` otherwise.                    |
///
/// ## Error type
///
/// `gpu_error_t` aliases `cudaError_t` under CUDA, `hipError_t` under
/// HIP, and `int` on the CPU stub (0 == success).

#include <cstring>    // memcpy
#include <cstdlib>    // malloc/free (stub path)
#include <stdexcept>  // std::runtime_error
#include <string>

// =========================================================================
// CUDA path
// =========================================================================
#if defined(KNNG_HAVE_CUDA) && KNNG_HAVE_CUDA

#include <cuda_runtime.h>

namespace knng::gpu {
using stream_t   = cudaStream_t;
using error_t    = cudaError_t;
constexpr error_t kSuccess = cudaSuccess;
} // namespace knng::gpu

/// @cond INTERNAL
inline void knng_gpu_check_impl(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " CUDA error: " + cudaGetErrorString(err));
    }
}
/// @endcond

#define GPU_CHECK(call)   knng_gpu_check_impl((call), __FILE__, __LINE__)

#define GPU_MALLOC(ptr, bytes)        GPU_CHECK(cudaMalloc(&(ptr), (bytes)))
#define GPU_FREE(ptr)                 GPU_CHECK(cudaFree(ptr))
#define GPU_MEMCPY_H2D(dst,src,n)    GPU_CHECK(cudaMemcpy((dst),(src),(n),cudaMemcpyHostToDevice))
#define GPU_MEMCPY_D2H(dst,src,n)    GPU_CHECK(cudaMemcpy((dst),(src),(n),cudaMemcpyDeviceToHost))
#define GPU_MEMCPY_D2D(dst,src,n)    GPU_CHECK(cudaMemcpy((dst),(src),(n),cudaMemcpyDeviceToDevice))
#define GPU_SYNC()                    GPU_CHECK(cudaDeviceSynchronize())
#define GPU_STREAM_CREATE(s)          GPU_CHECK(cudaStreamCreate(&(s)))
#define GPU_STREAM_DESTROY(s)         GPU_CHECK(cudaStreamDestroy(s))

#define GPU_LAUNCH(fn, grid, block, shmem, stream, ...)  \
    fn<<<(grid),(block),(shmem),(stream)>>>(__VA_ARGS__); \
    GPU_CHECK(cudaGetLastError())

#define GPU_DEVICE       __device__
#define GPU_GLOBAL       __global__
#define GPU_HOST         __host__
#define GPU_HOST_DEVICE  __host__ __device__
#define GPU_SHARED       __shared__
#define GPU_INLINE       __forceinline__

// =========================================================================
// HIP path
// =========================================================================
#elif defined(KNNG_HAVE_HIP) && KNNG_HAVE_HIP

#include <hip/hip_runtime.h>

namespace knng::gpu {
using stream_t   = hipStream_t;
using error_t    = hipError_t;
constexpr error_t kSuccess = hipSuccess;
} // namespace knng::gpu

inline void knng_gpu_check_impl(hipError_t err, const char* file, int line)
{
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " HIP error: " + hipGetErrorString(err));
    }
}

#define GPU_CHECK(call)   knng_gpu_check_impl((call), __FILE__, __LINE__)

#define GPU_MALLOC(ptr, bytes)        GPU_CHECK(hipMalloc(&(ptr), (bytes)))
#define GPU_FREE(ptr)                 GPU_CHECK(hipFree(ptr))
#define GPU_MEMCPY_H2D(dst,src,n)    GPU_CHECK(hipMemcpy((dst),(src),(n),hipMemcpyHostToDevice))
#define GPU_MEMCPY_D2H(dst,src,n)    GPU_CHECK(hipMemcpy((dst),(src),(n),hipMemcpyDeviceToHost))
#define GPU_MEMCPY_D2D(dst,src,n)    GPU_CHECK(hipMemcpy((dst),(src),(n),hipMemcpyDeviceToDevice))
#define GPU_SYNC()                    GPU_CHECK(hipDeviceSynchronize())
#define GPU_STREAM_CREATE(s)          GPU_CHECK(hipStreamCreate(&(s)))
#define GPU_STREAM_DESTROY(s)         GPU_CHECK(hipStreamDestroy(s))

#define GPU_LAUNCH(fn, grid, block, shmem, stream, ...)  \
    hipLaunchKernelGGL((fn),(grid),(block),(shmem),(stream),__VA_ARGS__); \
    GPU_CHECK(hipGetLastError())

#define GPU_DEVICE       __device__
#define GPU_GLOBAL       __global__
#define GPU_HOST         __host__
#define GPU_HOST_DEVICE  __host__ __device__
#define GPU_SHARED       __shared__
#define GPU_INLINE       __forceinline__

// =========================================================================
// CPU stub path — no GPU runtime available
// =========================================================================
#else

namespace knng::gpu {
using stream_t = int;   // opaque handle placeholder
using error_t  = int;   // 0 == success
constexpr error_t kSuccess = 0;
} // namespace knng::gpu

inline void knng_gpu_check_impl(int err, const char* file, int line)
{
    if (err != 0) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " GPU stub error: " + std::to_string(err));
    }
}

#define GPU_CHECK(call)   knng_gpu_check_impl((call), __FILE__, __LINE__)

// Stubs allocate on the host heap so that DeviceBuffer<T> and other
// RAII wrappers are testable on machines without a GPU.
#define GPU_MALLOC(ptr, bytes)        ((ptr) = static_cast<decltype(ptr)>(std::malloc(bytes)), (ptr) ? void() : throw std::bad_alloc())
#define GPU_FREE(ptr)                 (std::free(ptr))
#define GPU_MEMCPY_H2D(dst,src,n)    (std::memcpy((dst),(src),(n)))
#define GPU_MEMCPY_D2H(dst,src,n)    (std::memcpy((dst),(src),(n)))
#define GPU_MEMCPY_D2D(dst,src,n)    (std::memcpy((dst),(src),(n)))
#define GPU_SYNC()                    ((void)0)
#define GPU_STREAM_CREATE(s)          ((s) = 0)
#define GPU_STREAM_DESTROY(s)         ((void)(s))

// Kernel-launch stub: call as a plain function with no grid/block
// arguments — only valid for CPU-reference kernel equivalents.
#define GPU_LAUNCH(fn, grid, block, shmem, stream, ...)  fn(__VA_ARGS__)

#define GPU_DEVICE
#define GPU_GLOBAL
#define GPU_HOST
#define GPU_HOST_DEVICE
#define GPU_SHARED       static   // nearest equivalent in C++ file scope
#define GPU_INLINE       inline

#endif // backend selection
