# HIP Portability Layer

> **Status (Step 60):** The `backend.hpp` macro layer and `cmake/FindKnngHIP.cmake`
> are in place.  Actual HIP compilation and benchmarking happen when the project
> is pushed to an AMD GPU system (MI250X / MI350A target).

## Rationale

All GPU source files in `src/gpu/` include only `<knng/gpu/backend.hpp>` —
never `<cuda_runtime.h>` or `<hip/hip_runtime.h>` directly.  The macro layer
maps `GPU_CHECK`, `GPU_MALLOC`, `GPU_MEMCPY_*`, `GPU_LAUNCH`, etc. to their
CUDA or HIP equivalents at compile time.  This makes the hipify step
mechanical rather than architectural.

## Backend selection

```bash
# CUDA (NVIDIA):
cmake -S . -B build-cuda -DKNNG_BACKEND=CUDA
cmake --build build-cuda

# HIP (AMD ROCm):
cmake -S . -B build-hip -DKNNG_BACKEND=HIP
cmake --build build-hip

# CPU-only (Mac, CI):
cmake -S . -B build
cmake --build build
```

## CUDA → HIP macro equivalences

| `backend.hpp` macro        | CUDA expansion           | HIP expansion               |
|----------------------------|--------------------------|-----------------------------|
| `GPU_CHECK(call)`          | `knng_gpu_check_impl`    | `knng_gpu_check_impl`       |
| `GPU_MALLOC(ptr, bytes)`   | `cudaMalloc`             | `hipMalloc`                 |
| `GPU_FREE(ptr)`            | `cudaFree`               | `hipFree`                   |
| `GPU_MEMCPY_H2D`           | `cudaMemcpy(…, H→D)`    | `hipMemcpy(…, H→D)`        |
| `GPU_MEMCPY_D2H`           | `cudaMemcpy(…, D→H)`    | `hipMemcpy(…, D→H)`        |
| `GPU_MEMCPY_D2D`           | `cudaMemcpy(…, D→D)`    | `hipMemcpy(…, D→D)`        |
| `GPU_SYNC()`               | `cudaDeviceSynchronize`  | `hipDeviceSynchronize`      |
| `GPU_STREAM_CREATE(s)`     | `cudaStreamCreate`       | `hipStreamCreate`           |
| `GPU_STREAM_DESTROY(s)`    | `cudaStreamDestroy`      | `hipStreamDestroy`          |
| `GPU_LAUNCH(fn,g,b,sh,st)` | `fn<<<g,b,sh,st>>>(…)`  | `hipLaunchKernelGGL(fn,…)` |
| `GPU_DEVICE`               | `__device__`             | `__device__`                |
| `GPU_GLOBAL`               | `__global__`             | `__global__`                |
| `GPU_HOST`                 | `__host__`               | `__host__`                  |
| `GPU_HOST_DEVICE`          | `__host__ __device__`    | `__host__ __device__`       |
| `GPU_SHARED`               | `__shared__`             | `__shared__`                |
| `GPU_INLINE`               | `__forceinline__`        | `__forceinline__`           |

## Library equivalences (deferred to Phase 12)

| CUDA library   | HIP / ROCm equivalent       | Step used        |
|----------------|-----------------------------|------------------|
| `cublas`       | `hipBLAS` / `rocBLAS`      | Step 56          |
| `cuda_fp16.h`  | `hip_fp16.h`                | Step 57          |
| `mma.h` (WMMA) | `rocwmma/rocwmma.hpp`       | Step 58          |
| `cudaFreeHost` | `hipFreeHost`               | Step 59          |
| NCCL           | RCCL                        | Phase 11         |

## hipify-perl

As a smoke-test, `hipify-perl` can mechanically translate the CUDA-specific
source files:

```bash
cd src/gpu
for f in *.cu; do
  hipify-perl "$f" > "../../src/gpu_hip/${f%.cu}.cpp"
done
```

The expected residual differences after hipify:
1. `cublas_v2.h` → `hipblas.h` (Step 56)
2. `cuda_fp16.h` → `hip_fp16.h` (Step 57)
3. `mma.h` / `nvcuda::wmma` → `rocwmma/rocwmma.hpp` / `rocwmma::` (Step 58)
4. `cudaMallocHost` / `cudaFreeHost` → `hipMallocHost` / `hipFreeHost` (Step 59)
5. `cudaStreamSynchronize` → `hipStreamSynchronize` (Step 59)

All other CUDA API calls in `src/gpu/` go through `backend.hpp` macros and
translate automatically.

## MI350A target notes

The primary final target is the AMD MI350A APU.  MI350A is GFX942 (CDNA3
architecture), which supports:
- HIP 6.x
- ROCm 6.x
- rocBLAS GEMM (equivalent to cuBLAS)
- rocWMMA 16×16 tiles on Matrix Cores (equivalent to WMMA Tensor Cores)
- RCCL collectives (equivalent to NCCL)

The CMake `-DKNNG_BACKEND=HIP` path is designed so that a single rebuild
on MI350A produces a fully functional binary with no source modifications.
