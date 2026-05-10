/// @file
/// @brief Step 57 — fp16-storage pairwise squared-L2 GPU kernel.
///
/// Convert fp32 dataset to __half on device; compute distances as fp32.

#include <knng/gpu/fp16_distance.hpp>
#include <knng/gpu/device_buffer.hpp>

#include <cuda_fp16.h>

namespace knng::gpu {

GPU_GLOBAL void fp32_to_fp16_kernel(
    const float* d_src,
    unsigned int n,
    unsigned int d,
    __half*      d_dst)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * d) {
        d_dst[i] = __float2half(d_src[i]);
    }
}

GPU_GLOBAL void fp16_pairwise_l2sq_kernel(
    const __half* d_queries_f16,
    const __half* d_refs_f16,
    unsigned int  nq,
    unsigned int  nr,
    unsigned int  d,
    float*        d_out)
{
    const unsigned int qi = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq || ri >= nr) { return; }

    float acc = 0.f;
    for (unsigned int k = 0; k < d; ++k) {
        const float qv = __half2float(d_queries_f16[qi * d + k]);
        const float rv = __half2float(d_refs_f16   [ri * d + k]);
        const float diff = qv - rv;
        acc += diff * diff;
    }
    d_out[qi * nr + ri] = acc;
}

std::vector<float> gpu_fp16_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d)
{
    // Allocate fp32 device buffers.
    DeviceBuffer<float> d_q32(nq * d);
    DeviceBuffer<float> d_r32(nr * d);
    d_q32.copy_from_host(queries, nq * d);
    d_r32.copy_from_host(refs,    nr * d);

    // Convert to fp16.
    // DeviceBuffer<__half> uses GPU_MALLOC which calls cudaMalloc internally.
    // __half is trivially copyable — safe to use with DeviceBuffer.
    DeviceBuffer<__half> d_q16(nq * d);
    DeviceBuffer<__half> d_r16(nr * d);
    {
        constexpr unsigned int kB = 256u;
        const auto nqd = static_cast<unsigned int>(nq * d);
        const auto nrd = static_cast<unsigned int>(nr * d);
        GPU_LAUNCH(fp32_to_fp16_kernel, (nqd+kB-1u)/kB, kB, 0, nullptr,
                   d_q32.data(), 1u, nqd, d_q16.data());
        GPU_LAUNCH(fp32_to_fp16_kernel, (nrd+kB-1u)/kB, kB, 0, nullptr,
                   d_r32.data(), 1u, nrd, d_r16.data());
    }

    DeviceBuffer<float> d_out(nq * nr);
    constexpr unsigned int kB = 16u;
    dim3 block(kB, kB);
    dim3 grid((static_cast<unsigned int>(nr)+kB-1u)/kB,
              (static_cast<unsigned int>(nq)+kB-1u)/kB);
    GPU_LAUNCH(fp16_pairwise_l2sq_kernel, grid, block, 0, nullptr,
               d_q16.data(), d_r16.data(),
               static_cast<unsigned int>(nq),
               static_cast<unsigned int>(nr),
               static_cast<unsigned int>(d),
               d_out.data());
    GPU_SYNC();
    return d_out.to_host();
}

} // namespace knng::gpu
