/// @file
/// @brief Step 58 — Tensor Core WMMA pairwise squared-L2 distance.
///
/// Uses the WMMA C++ API (`<mma.h>`) to compute `-2 Q Rᵀ` via 16×16 fp16
/// matrix tiles.  Requires compute capability ≥ 70 (Volta+).
///
/// The WMMA fragment API:
///   - `wmma::fragment<matrix_a, 16,16,16, half, row_major>` — 16×16 query tile.
///   - `wmma::fragment<matrix_b, 16,16,16, half, col_major>` — 16×16 ref tile.
///   - `wmma::fragment<accumulator, 16,16,16, float>`         — fp32 result.
///   - `wmma::mma_sync(acc, a, b, acc)` — fused multiply-accumulate.
///
/// The 16×16 tile size is the only size supported on Volta; Ampere adds
/// 32×8 and 8×32.  We use 16×16 for maximum compatibility.

#include <knng/gpu/wmma_distance.hpp>
#include <knng/gpu/distance_gemm.hpp>   // norm kernels reused from Step 56
#include <knng/gpu/device_buffer.hpp>
#include <knng/gpu/fp16_distance.hpp>   // fp32_to_fp16_kernel

#include <cuda_fp16.h>
#include <mma.h>

#include <stdexcept>

namespace knng::gpu {

using namespace nvcuda;

static constexpr unsigned int kWmmaTile = 16u;

/// @brief WMMA GEMM kernel: C[nq,nr] = -2 * Q[nq,d] * R^T[d,nr] (fp16×fp16→fp32).
///
/// Grid: (ceil(nr/16), ceil(nq/16)) blocks; each block computes one 16×16 tile of C.
/// Block: one warp (32 threads) — the WMMA API is warp-granular.
///
/// @note nq, nr, d must all be multiples of kWmmaTile=16.
GPU_GLOBAL void wmma_gemm_kernel(
    const __half* d_Q_f16,
    const __half* d_R_f16,
    unsigned int  nq,
    unsigned int  nr,
    unsigned int  d,
    float*        d_out)   // C = -2 * Q * R^T
{
#if __CUDA_ARCH__ >= 700
    // Each warp handles one 16×16 output tile.
    const unsigned int warp_qi = (blockIdx.y * blockDim.y + threadIdx.y) * kWmmaTile;
    const unsigned int warp_ri = (blockIdx.x * blockDim.x + threadIdx.x / 32u) * kWmmaTile;

    if (warp_qi >= nq || warp_ri >= nr) { return; }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.f);

    // Accumulate over k-tiles.
    for (unsigned int k = 0; k < d; k += kWmmaTile) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

        // Load Q tile [warp_qi:warp_qi+16, k:k+16]
        wmma::load_matrix_sync(a_frag,
            d_Q_f16 + warp_qi * d + k,
            d);

        // Load R tile [warp_ri:warp_ri+16, k:k+16]  (col_major → Rᵀ row-major)
        wmma::load_matrix_sync(b_frag,
            d_R_f16 + warp_ri * d + k,
            d);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Multiply by -2 and store.
    for (unsigned int i = 0; i < acc_frag.num_elements; ++i) {
        acc_frag.x[i] *= -2.f;
    }
    wmma::store_matrix_sync(
        d_out + warp_qi * nr + warp_ri,
        acc_frag,
        nr,
        wmma::mem_row_major);
#else
    // Fallback for older compute capabilities: should not be called at runtime.
    (void)d_Q_f16; (void)d_R_f16; (void)nq; (void)nr; (void)d; (void)d_out;
#endif
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

std::vector<float> gpu_wmma_distance(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d)
{
    if (nr % kWmmaTile != 0 || d % kWmmaTile != 0) {
        throw std::invalid_argument(
            "gpu_wmma_distance: nr and d must be multiples of 16");
    }

    const auto nq_ = static_cast<unsigned int>(nq);
    const auto nr_ = static_cast<unsigned int>(nr);
    const auto d_  = static_cast<unsigned int>(d);

    // fp32 → device fp32 → fp16 conversion.
    DeviceBuffer<float> d_q32(nq * d), d_r32(nr * d);
    d_q32.copy_from_host(queries, nq * d);
    d_r32.copy_from_host(refs,    nr * d);

    DeviceBuffer<__half> d_q16(nq * d), d_r16(nr * d);
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

    // Norms.
    DeviceBuffer<float> d_q_norms(nq), d_r_norms(nr);
    {
        constexpr unsigned int kB = 256u;
        GPU_LAUNCH(compute_norms_kernel, (nq_+kB-1u)/kB, kB, 0, nullptr,
                   d_q32.data(), nq_, d_, d_q_norms.data());
        GPU_LAUNCH(compute_norms_kernel, (nr_+kB-1u)/kB, kB, 0, nullptr,
                   d_r32.data(), nr_, d_, d_r_norms.data());
    }

    // WMMA GEMM: one warp per 16×16 output tile.
    // blockDim: (32, 1) = one warp; gridDim: (nr/16, nq/16)
    {
        dim3 block(32u, 1u);
        dim3 grid(nr_ / kWmmaTile, nq_ / kWmmaTile);
        GPU_LAUNCH(wmma_gemm_kernel, grid, block, 0, nullptr,
                   d_q16.data(), d_r16.data(), nq_, nr_, d_, d_out.data());
    }

    // Epilogue: add norms.
    {
        constexpr unsigned int kB = 16u;
        dim3 block(kB, kB);
        dim3 grid((nr_+kB-1u)/kB, (nq_+kB-1u)/kB);
        GPU_LAUNCH(norm_epilogue_kernel, grid, block, 0, nullptr,
                   d_out.data(), d_q_norms.data(), d_r_norms.data(), nq_, nr_);
    }

    GPU_SYNC();
    return d_out.to_host();
}

} // namespace knng::gpu
