/// @file
/// @brief Step 56 — cuBLAS GEMM GPU distance computation.
///
/// Distance = ‖Q‖² + ‖R‖² − 2 Q Rᵀ
/// The `-2 Q Rᵀ` term is computed by `cublasSgemm`; norms are computed by
/// `compute_norms_kernel` and added by `norm_epilogue_kernel`.

#include <knng/gpu/distance_gemm.hpp>
#include <knng/gpu/device_buffer.hpp>

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// cuBLAS error check helper
// ---------------------------------------------------------------------------

inline void cublas_check(cublasStatus_t s, const char* file, int line)
{
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " cuBLAS error: " + std::to_string(static_cast<int>(s)));
    }
}
#define CUBLAS_CHECK(c) cublas_check((c), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

GPU_GLOBAL void compute_norms_kernel(
    const float* d_pts,
    unsigned int n,
    unsigned int d,
    float*       d_norms)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    float acc = 0.f;
    for (unsigned int k = 0; k < d; ++k) {
        const float v = d_pts[static_cast<std::size_t>(i) * d + k];
        acc += v * v;
    }
    d_norms[i] = acc;
}

GPU_GLOBAL void norm_epilogue_kernel(
    float*       d_out,
    const float* d_q_norms,
    const float* d_r_norms,
    unsigned int nq,
    unsigned int nr)
{
    const unsigned int qi = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq || ri >= nr) { return; }
    d_out[qi * nr + ri] += d_q_norms[qi] + d_r_norms[ri];
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

std::vector<float> gpu_distance_gemm(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d)
{
    const auto nq_ = static_cast<unsigned int>(nq);
    const auto nr_ = static_cast<unsigned int>(nr);
    const auto d_  = static_cast<unsigned int>(d);

    DeviceBuffer<float> d_queries(nq * d);
    DeviceBuffer<float> d_refs(nr * d);
    DeviceBuffer<float> d_out(nq * nr);
    DeviceBuffer<float> d_q_norms(nq);
    DeviceBuffer<float> d_r_norms(nr);

    d_queries.copy_from_host(queries, nq * d);
    d_refs.copy_from_host(refs, nr * d);

    // Compute norms.
    {
        constexpr unsigned int kB = 256u;
        GPU_LAUNCH(compute_norms_kernel, (nq_ + kB-1u)/kB, kB, 0, nullptr,
                   d_queries.data(), nq_, d_, d_q_norms.data());
        GPU_LAUNCH(compute_norms_kernel, (nr_ + kB-1u)/kB, kB, 0, nullptr,
                   d_refs.data(),    nr_, d_, d_r_norms.data());
    }

    // GEMM: out = -2 * queries * refs^T  (+ 0 * out)
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        const float alpha = -2.f;
        const float beta  =  0.f;

        // cublasSgemm computes C = alpha * A * B + beta * C in column-major.
        // To compute row-major  C = alpha * A * B^T,
        // pass B, A (reversed) with OP_T on B and OP_N on A.
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(nr), static_cast<int>(nq), static_cast<int>(d),
            &alpha,
            d_refs.data(),    static_cast<int>(d),
            d_queries.data(), static_cast<int>(d),
            &beta,
            d_out.data(),     static_cast<int>(nr)));

        CUBLAS_CHECK(cublasDestroy(handle));
    }

    // Epilogue: add ‖q_i‖² + ‖r_j‖².
    {
        constexpr unsigned int kB = 16u;
        dim3 block(kB, kB);
        dim3 grid((nr_ + kB-1u)/kB, (nq_ + kB-1u)/kB);
        GPU_LAUNCH(norm_epilogue_kernel, grid, block, 0, nullptr,
                   d_out.data(), d_q_norms.data(), d_r_norms.data(), nq_, nr_);
    }

    GPU_SYNC();
    return d_out.to_host();
}

} // namespace knng::gpu
