#pragma once

/// @file
/// @brief Step 56 — Distance computation via GEMM (distance = ‖q‖² + ‖r‖² − 2⟨q,r⟩).
///
/// Pairwise squared-L2 distance can be decomposed as:
///
///   D[i,j] = ‖q_i − r_j‖² = ‖q_i‖² + ‖r_j‖² − 2 ⟨q_i, r_j⟩
///
/// The dominant term is the dot-product matrix `-2 Q Rᵀ`, which maps
/// directly to a `cublasSgemm` call:
///
///   C = α A B + β C,  with α=-2, β=0, A=Q (nq×d), B=Rᵀ (d×nr) → C (nq×nr)
///
/// The norm terms are precomputed in O(n d) time and added via a cheap
/// epilogue kernel.  This moves the distance computation off the CUDA
/// cores and onto the Tensor Cores (when cuBLAS chooses a WMMA-based
/// algorithm), which on Ampere/Hopper can be 4–16× faster than the naive
/// kernel at large n and d.
///
/// ### CPU reference
/// `cpu_distance_gemm` uses the existing CPU BLAS integration (`cblas_sgemm`
/// when available, manual dot-product otherwise) so the reference matches
/// the GPU path exactly and exercises the same algorithmic decomposition.
///
/// ### Files
/// - `src/gpu/distance_gemm_cpu_ref.cpp` — always compiled into `knng::gpu_ref`.
/// - `src/gpu/distance_gemm.cu`          — CUDA-only; links `CUDA::cublas`.

#include <knng/gpu/backend.hpp>

#include <cstddef>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Compute norms ‖p_i‖² for all `n` points.
///
/// @param pts  Row-major `float[n][d]`.
/// @param n    Number of points.
/// @param d    Feature dimensionality.
/// @param out  Output `float[n]` (caller-allocated).
void cpu_compute_norms(
    const float* pts,
    std::size_t  n,
    std::size_t  d,
    float*       out) noexcept;

/// @brief Pairwise squared-L2 via GEMM decomposition (CPU).
///
/// Computes `-2 Q Rᵀ` using `cblas_sgemm` when BLAS is available,
/// or via a manual triple loop otherwise.  Adds norms in a second pass.
///
/// @param queries  Row-major `float[nq][d]`.
/// @param refs     Row-major `float[nr][d]`.
/// @param nq       Number of query points.
/// @param nr       Number of reference points.
/// @param d        Feature dimensionality.
/// @param out      Row-major output `float[nq][nr]` (caller-allocated).
void cpu_distance_gemm(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept;

// ---------------------------------------------------------------------------
// GPU path (CUDA/HIP only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Epilogue kernel: add precomputed norms to the GEMM result.
///
/// For each entry `out[qi][ri]`, adds `q_norms[qi] + r_norms[ri]`.
/// One thread per output element.
GPU_GLOBAL void norm_epilogue_kernel(
    float*       d_out,
    const float* d_q_norms,
    const float* d_r_norms,
    unsigned int nq,
    unsigned int nr);

/// @brief Norm kernel: compute ‖p_i‖² for each row.
GPU_GLOBAL void compute_norms_kernel(
    const float* d_pts,
    unsigned int n,
    unsigned int d,
    float*       d_norms);

/// @brief Host launcher: cuBLAS GEMM + norm epilogue.
///
/// @param queries  Host row-major `float[nq][d]`.
/// @param refs     Host row-major `float[nr][d]`.
/// @param nq       Number of query points.
/// @param nr       Number of reference points.
/// @param d        Feature dimensionality.
/// @return         Host result vector, row-major `float[nq*nr]`.
std::vector<float> gpu_distance_gemm(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d);

#endif

} // namespace knng::gpu
