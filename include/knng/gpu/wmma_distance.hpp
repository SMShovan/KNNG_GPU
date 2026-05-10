#pragma once

/// @file
/// @brief Step 58 — Tensor Core (WMMA) distance computation.
///
/// NVIDIA Volta+ GPUs have Tensor Core hardware that multiplies 16×16
/// matrices of fp16 in one instruction, delivering 8× the throughput of
/// CUDA core FP32 GEMM.  This step implements the GEMM portion of the
/// distance decomposition using the WMMA (Warp Matrix Multiply-Accumulate)
/// C++ API from `<mma.h>`.
///
/// ### Algorithm (same as Step 56 GEMM)
///   D[nq,nr] = ‖Q‖² + ‖R‖² − 2 Q Rᵀ
///
/// The `-2 Q Rᵀ` term is computed by a WMMA-based kernel:
///   - Each warp loads 16×16 tiles of Q and R into `wmma::fragment` objects.
///   - `wmma::mma_sync` accumulates the dot products in fp32.
///   - The epilogue adds norms as in Step 56.
///
/// ### Compute capability guard
/// WMMA requires CC ≥ 70 (Volta).  The kernel file uses
/// `#if __CUDA_ARCH__ >= 700` to suppress compilation on older GPUs and
/// to emit a readable error if called at runtime on CC < 70.
///
/// ### CPU reference
/// `cpu_wmma_distance` is identical to `cpu_distance_gemm` from Step 56 —
/// both are algorithmic GEMM decompositions and produce the same result.
/// There is no CPU WMMA instruction; the CPU path exercises the correctness
/// of the surrounding host code (norm computation, epilogue) without the
/// hardware tile computation.

#include <knng/gpu/backend.hpp>

#include <cstddef>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// CPU reference (always compiled — delegates to existing GEMM decomposition)
// ---------------------------------------------------------------------------

/// @brief WMMA-strategy pairwise squared-L2, CPU reference.
///
/// Identical to `cpu_distance_gemm`: the WMMA tile size (16) is irrelevant
/// on the CPU — any correct GEMM decomposition gives the same result.
void cpu_wmma_distance(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept;

// ---------------------------------------------------------------------------
// GPU path (CUDA-only, CC ≥ 70)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA)

/// @brief Host launcher: WMMA GEMM + norm epilogue.
///
/// @param queries  Host row-major `float[nq][d]`.
/// @param refs     Host row-major `float[nr][d]`.
/// @param nq       Number of query points.
/// @param nr       Number of reference points (must be multiple of 16).
/// @param d        Feature dimensionality (must be multiple of 16).
/// @return         Host result vector, row-major `float[nq*nr]`.
/// @throws std::invalid_argument if `nr` or `d` is not a multiple of 16.
std::vector<float> gpu_wmma_distance(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d);

#endif // KNNG_HAVE_CUDA

} // namespace knng::gpu
