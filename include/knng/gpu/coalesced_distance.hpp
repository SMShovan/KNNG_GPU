#pragma once

/// @file
/// @brief Step 52 — Coalesced pairwise squared-L2 distance.
///
/// The naive kernel (Step 47) loads reference vectors in row-major order:
/// thread 0 reads `refs[0][dim]`, thread 1 reads `refs[1][dim]`, …, so
/// thread `t` reads address `refs_base + t*d*4`.  For d=128 (SIFT) that is
/// 512-byte stride between consecutive threads — a fully non-coalesced warp
/// load.
///
/// The fix: transpose the reference matrix to **column-major** storage before
/// the kernel:
/// ```
///   refs_T[dim][ri]  =  refs[ri][dim]
/// ```
/// Now thread 0 reads `refs_T[dim][0]`, thread 1 reads `refs_T[dim][1]`, …,
/// which are contiguous 4-byte floats — a fully coalesced warp load.
///
/// The transposition is performed by a separate `transpose_kernel` (O(nr*d)
/// work, one global-memory pass).  The query matrix stays row-major because
/// each thread reads all `d` floats of *its own* query sequentially — that
/// access pattern is already sequential within a warp (broadcast from L1).
///
/// ### Functions
/// - `cpu_transpose`  — transpose row-major refs to column-major (CPU).
/// - `cpu_coalesced_pairwise_l2sq` — distance with column-major refs (CPU).
/// - `gpu_coalesced_pairwise_l2sq` — full GPU pipeline (CUDA-only).

#include <knng/gpu/backend.hpp>

#include <cstddef>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// CPU helpers (always compiled)
// ---------------------------------------------------------------------------

/// @brief Transpose a row-major float matrix to column-major in-place output.
///
/// @param src     Row-major input, `float[rows][cols]`.
/// @param rows    Number of rows in @p src.
/// @param cols    Number of columns in @p src.
/// @param dst     Column-major output, `float[cols][rows]` (caller-allocated).
void cpu_transpose(
    const float* src,
    std::size_t  rows,
    std::size_t  cols,
    float*       dst) noexcept;

/// @brief Pairwise squared-L2 with **column-major** reference matrix.
///
/// Functionally identical to `cpu_pairwise_l2sq` but reads references from
/// a column-major `refs_T[d][nr]` layout.
///
/// @param queries   Row-major `float[nq][d]`.
/// @param refs_T    Column-major `float[d][nr]` (transposed reference matrix).
/// @param nq        Number of query points.
/// @param nr        Number of reference points.
/// @param d         Feature dimensionality.
/// @param out       Row-major output `float[nq][nr]`.
void cpu_coalesced_pairwise_l2sq(
    const float* queries,
    const float* refs_T,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept;

// ---------------------------------------------------------------------------
// GPU path (CUDA/HIP only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Transpose kernel: row-major `src[rows][cols]` → col-major `dst[cols][rows]`.
GPU_GLOBAL void transpose_kernel(
    const float* src,
    unsigned int rows,
    unsigned int cols,
    float*       dst);

/// @brief Coalesced pairwise squared-L2 kernel.
///
/// One thread per (qi, ri) pair; references loaded from column-major layout.
GPU_GLOBAL void coalesced_pairwise_l2sq_kernel(
    const float* d_queries,
    const float* d_refs_T,
    unsigned int nq,
    unsigned int nr,
    unsigned int d,
    float*       d_out);

/// @brief Host launcher: transpose refs on device, run coalesced kernel.
///
/// @param queries  Host row-major `float[nq][d]`.
/// @param refs     Host row-major `float[nr][d]`.
/// @param nq       Number of query points.
/// @param nr       Number of reference points.
/// @param d        Feature dimensionality.
/// @return         Host result vector, row-major `float[nq*nr]`.
std::vector<float> gpu_coalesced_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d);

#endif // KNNG_HAVE_CUDA || KNNG_HAVE_HIP

} // namespace knng::gpu
