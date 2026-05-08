#pragma once

/// @file
/// @brief Step 47 — Naive GPU distance kernel declarations + CPU reference.
///
/// The kernel `knng::gpu::naive_pairwise_l2sq_kernel` assigns one GPU thread
/// to each (query, reference) pair and writes the squared-L2 distance into a
/// `nq × nr` output matrix stored in row-major order.  It is the simplest
/// possible GPU distance calculator — the GPU equivalent of Step 12's CPU
/// brute-force — and serves as the correctness baseline for all later GPU
/// distance optimisations.
///
/// ### Layout convention
/// - Queries:     row-major `float[nq][d]`
/// - References:  row-major `float[nr][d]`
/// - Output:      row-major `float[nq][nr]`, entry `[i][j]` = ||q_i - r_j||²
///
/// On the GPU, each thread computes one (i, j) pair by looping over all `d`
/// dimensions.  There is no shared-memory caching and no coalescing
/// optimisation — those land in Phase 8.
///
/// The CPU reference `knng::gpu::cpu_pairwise_l2sq` implements the same
/// algorithm in portable C++ so that tests can verify GPU output against a
/// known-good baseline without needing a GPU.

#include <knng/gpu/backend.hpp>

#include <cstddef>
#include <vector>  // gpu_pairwise_l2sq return type

namespace knng::gpu {

// ---------------------------------------------------------------------------
// CPU reference (always compiled, used by unit tests without a GPU)
// ---------------------------------------------------------------------------

/// @brief Compute all pairwise squared-L2 distances on the CPU.
///
/// @param queries   Row-major query matrix,    shape `[nq][d]`.
/// @param refs      Row-major reference matrix, shape `[nr][d]`.
/// @param nq        Number of query points.
/// @param nr        Number of reference points.
/// @param d         Feature dimensionality.
/// @param out       Output matrix, row-major `[nq][nr]` (caller-allocated).
void cpu_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept;

// ---------------------------------------------------------------------------
// GPU kernel (compiled only when KNNG_HAVE_CUDA or KNNG_HAVE_HIP is set)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Naive GPU pairwise squared-L2 kernel.
///
/// Grid / block sizing: caller should launch with a 2-D grid where each
/// thread handles one (query_idx, ref_idx) pair.  A common choice is
/// `block = dim3(16, 16)` and `grid = dim3(ceil(nr/16), ceil(nq/16))`.
///
/// @param d_queries  Device pointer, row-major `float[nq][d]`.
/// @param d_refs     Device pointer, row-major `float[nr][d]`.
/// @param nq         Number of query points.
/// @param nr         Number of reference points.
/// @param d          Feature dimensionality.
/// @param d_out      Device pointer, output row-major `float[nq][nr]`.
GPU_GLOBAL void naive_pairwise_l2sq_kernel(
    const float* d_queries,
    const float* d_refs,
    unsigned int nq,
    unsigned int nr,
    unsigned int d,
    float*       d_out);

/// @brief Host-side launcher: allocates device buffers, runs the kernel,
///        copies results back.
///
/// @param queries  Host query matrix, row-major `[nq][d]`.
/// @param refs     Host reference matrix, row-major `[nr][d]`.
/// @param nq       Number of query points.
/// @param nr       Number of reference points.
/// @param d        Feature dimensionality.
/// @return         Host-side result vector, row-major `[nq × nr]`.
std::vector<float> gpu_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d);

#endif // KNNG_HAVE_CUDA || KNNG_HAVE_HIP

} // namespace knng::gpu
