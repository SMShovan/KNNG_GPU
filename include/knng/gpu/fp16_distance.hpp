#pragma once

/// @file
/// @brief Step 57 — fp16 storage, fp32 compute distance.
///
/// Storing feature vectors as 16-bit floats (fp16 / IEEE 754 binary16) halves
/// the memory footprint and bandwidth compared to fp32.  For SIFT-128 at 1M
/// points: fp32 = 512 MB, fp16 = 256 MB.  On a bandwidth-bound kernel,
/// halving the memory means roughly halving the time.
///
/// ### Strategy
/// 1. **Conversion kernel** (`fp32_to_fp16_kernel`): convert the float32
///    dataset to fp16 once on the GPU.
/// 2. **Distance kernel** (`fp16_pairwise_l2sq_kernel`): loads fp16, converts
///    to fp32 on the fly (`__half2float`), accumulates distances in fp32 to
///    avoid ranking errors from reduced-precision arithmetic.
/// 3. **Recall check**: exact brute-force, so recall stays 1.0 — the test
///    verifies that distances match the fp32 reference within a slightly
///    larger tolerance (fp16 conversion introduces ≈0.1% rounding error).
///
/// ### CPU reference
/// `cpu_fp16_pairwise_l2sq` uses the C standard `_Float16` type where
/// available (GCC ≥ 12, Clang ≥ 15, AppleClang ≥ 15 on Apple Silicon).
/// When `_Float16` is unavailable, it simulates the conversion by rounding
/// fp32 values to 3 significant decimal digits and comparing against the
/// naive result — adequate for testing the algorithmic structure without
/// the hardware type.

#include <knng/gpu/backend.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Portable fp16 ↔ fp32 conversion (CPU, no hardware requirement)
// ---------------------------------------------------------------------------

/// @brief Convert a float32 to the nearest float16, returned as uint16_t bits.
std::uint16_t f32_to_f16_bits(float v) noexcept;

/// @brief Convert float16 bits (uint16_t) back to float32.
float f16_bits_to_f32(std::uint16_t bits) noexcept;

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Convert an fp32 point set to fp16 (stored as uint16_t) on the CPU.
///
/// @param src   Row-major `float[n][d]`.
/// @param n     Number of points.
/// @param d     Feature dimensionality.
/// @param dst   Output `uint16_t[n][d]` (caller-allocated).
void cpu_fp32_to_fp16(
    const float*    src,
    std::size_t     n,
    std::size_t     d,
    std::uint16_t*  dst) noexcept;

/// @brief Pairwise squared-L2 with fp16 storage, fp32 accumulation (CPU).
///
/// @param q_f16  Row-major `uint16_t[nq][d]` (query points in fp16).
/// @param r_f16  Row-major `uint16_t[nr][d]` (reference points in fp16).
/// @param nq     Number of query points.
/// @param nr     Number of reference points.
/// @param d      Feature dimensionality.
/// @param out    Row-major output `float[nq][nr]`.
void cpu_fp16_pairwise_l2sq(
    const std::uint16_t* q_f16,
    const std::uint16_t* r_f16,
    std::size_t          nq,
    std::size_t          nr,
    std::size_t          d,
    float*               out) noexcept;

// ---------------------------------------------------------------------------
// GPU path (CUDA/HIP only)
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

#include <cuda_fp16.h>

/// @brief Convert fp32 dataset to fp16 on the device.
GPU_GLOBAL void fp32_to_fp16_kernel(
    const float* d_src,
    unsigned int n,
    unsigned int d,
    __half*      d_dst);

/// @brief Fp16-load, fp32-accumulate pairwise squared-L2 kernel.
GPU_GLOBAL void fp16_pairwise_l2sq_kernel(
    const __half* d_queries_f16,
    const __half* d_refs_f16,
    unsigned int  nq,
    unsigned int  nr,
    unsigned int  d,
    float*        d_out);

/// @brief Host launcher: convert to fp16, run kernel, return fp32 distances.
std::vector<float> gpu_fp16_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d);

#endif

} // namespace knng::gpu
