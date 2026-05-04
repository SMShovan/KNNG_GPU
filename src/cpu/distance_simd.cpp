/// @file
/// @brief Hand-vectorised squared-L2 / dot-product primitives.
///
/// Compile-time dispatch picks AVX2 (x86_64), NEON (arm64), or
/// the scalar fallback based on which target macros the toolchain
/// defines. Runtime CPUID on x86 further degrades to scalar if the
/// binary was compiled with `__AVX2__` but runs on a CPU that does
/// not support it.
///
/// The intrinsic implementations follow the same shape:
///
///   * Initialise an accumulator of vector-width-many lanes
///     (`__m256` → 8, `float32x4_t` → 4).
///   * Stream pairs of input buffers through the loop, computing
///     each chunk's per-lane contribution with FMA.
///   * Horizontally reduce the accumulator after the main loop.
///   * Scalar tail loop for the remainder when `dim` is not a
///     multiple of the vector width.
///
/// The scalar tail is non-negotiable: brute-force feature
/// dimensionalities are not always 8/4-aligned (SIFT is 128, GIST
/// is 960 — both nice; Fashion-MNIST is 784 — needs the tail).

#include "knng/cpu/distance_simd.hpp"

#include "knng/core/distance.hpp"
#include "knng/cpu/distance.hpp"

#if defined(__AVX2__)
#  include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

#include <cstddef>

namespace knng::cpu {

namespace {

#if defined(__AVX2__)
/// Horizontal sum of an `__m256` register: 8 lanes → one float.
/// Sequence: split high/low halves, add, then `_mm_hadd_ps` twice
/// for the 4-lane reduction. Uses only AVX1 / SSE3 ops so it works
/// on every AVX2 host.
[[nodiscard]] inline float horizontal_sum_avx2(__m256 v) noexcept
{
    const __m128 hi   = _mm256_extractf128_ps(v, 1);
    const __m128 lo   = _mm256_castps256_ps128(v);
    __m128 sum128     = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

#if defined(__ARM_NEON)
/// Horizontal sum of a `float32x4_t` register. ARMv8 has
/// `vaddvq_f32` which does this in a single instruction.
[[nodiscard]] inline float horizontal_sum_neon(float32x4_t v) noexcept
{
    return vaddvq_f32(v);
}
#endif

#if defined(__AVX2__)
/// On x86 with AVX2 compiled in, this query is meaningful — we
/// may still be running on a CPU without AVX2 if the user copied
/// the binary across hosts.
[[nodiscard]] bool runtime_supports_avx2() noexcept
{
#  if defined(__GNUC__) || defined(__clang__)
    static const bool ok = __builtin_cpu_supports("avx2");
    return ok;
#  else
    return true;
#  endif
}

[[nodiscard]] float scalar_squared_l2(const float* a,
                                      const float* b,
                                      std::size_t dim) noexcept
{
    return ::knng::squared_l2(a, b, dim);
}

[[nodiscard]] float scalar_dot_product(const float* a,
                                       const float* b,
                                       std::size_t dim) noexcept
{
    return ::knng::cpu::dot_product(a, b, dim);
}
#endif // __AVX2__

#if defined(__AVX2__)
[[nodiscard]] float avx2_squared_l2(const float* a,
                                    const float* b,
                                    std::size_t dim) noexcept
{
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float result = horizontal_sum_avx2(acc);
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

[[nodiscard]] float avx2_dot_product(const float* a,
                                     const float* b,
                                     std::size_t dim) noexcept
{
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float result = horizontal_sum_avx2(acc);
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif // __AVX2__

#if defined(__ARM_NEON)
[[nodiscard]] float neon_squared_l2(const float* a,
                                    const float* b,
                                    std::size_t dim) noexcept
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        const float32x4_t d  = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, d, d);   // acc += d * d
    }
    float result = horizontal_sum_neon(acc);
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

[[nodiscard]] float neon_dot_product(const float* a,
                                     const float* b,
                                     std::size_t dim) noexcept
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb); // acc += va * vb
    }
    float result = horizontal_sum_neon(acc);
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif // __ARM_NEON

} // namespace

SimdPath active_simd_path() noexcept
{
#if defined(__AVX2__)
    return runtime_supports_avx2() ? SimdPath::kAvx2 : SimdPath::kScalar;
#elif defined(__ARM_NEON)
    return SimdPath::kNeon;
#else
    return SimdPath::kScalar;
#endif
}

float simd_squared_l2(const float* a, const float* b,
                      std::size_t dim) noexcept
{
#if defined(__AVX2__)
    if (runtime_supports_avx2()) {
        return avx2_squared_l2(a, b, dim);
    }
    return scalar_squared_l2(a, b, dim);
#elif defined(__ARM_NEON)
    return neon_squared_l2(a, b, dim);
#else
    return ::knng::squared_l2(a, b, dim);
#endif
}

float simd_dot_product(const float* a, const float* b,
                       std::size_t dim) noexcept
{
#if defined(__AVX2__)
    if (runtime_supports_avx2()) {
        return avx2_dot_product(a, b, dim);
    }
    return scalar_dot_product(a, b, dim);
#elif defined(__ARM_NEON)
    return neon_dot_product(a, b, dim);
#else
    return ::knng::cpu::dot_product(a, b, dim);
#endif
}

} // namespace knng::cpu
