#pragma once

/// @file
/// @brief Hand-vectorised squared-L2 / dot-product primitives.
///
/// `knng::cpu::dot_product` (Step 19) and `knng::squared_l2`
/// (Step 08) are the canonical scalar primitives every CPU
/// algorithm consumes. This header adds the *vectorised* variants
/// — `knng::cpu::simd_squared_l2` and `knng::cpu::simd_dot_product` —
/// which apply hand-written intrinsics when the build target
/// supports them and fall back to the scalar functions when it
/// does not.
///
/// Compile-time dispatch picks the widest path the build target
/// guarantees:
///
///   * **AVX2 (x86_64).** 8 floats per register, FMA via
///     `_mm256_fmadd_ps`. Selected when `__AVX2__` is defined
///     (i.e. the user passed `-mavx2` / `-march=native` on a
///     CPU that supports it).
///   * **NEON (arm64).** 4 floats per register, FMA via
///     `vfmaq_f32`. Selected when `__ARM_NEON` is defined
///     (mandatory on ARMv8, so always present on Apple Silicon
///     and on any modern arm64 Linux build).
///   * **Scalar fallback.** When neither vector ISA is
///     compile-time available, the SIMD entry points reduce to
///     calls into the scalar primitives — same answer, no
///     speedup. This keeps the public API stable across
///     toolchains.
///
/// Runtime CPUID dispatch on x86 is *also* applied: when the
/// binary was compiled with `__AVX2__` but happens to run on a
/// CPU that does not support AVX2, the SIMD entry points detect
/// this once at static-initialisation time via
/// `__builtin_cpu_supports("avx2")` and route to the scalar path
/// for the rest of the process's lifetime. ARM NEON is mandatory
/// on ARMv8 so no runtime check is necessary there.
///
/// Why a separate header rather than overloading the existing
/// `dot_product` / `squared_l2`? The scalar versions are `inline`
/// in their headers — every TU that includes them gets its own
/// copy, which is the right shape for the autovectoriser. The
/// SIMD versions are out-of-line in `src/cpu/distance_simd.cpp`
/// because each platform's intrinsics live behind their own
/// `#include` and the resulting machine code does not benefit
/// from per-TU specialisation. Step 28's Step-29 successor
/// (CPU scaling writeup) will compare the two paths line-by-line.
///
/// Mapping to GPU warp-level thinking: an `__m256` holds 8 lanes;
/// a CUDA warp holds 32. The AVX2 dot-product loop computes a
/// per-lane partial sum and horizontally reduces at the end —
/// exactly the same shape as a CUDA shuffle-based warp reduction
/// (`__shfl_xor_sync`). Step 53's GPU warp-level top-k will
/// translate the AVX2 code's structure almost verbatim into
/// shuffle ops; the only thing that changes is the lane count.

#include <cstddef>

namespace knng::cpu {

/// Compile-time tag for which path was selected at build time.
/// Useful for tests that want to assert "yes, the AVX2 path
/// actually compiled in" (not just "the API call succeeded").
enum class SimdPath {
    kAvx2,
    kNeon,
    kScalar,
};

/// Path the public `simd_*` entry points dispatch to *at compile
/// time*. Runtime CPUID may further degrade `kAvx2` to `kScalar`
/// on a host that lacks AVX2 — `active_simd_path()` reports the
/// runtime answer.
[[nodiscard]] constexpr SimdPath compiled_simd_path() noexcept;

/// Path the public `simd_*` entry points actually use *at this
/// runtime*. On x86 with `__AVX2__` defined but the running CPU
/// lacking AVX2, this returns `kScalar` even though
/// `compiled_simd_path()` returns `kAvx2`.
[[nodiscard]] SimdPath active_simd_path() noexcept;

/// Squared-L2 distance between two equal-length float buffers,
/// hand-vectorised when the build / runtime supports it. Returns
/// the same result (within fp accumulation reordering) as the
/// scalar `knng::squared_l2` from Step 08.
[[nodiscard]] float simd_squared_l2(const float* a,
                                    const float* b,
                                    std::size_t dim) noexcept;

/// Inner product of two equal-length float buffers, hand-vectorised
/// when the build / runtime supports it. Same result (within fp
/// accumulation reordering) as the scalar `knng::cpu::dot_product`
/// from Step 19.
[[nodiscard]] float simd_dot_product(const float* a,
                                     const float* b,
                                     std::size_t dim) noexcept;

// ---- compiled_simd_path implementation kept in the header so it
//      remains a `constexpr` query. ---------------------------------------

constexpr SimdPath compiled_simd_path() noexcept
{
#if defined(__AVX2__)
    return SimdPath::kAvx2;
#elif defined(__ARM_NEON)
    return SimdPath::kNeon;
#else
    return SimdPath::kScalar;
#endif
}

} // namespace knng::cpu
