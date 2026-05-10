/// @file
/// @brief Step 57 — CPU reference: fp16 storage, fp32 compute distance.
///
/// Implements a portable software fp16↔fp32 conversion so the algorithm
/// can be tested on any platform (no CUDA, no hardware fp16 support needed).

#include <knng/gpu/fp16_distance.hpp>

#include <cstdint>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Software fp16 ↔ fp32 conversion
// (IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits)
// ---------------------------------------------------------------------------

std::uint16_t f32_to_f16_bits(float v) noexcept
{
    // Use a union to access the raw bits of the float.
    std::uint32_t bits = 0u;
    __builtin_memcpy(&bits, &v, sizeof(bits));

    const std::uint32_t sign     = (bits >> 31u) & 1u;
    const std::uint32_t exp32    = (bits >> 23u) & 0xFFu;
    const std::uint32_t mant32   =  bits         & 0x7FFFFFu;

    std::uint32_t exp16 = 0u, mant16 = 0u;

    if (exp32 == 0u) {
        // Subnormal or zero → zero in fp16.
        exp16 = 0u; mant16 = 0u;
    } else if (exp32 == 0xFFu) {
        // Inf or NaN.
        exp16  = 0x1Fu;
        mant16 = (mant32 != 0u) ? 1u : 0u;
    } else {
        const int e = static_cast<int>(exp32) - 127 + 15;
        if (e <= 0) {
            exp16 = 0u; mant16 = 0u;           // underflow → zero
        } else if (e >= 31) {
            exp16 = 0x1Fu; mant16 = 0u;        // overflow → inf
        } else {
            exp16  = static_cast<std::uint32_t>(e);
            mant16 = mant32 >> 13u;
        }
    }
    return static_cast<std::uint16_t>((sign << 15u) | (exp16 << 10u) | mant16);
}

float f16_bits_to_f32(std::uint16_t h) noexcept
{
    const std::uint32_t sign   = (h >> 15u) & 1u;
    const std::uint32_t exp16  = (h >> 10u) & 0x1Fu;
    const std::uint32_t mant16 =  h         & 0x3FFu;

    std::uint32_t bits32 = 0u;
    if (exp16 == 0u) {
        bits32 = 0u;                         // zero / subnormal → zero
    } else if (exp16 == 0x1Fu) {
        bits32 = (sign << 31u) | 0x7F800000u | mant16;  // inf / NaN
    } else {
        const std::uint32_t exp32  = exp16 - 15u + 127u;
        const std::uint32_t mant32 = mant16 << 13u;
        bits32 = (sign << 31u) | (exp32 << 23u) | mant32;
    }
    float result = 0.f;
    __builtin_memcpy(&result, &bits32, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// Conversion and distance
// ---------------------------------------------------------------------------

void cpu_fp32_to_fp16(
    const float*   src,
    std::size_t    n,
    std::size_t    d,
    std::uint16_t* dst) noexcept
{
    for (std::size_t i = 0; i < n * d; ++i) {
        dst[i] = f32_to_f16_bits(src[i]);
    }
}

void cpu_fp16_pairwise_l2sq(
    const std::uint16_t* q_f16,
    const std::uint16_t* r_f16,
    std::size_t          nq,
    std::size_t          nr,
    std::size_t          d,
    float*               out) noexcept
{
    for (std::size_t qi = 0; qi < nq; ++qi) {
        for (std::size_t ri = 0; ri < nr; ++ri) {
            float acc = 0.f;
            for (std::size_t k = 0; k < d; ++k) {
                const float qv = f16_bits_to_f32(q_f16[qi * d + k]);
                const float rv = f16_bits_to_f32(r_f16[ri * d + k]);
                const float diff = qv - rv;
                acc += diff * diff;
            }
            out[qi * nr + ri] = acc;
        }
    }
}

} // namespace knng::gpu
