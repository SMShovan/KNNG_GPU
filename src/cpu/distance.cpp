/// @file
/// @brief Out-of-line implementations for `knng::cpu::distance.hpp`.
///
/// Only `compute_norms_squared` lives here; the inner-loop
/// `dot_product` is `inline` in the header. Splitting the two
/// matches the project convention from
/// `include/knng/core/distance.hpp` — primitives that an algorithm
/// wants to specialise per call live in headers; helpers that an
/// algorithm calls once outside the timed loop live in TUs.

#include "knng/cpu/distance.hpp"

#include <cassert>

namespace knng::cpu {

void compute_norms_squared(const Dataset& ds, std::vector<float>& out)
{
    assert(ds.is_contiguous());
    out.resize(ds.n);

    const float*      base   = ds.data_ptr();
    const std::size_t stride = ds.stride();

    for (std::size_t i = 0; i < ds.n; ++i) {
        const float* row = base + i * stride;
        // Single-pass squared-L2 norm. Hand-written rather than
        // delegating to dot_product(row, row, stride) so a future
        // SIMD specialisation can collapse the multiply into the
        // accumulate in one fused instruction.
        float acc = 0.0f;
        for (std::size_t j = 0; j < stride; ++j) {
            const float v = row[j];
            acc += v * v;
        }
        out[i] = acc;
    }
}

} // namespace knng::cpu
