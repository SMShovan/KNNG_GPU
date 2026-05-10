/// @file
/// @brief Step 52 — CPU reference: coalesced distance with column-major refs.

#include <knng/gpu/coalesced_distance.hpp>

namespace knng::gpu {

void cpu_transpose(
    const float* src,
    std::size_t  rows,
    std::size_t  cols,
    float*       dst) noexcept
{
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void cpu_coalesced_pairwise_l2sq(
    const float* queries,
    const float* refs_T,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept
{
    for (std::size_t qi = 0; qi < nq; ++qi) {
        for (std::size_t ri = 0; ri < nr; ++ri) {
            float acc = 0.f;
            for (std::size_t k = 0; k < d; ++k) {
                // query: row-major[qi][k], refs_T: col-major[k][ri]
                const float diff = queries[qi * d + k] - refs_T[k * nr + ri];
                acc += diff * diff;
            }
            out[qi * nr + ri] = acc;
        }
    }
}

} // namespace knng::gpu
