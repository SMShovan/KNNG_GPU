/// @file
/// @brief Step 58 — CPU reference for WMMA distance: delegates to GEMM decomposition.

#include <knng/gpu/wmma_distance.hpp>
#include <knng/gpu/distance_gemm.hpp>   // cpu_distance_gemm

namespace knng::gpu {

void cpu_wmma_distance(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept
{
    cpu_distance_gemm(queries, refs, nq, nr, d, out);
}

} // namespace knng::gpu
