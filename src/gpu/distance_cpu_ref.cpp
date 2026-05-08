/// @file
/// @brief Step 47 — CPU reference implementations for GPU distance kernels.
///
/// These functions are always compiled (no CUDA required) and serve as the
/// correctness oracle for the GPU kernels in `distance_kernel.cu`.
/// Tests on Mac run only these implementations; tests on a GPU machine
/// additionally compare against the GPU kernel output.

#include <knng/gpu/distance_kernel.hpp>

namespace knng::gpu {

void cpu_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept
{
    for (std::size_t i = 0; i < nq; ++i) {
        for (std::size_t j = 0; j < nr; ++j) {
            float acc = 0.f;
            for (std::size_t k = 0; k < d; ++k) {
                const float diff = queries[i * d + k] - refs[j * d + k];
                acc += diff * diff;
            }
            out[i * nr + j] = acc;
        }
    }
}

} // namespace knng::gpu
