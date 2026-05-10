/// @file
/// @brief Step 56 — CPU reference: GEMM-based pairwise squared-L2 distance.
///
/// Uses cblas_sgemm when KNNG_HAVE_BLAS is defined; falls back to a manual
/// triple loop otherwise.  The algorithmic decomposition
///   D = ‖Q‖² + ‖R‖² − 2 Q Rᵀ
/// is always exercised so the test validates the decomposition logic
/// independently of BLAS availability.

#include <knng/gpu/distance_gemm.hpp>

#ifdef KNNG_HAVE_BLAS
#  ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

#include <vector>

namespace knng::gpu {

void cpu_compute_norms(
    const float* pts,
    std::size_t  n,
    std::size_t  d,
    float*       out) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        float acc = 0.f;
        for (std::size_t k = 0; k < d; ++k) {
            acc += pts[i * d + k] * pts[i * d + k];
        }
        out[i] = acc;
    }
}

void cpu_distance_gemm(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d,
    float*       out) noexcept
{
    // Step 1: fill `out` with Q norms broadcast across each row.
    std::vector<float> q_norms(nq), r_norms(nr);
    cpu_compute_norms(queries, nq, d, q_norms.data());
    cpu_compute_norms(refs,    nr, d, r_norms.data());

    for (std::size_t qi = 0; qi < nq; ++qi) {
        for (std::size_t ri = 0; ri < nr; ++ri) {
            out[qi * nr + ri] = q_norms[qi] + r_norms[ri];
        }
    }

    // Step 2: add -2 Q Rᵀ  (the expensive dot-product step).
#ifdef KNNG_HAVE_BLAS
    // cblas_sgemm: C = alpha * A * B + beta * C
    // A = queries (nq × d), B = refs^T ≡ refs (d × nr), C = out (nq × nr)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(nq),
                static_cast<int>(nr),
                static_cast<int>(d),
                -2.0f,
                queries, static_cast<int>(d),
                refs,    static_cast<int>(d),
                1.0f,    // β = 1 because out already holds the norm terms
                out,     static_cast<int>(nr));
#else
    // Manual triple loop (no BLAS).
    for (std::size_t qi = 0; qi < nq; ++qi) {
        for (std::size_t ri = 0; ri < nr; ++ri) {
            float dot = 0.f;
            for (std::size_t k = 0; k < d; ++k) {
                dot += queries[qi * d + k] * refs[ri * d + k];
            }
            out[qi * nr + ri] -= 2.f * dot;
        }
    }
#endif
}

} // namespace knng::gpu
