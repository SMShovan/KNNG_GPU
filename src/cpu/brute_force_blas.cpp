/// @file
/// @brief BLAS-backed brute-force L2 KNN (Step 21).
///
/// Compiled only when `KNNG_HAVE_BLAS` is defined — see
/// `cmake/FindKnngBlas.cmake`. The algorithm is the matrix form of
/// the Step-19 identity:
///
///     D[i, j]  =  ||x_i||²  +  ||y_j||²  -  2 · (X · Yᵀ)[i, j]
///
/// laid out as a tile loop that mirrors Step 20's shape. Each outer
/// iteration picks a slice of `query_tile` queries `X` and a slice
/// of `ref_tile` references `Y`, hands `X` and `Y` to `cblas_sgemm`
/// to obtain `(QUERY_TILE × REF_TILE)` cross-products in one call,
/// and folds the precomputed norms in via a scalar epilogue. The
/// per-query `TopK` heap then absorbs the row.

#include "knng/cpu/brute_force.hpp"

#if defined(KNNG_HAVE_BLAS) && KNNG_HAVE_BLAS

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(KNNG_BLAS_USES_ACCELERATE)
#  include <Accelerate/Accelerate.h>
#else
#  include <cblas.h>
#endif

#include "knng/cpu/distance.hpp"

namespace knng::cpu {

Knng brute_force_knn_l2_blas(const Dataset& ds,
                              std::size_t k,
                              std::size_t query_tile,
                              std::size_t ref_tile)
{
    if (ds.n == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_blas: dataset is empty");
    }
    if (k == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_blas: k must be > 0");
    }
    if (k > ds.n - 1) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_blas: k ("
            + std::to_string(k) + ") must be <= ds.n - 1 ("
            + std::to_string(ds.n - 1) + ")");
    }
    if (query_tile == 0 || ref_tile == 0) {
        throw std::invalid_argument(
            "knng::cpu::brute_force_knn_l2_blas: tile sizes must be > 0");
    }
    assert(ds.is_contiguous());

    std::vector<float> norms;
    compute_norms_squared(ds, norms);

    const float*      base   = ds.data_ptr();
    const std::size_t stride = ds.stride();
    const int         d      = static_cast<int>(stride);

    Knng out(ds.n, k);

    // Per-tile distance scratchpad (`QUERY_TILE × REF_TILE`). Sized
    // once at function entry; the inner loops trust the
    // tile-boundary clip.
    std::vector<float> dist_block(query_tile * ref_tile, 0.0f);

    std::vector<TopK> heaps;
    heaps.reserve(query_tile);

    for (std::size_t q_lo = 0; q_lo < ds.n; q_lo += query_tile) {
        const std::size_t q_hi = std::min(q_lo + query_tile, ds.n);
        const std::size_t q_n  = q_hi - q_lo;

        heaps.clear();
        for (std::size_t i = 0; i < q_n; ++i) {
            heaps.emplace_back(k);
        }

        const float* X = base + q_lo * stride;

        for (std::size_t r_lo = 0; r_lo < ds.n; r_lo += ref_tile) {
            const std::size_t r_hi = std::min(r_lo + ref_tile, ds.n);
            const std::size_t r_n  = r_hi - r_lo;
            const float*      Y    = base + r_lo * stride;

            // C := -2 * X * Yᵀ where
            //   X is (q_n × d) row-major, leading dimension `stride`
            //   Y is (r_n × d) row-major, leading dimension `stride`
            //   C is (q_n × r_n) row-major, leading dimension `ref_tile`
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                static_cast<int>(q_n),
                static_cast<int>(r_n),
                d,
                /*alpha=*/-2.0f,
                X, /*lda=*/static_cast<int>(stride),
                Y, /*ldb=*/static_cast<int>(stride),
                /*beta=*/0.0f,
                dist_block.data(),
                /*ldc=*/static_cast<int>(ref_tile));

            // Norm fold-in + heap admission. The block stride is
            // `ref_tile` (the `ldc` we passed to BLAS), regardless
            // of `r_n`; we walk only the `r_n` populated columns.
            for (std::size_t qi = 0; qi < q_n; ++qi) {
                const std::size_t q  = q_lo + qi;
                const float       na = norms[q];
                TopK&             heap = heaps[qi];
                float* row = dist_block.data() + qi * ref_tile;

                for (std::size_t rj = 0; rj < r_n; ++rj) {
                    const std::size_t r = r_lo + rj;
                    if (r == q) {
                        continue;
                    }
                    float dist = na + norms[r] + row[rj];
                    if (dist < 0.0f) {
                        dist = 0.0f;
                    }
                    heap.push(static_cast<index_t>(r), dist);
                }
            }
        }

        for (std::size_t qi = 0; qi < q_n; ++qi) {
            const std::size_t q  = q_lo + qi;
            const auto sorted = heaps[qi].extract_sorted();
            auto neighbors_row = out.neighbors_of(q);
            auto distances_row = out.distances_of(q);
            for (std::size_t j = 0; j < sorted.size(); ++j) {
                neighbors_row[j] = sorted[j].first;
                distances_row[j] = sorted[j].second;
            }
        }
    }

    return out;
}

} // namespace knng::cpu

#endif  // KNNG_HAVE_BLAS
