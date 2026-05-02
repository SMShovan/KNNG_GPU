#pragma once

/// @file
/// @brief Recall@k — the canonical quality metric for an approximate KNNG.
///
/// `recall_at_k(approx, truth)` is the fraction of (point, neighbor)
/// pairs in the approximate graph that also appear among the top-k
/// neighbors of the same point in the exact graph. It is the standard
/// quality metric used throughout the ANN literature (FAISS,
/// `ann-benchmarks`, NEO-DNND) and the value every later builder in
/// this project will report alongside its wall time.
///
/// ## Definition
///
/// For a `Knng approx` and a ground-truth `Knng truth` over the same
/// `n` points and the same `k`:
///
///     recall@k =  ( Σ_q |approx_neighbors(q) ∩ truth_neighbors(q)| )
///                 ────────────────────────────────────────────────
///                                       n * k
///
/// The intersection is over the *sets* of neighbor IDs — order does
/// not matter, but per-row uniqueness does (a neighbor that appears
/// twice in `approx` cannot inflate the score).
///
/// The function returns a value in `[0, 1]`. `1.0` means the
/// approximate graph is exact. The value is reported as a `double`
/// (not `float`) because at the n*k counts of interest (e.g. SIFT1M
/// k=100 ⇒ 1e8 pairs) `float`'s 24-bit mantissa would lose
/// resolution — every test in this project's regression suite needs
/// to detect a single-pair regression.

#include <cstddef>

#include "knng/core/graph.hpp"

namespace knng::bench {

/// Recall@k as defined in the file-level docs.
///
/// Both graphs must have identical `(n, k)` shape; mismatches throw
/// `std::invalid_argument` so a caller never silently compares two
/// graphs that disagree on what "the answer" is supposed to look
/// like. A degenerate `n == 0` graph returns `1.0` — a vacuous truth
/// that prevents callers from special-casing the empty-input case.
///
/// @param approx The graph under test — usually the output of an
///         approximate builder (NN-Descent, GPU brute-force, etc).
/// @param truth The exact reference, typically produced by
///         `knng::cpu::brute_force_knn` or `load_or_compute_ground_truth`.
/// @return Fraction in `[0, 1]` of approximate neighbors that appear
///         in the corresponding exact row.
/// @throws std::invalid_argument if `approx.n != truth.n` or
///         `approx.k != truth.k`.
[[nodiscard]] double recall_at_k(const Knng& approx, const Knng& truth);

/// Per-row recall as the integer count of `approx` neighbors that
/// also appear in `truth`'s row. Useful for histograms and for tests
/// that want to assert "every row is fully recalled" without a
/// floating-point tolerance.
///
/// Returns the number of matches in `[0, k]`. The same `(n, k)`
/// matching constraint as `recall_at_k` applies; mismatches throw.
[[nodiscard]] std::size_t recall_at_k_row(const Knng& approx,
                                          const Knng& truth,
                                          std::size_t row);

} // namespace knng::bench
