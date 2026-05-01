#pragma once

/// @file
/// @brief Row-major feature-vector container — the universal input shape.
///
/// Every algorithm in the library — CPU brute-force, CPU NN-Descent, GPU
/// kernels, and the multi-GPU pipelines — consumes a `Dataset`. Pinning
/// the layout here means every later stage can pass a `const Dataset&`
/// (or a `std::span<const float>` slice of it) without converting between
/// flavours of "matrix of floats."
///
/// Storage contract:
///   * `data` is a single flat `std::vector<float>` of length `n * d`.
///   * Element `(i, j)` lives at `data[i * d + j]` — row-major, where
///     row `i` is the i-th feature vector and column `j` is its j-th
///     coordinate.
///   * The row stride is exactly `d` floats; there is no padding. This
///     mirrors `Knng`'s `(n, k)` layout so that callers reason about
///     "row of the dataset" and "row of the graph" the same way.
///
/// Why a plain struct, not a class with private members? `Dataset` is a
/// value type with no invariants beyond `data.size() == n * d`, and
/// every consumer wants direct access to the underlying buffer for
/// memcpy, BLAS calls, and GPU H2D transfers. Hiding the buffer behind
/// accessors would only obscure that.

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace knng {

/// Plain-old-data row-major feature matrix. See file-level docs for
/// the storage contract.
struct Dataset {
    std::size_t        n{};  ///< Number of feature vectors (rows).
    std::size_t        d{};  ///< Dimensionality of each feature vector.
    std::vector<float> data;

    /// Default-construct an empty (0×0) dataset. Useful as a placeholder
    /// before a loader fills it in.
    Dataset() = default;

    /// Allocate a dataset of shape (n, d). Storage is value-initialized
    /// to 0.0f; the caller is expected to overwrite every cell before
    /// any algorithm reads from it.
    Dataset(std::size_t n_points, std::size_t dimensions)
        : n{n_points}
        , d{dimensions}
        , data(n_points * dimensions)
    {
    }

    /// Read-only view of row `i` as a contiguous span of `d` floats.
    [[nodiscard]] std::span<const float> row(std::size_t i) const noexcept
    {
        assert(i < n);
        return {data.data() + i * d, d};
    }

    /// Mutable view of row `i` as a contiguous span of `d` floats.
    [[nodiscard]] std::span<float> row(std::size_t i) noexcept
    {
        assert(i < n);
        return {data.data() + i * d, d};
    }
};

} // namespace knng
