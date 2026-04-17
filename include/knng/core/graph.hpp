#pragma once

/// @file
/// @brief The in-memory representation of a K-nearest-neighbor graph.
///
/// A `Knng` is a fixed-K adjacency structure stored as two parallel
/// flat arrays of shape `(n, k)` in row-major order:
///
///     neighbors[i*k + j]   — global index of the j-th neighbor of i
///     distances[i*k + j]   — distance from i to that neighbor, under
///                            whichever `Distance` metric was used to
///                            build the graph
///
/// The layout is deliberately plain. Every later pipeline stage — CPU
/// refinement, GPU kernels, multi-GPU exchange, MPI all-to-all — will
/// want to memcpy entire rows or entire arrays, and the row-major
/// `(n, k)` layout is exactly the shape those stages consume.
///
/// `Knng` makes no claim about neighbor ordering. Some builders emit
/// rows sorted ascending by distance; others (including most
/// incremental builders) leave rows in arbitrary order. Callers that
/// need a sorted row should sort it explicitly.

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

#include "knng/core/types.hpp"

namespace knng {

/// Plain-old-data K-nearest-neighbor graph. See file-level docs for
/// the storage contract.
struct Knng {
    std::size_t          n{};  ///< Number of points (rows).
    std::size_t          k{};  ///< Neighbors per point (columns).
    std::vector<index_t> neighbors;
    std::vector<float>   distances;

    /// Default-construct an empty (0x0) graph. Useful as a placeholder
    /// before a builder fills it in.
    Knng() = default;

    /// Allocate a graph of shape (n, k). Storage is value-initialized
    /// (neighbor indices to 0, distances to 0.0f); the caller is
    /// expected to overwrite every cell before any search or
    /// refinement routine reads the graph.
    Knng(std::size_t n_points, std::size_t k_neighbors)
        : n{n_points}
        , k{k_neighbors}
        , neighbors(n_points * k_neighbors)
        , distances(n_points * k_neighbors)
    {
    }

    /// Mutable view over the neighbor indices of row `i`.
    [[nodiscard]] std::span<index_t> neighbors_of(std::size_t i) noexcept
    {
        assert(i < n);
        return {neighbors.data() + i * k, k};
    }

    /// Read-only view over the neighbor indices of row `i`.
    [[nodiscard]] std::span<const index_t> neighbors_of(std::size_t i) const noexcept
    {
        assert(i < n);
        return {neighbors.data() + i * k, k};
    }

    /// Mutable view over the neighbor distances of row `i`.
    [[nodiscard]] std::span<float> distances_of(std::size_t i) noexcept
    {
        assert(i < n);
        return {distances.data() + i * k, k};
    }

    /// Read-only view over the neighbor distances of row `i`.
    [[nodiscard]] std::span<const float> distances_of(std::size_t i) const noexcept
    {
        assert(i < n);
        return {distances.data() + i * k, k};
    }
};

} // namespace knng
