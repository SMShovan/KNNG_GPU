#pragma once

/// @file
/// @brief Fundamental scalar and index types used throughout `knng`.
///
/// Every public API in the library — CPU and GPU — consumes and
/// produces these types, so pinning them here keeps the ABI stable.
/// The choices below are deliberately narrow:
///
///   * `index_t` is 32-bit unsigned. 2^32 points (~4.3 billion) is far
///     more than any single-node KNNG workload we expect, and fixing
///     the width keeps graph memory footprint predictable across
///     32-bit and 64-bit hosts. Using an unsigned type also makes the
///     "sentinel == max()" idiom for "no neighbor yet" unambiguous.
///   * `dim_t` is a separate alias of the same underlying type. The
///     distinction is purely documentary: a function signature that
///     says `dim_t dimensions` tells the reader something that
///     `index_t dimensions` would not.
///
/// When the library grows a 16-bit ID path (for quantized sub-graph
/// batches on GPU), it will live alongside `index_t` rather than
/// replace it — consumers must continue to see a stable `knng::index_t`.

#include <cstdint>

namespace knng {

/// Zero-based index into a point set / row of a KNNG.
using index_t = std::uint32_t;

/// Dimensionality of a feature vector.
using dim_t = std::uint32_t;

} // namespace knng
