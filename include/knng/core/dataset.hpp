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
/// ## Storage contract (formalised at Step 18)
///
///   * `data` is a single flat `std::vector<float>` of length `n * d`.
///   * Element `(i, j)` lives at `data[i * d + j]` — row-major, where
///     row `i` is the i-th feature vector and column `j` is its j-th
///     coordinate.
///   * The row stride is exactly `d` floats; there is no inter-row
///     padding. `stride()` returns this constant; `byte_stride()`
///     multiplies by `sizeof(float)`. Future GPU paths that demand a
///     wider stride for alignment will introduce a *separate* type
///     (`PaddedDataset` or similar) rather than complicate this one.
///   * The buffer is *contiguous*. `data_ptr()` returns a `float*`
///     suitable for direct memcpy, `cublasSgemm`, `mmap`, and
///     CUDA / HIP H2D transfer. No padded-row gymnastics, no
///     vector-of-vectors. This is the property every later
///     vectorisation step (Step 19's `||p||²` precompute, Step 20's
///     L1-tile blocking, Step 21's BLAS GEMM, Step 27's hand-vectorised
///     SIMD distance kernel, Step 49's GPU brute-force) relies on.
///   * The row stride `d` is identical to the row stride of every
///     row of `Knng`'s `(n, k)` adjacency layout. Callers reason
///     about "row of the dataset" and "row of the graph" with the
///     same arithmetic.
///
/// ## Why a single flat float buffer, not a `std::vector<std::vector<float>>`?
///
/// Three reasons that compound:
///
///   1. **Vectorisation.** A SIMD load (`_mm256_loadu_ps`,
///      `vld1q_f32`, `__shfl_sync` of contiguous floats) needs a
///      contiguous source. A vector-of-vectors gives the compiler
///      no information about whether two rows live in adjacent
///      memory; the autovectoriser bails out on the inner loop and
///      we lose 4–8× throughput before the first hand-tuned
///      intrinsic appears.
///   2. **Cache locality.** Row stride `d` means scanning the
///      reference set is a single forward stride over a contiguous
///      block. The hardware prefetcher latches onto it; the
///      vector-of-vectors variant emits a pointer dereference per
///      row, every one of which can stall on an L1 miss.
///   3. **Zero-copy GPU transfer.** `cudaMemcpy(dst, ds.data_ptr(),
///      n * d * sizeof(float), cudaMemcpyHostToDevice)` is one
///      syscall against a contiguous source. Anything else
///      requires staging through a packed buffer first — twice
///      the memory bandwidth, twice the latency.
///
/// ## Why a plain struct, not a class with private members?
///
/// `Dataset` is a value type with no invariants beyond
/// `data.size() == n * d` (verified by `is_contiguous()`), and
/// every consumer wants direct access to the underlying buffer for
/// memcpy, BLAS calls, and GPU H2D transfers. Hiding the buffer
/// behind accessors would only obscure that. The accessors below
/// (`data_ptr()`, `stride()`, `row(i)`, `is_contiguous()`) are
/// *additions* on top of the public field, not a substitute.

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

    /// Row stride, in elements (floats). Identical to `d` for the
    /// canonical layout; exposed as a function so future
    /// alignment-padded variants can be a one-line change at every
    /// call site.
    [[nodiscard]] std::size_t stride() const noexcept { return d; }

    /// Row stride, in bytes. The natural denominator for any
    /// `cudaMemcpy2D` / `cublasSgemm` LDA argument later phases will
    /// pass it to.
    [[nodiscard]] std::size_t byte_stride() const noexcept
    {
        return d * sizeof(float);
    }

    /// Direct pointer to the start of the row-major buffer. Used
    /// where a `std::span` would be syntactic noise — BLAS calls,
    /// GPU H2D transfers, the `mmap`-backed loader. Read-only
    /// overload first; the const-correctness lets a `const Dataset&`
    /// hand its buffer to a BLAS routine that takes
    /// `const float*` without an explicit cast.
    [[nodiscard]] const float* data_ptr() const noexcept
    {
        return data.data();
    }

    /// Mutable pointer to the start of the row-major buffer.
    [[nodiscard]] float* data_ptr() noexcept { return data.data(); }

    /// Total number of float elements in the contiguous buffer.
    /// Equivalent to `n * d`; pre-named so callers do not have to
    /// compute the product (and risk a `size_t` overflow on
    /// pathological inputs).
    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }

    /// True iff the storage contract is intact: `data.size() == n * d`.
    /// Cheap (one multiply, one compare); intended for the
    /// preconditions of every algorithm that consumes a `Dataset`.
    /// The function is `noexcept` and does not allocate, so a debug
    /// build's `assert(ds.is_contiguous())` is essentially free.
    [[nodiscard]] bool is_contiguous() const noexcept
    {
        return data.size() == n * d;
    }
};

} // namespace knng
