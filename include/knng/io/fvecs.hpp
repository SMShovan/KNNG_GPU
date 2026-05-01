#pragma once

/// @file
/// @brief Loaders for the `.fvecs` / `.ivecs` / `.bvecs` benchmark formats.
///
/// These are the three file formats used by every standard ANN
/// benchmark dataset the project targets — SIFT1M / SIFT10M / SIFT1B
/// (Jégou et al., the `corpus-texmex` site), GIST1M, Fashion-MNIST.
///
/// File format (common shape, varying element type):
///
/// ```
///   ┌─────────────┬───────────────────────────────┐
///   │ uint32 dim  │ dim × element                 │  ← one record
///   └─────────────┴───────────────────────────────┘
///   ┌─────────────┬───────────────────────────────┐
///   │ uint32 dim  │ dim × element                 │  ← next record
///   └─────────────┴───────────────────────────────┘
///   ...
/// ```
///
/// All integers are little-endian. The element type is:
///   * `.fvecs` → 32-bit IEEE float
///   * `.ivecs` → 32-bit signed int
///   * `.bvecs` → 8-bit unsigned int
///
/// The dimension prefix is repeated for every record. The number of
/// records `n` is not stored anywhere — it is recovered from the
/// total file size: `n = file_size / (4 + dim * sizeof(element))`. The
/// loaders below validate that division is exact and that every
/// record's dim prefix matches the first; mismatches throw
/// `std::runtime_error`.
///
/// Loading strategy: each loader memory-maps the file via POSIX
/// `mmap`, walks it once stripping the per-record dim prefix, and
/// copies the elements into a row-major destination buffer. Copying
/// (rather than zero-copy aliasing) is unavoidable because the
/// destination layout has stride `dim` while the on-disk layout has
/// stride `4 + dim * sizeof(element)`. The mmap saves the kernel-to-
/// userspace `read()` copy and lets the OS demand-page the file under
/// memory pressure.

#include <cstdint>
#include <filesystem>
#include <vector>

#include "knng/core/dataset.hpp"

namespace knng::io {

/// In-memory representation of a `.ivecs` file. Row-major, stride
/// `d` int32 values per row. Used both for ground-truth neighbor
/// lists (where each row is a query's exact neighbor IDs) and for
/// generic int32 datasets.
struct IvecsData {
    std::size_t               n{};
    std::size_t               d{};
    std::vector<std::int32_t> data;
};

/// In-memory representation of a `.bvecs` file in its native uint8
/// element type. Most algorithms in this project consume floats, so
/// `load_bvecs_as_float` is also provided — but a few benchmarks
/// (quantised search) need the raw byte width preserved.
struct BvecsData {
    std::size_t              n{};
    std::size_t              d{};
    std::vector<std::uint8_t> data;
};

/// Load a `.fvecs` file as a row-major float `Dataset`.
///
/// @param path Filesystem path to the `.fvecs` file.
/// @return A `Dataset` whose `data` is `n*d` floats laid out row-major.
/// @throws std::runtime_error if the file cannot be opened, mmap'd,
///         is empty, has an inconsistent record dimension, or has a
///         size that does not divide cleanly into records.
[[nodiscard]] Dataset load_fvecs(const std::filesystem::path& path);

/// Load a `.ivecs` file in its native int32 element type. See
/// `IvecsData` for the storage shape.
[[nodiscard]] IvecsData load_ivecs(const std::filesystem::path& path);

/// Load a `.bvecs` file in its native uint8 element type. See
/// `BvecsData` for the storage shape.
[[nodiscard]] BvecsData load_bvecs(const std::filesystem::path& path);

/// Convenience: load a `.bvecs` file and widen each byte to float so
/// the result drops into the same `Dataset` consumers as `.fvecs`.
/// Useful when running CPU / GPU brute-force over the SIFT1B base.
[[nodiscard]] Dataset load_bvecs_as_float(const std::filesystem::path& path);

} // namespace knng::io
