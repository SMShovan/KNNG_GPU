#pragma once

/// @file
/// @brief Step 59 — Stream-pipelined GPU brute-force KNN.
///
/// Datasets large enough not to fit in GPU VRAM require chunked processing:
/// the query set is split into chunks, each chunk is transferred to the GPU,
/// KNN is computed on-device, and the result is transferred back.  Done
/// naively (transfer, compute, transfer back, repeat) the GPU is idle during
/// transfers.  Stream pipelining overlaps the three stages using `N_STREAMS`
/// independent CUDA streams:
///
/// ```
///   time →
///   stream 0: [H2D chunk 0] [compute 0] [D2H 0]
///   stream 1:               [H2D chunk 1] [compute 1] [D2H 1]
///   stream 2:                             [H2D chunk 2] ...
/// ```
///
/// The expected speedup on large datasets is `min(N_STREAMS, stage_time_ratio)`.
/// For typical SIFT-128 n=1M, compute dominates; stream overlap is most
/// valuable for smaller d where transfer and compute balance.
///
/// ### Chunking
/// The query set (n_queries × d floats) is partitioned into `ceil(n / chunk_sz)`
/// chunks of `chunk_sz` queries each.  The reference set is always fully
/// on-device.  Chunk size is tunable; the default (`kDefaultChunkSz = 1024`)
/// fits one chunk in ~512 KB of pinned host memory for d=128.
///
/// ### CPU reference
/// `cpu_streamed_brute_force_knn` processes chunks sequentially without any
/// actual streaming (no CUDA streams) — it validates correctness of the
/// chunked algorithm independently of the pipelining mechanism.

#include <knng/core/dataset.hpp>
#include <knng/core/graph.hpp>

#include <cstddef>

namespace knng::gpu {

/// Default query chunk size for the streamed brute-force kernel.
inline constexpr std::size_t kDefaultChunkSz = 1024;

/// Number of streams used in the pipelined launcher.
inline constexpr std::size_t kNStreams = 3;

// ---------------------------------------------------------------------------
// CPU reference (always compiled)
// ---------------------------------------------------------------------------

/// @brief Chunked brute-force KNN on the CPU — same output as cpu_brute_force_knn.
///
/// Validates the correctness of the chunked algorithm without streaming.
///
/// @param dataset    Input point set.
/// @param k          Neighbours per query.
/// @param chunk_sz   Queries per chunk (default: kDefaultChunkSz).
/// @return           Knng sorted by distance.
[[nodiscard]] knng::Knng cpu_streamed_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::size_t          chunk_sz = kDefaultChunkSz);

// ---------------------------------------------------------------------------
// GPU path
// ---------------------------------------------------------------------------

#if defined(KNNG_HAVE_CUDA) || defined(KNNG_HAVE_HIP)

/// @brief Stream-pipelined brute-force KNN.
///
/// Splits the query set into chunks of `chunk_sz`, uses `kNStreams` CUDA
/// streams to overlap H2D, compute, and D2H stages.
///
/// @param dataset    Input point set.
/// @param k          Neighbours per query.
/// @param chunk_sz   Queries per chunk.
/// @return           Knng sorted by distance.
[[nodiscard]] knng::Knng gpu_streamed_brute_force_knn(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::size_t          chunk_sz = kDefaultChunkSz);

#endif

} // namespace knng::gpu
