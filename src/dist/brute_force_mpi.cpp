/// @file
/// @brief Distributed brute-force KNN — ring-shift reference columns.

#include "knng/dist/brute_force_mpi.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/top_k.hpp"

namespace knng::dist {

namespace {

/// Compute squared-L2 distance between two float vectors of length `d`.
inline float l2sq(const float* __restrict__ a,
                  const float* __restrict__ b,
                  std::size_t d) noexcept
{
    float acc = 0.0f;
    for (std::size_t j = 0; j < d; ++j) {
        const float diff = a[j] - b[j];
        acc += diff * diff;
    }
    return acc;
}

} // namespace

Knng brute_force_knn_mpi(const ShardedDataset& shard,
                         std::size_t           k,
                         MPI_Comm              comm)
{
    const std::size_t local_n  = shard.local_n();
    const std::size_t global_n = shard.global_n();
    const std::size_t d        = shard.d();
    const int         rank     = shard.rank();
    const int         size     = shard.size();

    if (global_n == 0 || k == 0) {
        throw std::invalid_argument{
            "knng::dist::brute_force_knn_mpi: empty dataset or k=0"};
    }
    if (k >= global_n) {
        throw std::invalid_argument{
            "knng::dist::brute_force_knn_mpi: k must be < global_n"};
    }

    // Per-query top-k accumulators (global neighbor indices).
    std::vector<TopK> heaps;
    heaps.reserve(local_n);
    for (std::size_t i = 0; i < local_n; ++i) {
        heaps.emplace_back(k);
    }

    // Ring buffer holds the current reference block being scored.
    // We use two buffers to overlap send and recv via MPI_Sendrecv.
    // The first shard in the ring is this rank's own data.
    std::vector<float> ring_buf(shard.local_dataset().data);
    std::size_t ring_start = shard.local_start(); // global index of ring_buf[0]

    // The actual count of points in the ring buffer this iteration.
    std::size_t ring_n = local_n;

    for (int step = 0; step < size; ++step) {
        // Score all local queries against ring_buf.
        for (std::size_t qi = 0; qi < local_n; ++qi) {
            const float* q_ptr =
                shard.local_dataset().data_ptr() + qi * d;
            const std::size_t q_global = shard.global_index(qi);

            for (std::size_t ri = 0; ri < ring_n; ++ri) {
                const std::size_t r_global = ring_start + ri;
                if (r_global == q_global) {
                    continue; // skip self
                }
                const float* r_ptr = ring_buf.data() + ri * d;
                const float  dist  = l2sq(q_ptr, r_ptr, d);
                heaps[qi].push(static_cast<knng::index_t>(r_global), dist);
            }
        }

        if (step == size - 1) {
            break; // last step — no ring shift needed
        }

        // Ring shift: send ring_buf to the right neighbour, receive
        // the next reference block from the left neighbour.
        const int right = (rank + 1) % size;
        const int left  = (rank - 1 + size) % size;

        // The left neighbour will send us its shard (which originated
        // from rank `(rank - step - 1 + size) % size`).
        const int source_rank =
            (rank - step - 1 + size) % size;
        const auto [next_start, next_count] =
            compute_shard(global_n, size, source_rank);

        std::vector<float> recv_buf(next_count * d);

        MPI_Sendrecv(
            ring_buf.data(),
            static_cast<int>(ring_n * d),
            MPI_FLOAT,
            right,
            /* sendtag = */ step,
            recv_buf.data(),
            static_cast<int>(next_count * d),
            MPI_FLOAT,
            left,
            /* recvtag = */ step,
            comm,
            MPI_STATUS_IGNORE);

        ring_buf   = std::move(recv_buf);
        ring_start = next_start;
        ring_n     = next_count;
    }

    // Materialise top-k results into a Knng (rows in local-query order).
    Knng out(local_n, k);
    for (std::size_t qi = 0; qi < local_n; ++qi) {
        const auto sorted       = heaps[qi].extract_sorted();
        auto       neighbor_row = out.neighbors_of(qi);
        auto       dist_row     = out.distances_of(qi);
        for (std::size_t j = 0; j < sorted.size(); ++j) {
            neighbor_row[j] = sorted[j].first;
            dist_row[j]     = sorted[j].second;
        }
        for (std::size_t j = sorted.size(); j < k; ++j) {
            neighbor_row[j] = std::numeric_limits<knng::index_t>::max();
            dist_row[j]     = std::numeric_limits<float>::infinity();
        }
    }
    return out;
}

Knng gather_graph(const Knng&           local_graph,
                  const ShardedDataset& shard,
                  int                   root_rank,
                  MPI_Comm              comm)
{
    const std::size_t k       = local_graph.k();
    const std::size_t global_n = shard.global_n();
    const int         size    = shard.size();
    const int         rank    = shard.rank();

    // Each rank sends 2*local_n*k values: k neighbor ids + k distances.
    // We pack (index, distance) pairs interleaved into a float buffer
    // (casting index_t to float and back is lossless for values < 2^24;
    // for correctness we send index and distance as separate MPI messages).

    // Send neighbor IDs (as uint32_t) and distances (as float) separately.
    const std::size_t local_n = shard.local_n();

    // Build per-rank send counts and displacements.
    std::vector<int> recv_counts(static_cast<std::size_t>(size));
    std::vector<int> displs(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        const auto [s, c] = compute_shard(global_n, size, r);
        recv_counts[static_cast<std::size_t>(r)] =
            static_cast<int>(c * k);
        displs[static_cast<std::size_t>(r)] =
            static_cast<int>(s * k);
    }

    Knng global_graph;
    if (rank == root_rank) {
        global_graph = Knng(global_n, k);
    }

    // Gather neighbor IDs.
    MPI_Gatherv(
        local_graph.neighbors_of(0).data(),
        static_cast<int>(local_n * k),
        MPI_UNSIGNED,
        rank == root_rank ? global_graph.neighbors_of(0).data() : nullptr,
        recv_counts.data(),
        displs.data(),
        MPI_UNSIGNED,
        root_rank,
        comm);

    // Gather distances.
    MPI_Gatherv(
        local_graph.distances_of(0).data(),
        static_cast<int>(local_n * k),
        MPI_FLOAT,
        rank == root_rank ? global_graph.distances_of(0).data() : nullptr,
        recv_counts.data(),
        displs.data(),
        MPI_FLOAT,
        root_rank,
        comm);

    return global_graph;
}

} // namespace knng::dist
