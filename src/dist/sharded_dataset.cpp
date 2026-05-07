/// @file
/// @brief `ShardedDataset` — construction, scatter, and gather.

#include "knng/dist/sharded_dataset.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace knng::dist {

ShardedDataset::ShardedDataset(Dataset     local_data,
                               std::size_t global_n,
                               std::size_t local_start,
                               int         rank,
                               int         size)
    : local_{std::move(local_data)}
    , global_n_{global_n}
    , local_start_{local_start}
    , rank_{rank}
    , size_{size}
{
    assert(local_.is_contiguous());
}

ShardedDataset ShardedDataset::scatter(const Dataset& root_data,
                                       int            root_rank,
                                       MPI_Comm       comm)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Broadcast global_n and d from root so every rank can compute
    // shard bounds without extra rounds of communication.
    std::size_t header[2] = {0, 0}; // {global_n, d}
    if (rank == root_rank) {
        header[0] = root_data.n;
        header[1] = root_data.d;
    }
    MPI_Bcast(header, 2, MPI_UNSIGNED_LONG, root_rank, comm);

    const std::size_t global_n = header[0];
    const std::size_t d        = header[1];

    if (global_n == 0 || d == 0) {
        throw std::runtime_error{
            "knng::dist::ShardedDataset::scatter: empty dataset"};
    }

    // Compute per-rank shard bounds.
    const auto [my_start, my_count] =
        compute_shard(global_n, size, rank);

    // Build send counts and displacements (in float elements) for
    // MPI_Scatterv from root.
    std::vector<int> send_counts(static_cast<std::size_t>(size));
    std::vector<int> displs(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        const auto [s, c] = compute_shard(global_n, size, r);
        send_counts[static_cast<std::size_t>(r)] =
            static_cast<int>(c * d);
        displs[static_cast<std::size_t>(r)] =
            static_cast<int>(s * d);
    }

    const int recv_count = static_cast<int>(my_count * d);

    Dataset local(my_count, d);
    MPI_Scatterv(
        rank == root_rank ? root_data.data_ptr() : nullptr,
        send_counts.data(),
        displs.data(),
        MPI_FLOAT,
        local.data_ptr(),
        recv_count,
        MPI_FLOAT,
        root_rank,
        comm);

    return ShardedDataset{std::move(local), global_n, my_start, rank, size};
}

Dataset ShardedDataset::gather(int root_rank, MPI_Comm comm) const
{
    // Compute receive counts and displacements for root.
    std::vector<int> recv_counts(static_cast<std::size_t>(size_));
    std::vector<int> displs(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
        const auto [s, c] = compute_shard(global_n_, size_, r);
        recv_counts[static_cast<std::size_t>(r)] =
            static_cast<int>(c * local_.d);
        displs[static_cast<std::size_t>(r)] =
            static_cast<int>(s * local_.d);
    }

    const int send_count = static_cast<int>(local_.n * local_.d);

    Dataset result;
    if (rank_ == root_rank) {
        result = Dataset(global_n_, local_.d);
    }

    MPI_Gatherv(
        local_.data_ptr(),
        send_count,
        MPI_FLOAT,
        rank_ == root_rank ? result.data_ptr() : nullptr,
        recv_counts.data(),
        displs.data(),
        MPI_FLOAT,
        root_rank,
        comm);

    return result;
}

} // namespace knng::dist
