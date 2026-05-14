/// @file
/// @brief Step 90 — DistTopology implementation (CPU reference / portable).
///
/// Uses MPI_Get_processor_name to discover node co-location and assigns
/// GPU indices as rank % gpus_per_node.  This is a pure C++/MPI file —
/// no CUDA required — so it lives in knng_dist_gpu_ref and compiles on Mac.

#include <knng/dist_gpu/topology.hpp>

#include <mpi.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace knng::dist_gpu {

// ---------------------------------------------------------------------------
DistTopology DistTopology::build(MPI_Comm comm, int gpus_per_node)
{
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather processor names from all ranks.
    char local_name[MPI_MAX_PROCESSOR_NAME] = {};
    int  name_len = 0;
    MPI_Get_processor_name(local_name, &name_len);

    // Allgather names (each slot MPI_MAX_PROCESSOR_NAME chars).
    std::vector<char> all_names(static_cast<std::size_t>(size) *
                                MPI_MAX_PROCESSOR_NAME, '\0');
    MPI_Allgather(local_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);

    // Assign node IDs: same hostname ↔ same node_id.
    std::map<std::string, int> hostname_to_node;
    int next_node = 0;

    DistTopology topo;
    topo.my_rank_ = rank;
    topo.ranks_.resize(static_cast<std::size_t>(size));

    for (int r = 0; r < size; ++r) {
        const char* base =
            all_names.data() + static_cast<std::ptrdiff_t>(r) * MPI_MAX_PROCESSOR_NAME;
        std::string hn(base);

        auto [it, inserted] = hostname_to_node.emplace(hn, next_node);
        if (inserted) ++next_node;

        topo.ranks_[static_cast<std::size_t>(r)].rank     = r;
        topo.ranks_[static_cast<std::size_t>(r)].node_id  = it->second;
        topo.ranks_[static_cast<std::size_t>(r)].hostname = std::move(hn);
    }
    topo.num_nodes_ = next_node;

    // Assign GPU indices within each node (round-robin by local rank order).
    // node_local_counter[node_id] counts how many ranks have been assigned
    // to that node so far.
    if (gpus_per_node > 0) {
        std::vector<int> local_ctr(static_cast<std::size_t>(topo.num_nodes_), 0);
        for (int r = 0; r < size; ++r) {
            auto& ri       = topo.ranks_[static_cast<std::size_t>(r)];
            int   node     = ri.node_id;
            ri.local_gpu   = local_ctr[static_cast<std::size_t>(node)] % gpus_per_node;
            ++local_ctr[static_cast<std::size_t>(node)];
        }
    } else {
        for (auto& ri : topo.ranks_) ri.local_gpu = -1;
    }

    // Build node_peers_: node_id → sorted list of ranks on that node.
    topo.node_peers_.resize(static_cast<std::size_t>(topo.num_nodes_));
    for (int r = 0; r < size; ++r) {
        int nid = topo.ranks_[static_cast<std::size_t>(r)].node_id;
        topo.node_peers_[static_cast<std::size_t>(nid)].push_back(r);
    }

    return topo;
}

// ---------------------------------------------------------------------------
const std::vector<int>& DistTopology::intra_node_ranks(int rank) const
{
    int nid = ranks_[static_cast<std::size_t>(rank)].node_id;
    return node_peers_[static_cast<std::size_t>(nid)];
}

// ---------------------------------------------------------------------------
std::string DistTopology::to_string() const
{
    if (my_rank_ != 0) return {};

    std::ostringstream oss;
    oss << "DistTopology: " << size() << " ranks, "
        << num_nodes_ << " node(s)\n";
    for (int n = 0; n < num_nodes_; ++n) {
        oss << "  node " << n << " [" << ranks_[static_cast<std::size_t>(
                node_peers_[static_cast<std::size_t>(n)][0])].hostname << "]: ranks {";
        for (int r : node_peers_[static_cast<std::size_t>(n)])
            oss << r << ' ';
        oss << "}\n";
    }
    return oss.str();
}

} // namespace knng::dist_gpu
