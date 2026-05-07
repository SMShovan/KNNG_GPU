/// @file
/// @brief Distributed NN-Descent — gather-scatter baseline (Step 41).
///
/// Each iteration:
///   1. Build local new/old snapshots from the rank-local NnDescentGraph.
///   2. Determine which remote global IDs are needed (neighbors + reverse).
///   3. AlltoAll exchange: ranks send their needed remote-ID lists; each
///      rank responds with the requested feature vectors.
///   4. Run the local join using the fetched features.
///   5. Allreduce the update count; check convergence.

#include "knng/dist/nn_descent_mpi.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "knng/core/dataset.hpp"
#include "knng/core/distance.hpp"
#include "knng/cpu/neighbor_list.hpp"
#include "knng/cpu/nn_descent.hpp"
#include "knng/random.hpp"

namespace knng::dist {

namespace {

/// For each local neighbor-list entry with a global ID owned by another rank,
/// collect those IDs into per-rank request lists.
/// Returns a vector of size `num_ranks`; entry `r` lists global IDs
/// whose owning rank is `r` and that we need but do not own.
std::vector<std::vector<knng::index_t>>
build_remote_requests(const knng::cpu::NnDescentGraph& graph,
                      const ShardedDataset&            shard)
{
    const int size  = shard.size();
    const std::size_t global_n = shard.global_n();
    const std::size_t local_n  = shard.local_n();

    std::vector<std::vector<knng::index_t>> requests(
        static_cast<std::size_t>(size));

    for (std::size_t li = 0; li < local_n; ++li) {
        const auto& list = graph.at(li);
        for (const auto& entry : list.view()) {
            const auto gid = entry.id;
            if (gid >= static_cast<knng::index_t>(global_n)) {
                continue; // sentinel
            }
            // Find owning rank by linear scan (fine for Phase 6's small P).
            for (int r = 0; r < size; ++r) {
                const auto [rs, rc] = compute_shard(global_n, size, r);
                if (gid >= static_cast<knng::index_t>(rs) &&
                    gid < static_cast<knng::index_t>(rs + rc))
                {
                    if (r != shard.rank()) {
                        requests[static_cast<std::size_t>(r)].push_back(gid);
                    }
                    break;
                }
            }
        }
    }

    // Deduplicate each per-rank list (basic alloc-and-sort approach).
    for (auto& req : requests) {
        std::sort(req.begin(), req.end());
        req.erase(std::unique(req.begin(), req.end()), req.end());
    }
    return requests;
}

/// Exchange ID request lists and return a map from global_id → feature row.
/// Uses MPI_Alltoall for the counts, then MPI_Alltoallv for the IDs,
/// then a second MPI_Alltoallv to return the feature vectors.
std::unordered_map<knng::index_t, std::vector<float>>
exchange_features(const std::vector<std::vector<knng::index_t>>& requests,
                  const ShardedDataset&                          shard,
                  MPI_Comm                                       comm,
                  std::size_t& bytes_sent_out)
{
    const int    size = shard.size();
    const int    rank = shard.rank();
    const std::size_t d = shard.d();

    // --- Round 1: exchange request counts ---
    std::vector<int> send_counts(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        send_counts[static_cast<std::size_t>(r)] =
            static_cast<int>(requests[static_cast<std::size_t>(r)].size());
    }
    std::vector<int> recv_counts(static_cast<std::size_t>(size));
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 comm);

    // --- Round 2: exchange the actual IDs (MPI_Alltoallv) ---
    std::vector<int> send_displs(static_cast<std::size_t>(size), 0);
    std::vector<int> recv_displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        send_displs[static_cast<std::size_t>(r)] =
            send_displs[static_cast<std::size_t>(r - 1)] +
            send_counts[static_cast<std::size_t>(r - 1)];
        recv_displs[static_cast<std::size_t>(r)] =
            recv_displs[static_cast<std::size_t>(r - 1)] +
            recv_counts[static_cast<std::size_t>(r - 1)];
    }

    // Pack send IDs.
    const int total_send_ids = send_displs.back() + send_counts.back();
    const int total_recv_ids = recv_displs.back() + recv_counts.back();
    std::vector<knng::index_t> send_ids;
    send_ids.reserve(static_cast<std::size_t>(total_send_ids));
    for (int r = 0; r < size; ++r) {
        for (auto id : requests[static_cast<std::size_t>(r)]) {
            send_ids.push_back(id);
        }
    }
    std::vector<knng::index_t> recv_ids(
        static_cast<std::size_t>(total_recv_ids));

    MPI_Alltoallv(send_ids.data(),  send_counts.data(), send_displs.data(),
                  MPI_UNSIGNED,
                  recv_ids.data(),  recv_counts.data(), recv_displs.data(),
                  MPI_UNSIGNED,
                  comm);

    // --- Round 3: reply with feature vectors ---
    // For each received ID, we own that point, so look it up locally.
    // Build per-rank feature reply buffers.
    std::vector<std::vector<float>> reply_bufs(
        static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        const int start = recv_displs[static_cast<std::size_t>(r)];
        const int count = recv_counts[static_cast<std::size_t>(r)];
        auto& buf = reply_bufs[static_cast<std::size_t>(r)];
        buf.resize(static_cast<std::size_t>(count) * d);
        for (int idx = 0; idx < count; ++idx) {
            const auto gid = recv_ids[static_cast<std::size_t>(start + idx)];
            const std::size_t local_i = gid - static_cast<knng::index_t>(
                shard.local_start());
            const float* src = shard.local_dataset().data_ptr() +
                               local_i * d;
            float* dst = buf.data() +
                         static_cast<std::size_t>(idx) * d;
            std::copy(src, src + d, dst);
        }
    }

    // Pack reply send buffer.
    std::vector<int> reply_send_counts(static_cast<std::size_t>(size));
    std::vector<int> reply_send_displs(static_cast<std::size_t>(size), 0);
    for (int r = 0; r < size; ++r) {
        reply_send_counts[static_cast<std::size_t>(r)] =
            recv_counts[static_cast<std::size_t>(r)] * static_cast<int>(d);
    }
    for (int r = 1; r < size; ++r) {
        reply_send_displs[static_cast<std::size_t>(r)] =
            reply_send_displs[static_cast<std::size_t>(r - 1)] +
            reply_send_counts[static_cast<std::size_t>(r - 1)];
    }
    std::vector<float> packed_reply;
    packed_reply.reserve(static_cast<std::size_t>(
        reply_send_displs.back() + reply_send_counts.back()));
    for (int r = 0; r < size; ++r) {
        for (float f : reply_bufs[static_cast<std::size_t>(r)]) {
            packed_reply.push_back(f);
        }
    }

    // Track bytes sent (feature vectors we're sending in response).
    bytes_sent_out += packed_reply.size() * sizeof(float);

    // Compute receive-side counts/displs for feature vectors.
    std::vector<int> feat_recv_counts(static_cast<std::size_t>(size));
    std::vector<int> feat_recv_displs(static_cast<std::size_t>(size), 0);
    for (int r = 0; r < size; ++r) {
        feat_recv_counts[static_cast<std::size_t>(r)] =
            send_counts[static_cast<std::size_t>(r)] * static_cast<int>(d);
    }
    for (int r = 1; r < size; ++r) {
        feat_recv_displs[static_cast<std::size_t>(r)] =
            feat_recv_displs[static_cast<std::size_t>(r - 1)] +
            feat_recv_counts[static_cast<std::size_t>(r - 1)];
    }
    const int total_feat_recv = feat_recv_displs.back() +
                                feat_recv_counts.back();
    std::vector<float> feat_recv(static_cast<std::size_t>(total_feat_recv));

    MPI_Alltoallv(packed_reply.data(),
                  reply_send_counts.data(), reply_send_displs.data(),
                  MPI_FLOAT,
                  feat_recv.data(),
                  feat_recv_counts.data(), feat_recv_displs.data(),
                  MPI_FLOAT,
                  comm);

    // Build global_id → feature-vector map.
    std::unordered_map<knng::index_t, std::vector<float>> cache;
    cache.reserve(send_ids.size());
    for (std::size_t i = 0; i < send_ids.size(); ++i) {
        std::vector<float> fvec(
            feat_recv.begin() + static_cast<std::ptrdiff_t>(i * d),
            feat_recv.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
        cache[send_ids[i]] = std::move(fvec);
    }

    // Also add locally-owned features to the cache for uniform lookups.
    for (std::size_t li = 0; li < shard.local_n(); ++li) {
        const auto gid = static_cast<knng::index_t>(shard.global_index(li));
        const float* ptr = shard.local_dataset().data_ptr() + li * d;
        cache[gid] = std::vector<float>(ptr, ptr + d);
    }

    return cache;
}

/// Compute squared-L2 distance between two float vectors.
inline float l2sq(const float* a, const float* b, std::size_t d) noexcept
{
    float acc = 0.0f;
    for (std::size_t j = 0; j < d; ++j) {
        const float diff = a[j] - b[j];
        acc += diff * diff;
    }
    return acc;
}

/// One iteration of the distributed local join.
/// Uses the feature cache built by `exchange_features`.
std::size_t local_join_distributed(
    const ShardedDataset& shard,
    knng::cpu::NnDescentGraph& graph,
    const std::unordered_map<knng::index_t, std::vector<float>>& cache)
{
    const std::size_t d = shard.d();
    const std::size_t local_n = shard.local_n();
    std::size_t updates = 0;

    // Snapshot new/old lists per point.
    std::vector<std::vector<knng::index_t>> new_ids(local_n);
    std::vector<std::vector<knng::index_t>> old_ids(local_n);

    for (std::size_t li = 0; li < local_n; ++li) {
        auto& list = graph.at(li);
        for (const auto& entry : list.view()) {
            const auto gid = entry.id;
            if (gid == std::numeric_limits<knng::index_t>::max()) continue;
            if (entry.is_new) {
                new_ids[li].push_back(gid);
            } else {
                old_ids[li].push_back(gid);
            }
        }
        list.mark_all_old();
    }

    // Local join: for each point, process new×new and new×old pairs.
    for (std::size_t li = 0; li < local_n; ++li) {
        const auto& ni = new_ids[li];
        const auto& oi = old_ids[li];

        auto process_pair = [&](knng::index_t u, knng::index_t v) {
            if (u == v) return;
            const auto it_u = cache.find(u);
            const auto it_v = cache.find(v);
            if (it_u == cache.end() || it_v == cache.end()) return;
            const float dist = l2sq(it_u->second.data(),
                                    it_v->second.data(), d);
            // Determine if u is local; if so, insert into u's list.
            const std::size_t u_start = shard.local_start();
            const std::size_t u_end   = shard.local_end();
            const auto u_sz  = static_cast<std::size_t>(u);
            if (u_sz >= u_start && u_sz < u_end) {
                const std::size_t u_li = u_sz - u_start;
                if (graph.at(u_li).insert(v, dist, true)) {
                    ++updates;
                }
            }
            // Similarly for v.
            const auto v_sz = static_cast<std::size_t>(v);
            if (v_sz >= u_start && v_sz < u_end) {
                const std::size_t v_li = v_sz - u_start;
                if (graph.at(v_li).insert(u, dist, true)) {
                    ++updates;
                }
            }
        };

        // new × new (avoid counting (u,v) and (v,u) separately).
        for (std::size_t a = 0; a < ni.size(); ++a) {
            for (std::size_t b = a + 1; b < ni.size(); ++b) {
                process_pair(ni[a], ni[b]);
            }
        }
        // new × old.
        for (auto u : ni) {
            for (auto v : oi) {
                process_pair(u, v);
            }
        }
    }

    return updates;
}

} // namespace

Knng nn_descent_mpi(const ShardedDataset&     shard,
                    std::size_t               k,
                    const NnDescentMpiConfig& cfg,
                    MPI_Comm                  comm)
{
    std::vector<NnDescentMpiLog> log;
    return nn_descent_mpi_with_log(shard, k, cfg, comm, log);
}

Knng nn_descent_mpi_with_log(const ShardedDataset&         shard,
                              std::size_t                   k,
                              const NnDescentMpiConfig&     cfg,
                              MPI_Comm                      comm,
                              std::vector<NnDescentMpiLog>& log_out)
{
    const std::size_t local_n  = shard.local_n();
    const std::size_t global_n = shard.global_n();

    if (global_n == 0 || k == 0) {
        throw std::invalid_argument{
            "knng::dist::nn_descent_mpi: empty dataset or k=0"};
    }
    if (k >= global_n) {
        throw std::invalid_argument{
            "knng::dist::nn_descent_mpi: k must be < global_n"};
    }

    log_out.clear();

    // Each rank initialises its local graph with random neighbors drawn
    // from the *global* index space.  We use the local brute-force
    // random init adapted for global indices.
    const std::uint64_t rank_seed =
        cfg.seed ^ (static_cast<std::uint64_t>(shard.rank()) * 0x9e3779b97f4a7c15ULL);
    knng::random::XorShift64 rng{rank_seed};

    knng::cpu::NnDescentGraph graph(local_n, k);
    // Random init: each local point gets k random global neighbours.
    for (std::size_t li = 0; li < local_n; ++li) {
        const auto q_global = static_cast<knng::index_t>(shard.global_index(li));
        std::size_t attempts = 0;
        const std::size_t max_attempts = k * 10 + 100;
        while (graph.at(li).size() < k && attempts < max_attempts) {
            ++attempts;
            const auto gid = static_cast<knng::index_t>(rng.next() % global_n);
            if (gid == q_global) continue;
            // Use a placeholder distance of 0 for random init;
            // the local join will refine on first iteration.
            graph.at(li).insert(gid, 0.0f, true);
        }
    }

    const double threshold =
        cfg.delta * static_cast<double>(global_n) *
        static_cast<double>(k);

    for (std::size_t iter = 0; iter < cfg.max_iters; ++iter) {
        // Build and exchange remote feature requests.
        auto requests = build_remote_requests(graph, shard);
        std::size_t bytes_sent = 0;
        auto cache = exchange_features(requests, shard, comm, bytes_sent);

        // Local join.
        const std::size_t local_updates =
            local_join_distributed(shard, graph, cache);

        // Global convergence check via Allreduce.
        std::size_t global_updates = 0;
        MPI_Allreduce(&local_updates, &global_updates, 1,
                      MPI_UNSIGNED_LONG, MPI_SUM, comm);

        // Aggregate bytes_sent.
        std::size_t total_bytes = 0;
        MPI_Allreduce(&bytes_sent, &total_bytes, 1,
                      MPI_UNSIGNED_LONG, MPI_SUM, comm);

        log_out.push_back({
            iter + 1,
            global_updates,
            static_cast<double>(global_updates) /
                (static_cast<double>(global_n) * static_cast<double>(k)),
            total_bytes
        });

        if (static_cast<double>(global_updates) < threshold) {
            break;
        }
    }

    // Convert NnDescentGraph → Knng with global indices already in place.
    return graph.to_knng();
}

} // namespace knng::dist
