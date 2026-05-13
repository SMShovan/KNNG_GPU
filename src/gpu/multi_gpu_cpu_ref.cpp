/// @file
/// @brief Steps 82–85 — Multi-GPU CPU reference implementations.
///
/// Always compiled (no CUDA/NCCL dependency).  Each function simulates the
/// corresponding multi-GPU algorithm using CPU arrays and `CpuSimCollective`
/// operations so tests run on Mac without hardware.

#include <knng/gpu/multi_gpu.hpp>
#include <knng/gpu/nccl_comm.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Partition helpers
// ---------------------------------------------------------------------------

std::vector<PointShard> partition_points(std::size_t n,
                                          const MultiGpuConfig& cfg) {
    const std::size_t ng     = static_cast<std::size_t>(cfg.n_gpus);
    const std::size_t chunk  = n / ng;
    const std::size_t rem    = n % ng;
    std::vector<PointShard> shards(ng);
    std::size_t offset = 0;
    for (std::size_t r = 0; r < ng; ++r) {
        const std::size_t sz = chunk + (r < rem ? 1u : 0u);
        shards[r] = {offset, offset + sz, static_cast<int>(r)};
        offset += sz;
    }
    return shards;
}

// ---------------------------------------------------------------------------
// Step 82 — Point-sharded brute-force CPU reference
// ---------------------------------------------------------------------------

namespace {
inline float sq_l2_vec(const float* a, const float* b, std::size_t dim) {
    float s = 0.f;
    for (std::size_t d = 0; d < dim; ++d) { float diff = a[d]-b[d]; s += diff*diff; }
    return s;
}
} // anonymous namespace

knng::Knng cpu_multi_brute_force_point(const knng::Dataset& dataset,
                                        const MultiGpuConfig& cfg) {
    const std::size_t n   = dataset.n;
    const std::size_t dim = dataset.d;
    const std::size_t k   = static_cast<std::size_t>(cfg.k);
    const std::size_t ng  = static_cast<std::size_t>(cfg.n_gpus);
    const float* vecs     = dataset.data.data();

    // Each rank's top-k partial results (dist, id) per query in its shard.
    // We simulate: each rank broadcasts full reference set (bcast of all vecs),
    // computes local distances, then rank 0 gathers and merges.

    const auto shards = partition_points(n, cfg);

    // Global result: for each query point, top-k (dist, id) pairs.
    std::vector<std::vector<std::pair<float,knng::index_t>>> global(n);
    for (auto& g : global) g.resize(k, {std::numeric_limits<float>::infinity(), knng::index_t(-1)});

    for (std::size_t r = 0; r < ng; ++r) {
        const auto& shard = shards[r];
        for (std::size_t qi = shard.begin; qi < shard.end; ++qi) {
            const float* qv = vecs + qi * dim;
            // Compute distance to every reference point
            for (std::size_t j = 0; j < n; ++j) {
                if (j == qi) continue;
                const float d = sq_l2_vec(qv, vecs + j * dim, dim);
                // Insertion into top-k (max-heap semantics: evict worst)
                if (d < global[qi].front().first) {
                    global[qi].front() = {d, static_cast<knng::index_t>(j)};
                    // Re-heapify: bubble down
                    std::make_heap(global[qi].begin(), global[qi].end());
                }
            }
            std::sort(global[qi].begin(), global[qi].end());
        }
    }

    // Build Knng
    knng::Knng result(n, k);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < k; ++r) {
            result.neighbors[i * k + r] = global[i][r].second;
            result.distances[i * k + r] = global[i][r].first;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Step 83 — Tile-sharded brute-force CPU reference
// ---------------------------------------------------------------------------

knng::Knng cpu_multi_brute_force_tile(const knng::Dataset& dataset,
                                       const MultiGpuConfig& cfg) {
    // 2-D partitioning: sqrt(n_gpus) tiles per dimension.
    // For simplicity in the CPU reference, we compute all tiles sequentially
    // and use CpuSimCollective::allreduce_sum to merge distance matrices.
    const std::size_t n   = dataset.n;
    const std::size_t dim = dataset.d;
    const std::size_t k   = static_cast<std::size_t>(cfg.k);
    const float* vecs     = dataset.data.data();

    // Each GPU owns a (query_tile × ref_tile) sub-block of the n×n distance matrix.
    // CPU reference: compute full n×n distance matrix, tile across n_gpus, then merge.
    const std::size_t ng      = static_cast<std::size_t>(cfg.n_gpus);
    const std::size_t q_tiles = std::max<std::size_t>(1, ng / 2);
    const std::size_t r_tiles = std::max<std::size_t>(1, ng / q_tiles);
    const std::size_t q_chunk = (n + q_tiles - 1) / q_tiles;
    const std::size_t r_chunk = (n + r_tiles - 1) / r_tiles;

    knng::Knng result(n, k);
    std::fill(result.neighbors.begin(), result.neighbors.end(), knng::index_t(-1));
    std::fill(result.distances.begin(), result.distances.end(),
              std::numeric_limits<float>::infinity());

    // Per-query top-k state
    std::vector<std::vector<std::pair<float,knng::index_t>>> topk(n);
    for (auto& t : topk) t.resize(k, {std::numeric_limits<float>::infinity(), knng::index_t(-1)});

    for (std::size_t qt = 0; qt < q_tiles; ++qt) {
        const std::size_t q_beg = qt * q_chunk;
        const std::size_t q_end = std::min(n, q_beg + q_chunk);
        for (std::size_t rt = 0; rt < r_tiles; ++rt) {
            const std::size_t r_beg = rt * r_chunk;
            const std::size_t r_end = std::min(n, r_beg + r_chunk);
            // Simulate one GPU processing this tile
            for (std::size_t qi = q_beg; qi < q_end; ++qi) {
                const float* qv = vecs + qi * dim;
                for (std::size_t j = r_beg; j < r_end; ++j) {
                    if (j == qi) continue;
                    const float d = sq_l2_vec(qv, vecs + j * dim, dim);
                    if (d < topk[qi].front().first) {
                        topk[qi].front() = {d, static_cast<knng::index_t>(j)};
                        std::make_heap(topk[qi].begin(), topk[qi].end());
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        std::sort(topk[i].begin(), topk[i].end());
        for (std::size_t r = 0; r < k; ++r) {
            result.neighbors[i * k + r] = topk[i][r].second;
            result.distances[i * k + r] = topk[i][r].first;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Step 85 — Multi-GPU NN-Descent CPU reference
// ---------------------------------------------------------------------------

knng::Knng cpu_multi_nn_descent(const knng::Dataset& dataset,
                                  const MultiGpuConfig& cfg,
                                  std::uint64_t seed) {
    const std::size_t n   = dataset.n;
    const std::size_t dim = dataset.d;
    const std::size_t k   = static_cast<std::size_t>(cfg.k);
    const std::size_t ng  = static_cast<std::size_t>(cfg.n_gpus);
    const float* vecs     = dataset.data.data();

    // Initialise: each rank owns n/P points; random graph with XorShift.
    // For simplicity we initialise globally then partition.
    auto xor_shift = [](std::uint64_t& s) -> std::uint64_t {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    };

    // ids[i * k + r] = neighbor id; dists[i * k + r] = sq_l2
    std::vector<knng::index_t> ids  (n * k, knng::index_t(-1));
    std::vector<float>         dists(n * k, std::numeric_limits<float>::infinity());

    // Random initialization
    std::uint64_t rng = seed;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t r = 0; r < k; ++r) {
            const std::size_t j = static_cast<std::size_t>(xor_shift(rng) % n);
            ids  [i * k + r] = static_cast<knng::index_t>(j);
            dists[i * k + r] = sq_l2_vec(vecs + i * dim, vecs + j * dim, dim);
        }
    }

    const auto shards = partition_points(n, cfg);

    // NN-Descent iterations — each rank processes its shard; then simulated
    // allgather refreshes remote entries.
    for (int iter = 0; iter < cfg.n_iterations; ++iter) {
        bool any_update = false;
        // Each rank processes its shard
        for (std::size_t rank = 0; rank < ng; ++rank) {
            const auto& sh = shards[rank];
            for (std::size_t i = sh.begin; i < sh.end; ++i) {
                for (std::size_t ri = 0; ri < k; ++ri) {
                    const knng::index_t u = ids[i * k + ri];
                    if (u == knng::index_t(-1)) continue;
                    for (std::size_t rj = ri + 1; rj < k; ++rj) {
                        const knng::index_t v = ids[i * k + rj];
                        if (v == knng::index_t(-1)) continue;
                        const float duv = sq_l2_vec(
                            vecs + static_cast<std::size_t>(u) * dim,
                            vecs + static_cast<std::size_t>(v) * dim, dim);
                        // Try inserting u into v's list
                        float worst = -1.f;
                        std::size_t ws = k;
                        for (std::size_t s = 0; s < k; ++s) {
                            if (dists[static_cast<std::size_t>(v)*k+s] > worst) {
                                worst = dists[static_cast<std::size_t>(v)*k+s];
                                ws = s;
                            }
                        }
                        if (ws < k && duv < worst) {
                            ids  [static_cast<std::size_t>(v)*k+ws] = u;
                            dists[static_cast<std::size_t>(v)*k+ws] = duv;
                            any_update = true;
                        }
                    }
                }
            }
        }
        // Simulated AllGather: all ranks see all updates (already in shared CPU arrays)
        if (!any_update) break;
    }

    // Sort each row by distance and build Knng
    knng::Knng result(n, k);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::size_t> order(k);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return dists[i*k+a] < dists[i*k+b];
        });
        for (std::size_t r = 0; r < k; ++r) {
            result.neighbors[i*k+r] = ids[i*k+order[r]];
            result.distances[i*k+r] = dists[i*k+order[r]];
        }
    }
    return result;
}

} // namespace knng::gpu
