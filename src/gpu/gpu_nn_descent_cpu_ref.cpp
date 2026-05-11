/// @file
/// @brief Steps 63–70 — CPU reference implementations for GPU NN-Descent.
///
/// Always compiled (no CUDA dependency).  Each function mirrors the GPU
/// kernel's algorithm exactly so that correctness can be verified on Mac.

#include <knng/gpu/gpu_nn_descent.hpp>
#include <knng/gpu/fp16_distance.hpp>   // f32_to_f16_bits, f16_bits_to_f32
#include <knng/random.hpp>              // XorShift64

#include <cmath>
#include <cstdint>
#include <vector>

namespace knng::gpu {

// ===========================================================================
// Step 63 — Random graph init (CPU reference)
// ===========================================================================

CpuDeviceGraph cpu_init_random_graph(
    const knng::Dataset& dataset,
    std::size_t          k,
    std::uint64_t        seed)
{
    const std::size_t n = dataset.n;
    const std::size_t d = dataset.d;
    CpuDeviceGraph graph(n, k);

    for (std::size_t qi = 0; qi < n; ++qi) {
        // Per-point seed: mix global seed with point index.
        knng::random::XorShift64 rng(seed ^ (static_cast<std::uint64_t>(qi + 1) * 6364136223846793005ULL));

        std::size_t filled = 0;
        std::size_t tries  = 0;
        const std::size_t max_tries = k * 16 + 64;

        while (filled < k && tries < max_tries) {
            ++tries;
            const auto rid = static_cast<knng::index_t>(rng.next_below(static_cast<std::uint64_t>(n)));
            if (rid == static_cast<knng::index_t>(qi)) { continue; }

            // Check duplicate.
            bool dup = false;
            for (std::size_t r = 0; r < filled; ++r) {
                if (graph.ids[qi * k + r] == rid) { dup = true; break; }
            }
            if (dup) { continue; }

            // Compute exact squared-L2 distance.
            const float* q = dataset.data.data() + qi * d;
            const float* r = dataset.data.data() + rid * d;
            float acc = 0.f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                const float diff = q[dim] - r[dim];
                acc += diff * diff;
            }

            graph.ids  [qi * k + filled] = rid;
            graph.dists[qi * k + filled] = acc;
            graph.flags[qi * k + filled] = 1u;   // is_new
            ++filled;
        }
        // If we couldn't fill k slots (very small n), leave sentinels.
    }
    return graph;
}

// ===========================================================================
// Step 64 — Naive local-join (CPU reference)
// ===========================================================================

std::size_t cpu_local_join(
    const knng::Dataset& dataset,
    CpuDeviceGraph&      graph)
{
    const std::size_t n = dataset.n;
    const std::size_t d = dataset.d;
    const std::size_t k = graph.k;

    // Collect snapshot of new/old per point.
    using IdVec = std::vector<knng::index_t>;
    std::vector<IdVec> new_nbrs(n), old_nbrs(n);
    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t r = 0; r < k; ++r) {
            const auto id = graph.ids[p * k + r];
            if (id == static_cast<knng::index_t>(-1)) { continue; }
            if (graph.flags[p * k + r] & 1u) {
                new_nbrs[p].push_back(id);
            } else {
                old_nbrs[p].push_back(id);
            }
        }
        graph.mark_old(p);   // flip all to old after snapshot
    }

    std::size_t updates = 0;

    // Local-join: for each point p, process (new×new, u<v) and (new×old).
    for (std::size_t p = 0; p < n; ++p) {
        const IdVec& nw = new_nbrs[p];
        const IdVec& od = old_nbrs[p];

        auto process_pair = [&](knng::index_t u, knng::index_t v) {
            if (u == v) { return; }
            const float* pu = dataset.data.data() + static_cast<std::size_t>(u) * d;
            const float* pv = dataset.data.data() + static_cast<std::size_t>(v) * d;
            float acc = 0.f;
            for (std::size_t dim = 0; dim < d; ++dim) {
                const float diff = pu[dim] - pv[dim];
                acc += diff * diff;
            }
            if (graph.try_insert(u, v, acc)) { ++updates; }
            if (graph.try_insert(v, u, acc)) { ++updates; }
        };

        // new × new (u < v to avoid double-processing).
        for (std::size_t i = 0; i < nw.size(); ++i) {
            for (std::size_t j = i + 1; j < nw.size(); ++j) {
                process_pair(nw[i], nw[j]);
            }
        }
        // new × old.
        for (auto u : nw) {
            for (auto v : od) {
                process_pair(u, v);
            }
        }
    }
    return updates;
}

// ===========================================================================
// Step 67 — Reverse-neighbor accumulation (CPU reference)
// ===========================================================================

void cpu_build_reverse_graph(
    const CpuDeviceGraph&          fwd,
    std::vector<knng::index_t>&    rev_ids,
    std::vector<std::uint32_t>&    rev_offsets)
{
    const std::size_t n = fwd.n;
    const std::size_t k = fwd.k;

    // Count in-degrees.
    rev_offsets.assign(n + 1, 0u);
    for (std::size_t qi = 0; qi < n; ++qi) {
        for (std::size_t r = 0; r < k; ++r) {
            const auto id = fwd.ids[qi * k + r];
            if (id < static_cast<knng::index_t>(n)) {
                ++rev_offsets[id + 1];
            }
        }
    }
    // Prefix sum → CSR offsets.
    for (std::size_t i = 1; i <= n; ++i) {
        rev_offsets[i] += rev_offsets[i - 1];
    }
    rev_ids.resize(rev_offsets[n]);

    // Fill CSR.
    std::vector<std::uint32_t> cursor(rev_offsets.begin(), rev_offsets.begin() + static_cast<std::ptrdiff_t>(n));
    for (std::size_t qi = 0; qi < n; ++qi) {
        for (std::size_t r = 0; r < k; ++r) {
            const auto id = fwd.ids[qi * k + r];
            if (id < static_cast<knng::index_t>(n)) {
                rev_ids[cursor[id]++] = static_cast<knng::index_t>(qi);
            }
        }
    }
}

// ===========================================================================
// Step 68 — Sampling (CPU reference)
// ===========================================================================

void cpu_sample_new_neighbors(
    CpuDeviceGraph& graph,
    double          rho,
    std::uint64_t   iter_seed)
{
    const std::size_t n = graph.n;
    const std::size_t k = graph.k;
    const auto        sample_k = static_cast<std::size_t>(std::ceil(rho * static_cast<double>(k)));

    for (std::size_t qi = 0; qi < n; ++qi) {
        knng::random::XorShift64 rng(iter_seed ^ ((qi + 1) * 2654435761ULL));

        std::size_t new_count = 0;
        for (std::size_t r = 0; r < k; ++r) {
            if (graph.flags[qi * k + r] & 1u) { ++new_count; }
        }

        if (new_count <= sample_k) { continue; }   // nothing to drop

        // Reservoir: randomly drop (new_count - sample_k) new entries.
        std::size_t to_drop = new_count - sample_k;
        for (std::size_t r = 0; r < k && to_drop > 0; ++r) {
            if (!(graph.flags[qi * k + r] & 1u)) { continue; }
            // Drop this entry with probability to_drop / remaining_new.
            const std::size_t remaining = new_count - r;  // approx
            if (rng.next_below(static_cast<std::uint64_t>(remaining)) < to_drop) {
                graph.flags[qi * k + r] &= ~1u;  // mark old
                --to_drop;
            }
        }
    }
}

// ===========================================================================
// Step 69 — Full driver (CPU reference)
// ===========================================================================

knng::Knng cpu_gpu_nn_descent(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg)
{
    CpuDeviceGraph graph = cpu_init_random_graph(dataset, k, cfg.seed);
    const double threshold = cfg.delta * static_cast<double>(dataset.n * k);

    for (std::size_t iter = 0; iter < cfg.max_iters; ++iter) {
        if (cfg.rho < 1.0) {
            cpu_sample_new_neighbors(graph, cfg.rho,
                cfg.seed ^ (static_cast<std::uint64_t>(iter + 1) * 1000003ULL));
        }
        const std::size_t updates = cpu_local_join(dataset, graph);
        if (static_cast<double>(updates) < threshold) { break; }
    }
    return graph.to_knng();
}

// ===========================================================================
// Step 70 — Mixed-precision NN-Descent (CPU reference)
// ===========================================================================

// Uses a CpuDeviceGraph variant where dists are stored as fp16 bits
// (in a uint16_t-sized float: we store actual fp16 bits in the LSBs of
// the uint32_t flags field's upper 16 bits).  Simpler approach: keep a
// parallel uint16_t array and convert on every comparison.

knng::Knng cpu_fp16_nn_descent(
    const knng::Dataset& dataset,
    std::size_t          k,
    const GpuNNDConfig&  cfg)
{
    // Initialize with fp32, then run the driver with fp16-converted distances.
    CpuDeviceGraph graph = cpu_init_random_graph(dataset, k, cfg.seed);

    // Convert all distances to fp16 and back to fp32 (simulates rounding).
    for (std::size_t i = 0; i < graph.n * graph.k; ++i) {
        graph.dists[i] = f16_bits_to_f32(f32_to_f16_bits(graph.dists[i]));
    }

    const double threshold = cfg.delta * static_cast<double>(dataset.n * k);
    for (std::size_t iter = 0; iter < cfg.max_iters; ++iter) {
        // Run local join; after each insertion, round distance to fp16.
        const std::size_t updates = cpu_local_join(dataset, graph);
        // Quantize newly inserted distances.
        for (std::size_t i = 0; i < graph.n * graph.k; ++i) {
            if (graph.flags[i] & 1u) {
                graph.dists[i] = f16_bits_to_f32(f32_to_f16_bits(graph.dists[i]));
            }
        }
        if (static_cast<double>(updates) < threshold) { break; }
    }
    return graph.to_knng();
}

} // namespace knng::gpu
