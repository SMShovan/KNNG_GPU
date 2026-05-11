/// @file
/// @brief Step 62 — DeviceGraph and CpuDeviceGraph implementations.
///
/// Always compiled (no CUDA dependency).  The `DeviceGraph` implementation
/// uses `DeviceBuffer` (which maps to `GPU_MALLOC` / `GPU_FREE`), so on Mac
/// it allocates from the host heap — the same code is used on GPU without
/// change.

#include <knng/gpu/device_graph.hpp>


namespace knng::gpu {

// ===========================================================================
// CpuDeviceGraph
// ===========================================================================

CpuDeviceGraph::CpuDeviceGraph(std::size_t n_, std::size_t k_)
    : ids  (n_ * k_, static_cast<knng::index_t>(-1))
    , dists(n_ * k_, kSentinelDist)
    , flags(n_ * k_, 0u)
    , n(n_), k(k_)
{}

CpuDeviceGraph CpuDeviceGraph::from_knng(const knng::Knng& g)
{
    CpuDeviceGraph cg(g.n, g.k);
    for (std::size_t qi = 0; qi < g.n; ++qi) {
        for (std::size_t r = 0; r < g.k; ++r) {
            cg.ids  [qi * g.k + r] = g.neighbors[qi * g.k + r];
            cg.dists[qi * g.k + r] = g.distances[qi * g.k + r];
            cg.flags[qi * g.k + r] = 1u;   // mark all as new
        }
    }
    return cg;
}

knng::Knng CpuDeviceGraph::to_knng() const
{
    knng::Knng g(n, k);
    for (std::size_t qi = 0; qi < n; ++qi) {
        for (std::size_t r = 0; r < k; ++r) {
            g.neighbors[qi * k + r] = ids  [qi * k + r];
            g.distances[qi * k + r] = dists[qi * k + r];
        }
    }
    return g;
}

bool CpuDeviceGraph::try_insert(std::size_t qi, knng::index_t new_id, float d)
{
    // Check if new_id already in this row.
    for (std::size_t r = 0; r < k; ++r) {
        if (ids[qi * k + r] == new_id) { return false; }
    }
    // Check against worst.
    float worst = worst_dist(qi);
    if (d >= worst) { return false; }

    // Find worst slot and replace.
    std::size_t worst_pos = 0;
    float       worst_d   = dists[qi * k];
    for (std::size_t r = 1; r < k; ++r) {
        if (dists[qi * k + r] > worst_d) {
            worst_d   = dists[qi * k + r];
            worst_pos = r;
        }
    }
    ids  [qi * k + worst_pos] = new_id;
    dists[qi * k + worst_pos] = d;
    flags[qi * k + worst_pos] = 1u;   // new insertion
    return true;
}

float CpuDeviceGraph::worst_dist(std::size_t qi) const noexcept
{
    float w = dists[qi * k];
    for (std::size_t r = 1; r < k; ++r) {
        if (dists[qi * k + r] > w) { w = dists[qi * k + r]; }
    }
    return w;
}

void CpuDeviceGraph::mark_old(std::size_t qi) noexcept
{
    for (std::size_t r = 0; r < k; ++r) {
        flags[qi * k + r] &= ~1u;
    }
}

// ===========================================================================
// DeviceGraph
// ===========================================================================

DeviceGraph::DeviceGraph(std::size_t n_, std::size_t k_)
    : ids  (n_ * k_)
    , dists(n_ * k_)
    , flags(n_ * k_)
    , n(n_), k(k_)
{}

DeviceGraph::DeviceGraph(const CpuDeviceGraph& cpu)
    : DeviceGraph(cpu.n, cpu.k)
{
    ids.copy_from_host  (cpu.ids  .data(), cpu.n * cpu.k);
    dists.copy_from_host(cpu.dists.data(), cpu.n * cpu.k);
    flags.copy_from_host(cpu.flags.data(), cpu.n * cpu.k);
}

CpuDeviceGraph DeviceGraph::to_cpu() const
{
    CpuDeviceGraph cpu(n, k);
    ids.copy_to_host  (cpu.ids);
    dists.copy_to_host(cpu.dists);
    flags.copy_to_host(cpu.flags);
    return cpu;
}

DeviceGraph DeviceGraph::from_knng(const knng::Knng& g)
{
    return DeviceGraph(CpuDeviceGraph::from_knng(g));
}

knng::Knng DeviceGraph::to_knng() const
{
    return to_cpu().to_knng();
}

} // namespace knng::gpu
