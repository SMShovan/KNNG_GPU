/// @file
/// @brief Step 81 — Topology CPU reference implementation.
///
/// Always compiled.  `Topology::simulate()` builds a purely CPU-side topology
/// for testing.  The GPU probe path (`Topology::probe()`) lives in
/// `topology_gpu.cu` (CUDA-gated).

#include <knng/gpu/topology.hpp>

#include <sstream>
#include <stdexcept>

namespace knng::gpu {

const char* peer_link_type_name(PeerLinkType t) noexcept {
    switch (t) {
        case PeerLinkType::Same:       return "Same";
        case PeerLinkType::NVLink:     return "NVLink";
        case PeerLinkType::PCIe:       return "PCIe";
        case PeerLinkType::HostBridge: return "HostBridge";
        case PeerLinkType::Simulated:  return "Simulated";
    }
    return "Unknown";
}

Topology::Topology(int n_gpus)
    : n_(n_gpus),
      matrix_(static_cast<std::size_t>(n_gpus * n_gpus), PeerLinkType::Simulated)
{}

Topology Topology::simulate(int n_gpus) {
    if (n_gpus <= 0)
        throw std::invalid_argument("n_gpus must be positive");
    Topology t(n_gpus);
    for (int i = 0; i < n_gpus; ++i)
        t.matrix_[static_cast<std::size_t>(i * n_gpus + i)] = PeerLinkType::Same;
    return t;
}

PeerLinkType Topology::link(int i, int j) const noexcept {
    return matrix_[static_cast<std::size_t>(i * n_ + j)];
}

bool Topology::has_p2p(int i, int j) const noexcept {
    const auto lt = link(i, j);
    return lt == PeerLinkType::NVLink || lt == PeerLinkType::PCIe;
}

std::string Topology::to_string() const {
    std::ostringstream oss;
    oss << "Topology (" << n_ << " GPUs)\n";
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            oss << peer_link_type_name(link(i, j));
            if (j < n_ - 1) oss << '\t';
        }
        oss << '\n';
    }
    return oss.str();
}

} // namespace knng::gpu
