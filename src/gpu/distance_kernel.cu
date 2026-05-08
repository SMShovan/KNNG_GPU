/// @file
/// @brief Step 47 — Naive GPU pairwise squared-L2 distance kernel.
///
/// This is the simplest GPU distance kernel possible: one CUDA thread per
/// (query, reference) pair, no shared memory, no coalescing considerations.
/// Phase 8 will replace this with a tiled, shared-memory, coalesced version.
/// The kernel exists here to establish a correct GPU baseline and a
/// Nsight Compute starting point.

#include <knng/gpu/distance_kernel.hpp>
#include <knng/gpu/device_buffer.hpp>

namespace knng::gpu {

// CPU reference lives in distance_cpu_ref.cpp (always compiled).

// ---------------------------------------------------------------------------
// GPU kernel
// ---------------------------------------------------------------------------

GPU_GLOBAL void naive_pairwise_l2sq_kernel(
    const float* d_queries,
    const float* d_refs,
    unsigned int nq,
    unsigned int nr,
    unsigned int d,
    float*       d_out)
{
    // Each thread owns one (query_row, ref_row) pair.
    const unsigned int qi = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    const unsigned int ri = blockIdx.x * blockDim.x + threadIdx.x;  // ref index

    if (qi >= nq || ri >= nr) {
        return;
    }

    float acc = 0.f;
    for (unsigned int k = 0; k < d; ++k) {
        float diff = d_queries[qi * d + k] - d_refs[ri * d + k];
        acc += diff * diff;
    }
    d_out[qi * nr + ri] = acc;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

std::vector<float> gpu_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d)
{
    DeviceBuffer<float> d_queries(nq * d);
    DeviceBuffer<float> d_refs(nr * d);
    DeviceBuffer<float> d_out(nq * nr);

    d_queries.copy_from_host(queries, nq * d);
    d_refs.copy_from_host(refs, nr * d);

    constexpr unsigned int kBlock = 16;
    dim3 block(kBlock, kBlock);
    dim3 grid(
        static_cast<unsigned int>((nr + kBlock - 1) / kBlock),
        static_cast<unsigned int>((nq + kBlock - 1) / kBlock));

    GPU_LAUNCH(naive_pairwise_l2sq_kernel, grid, block, 0, nullptr,
               d_queries.data(), d_refs.data(),
               static_cast<unsigned int>(nq),
               static_cast<unsigned int>(nr),
               static_cast<unsigned int>(d),
               d_out.data());
    GPU_SYNC();

    return d_out.to_host();
}

} // namespace knng::gpu
