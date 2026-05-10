/// @file
/// @brief Step 52 — Coalesced pairwise squared-L2 GPU kernel.
///
/// Transpose the reference matrix to column-major on the device, then run
/// the distance kernel with coalesced global loads on the reference side.

#include <knng/gpu/coalesced_distance.hpp>
#include <knng/gpu/device_buffer.hpp>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Transpose kernel — tiled 32×32 for shared-memory bank-conflict avoidance
// ---------------------------------------------------------------------------

static constexpr unsigned int kTransposeTile = 32u;

GPU_GLOBAL void transpose_kernel(
    const float* src,
    unsigned int rows,
    unsigned int cols,
    float*       dst)
{
    GPU_SHARED float tile[kTransposeTile][kTransposeTile + 1]; // +1 avoids bank conflict

    unsigned int r = blockIdx.y * kTransposeTile + threadIdx.y;
    unsigned int c = blockIdx.x * kTransposeTile + threadIdx.x;

    if (r < rows && c < cols) {
        tile[threadIdx.y][threadIdx.x] = src[r * cols + c];
    }
    __syncthreads();

    // Write transposed tile.
    r = blockIdx.x * kTransposeTile + threadIdx.y;
    c = blockIdx.y * kTransposeTile + threadIdx.x;
    if (r < cols && c < rows) {
        dst[r * rows + c] = tile[threadIdx.x][threadIdx.y];
    }
}

// ---------------------------------------------------------------------------
// Coalesced distance kernel
// ---------------------------------------------------------------------------

GPU_GLOBAL void coalesced_pairwise_l2sq_kernel(
    const float* d_queries,
    const float* d_refs_T,
    unsigned int nq,
    unsigned int nr,
    unsigned int d,
    float*       d_out)
{
    const unsigned int qi = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int ri = blockIdx.x * blockDim.x + threadIdx.x;

    if (qi >= nq || ri >= nr) {
        return;
    }

    float acc = 0.f;
    for (unsigned int k = 0; k < d; ++k) {
        // refs_T layout: [d][nr], so refs_T[k][ri] = refs_T[k*nr + ri]
        // Consecutive threads have consecutive ri → coalesced load.
        const float diff = d_queries[qi * d + k] - d_refs_T[k * nr + ri];
        acc += diff * diff;
    }
    d_out[qi * nr + ri] = acc;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

std::vector<float> gpu_coalesced_pairwise_l2sq(
    const float* queries,
    const float* refs,
    std::size_t  nq,
    std::size_t  nr,
    std::size_t  d)
{
    DeviceBuffer<float> d_queries(nq * d);
    DeviceBuffer<float> d_refs(nr * d);
    DeviceBuffer<float> d_refs_T(nr * d);
    DeviceBuffer<float> d_out(nq * nr);

    d_queries.copy_from_host(queries, nq * d);
    d_refs.copy_from_host(refs, nr * d);

    // Transpose refs: [nr][d] → [d][nr]
    {
        dim3 block(kTransposeTile, kTransposeTile);
        dim3 grid(
            (static_cast<unsigned int>(d)  + kTransposeTile - 1u) / kTransposeTile,
            (static_cast<unsigned int>(nr) + kTransposeTile - 1u) / kTransposeTile);
        GPU_LAUNCH(transpose_kernel, grid, block, 0, nullptr,
                   d_refs.data(),
                   static_cast<unsigned int>(nr),
                   static_cast<unsigned int>(d),
                   d_refs_T.data());
    }

    // Coalesced distance computation
    {
        constexpr unsigned int kBlock = 16u;
        dim3 block(kBlock, kBlock);
        dim3 grid(
            (static_cast<unsigned int>(nr) + kBlock - 1u) / kBlock,
            (static_cast<unsigned int>(nq) + kBlock - 1u) / kBlock);
        GPU_LAUNCH(coalesced_pairwise_l2sq_kernel, grid, block, 0, nullptr,
                   d_queries.data(), d_refs_T.data(),
                   static_cast<unsigned int>(nq),
                   static_cast<unsigned int>(nr),
                   static_cast<unsigned int>(d),
                   d_out.data());
    }

    GPU_SYNC();
    return d_out.to_host();
}

} // namespace knng::gpu
