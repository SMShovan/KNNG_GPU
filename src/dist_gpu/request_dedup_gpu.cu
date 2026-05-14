/// @file
/// @brief Step 93 — GPU-side bitset request deduplication.
///
/// Algorithm:
///   1. mark_ids_kernel: atomically sets seen[id] = 1 for each request ID.
///   2. CUB DeviceScan::ExclusiveSum: prefix-sum of the seen array gives
///      the output position for each unique ID.
///   3. collect_ids_kernel: writes each marked ID to its prefix-sum position.
///
/// Result: a compact, sorted list of unique global IDs ready for MPI exchange.
/// The deduplication ratio (raw - unique) / raw measures communication savings.

#include <knng/dist_gpu/replication_policy.hpp>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <stdexcept>
#include <string>
#include <vector>

namespace knng::dist_gpu {

namespace {
inline void cuda_chk(cudaError_t e, const char* w)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}

// Mark each requested ID in a binary seen[] array.
__global__
static void mark_ids_kernel(const knng::index_t* __restrict__ d_reqs,
                             std::size_t  n_raw,
                             int*         d_seen,
                             std::size_t  global_n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
    if (i < n_raw) {
        const knng::index_t id = d_reqs[i];
        if (static_cast<std::size_t>(id) < global_n)
            d_seen[id] = 1;
    }
}

// Write unique ID i to output at position given by the prefix sum.
__global__
static void collect_ids_kernel(const int*          d_seen,
                                const int*          d_prefix,
                                knng::index_t*      d_out,
                                std::size_t         global_n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
    if (i < global_n && d_seen[i])
        d_out[static_cast<std::size_t>(d_prefix[i])] =
            static_cast<knng::index_t>(i);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
std::pair<std::size_t, GpuDedupStats>
gpu_dedup_requests(const knng::index_t* d_requests,
                   std::size_t          n_raw,
                   std::size_t          global_n,
                   knng::index_t*       d_out)
{
    // 1. Allocate seen[] (binary) and prefix[] (prefix-sum output).
    int* d_seen   = nullptr;
    int* d_prefix = nullptr;
    cuda_chk(cudaMalloc(&d_seen,   global_n * sizeof(int)), "cudaMalloc seen");
    cuda_chk(cudaMalloc(&d_prefix, global_n * sizeof(int)), "cudaMalloc prefix");
    cuda_chk(cudaMemset(d_seen, 0, global_n * sizeof(int)), "memset seen");

    // 2. Mark requested IDs.
    const int threads = 256;
    const int blocks  = static_cast<int>((n_raw + threads - 1) / threads);
    if (blocks > 0)
        mark_ids_kernel<<<blocks, threads>>>(d_requests, n_raw, d_seen, global_n);
    cuda_chk(cudaDeviceSynchronize(), "mark_ids sync");

    // 3. Exclusive prefix-sum via CUB.
    std::size_t tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, d_seen, d_prefix,
                                  static_cast<int>(global_n));
    void* d_tmp = nullptr;
    cuda_chk(cudaMalloc(&d_tmp, tmp_bytes), "cudaMalloc cub_tmp");
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_seen, d_prefix,
                                  static_cast<int>(global_n));
    cuda_chk(cudaDeviceSynchronize(), "ExclusiveSum sync");
    cuda_chk(cudaFree(d_tmp), "cudaFree cub_tmp");

    // 4. Read total unique count from host = prefix[N-1] + seen[N-1].
    int last_prefix = 0, last_seen = 0;
    cuda_chk(cudaMemcpy(&last_prefix, d_prefix + global_n - 1, sizeof(int),
                         cudaMemcpyDeviceToHost), "D2H last_prefix");
    cuda_chk(cudaMemcpy(&last_seen, d_seen + global_n - 1, sizeof(int),
                         cudaMemcpyDeviceToHost), "D2H last_seen");
    const std::size_t n_unique = static_cast<std::size_t>(last_prefix + last_seen);

    // 5. Collect unique IDs.
    const int cblocks = static_cast<int>((global_n + threads - 1) / threads);
    collect_ids_kernel<<<cblocks, threads>>>(d_seen, d_prefix, d_out, global_n);
    cuda_chk(cudaDeviceSynchronize(), "collect_ids sync");

    cuda_chk(cudaFree(d_prefix), "cudaFree prefix");
    cuda_chk(cudaFree(d_seen),   "cudaFree seen");

    GpuDedupStats stats{n_raw, n_unique};
    return {n_unique, stats};
}

} // namespace knng::dist_gpu
