/// @file
/// @brief Step 80 — NcclComm RAII wrapper implementation.
///
/// Only compiled when KNNG_HAVE_CUDA and KNNG_HAVE_NCCL are both defined.

#include <knng/gpu/nccl_comm.hpp>
#include <knng/gpu/backend.hpp>

#if defined(KNNG_HAVE_CUDA) && defined(KNNG_HAVE_NCCL)

#include <stdexcept>
#include <string>
#include <vector>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void nccl_check(ncclResult_t res, const char* file, int line) {
    if (res != ncclSuccess) {
        throw std::runtime_error(
            std::string("NCCL error at ") + file + ":" + std::to_string(line)
            + ": " + ncclGetErrorString(res));
    }
}
#define NCCL_CHECK(call) nccl_check((call), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// NcclComm
// ---------------------------------------------------------------------------

NcclComm::~NcclComm() {
    if (comm_) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
}

NcclComm::NcclComm(NcclComm&& o) noexcept
    : comm_(o.comm_), rank_(o.rank_), n_ranks_(o.n_ranks_) {
    o.comm_ = nullptr;
}

NcclComm& NcclComm::operator=(NcclComm&& o) noexcept {
    if (this != &o) {
        if (comm_) ncclCommDestroy(comm_);
        comm_    = o.comm_;
        rank_    = o.rank_;
        n_ranks_ = o.n_ranks_;
        o.comm_  = nullptr;
    }
    return *this;
}

std::vector<NcclComm> NcclComm::init_all(int n_gpus) {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    std::vector<NcclComm> comms(static_cast<std::size_t>(n_gpus));
    // ncclCommInitAll: initialise communicators for all local GPUs in one call.
    std::vector<ncclComm_t> raw(static_cast<std::size_t>(n_gpus));
    NCCL_CHECK(ncclCommInitAll(raw.data(), n_gpus, nullptr));
    for (int r = 0; r < n_gpus; ++r) {
        comms[static_cast<std::size_t>(r)].comm_    = raw[static_cast<std::size_t>(r)];
        comms[static_cast<std::size_t>(r)].rank_    = r;
        comms[static_cast<std::size_t>(r)].n_ranks_ = n_gpus;
    }
    return comms;
}

void NcclComm::allreduce_sum(float* d_buf, std::size_t count,
                              cudaStream_t stream) {
    NCCL_CHECK(ncclAllReduce(d_buf, d_buf,
                              count, ncclFloat,
                              ncclSum, comm_, stream));
}

void NcclComm::bcast(float* d_buf, std::size_t count, int root,
                      cudaStream_t stream) {
    NCCL_CHECK(ncclBcast(d_buf, count, ncclFloat, root, comm_, stream));
}

void NcclComm::allgather(const float* d_sendbuf, float* d_recvbuf,
                          std::size_t count, cudaStream_t stream) {
    NCCL_CHECK(ncclAllGather(d_sendbuf, d_recvbuf,
                              count, ncclFloat, comm_, stream));
}

} // namespace knng::gpu

#endif // KNNG_HAVE_CUDA && KNNG_HAVE_NCCL
