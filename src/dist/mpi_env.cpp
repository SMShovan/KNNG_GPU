/// @file
/// @brief `MpiEnv` — RAII guard for MPI init / finalize.

#include "knng/dist/mpi_env.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace knng::dist {

MpiEnv::MpiEnv(int* argc, char*** argv)
{
    int already_init = 0;
    MPI_Initialized(&already_init);
    if (already_init) {
        throw std::runtime_error{
            "knng::dist::MpiEnv: MPI is already initialised — "
            "only one MpiEnv may exist at a time"};
    }

    if (MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED,
                        &thread_support_) != MPI_SUCCESS) {
        throw std::runtime_error{
            "knng::dist::MpiEnv: MPI_Init_thread failed"};
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    owns_ = true;
}

MpiEnv::~MpiEnv()
{
    if (owns_) {
        MPI_Finalize();
    }
}

MpiEnv::MpiEnv(MpiEnv&& other) noexcept
    : rank_{other.rank_},
      size_{other.size_},
      thread_support_{other.thread_support_},
      owns_{other.owns_}
{
    other.owns_ = false;
}

MpiEnv& MpiEnv::operator=(MpiEnv&& other) noexcept
{
    if (this != &other) {
        if (owns_) {
            MPI_Finalize();
        }
        rank_           = other.rank_;
        size_           = other.size_;
        thread_support_ = other.thread_support_;
        owns_           = other.owns_;
        other.owns_     = false;
    }
    return *this;
}

void MpiEnv::barrier() const
{
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace knng::dist
