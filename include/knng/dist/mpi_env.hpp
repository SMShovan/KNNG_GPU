#pragma once

/// @file
/// @brief RAII wrapper around MPI initialisation / finalisation.
///
/// `knng::dist::MpiEnv` is the *only* object in the dist library that
/// calls `MPI_Init_thread` / `MPI_Finalize`. Construct exactly one
/// instance at the top of `main()`; it may not be copied, but it may
/// be moved (the moved-from instance becomes a no-op on destruction).
///
/// **Usage pattern:**
/// @code
/// int main(int argc, char** argv) {
///     knng::dist::MpiEnv env{&argc, &argv};
///     if (env.is_root()) { /* rank-0-only logging */ }
///     env.barrier();
///     return 0;
/// }
/// @endcode
///
/// **Thread support.** The constructor requests `MPI_THREAD_FUNNELED`:
/// the application is multi-threaded but only the main thread will
/// make MPI calls. The *provided* thread level is stored and available
/// via `thread_support()`. If an MPI operation later requires a higher
/// level (e.g., progress threads in Phase 12), the constructor must be
/// updated to request `MPI_THREAD_MULTIPLE`.

#if !defined(KNNG_HAVE_MPI) || !KNNG_HAVE_MPI
#  error "mpi_env.hpp included without MPI — guard the include with KNNG_HAVE_MPI"
#endif

#include <mpi.h>

namespace knng::dist {

/// RAII guard for the MPI execution environment.
///
/// Exactly one `MpiEnv` should exist for the lifetime of any
/// distributed run. Constructing a second live instance (before the
/// first is destroyed) produces undefined behaviour — MPI does not
/// support nested `MPI_Init` calls.
class MpiEnv {
public:
    /// Initialise MPI with thread level `MPI_THREAD_FUNNELED`.
    ///
    /// @param argc Pointer to the process's `argc`. Forwarded to
    ///             `MPI_Init_thread`. May be `nullptr` when the
    ///             binary has no command-line arguments.
    /// @param argv Pointer to the process's `argv`. Forwarded to
    ///             `MPI_Init_thread`. May be `nullptr`.
    /// @throws std::runtime_error if `MPI_Init_thread` fails or if
    ///         MPI reports it is already initialised.
    explicit MpiEnv(int* argc = nullptr, char*** argv = nullptr);

    /// Call `MPI_Finalize()` unless this object was moved from.
    ~MpiEnv();

    MpiEnv(const MpiEnv&)            = delete;
    MpiEnv& operator=(const MpiEnv&) = delete;

    /// Move construction: the moved-from object becomes inert (will
    /// not call `MPI_Finalize` on destruction).
    MpiEnv(MpiEnv&&) noexcept;
    MpiEnv& operator=(MpiEnv&&) noexcept;

    /// Rank of this process in `MPI_COMM_WORLD`.
    [[nodiscard]] int rank() const noexcept { return rank_; }

    /// Total number of processes in `MPI_COMM_WORLD`.
    [[nodiscard]] int size() const noexcept { return size_; }

    /// `true` iff this is the root process (`rank() == 0`).
    [[nodiscard]] bool is_root() const noexcept { return rank_ == 0; }

    /// The thread-support level MPI actually provided. May be higher
    /// or lower than `MPI_THREAD_FUNNELED` depending on the
    /// implementation.
    [[nodiscard]] int thread_support() const noexcept
    {
        return thread_support_;
    }

    /// Blocking barrier on `MPI_COMM_WORLD`.
    void barrier() const;

private:
    int  rank_           = 0;
    int  size_           = 1;
    int  thread_support_ = MPI_THREAD_SINGLE;
    bool owns_           = false; ///< False after move.
};

} // namespace knng::dist
