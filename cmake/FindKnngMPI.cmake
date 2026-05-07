# FindKnngMPI.cmake
#
# Discovers an MPI implementation and exposes it as the
# `knng::mpi_iface` INTERFACE target. Sets the cache variable
# `KNNG_HAVE_MPI` to ON when discovery succeeded.
#
# We rely entirely on CMake's built-in `FindMPI` module; our wrapper
# adds:
#   1. An opt-in option (`KNNG_ENABLE_MPI`), consistent with the
#      BLAS and OpenMP patterns elsewhere in the tree.
#   2. The `knng::mpi_iface` INTERFACE alias so dist-library targets
#      spell their dependency uniformly.
#   3. A compile definition `KNNG_HAVE_MPI=1` propagated to every
#      consumer, mirroring the `KNNG_HAVE_OPENMP` / `KNNG_HAVE_BLAS`
#      pattern.
#
# When discovery fails (or `KNNG_ENABLE_MPI=OFF`), `KNNG_HAVE_MPI`
# is set to OFF and the `src/dist/` subtree is silently skipped.
# The non-MPI build remains fully functional.
#
# Thread-level requested: `MPI_THREAD_FUNNELED` — the dist library
# calls MPI from a single thread at a time (the main thread). The
# actual provided level is stored in `MpiEnv::thread_support()` and
# checked at runtime; callers that want `MPI_THREAD_MULTIPLE` will
# need to upgrade the request here.

option(KNNG_ENABLE_MPI "Enable MPI distributed support (Phase 6+)" ON)

if(NOT KNNG_ENABLE_MPI)
    set(KNNG_HAVE_MPI OFF CACHE INTERNAL
        "MPI disabled by KNNG_ENABLE_MPI=OFF")
    return()
endif()

find_package(MPI QUIET COMPONENTS CXX)

if(MPI_CXX_FOUND)
    add_library(knng_mpi_iface INTERFACE)
    target_link_libraries(knng_mpi_iface INTERFACE MPI::MPI_CXX)
    target_compile_definitions(knng_mpi_iface INTERFACE
        KNNG_HAVE_MPI=1)
    add_library(knng::mpi_iface ALIAS knng_mpi_iface)
    set(KNNG_HAVE_MPI ON CACHE INTERNAL "MPI via find_package")
    message(STATUS "knng: MPI ${MPI_CXX_VERSION} (${MPI_CXX_LIBRARIES})")
    return()
endif()

set(KNNG_HAVE_MPI OFF CACHE INTERNAL "MPI not found")
message(STATUS "knng: MPI not found — Phase 6+ distributed targets skipped "
               "(install OpenMPI or MPICH and re-run cmake to enable)")
