# FindKnngCudaAwareMPI.cmake
#
# Probes for CUDA-aware MPI: the ability to pass GPU device pointers
# directly to MPI_Send / MPI_Recv without a host-side bounce buffer.
#
# Prerequisites: cmake/FindKnngMPI.cmake and cmake/FindKnngCUDA.cmake
# must have been included first (KNNG_HAVE_MPI and KNNG_HAVE_CUDA set).
#
# Detection strategy:
#   1. Guard: if MPI or CUDA is absent, set OFF and return.
#   2. Try to compile a minimal snippet that includes <mpi-ext.h> and
#      calls MPIX_Query_cuda_support() — the de-facto standard extension
#      from Open MPI ≥ 1.7 and MVAPICH2.
#   3. Fall back to OFF (plain MPI without CUDA extension headers).
#
# Outputs:
#   KNNG_HAVE_CUDA_AWARE_MPI    BOOL cache variable (ON / OFF)
#   knng::cuda_aware_mpi_iface  INTERFACE target (created when MPI+CUDA
#                               are both present); propagates
#                               KNNG_CUDA_AWARE_MPI=1 when supported.

option(KNNG_ENABLE_CUDA_AWARE_MPI
    "Probe for CUDA-aware MPI device-pointer support (Phase 12+)" ON)

if(NOT KNNG_HAVE_MPI OR NOT KNNG_HAVE_CUDA OR NOT KNNG_ENABLE_CUDA_AWARE_MPI)
    set(KNNG_HAVE_CUDA_AWARE_MPI OFF CACHE INTERNAL
        "CUDA-aware MPI requires both MPI and CUDA")
    return()
endif()

include(CheckCXXSourceCompiles)

set(_save_libs  "${CMAKE_REQUIRED_LIBRARIES}")
set(_save_flags "${CMAKE_REQUIRED_FLAGS}")

get_target_property(_mpi_incs MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
if(_mpi_incs)
    foreach(_d IN LISTS _mpi_incs)
        string(APPEND CMAKE_REQUIRED_FLAGS " -I${_d}")
    endforeach()
endif()
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)

check_cxx_source_compiles(
    "#include <mpi.h>\n#include <mpi-ext.h>\nint main(){return MPIX_Query_cuda_support();}\n"
    KNNG_HAVE_CUDA_AWARE_MPI_DETECTED)

set(CMAKE_REQUIRED_LIBRARIES  "${_save_libs}")
set(CMAKE_REQUIRED_FLAGS      "${_save_flags}")

if(KNNG_HAVE_CUDA_AWARE_MPI_DETECTED)
    set(KNNG_HAVE_CUDA_AWARE_MPI ON CACHE INTERNAL
        "CUDA-aware MPI detected (MPIX_Query_cuda_support available)")
    message(STATUS "knng: CUDA-aware MPI — device-pointer transfers enabled")
else()
    set(KNNG_HAVE_CUDA_AWARE_MPI OFF CACHE INTERNAL
        "CUDA-aware MPI not found — device sends fall back to pinned staging buffer")
    message(STATUS "knng: CUDA-aware MPI not found — pinned-buffer fallback active")
endif()

if(NOT TARGET knng_cuda_aware_mpi_iface)
    add_library(knng_cuda_aware_mpi_iface INTERFACE)
    if(KNNG_HAVE_CUDA_AWARE_MPI)
        target_compile_definitions(knng_cuda_aware_mpi_iface INTERFACE
            KNNG_CUDA_AWARE_MPI=1)
    endif()
    add_library(knng::cuda_aware_mpi_iface ALIAS knng_cuda_aware_mpi_iface)
endif()
