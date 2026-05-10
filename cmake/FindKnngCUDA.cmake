# cmake/FindKnngCUDA.cmake
#
# Detect the CUDA Toolkit when the project is configured with
# -DKNNG_ENABLE_CUDA=ON.  Exports:
#
#   KNNG_HAVE_CUDA    — BOOL cache var, TRUE if CUDA is available and enabled
#   knng::cuda_iface  — INTERFACE target carrying CUDA include dirs and
#                       the `KNNG_HAVE_CUDA` compile definition, linked by
#                       every target that needs GPU headers.
#
# If the CUDA compiler is not found on the current machine the variable is
# set to OFF and no error is raised — callers guard with `if(KNNG_HAVE_CUDA)`.

set(KNNG_HAVE_CUDA OFF CACHE BOOL "CUDA Toolkit found and GPU build enabled" FORCE)

# Accept both -DKNNG_BACKEND=CUDA and legacy -DKNNG_ENABLE_CUDA=ON.
if(NOT KNNG_ENABLE_CUDA AND NOT KNNG_BACKEND STREQUAL "CUDA")
    return()
endif()

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "knng: KNNG_ENABLE_CUDA=ON but no CUDA compiler found — GPU build disabled")
    return()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(KNNG_HAVE_CUDA ON CACHE BOOL "CUDA Toolkit found and GPU build enabled" FORCE)
message(STATUS "knng: CUDA enabled — compiler ${CMAKE_CUDA_COMPILER}, toolkit ${CUDAToolkit_VERSION}")

# Portable CUDA architecture list: Volta (70), Ampere (80), Hopper (90).
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=<list>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90 CACHE STRING "CUDA compute capabilities")
endif()

# Interface target that propagates the CUDA include path and the
# KNNG_HAVE_CUDA compile definition to every translation unit that
# includes `include/knng/gpu/backend.hpp`.
add_library(knng_cuda_iface INTERFACE)
target_include_directories(knng_cuda_iface INTERFACE
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(knng_cuda_iface INTERFACE KNNG_HAVE_CUDA=1)
add_library(knng::cuda_iface ALIAS knng_cuda_iface)
