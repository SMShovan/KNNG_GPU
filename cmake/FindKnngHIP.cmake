# cmake/FindKnngHIP.cmake
#
# Detect the HIP / ROCm toolchain when the project is configured with
# -DKNNG_BACKEND=HIP.  Exports:
#
#   KNNG_HAVE_HIP      — BOOL cache var, TRUE if HIP is available and enabled
#   knng::hip_iface    — INTERFACE target carrying HIP include dirs and the
#                        KNNG_HAVE_HIP compile definition.
#
# If the HIP compiler is not found, KNNG_HAVE_HIP is set to OFF without error.
#
# Usage:
#   cmake -S . -B build-hip -DKNNG_BACKEND=HIP
#   cmake --build build-hip
#
# The backend.hpp macro layer already contains HIP paths guarded by
# #ifdef KNNG_HAVE_HIP, so no source changes are needed — only this module
# and the CMakeLists.txt wiring below are required.

set(KNNG_HAVE_HIP OFF CACHE BOOL "HIP/ROCm found and GPU build enabled" FORCE)

if(NOT KNNG_BACKEND STREQUAL "HIP")
    return()
endif()

# ROCm ships find modules under /opt/rocm/lib/cmake.
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

find_package(HIP QUIET)

if(NOT HIP_FOUND)
    message(STATUS "knng: KNNG_BACKEND=HIP but HIP package not found — GPU build disabled")
    return()
endif()

set(KNNG_HAVE_HIP ON CACHE BOOL "HIP/ROCm found and GPU build enabled" FORCE)
message(STATUS "knng: HIP enabled — version ${HIP_VERSION}")

# Interface target (parallel to knng::cuda_iface).
add_library(knng_hip_iface INTERFACE)
target_include_directories(knng_hip_iface INTERFACE ${HIP_INCLUDE_DIRS})
target_compile_definitions(knng_hip_iface INTERFACE KNNG_HAVE_HIP=1)
add_library(knng::hip_iface ALIAS knng_hip_iface)
