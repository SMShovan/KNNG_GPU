# cmake/FindKnngNCCL.cmake
#
# Discovers NCCL (NVIDIA Collective Communications Library) and exposes it
# as the `knng::nccl_iface` INTERFACE target. Sets `KNNG_HAVE_NCCL` to ON
# when discovery succeeded.
#
# Requires CUDA to already be enabled (KNNG_HAVE_CUDA=ON); NCCL without CUDA
# is meaningless and silently skipped.
#
# Search order:
#   1. NCCL_ROOT / NCCL_DIR CMake/env variables
#   2. CUDAToolkit prefix (common when NCCL ships alongside CUDA)
#   3. Standard system paths
#
# Exposed compile definition: KNNG_HAVE_NCCL=1

option(KNNG_ENABLE_NCCL "Enable NCCL multi-GPU collectives (Phase 11+)" ON)

if(NOT KNNG_ENABLE_NCCL OR NOT KNNG_HAVE_CUDA)
    set(KNNG_HAVE_NCCL OFF CACHE INTERNAL "NCCL disabled or CUDA unavailable")
    return()
endif()

# Search hints: honour explicit roots first.
set(_nccl_hints "")
if(DEFINED NCCL_ROOT)
    list(APPEND _nccl_hints "${NCCL_ROOT}")
endif()
if(DEFINED ENV{NCCL_ROOT})
    list(APPEND _nccl_hints "$ENV{NCCL_ROOT}")
endif()
if(DEFINED CUDAToolkit_ROOT)
    list(APPEND _nccl_hints "${CUDAToolkit_ROOT}")
endif()

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${_nccl_hints}
    PATH_SUFFIXES include
)
find_library(NCCL_LIBRARY
    NAMES nccl nccl_static
    HINTS ${_nccl_hints}
    PATH_SUFFIXES lib lib64
)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    # Extract version from nccl.h
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_ver_line
         REGEX "#define NCCL_VERSION_CODE")
    if(_nccl_ver_line)
        string(REGEX REPLACE ".*NCCL_VERSION_CODE +([0-9]+).*" "\\1"
               NCCL_VERSION_CODE "${_nccl_ver_line}")
    else()
        set(NCCL_VERSION_CODE "unknown")
    endif()

    add_library(knng_nccl_iface INTERFACE)
    target_include_directories(knng_nccl_iface INTERFACE "${NCCL_INCLUDE_DIR}")
    target_link_libraries(knng_nccl_iface INTERFACE "${NCCL_LIBRARY}")
    target_compile_definitions(knng_nccl_iface INTERFACE KNNG_HAVE_NCCL=1)
    add_library(knng::nccl_iface ALIAS knng_nccl_iface)

    set(KNNG_HAVE_NCCL ON CACHE INTERNAL "NCCL found via find_path/find_library")
    message(STATUS "knng: NCCL found (version code ${NCCL_VERSION_CODE}) — ${NCCL_LIBRARY}")
    return()
endif()

set(KNNG_HAVE_NCCL OFF CACHE INTERNAL "NCCL not found")
message(STATUS "knng: NCCL not found — Phase 11+ multi-GPU collectives skipped "
               "(set NCCL_ROOT or install NCCL and re-run cmake to enable)")
