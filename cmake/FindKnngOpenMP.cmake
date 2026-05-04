# FindKnngOpenMP.cmake
#
# Discovers an OpenMP runtime and exposes it as the
# `knng::openmp_iface` INTERFACE target. Sets the cache variable
# `KNNG_HAVE_OPENMP` to ON when discovery succeeded.
#
# Discovery is non-trivial because the dev machine matrix includes
# AppleClang, which does NOT ship libomp out of the box but DOES
# accept it via `-Xpreprocessor -fopenmp` plus a manual link to
# Homebrew's libomp. CMake's stock `find_package(OpenMP)` handles
# this only when `OpenMP_ROOT` is pre-set or when the headers /
# library happen to live in a hard-coded list of paths the module
# probes. We pre-set `OpenMP_ROOT` to the Homebrew layout on Apple
# so the user does not have to configure it themselves.
#
# When discovery fails, `KNNG_HAVE_OPENMP` is left OFF and the
# project's parallel-builder TUs gate themselves on the cache
# variable. The non-OpenMP build is still functional — every
# `#pragma omp parallel for` falls back to a serial loop.

option(KNNG_ENABLE_OPENMP "Enable OpenMP-parallel CPU builders (Step 24)" ON)

if(NOT KNNG_ENABLE_OPENMP)
    set(KNNG_HAVE_OPENMP OFF CACHE INTERNAL
        "OpenMP disabled by KNNG_ENABLE_OPENMP=OFF")
    return()
endif()

# CMake's FindOpenMP needs an enabled language to probe with.
include(CheckLanguage)

# Hint at Homebrew's libomp on Apple before find_package runs.
# `OpenMP_ROOT` is the standard hint variable; setting it here means
# the standard FindOpenMP module sees the right include + library
# directory without the user touching `cmake/`.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
    if(EXISTS /opt/homebrew/opt/libomp)
        set(OpenMP_ROOT /opt/homebrew/opt/libomp)
    elseif(EXISTS /usr/local/opt/libomp)
        set(OpenMP_ROOT /usr/local/opt/libomp)
    endif()
endif()

find_package(OpenMP QUIET COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
    add_library(knng_openmp_iface INTERFACE)
    target_link_libraries(knng_openmp_iface INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(knng_openmp_iface INTERFACE
        KNNG_HAVE_OPENMP=1)
    add_library(knng::openmp_iface ALIAS knng_openmp_iface)
    set(KNNG_HAVE_OPENMP ON CACHE INTERNAL "OpenMP via find_package")
    message(STATUS "knng: OpenMP ${OpenMP_CXX_VERSION} via "
                   "${OpenMP_CXX_LIBRARIES}")
    return()
endif()

set(KNNG_HAVE_OPENMP OFF CACHE INTERNAL "OpenMP not found")
message(STATUS "knng: OpenMP not found — Step 24+ parallel builders "
               "will fall back to serial loops")
