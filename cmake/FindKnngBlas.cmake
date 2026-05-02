# FindKnngBlas.cmake
#
# Discovers a CBLAS-providing library and exposes it as the
# `knng::blas_iface` INTERFACE target. Sets the cache variable
# `KNNG_HAVE_BLAS` to ON when discovery succeeded.
#
# Discovery order:
#   * Apple (macOS):  the system Accelerate framework. Always
#     present on every supported macOS version, no third-party
#     install needed. The framework header chain pulls in
#     `<Accelerate/Accelerate.h>`, which transitively defines the
#     standard `cblas_sgemm` symbols.
#   * Otherwise:      `find_package(BLAS QUIET)`. On Linux this
#     picks up OpenBLAS / MKL / netlib-blas — whichever the user
#     has installed via apt / yum / brew. We additionally
#     `find_path(... cblas.h)` so the include directory is wired
#     up; some BLAS providers (notably MKL) place the header in a
#     non-default location.
#
# When neither succeeds, `KNNG_HAVE_BLAS` is left OFF and
# `knng::blas_iface` is not created. The caller is expected to gate
# the BLAS-using sources behind the cache variable.

option(KNNG_ENABLE_BLAS "Enable BLAS-backed brute-force builder (Step 21)" ON)

if(NOT KNNG_ENABLE_BLAS)
    set(KNNG_HAVE_BLAS OFF CACHE INTERNAL "BLAS disabled by KNNG_ENABLE_BLAS=OFF")
    return()
endif()

if(APPLE)
    find_library(KNNG_ACCELERATE_FRAMEWORK Accelerate)
    if(KNNG_ACCELERATE_FRAMEWORK)
        add_library(knng_blas_iface INTERFACE)
        target_link_libraries(knng_blas_iface INTERFACE
            ${KNNG_ACCELERATE_FRAMEWORK})
        # `ACCELERATE_NEW_LAPACK=1` opts in to the post-macOS-13.3
        # CBLAS interface; without it `cblas_sgemm` is marked
        # `deprecated` and `-Werror` rejects it. The new header
        # signatures are source-compatible with reference CBLAS,
        # so user code does not change.
        target_compile_definitions(knng_blas_iface INTERFACE
            KNNG_HAVE_BLAS=1
            KNNG_BLAS_USES_ACCELERATE=1
            ACCELERATE_NEW_LAPACK=1)
        add_library(knng::blas_iface ALIAS knng_blas_iface)
        set(KNNG_HAVE_BLAS ON CACHE INTERNAL "BLAS via Accelerate")
        message(STATUS "knng: BLAS via Accelerate framework "
                       "(${KNNG_ACCELERATE_FRAMEWORK})")
        return()
    endif()
endif()

# Generic path: enable C so find_package(BLAS) has a compiler to
# probe with, then ask CMake's standard finder.
include(CheckLanguage)
check_language(C)
if(CMAKE_C_COMPILER)
    enable_language(C)
endif()

find_package(BLAS QUIET)
if(BLAS_FOUND)
    # Locate cblas.h. The default search paths cover the common
    # Linux layouts; explicit hints handle Homebrew and MKL.
    find_path(KNNG_CBLAS_INCLUDE_DIR cblas.h
        HINTS
            /usr/include
            /usr/include/openblas
            /usr/local/include
            /opt/homebrew/include
            /opt/homebrew/opt/openblas/include
            $ENV{MKLROOT}/include
        PATH_SUFFIXES openblas)
    if(KNNG_CBLAS_INCLUDE_DIR)
        add_library(knng_blas_iface INTERFACE)
        target_link_libraries(knng_blas_iface INTERFACE ${BLAS_LIBRARIES})
        target_include_directories(knng_blas_iface INTERFACE
            ${KNNG_CBLAS_INCLUDE_DIR})
        target_compile_definitions(knng_blas_iface INTERFACE
            KNNG_HAVE_BLAS=1)
        add_library(knng::blas_iface ALIAS knng_blas_iface)
        set(KNNG_HAVE_BLAS ON CACHE INTERNAL "BLAS via find_package")
        message(STATUS "knng: BLAS via find_package (${BLAS_LIBRARIES}); "
                       "cblas.h in ${KNNG_CBLAS_INCLUDE_DIR}")
        return()
    endif()
endif()

set(KNNG_HAVE_BLAS OFF CACHE INTERNAL "BLAS not found")
message(STATUS "knng: no CBLAS provider found — Step 21's "
               "brute_force_knn_l2_blas will not be compiled")
