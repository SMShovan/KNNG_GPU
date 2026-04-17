# CompilerWarnings.cmake
#
# Provides a single helper, `knng_set_warnings(<target>)`, that applies the
# project's agreed warning policy to a target. Centralizing the flags here
# keeps every `CMakeLists.txt` short and guarantees that new subdirectories
# inherit the same strictness without copy-paste drift.
#
# Policy:
#   * GCC / Clang: -Wall -Wextra -Wpedantic -Wshadow -Wconversion
#                  -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
#                  -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
#                  -Wformat=2 -Wimplicit-fallthrough
#                  -Werror (treat warnings as errors)
#   * MSVC       : /W4 /WX /permissive-
#
# Escape hatch: set -DKNNG_WARNINGS_AS_ERRORS=OFF at configure time to keep
# warnings visible without failing the build. Useful when bisecting bugs or
# working with a stricter compiler version than CI expects.

option(KNNG_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

function(knng_set_warnings target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "knng_set_warnings: '${target}' is not a target")
    endif()

    set(_gnu_flags
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )

    set(_msvc_flags
        /W4
        /permissive-
    )

    if(KNNG_WARNINGS_AS_ERRORS)
        list(APPEND _gnu_flags  -Werror)
        list(APPEND _msvc_flags /WX)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE ${_gnu_flags})
    elseif(MSVC)
        target_compile_options(${target} PRIVATE ${_msvc_flags})
    endif()
endfunction()
