# docs/CMakeLists.txt
#
# Doxygen HTML documentation target.
#
# Behaviour:
#   * If Doxygen is not installed, this file returns early with a
#     STATUS message. Configure continues normally; `docs` is a
#     developer nicety, not a build requirement.
#   * If graphviz (`dot`) is installed, class graphs are enabled.
#   * Output lands in `<build>/docs/html/`. Open `index.html` in a
#     browser to read.

find_package(Doxygen OPTIONAL_COMPONENTS dot)

if(NOT DOXYGEN_FOUND)
    message(STATUS "knng: Doxygen not found — the `docs` target is disabled.")
    message(STATUS "      Install via: brew install doxygen         (macOS)")
    message(STATUS "                    sudo apt install doxygen    (Debian/Ubuntu)")
    return()
endif()

# Doxygen expects INPUT as a whitespace-separated list. Concatenating
# with " \\\n" keeps the generated Doxyfile readable if a human opens
# it. All paths are absolute so Doxygen does not depend on its working
# directory.
set(_knng_doxygen_inputs
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/tools"
    "${PROJECT_SOURCE_DIR}/README.md"
    "${PROJECT_SOURCE_DIR}/CHANGELOG.md"
    "${PROJECT_SOURCE_DIR}/docs/STYLE.md"
)
list(JOIN _knng_doxygen_inputs " \\\n                         " KNNG_DOXYGEN_INPUT)

set(KNNG_DOXYGEN_OUTPUT   "${CMAKE_CURRENT_BINARY_DIR}")
set(KNNG_DOXYGEN_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")

if(DOXYGEN_DOT_FOUND)
    set(KNNG_DOXYGEN_HAVE_DOT "YES")
else()
    set(KNNG_DOXYGEN_HAVE_DOT "NO")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY
)

# Using `add_custom_target` (rather than `add_custom_command` + a
# driver target) means the docs are never rebuilt on an ordinary
# `cmake --build build` — they are opt-in via
# `cmake --build build --target docs`. This keeps test/CI runtimes
# from being inflated by a Doxygen pass nobody asked for.
add_custom_target(docs
    COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "knng: generating Doxygen HTML documentation"
    VERBATIM
)

message(STATUS
    "knng: Doxygen found — `cmake --build build --target docs` "
    "writes HTML to ${CMAKE_CURRENT_BINARY_DIR}/html/")
