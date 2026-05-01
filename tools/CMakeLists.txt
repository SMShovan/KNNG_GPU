# tools/CMakeLists.txt
#
# Command-line utilities built on top of the knng library. At this early
# stage the only entry is `hello_knng`, a smoke-test executable that
# exercises the build system end-to-end. Future steps will add dataset
# download/conversion utilities and the `build_knng` CLI.

add_executable(hello_knng hello_knng.cpp)
target_link_libraries(hello_knng PRIVATE knng::headers)
knng_set_warnings(hello_knng)

set_target_properties(hello_knng PROPERTIES
    OUTPUT_NAME "hello_knng"
)

# `build_knng` — end-to-end CLI builder. Loads a .fvecs dataset,
# runs the CPU brute-force algorithm, writes a binary .knng graph.
# See tools/build_knng.cpp for the output format documentation.
add_executable(build_knng build_knng.cpp)
target_link_libraries(build_knng
    PRIVATE
        knng::cpu
        knng::io
)
knng_set_warnings(build_knng)

set_target_properties(build_knng PROPERTIES
    OUTPUT_NAME "build_knng"
)
