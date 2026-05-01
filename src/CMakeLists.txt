# src/CMakeLists.txt
#
# Library targets:
#   * `knng::core` — INTERFACE library covering the public type
#     scaffold (types, Distance concept, Dataset, Knng, TopK). Header-
#     only by design; consumers link it to inherit include paths and
#     the C++20 requirement.
#   * `knng::cpu`  — STATIC library for CPU-side algorithms whose
#     implementations need real translation units (brute_force from
#     Step 10 onward). Transitively brings in `knng::core` so a
#     downstream consumer that links `knng::cpu` does not also need
#     to spell out `knng::core`.

add_library(knng_core INTERFACE)
target_link_libraries(knng_core INTERFACE knng::headers)
add_library(knng::core ALIAS knng_core)

add_library(knng_cpu STATIC
    cpu/brute_force.cpp
)
target_link_libraries(knng_cpu PUBLIC knng::core)
knng_set_warnings(knng_cpu)
add_library(knng::cpu ALIAS knng_cpu)
