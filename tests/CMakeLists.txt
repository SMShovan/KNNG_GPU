# tests/CMakeLists.txt
#
# GoogleTest-based unit and integration tests. Each test binary is kept
# small and topical; `gtest_discover_tests` registers its TEST()s with
# CTest so `ctest` in the build tree picks them up automatically.
#
# Convention:
#   * One test binary per logical area (version, distance, graph, ...).
#   * Binaries are named `test_<area>` and installed under build/bin for
#     easy ad-hoc invocation (`./build/bin/test_smoke`).
#   * Every test target links `knng::headers` + `GTest::gtest_main` and
#     opts in to the project-wide warning policy via `knng_set_warnings`.

add_executable(test_smoke smoke_test.cpp)
target_link_libraries(test_smoke
    PRIVATE
        knng::headers
        GTest::gtest_main
)
knng_set_warnings(test_smoke)

set_target_properties(test_smoke PROPERTIES
    OUTPUT_NAME "test_smoke"
)

gtest_discover_tests(test_smoke
    PROPERTIES
        LABELS "smoke"
    DISCOVERY_TIMEOUT 10
)

# `test_core` exercises the `knng::core` public API (types, Distance
# concept, built-in metrics, the Knng adjacency struct). Linking
# `knng::core` rather than `knng::headers` directly mirrors how a
# downstream consumer of the library will build against us.
add_executable(test_core core_test.cpp)
target_link_libraries(test_core
    PRIVATE
        knng::core
        GTest::gtest_main
)
knng_set_warnings(test_core)

set_target_properties(test_core PROPERTIES
    OUTPUT_NAME "test_core"
)

gtest_discover_tests(test_core
    PROPERTIES
        LABELS "core"
    DISCOVERY_TIMEOUT 10
)

# `test_top_k` exercises the bounded-size top-k buffer that every
# nearest-neighbor builder will push candidates into. Header-only at
# Step 09; lives outside `knng::core` because top-k is a generic
# utility rather than a member of the public-API surface.
add_executable(test_top_k top_k_test.cpp)
target_link_libraries(test_top_k
    PRIVATE
        knng::core
        GTest::gtest_main
)
knng_set_warnings(test_top_k)

set_target_properties(test_top_k PROPERTIES
    OUTPUT_NAME "test_top_k"
)

gtest_discover_tests(test_top_k
    PROPERTIES
        LABELS "top_k"
    DISCOVERY_TIMEOUT 10
)

# `test_brute_force` exercises the CPU brute-force KNN reference
# builder. Links `knng::cpu` so the explicit-template-instantiation
# symbols come from the static library rather than re-instantiating
# in the test TU.
add_executable(test_brute_force brute_force_test.cpp)
target_link_libraries(test_brute_force
    PRIVATE
        knng::cpu
        GTest::gtest_main
)
knng_set_warnings(test_brute_force)

set_target_properties(test_brute_force PROPERTIES
    OUTPUT_NAME "test_brute_force"
)

gtest_discover_tests(test_brute_force
    PROPERTIES
        LABELS "brute_force"
    DISCOVERY_TIMEOUT 10
)
