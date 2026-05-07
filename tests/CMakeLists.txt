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

# `test_fvecs` exercises the .fvecs / .ivecs / .bvecs loaders against
# tiny in-memory fixtures written to the OS temp directory. No
# network I/O — the SIFT-small download is a separate ad-hoc tool.
add_executable(test_fvecs fvecs_test.cpp)
target_link_libraries(test_fvecs
    PRIVATE
        knng::io
        GTest::gtest_main
)
knng_set_warnings(test_fvecs)

set_target_properties(test_fvecs PROPERTIES
    OUTPUT_NAME "test_fvecs"
)

gtest_discover_tests(test_fvecs
    PROPERTIES
        LABELS "fvecs"
    DISCOVERY_TIMEOUT 10
)

# `test_ground_truth` exercises the brute-force ground-truth cache
# (Step 14): dataset hash stability, round-trip serialisation,
# cache-key validation, and the load-or-compute convenience.
add_executable(test_ground_truth ground_truth_test.cpp)
target_link_libraries(test_ground_truth
    PRIVATE
        knng::bench
        GTest::gtest_main
)
knng_set_warnings(test_ground_truth)

set_target_properties(test_ground_truth PROPERTIES
    OUTPUT_NAME "test_ground_truth"
)

gtest_discover_tests(test_ground_truth
    PROPERTIES
        LABELS "ground_truth"
    DISCOVERY_TIMEOUT 10
)

# `test_recall` exercises `knng::bench::recall_at_k` (Step 15) on
# small hand-built graphs: exact match, scrambled order, partial
# overlap, duplicates, shape-mismatch errors.
add_executable(test_recall recall_test.cpp)
target_link_libraries(test_recall
    PRIVATE
        knng::bench
        GTest::gtest_main
)
knng_set_warnings(test_recall)

set_target_properties(test_recall PROPERTIES
    OUTPUT_NAME "test_recall"
)

gtest_discover_tests(test_recall
    PROPERTIES
        LABELS "recall"
    DISCOVERY_TIMEOUT 10
)

# `test_random` exercises the `knng::random::XorShift64` PRNG
# (Step 17): seed → sequence determinism, non-zero state invariant,
# float-in-[0,1), bounded-integer helper, drop-in compatibility
# with `std::uniform_int_distribution`.
add_executable(test_random random_test.cpp)
target_link_libraries(test_random
    PRIVATE
        knng::core
        GTest::gtest_main
)
knng_set_warnings(test_random)

set_target_properties(test_random PROPERTIES
    OUTPUT_NAME "test_random"
)

gtest_discover_tests(test_random
    PROPERTIES
        LABELS "random"
    DISCOVERY_TIMEOUT 10
)

# `test_numa` exercises the cross-platform first-touch helper
# (Step 26): no-op on null/zero, contents preserved, sub-page
# buffer tolerated, platform flag returns a stable value.
add_executable(test_numa numa_test.cpp)
target_link_libraries(test_numa
    PRIVATE
        knng::cpu
        GTest::gtest_main
)
knng_set_warnings(test_numa)

set_target_properties(test_numa PROPERTIES
    OUTPUT_NAME "test_numa"
)

gtest_discover_tests(test_numa
    PROPERTIES
        LABELS "numa"
    DISCOVERY_TIMEOUT 10
)

# `test_neighbor_list` exercises the per-point sorted neighbour
# list (Step 30) — sorted invariant, tie-break, eviction, duplicate
# handling, is_new accounting.
add_executable(test_neighbor_list neighbor_list_test.cpp)
target_link_libraries(test_neighbor_list
    PRIVATE
        knng::cpu
        GTest::gtest_main
)
knng_set_warnings(test_neighbor_list)

set_target_properties(test_neighbor_list PROPERTIES
    OUTPUT_NAME "test_neighbor_list"
)

gtest_discover_tests(test_neighbor_list
    PROPERTIES
        LABELS "neighbor_list"
    DISCOVERY_TIMEOUT 10
)

# `test_nn_descent` exercises the CPU NN-Descent surface across
# Phase 5 — Step 31 covers the random-graph initialiser; later
# steps append more cases to the same TU.
add_executable(test_nn_descent nn_descent_test.cpp)
target_link_libraries(test_nn_descent
    PRIVATE
        knng::bench
        knng::cpu
        GTest::gtest_main
)
knng_set_warnings(test_nn_descent)

set_target_properties(test_nn_descent PROPERTIES
    OUTPUT_NAME "test_nn_descent"
)

gtest_discover_tests(test_nn_descent
    PROPERTIES
        LABELS "nn_descent"
    DISCOVERY_TIMEOUT 10
)

# `test_mpi_env` exercises `knng::dist::MpiEnv` (Step 38).
# Launched via `mpirun -np 1` so CTest can run it without a full
# multi-rank setup. MPI_Allreduce with one rank degenerates to a
# local identity, which validates the MPI wiring end-to-end.
# Only built when `KNNG_HAVE_MPI` is ON.
if(KNNG_HAVE_MPI)
    find_program(MPIRUN_EXEC
        NAMES mpirun mpiexec
        DOC   "MPI launcher for test_mpi_env")

    add_executable(test_mpi_env mpi_env_test.cpp)
    target_link_libraries(test_mpi_env
        PRIVATE
            knng::dist
            GTest::gtest_main
    )
    knng_set_warnings(test_mpi_env)
    set_target_properties(test_mpi_env PROPERTIES
        OUTPUT_NAME "test_mpi_env")

    if(MPIRUN_EXEC)
        # Register the test to run under `mpirun -np 1`.
        add_test(
            NAME    mpi_env
            COMMAND "${MPIRUN_EXEC}" -np 1
                    "$<TARGET_FILE:test_mpi_env>"
                    --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/mpi_env.xml
        )
        set_tests_properties(mpi_env PROPERTIES LABELS "mpi;dist")
    else()
        # No launcher found — register a direct invocation (works for
        # single-process MPI implementations that don't need mpirun).
        gtest_discover_tests(test_mpi_env
            PROPERTIES
                LABELS "mpi;dist"
            DISCOVERY_TIMEOUT 10
        )
    endif()

    # `test_mpi_sharded` — Step 39: ShardedDataset scatter/gather.
    add_executable(test_mpi_sharded mpi_sharded_test.cpp)
    target_link_libraries(test_mpi_sharded
        PRIVATE
            knng::dist
            GTest::gtest_main
    )
    knng_set_warnings(test_mpi_sharded)
    set_target_properties(test_mpi_sharded PROPERTIES
        OUTPUT_NAME "test_mpi_sharded")

    if(MPIRUN_EXEC)
        add_test(
            NAME    mpi_sharded
            COMMAND "${MPIRUN_EXEC}" -np 1
                    "$<TARGET_FILE:test_mpi_sharded>"
        )
        set_tests_properties(mpi_sharded PROPERTIES LABELS "mpi;dist")
    else()
        gtest_discover_tests(test_mpi_sharded
            PROPERTIES
                LABELS "mpi;dist"
            DISCOVERY_TIMEOUT 10
        )
    endif()

    # `test_mpi_brute_force` — Step 40: distributed brute-force KNN.
    add_executable(test_mpi_brute_force mpi_brute_force_test.cpp)
    target_link_libraries(test_mpi_brute_force
        PRIVATE
            knng::dist
            GTest::gtest_main
    )
    knng_set_warnings(test_mpi_brute_force)
    set_target_properties(test_mpi_brute_force PROPERTIES
        OUTPUT_NAME "test_mpi_brute_force")

    if(MPIRUN_EXEC)
        add_test(
            NAME    mpi_brute_force
            COMMAND "${MPIRUN_EXEC}" -np 1
                    "$<TARGET_FILE:test_mpi_brute_force>"
        )
        set_tests_properties(mpi_brute_force PROPERTIES LABELS "mpi;dist")
    else()
        gtest_discover_tests(test_mpi_brute_force
            PROPERTIES
                LABELS "mpi;dist"
            DISCOVERY_TIMEOUT 10
        )
    endif()

    # `test_mpi_nn_descent` — Step 41: distributed NN-Descent.
    add_executable(test_mpi_nn_descent mpi_nn_descent_test.cpp)
    target_link_libraries(test_mpi_nn_descent
        PRIVATE
            knng::dist
            knng::bench
            GTest::gtest_main
    )
    knng_set_warnings(test_mpi_nn_descent)
    set_target_properties(test_mpi_nn_descent PROPERTIES
        OUTPUT_NAME "test_mpi_nn_descent")

    if(MPIRUN_EXEC)
        add_test(
            NAME    mpi_nn_descent
            COMMAND "${MPIRUN_EXEC}" -np 1
                    "$<TARGET_FILE:test_mpi_nn_descent>"
        )
        set_tests_properties(mpi_nn_descent PROPERTIES LABELS "mpi;dist")
    else()
        gtest_discover_tests(test_mpi_nn_descent
            PROPERTIES
                LABELS "mpi;dist"
            DISCOVERY_TIMEOUT 10
        )
    endif()

    # `test_mpi_request_dedup` — Step 42: duplicate-request reduction.
    add_executable(test_mpi_request_dedup mpi_request_dedup_test.cpp)
    target_link_libraries(test_mpi_request_dedup
        PRIVATE
            knng::dist
            GTest::gtest_main
    )
    knng_set_warnings(test_mpi_request_dedup)
    set_target_properties(test_mpi_request_dedup PROPERTIES
        OUTPUT_NAME "test_mpi_request_dedup")

    if(MPIRUN_EXEC)
        add_test(
            NAME    mpi_request_dedup
            COMMAND "${MPIRUN_EXEC}" -np 1
                    "$<TARGET_FILE:test_mpi_request_dedup>"
        )
        set_tests_properties(mpi_request_dedup PROPERTIES LABELS "mpi;dist")
    else()
        gtest_discover_tests(test_mpi_request_dedup
            PROPERTIES
                LABELS "mpi;dist"
            DISCOVERY_TIMEOUT 10
        )
    endif()

    # `test_mpi_shm_replication` — Step 43: intra-node SHM replication.
    add_executable(test_mpi_shm_replication mpi_shm_replication_test.cpp)
    target_link_libraries(test_mpi_shm_replication
        PRIVATE
            knng::dist
            GTest::gtest_main
    )
    knng_set_warnings(test_mpi_shm_replication)
    set_target_properties(test_mpi_shm_replication PROPERTIES
        OUTPUT_NAME "test_mpi_shm_replication")

    if(MPIRUN_EXEC)
        add_test(
            NAME    mpi_shm_replication
            COMMAND "${MPIRUN_EXEC}" -np 1
                    "$<TARGET_FILE:test_mpi_shm_replication>"
        )
        set_tests_properties(mpi_shm_replication
            PROPERTIES LABELS "mpi;dist")
    else()
        gtest_discover_tests(test_mpi_shm_replication
            PROPERTIES
                LABELS "mpi;dist"
            DISCOVERY_TIMEOUT 10
        )
    endif()
endif()
