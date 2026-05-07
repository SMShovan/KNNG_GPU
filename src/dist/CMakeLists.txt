# src/dist/CMakeLists.txt
#
# `knng::dist` — MPI-backed distributed algorithms.
#
# This subdirectory is only included when `KNNG_HAVE_MPI` is ON
# (set by `cmake/FindKnngMPI.cmake`). Every target here links
# `knng::mpi_iface`, which propagates `MPI::MPI_CXX` and the
# `KNNG_HAVE_MPI=1` compile definition.
#
# Target layout mirrors `knng::cpu`:
#   * `knng::dist` — STATIC library for the distributed algorithms.
#
# Phase 6 adds to this list incrementally:
#   Step 38 — mpi_env       (init / finalize RAII)
#   Step 39 — sharded_dataset
#   Step 40 — brute_force_mpi
#   Step 41 — nn_descent_mpi (basic gather-scatter)
#   Step 42 — request_dedup  (NEO-DNND duplicate reduction)
#   Step 43 — shm_replication (MPI-3 SHM intra-node)

add_library(knng_dist STATIC
    mpi_env.cpp
    sharded_dataset.cpp
    brute_force_mpi.cpp
    nn_descent_mpi.cpp
    request_dedup.cpp
    shm_replication.cpp
)
target_link_libraries(knng_dist PUBLIC
    knng::cpu
    knng::mpi_iface
)
knng_set_warnings(knng_dist)
add_library(knng::dist ALIAS knng_dist)
