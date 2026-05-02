# benchmarks/CMakeLists.txt
#
# Google-Benchmark-driven micro and end-to-end benchmarks. Opt-in via
# `-DKNNG_BUILD_BENCHMARKS=ON` (default OFF) so the standard developer
# loop and the CI matrix don't pay the FetchContent + build cost
# unless they actually want benchmark numbers.
#
# Convention:
#   * One bench binary per topic (`bench_brute_force`, later
#     `bench_distance_kernel`, etc.). Binaries live under
#     `build/bin/`.
#   * Every bench target links the library it benchmarks plus
#     `benchmark::benchmark_main` (Google Benchmark's prebuilt main).
#   * `knng_set_warnings()` applies to our bench TUs but never to
#     Google Benchmark's own targets.

add_executable(bench_brute_force bench_brute_force.cpp)
target_link_libraries(bench_brute_force
    PRIVATE
        knng::bench
        knng::cpu
        knng::io
        benchmark::benchmark_main
)
knng_set_warnings(bench_brute_force)

set_target_properties(bench_brute_force PROPERTIES
    OUTPUT_NAME "bench_brute_force"
)
