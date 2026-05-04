#!/usr/bin/env bash
# tools/run_bench_numa.sh — wrap bench_brute_force in numactl when
# the host is a Linux multi-socket system, run it bare otherwise.
#
# Usage:
#   tools/run_bench_numa.sh [bench_brute_force flags...]
#
# What it does:
#   * Detects `numactl` availability (Linux only).
#   * On Linux: invokes `numactl --interleave=all build/bin/bench_brute_force`.
#     The interleave policy spreads pages across all NUMA nodes
#     round-robin, which is the right baseline when the bench
#     produces wall-time numbers that do not assume the
#     `first_touch` pass has aligned the layout.
#   * On macOS or any host without numactl: invokes the bench
#     directly.
#
# Why the wrapper exists:
#   Step 16's bench harness emits JSON; Step 29's scaling writeup
#   ingests the JSON. A NUMA-blind run on a multi-socket host
#   reports numbers dominated by remote-DRAM stalls and would lie
#   about strong-scaling efficiency. `--interleave=all` is the
#   smallest knob that produces interpretable numbers; the
#   `first_touch`-aware path (`brute_force_knn_l2_omp_scratch` plus
#   a pre-bench `first_touch` call) is the *real* fix that lands
#   in Step 29.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${REPO_ROOT}/build/bin/bench_brute_force"

if [[ ! -x "${BENCH}" ]]; then
    echo "error: ${BENCH} not built." >&2
    echo "  hint: cmake --build build --target bench_brute_force --parallel" >&2
    exit 2
fi

if command -v numactl >/dev/null 2>&1; then
    echo "+ numactl --interleave=all ${BENCH} $*" >&2
    exec numactl --interleave=all "${BENCH}" "$@"
else
    echo "+ ${BENCH} $*  # no numactl on this host" >&2
    exec "${BENCH}" "$@"
fi
