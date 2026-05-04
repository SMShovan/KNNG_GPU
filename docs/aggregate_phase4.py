#!/usr/bin/env python3
"""
aggregate_phase4.py — flatten Phase-4 bench JSON into the
single-threaded ladder + strong-scaling tables shown in
docs/SCALING_CPU.md.

Usage:
    docs/aggregate_phase4.py path/to/phase4.json

The script is intentionally short and dependency-light (json +
argparse + pathlib only). It is committed alongside SCALING_CPU.md
so the writeup is reproducible: a future contributor running the
bench on a different host can rerun this script against the new
JSON and produce a structurally identical writeup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Field names match Step-16's counter set + Step-24's `threads` knob.
F_NAME    = "run_name"
F_TIME    = "real_time"
F_RECALL  = "recall_at_k"
F_THREADS = "threads"


def _counter(row: dict[str, Any], key: str) -> Any:
    # Google Benchmark JSON places counter values either inside a
    # `counters` sub-dict or flattened at the row level depending
    # on version. Check both.
    counters = row.get("counters") or {}
    return counters.get(key) if key in counters else row.get(key)


# Family names that report a `threads` counter — i.e. the parallel
# variants that contribute to the strong-scaling sweep.
_PARALLEL_FAMILIES = (
    "BM_BruteForceL2Omp_",
    "BM_BruteForceL2OmpScratch_",
    "BM_BruteForceL2Threaded_",
)


def _is_parallel(row: dict[str, Any]) -> bool:
    name = row.get(F_NAME, "")
    return any(name.startswith(prefix) for prefix in _PARALLEL_FAMILIES)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Bench JSON output.")
    args = p.parse_args(argv)

    with args.input.open("r") as f:
        d = json.load(f)

    rows = [
        r for r in d.get("benchmarks", [])
        if r.get("run_type") == "iteration"
    ]

    print("# Single-threaded ladder")
    for r in rows:
        if "/1024/128" not in r[F_NAME]:
            continue
        if _is_parallel(r):
            continue
        rec = _counter(r, F_RECALL)
        print(f"  {r[F_NAME]:55s} {r[F_TIME]:8.2f} ms   "
              f"recall={rec}")

    print()
    print("# Strong-scaling sweep")
    for r in rows:
        if not _is_parallel(r):
            continue
        if "/1024/128" not in r[F_NAME]:
            continue
        th = _counter(r, F_THREADS)
        rec = _counter(r, F_RECALL)
        th_s = f" t={int(th)}" if th is not None else ""
        print(f"  {r[F_NAME]:55s} {r[F_TIME]:8.2f} ms"
              f"{th_s}   recall={rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
