#!/usr/bin/env python3
"""
aggregate_phase5.py — flatten NN-Descent bench JSON into the
recall-vs-time table shown in docs/NN_DESCENT.md.

Usage:
    docs/aggregate_phase5.py path/to/phase5_nnd.json

Mirrors docs/aggregate_phase4.py's shape: standard library only,
field names mirror the C++ counter set so a counter rename is a
two-edit change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


F_NAME    = "run_name"
F_TIME    = "real_time"
F_RECALL  = "recall_at_k"
F_THREADS = "threads"
F_REVERSE = "use_reverse"
F_RHO     = "rho"
F_ITERS   = "iterations"


def _counter(row: dict[str, Any], key: str) -> Any:
    counters = row.get("counters") or {}
    return counters.get(key) if key in counters else row.get(key)


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

    print(f"{'Variant':52s} {'t':>2} {'rev':>3} {'rho':>4} "
          f"{'iters':>5} {'time(ms)':>9} {'recall':>7}")
    for r in rows:
        rt    = r[F_TIME]
        rec   = _counter(r, F_RECALL)
        th    = _counter(r, F_THREADS)
        rev   = _counter(r, F_REVERSE)
        rho   = _counter(r, F_RHO)
        iters = _counter(r, F_ITERS)
        rec_s = f"{rec:.3f}" if rec is not None else "-"
        rho_s = f"{rho:.2f}" if rho is not None else "-"
        print(f"{r[F_NAME]:52s} {int(th) if th else '-':>2} "
              f"{int(rev) if rev is not None else '-':>3} "
              f"{rho_s:>4} {int(iters) if iters else '-':>5} "
              f"{rt:>9.2f} {rec_s:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
