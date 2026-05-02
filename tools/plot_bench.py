#!/usr/bin/env python3
"""
plot_bench.py — render Google-Benchmark JSON output as Matplotlib plots.

This script is *tooling*, not part of the C++ build. It is committed
alongside the bench targets so the same JSON shape Step 16 emits has
a reference renderer; downstream consumers (a CI artefact upload,
the Phase 13 Pareto plot in `docs/`, an ad-hoc developer who just
wants to look at one run) can either use this script or treat it as
a worked example for their own pipeline.

Counters consumed (each run benchmark family carries them):
    * recall_at_k              — quality of the approximate graph
    * peak_memory_mb           — RSS high-watermark
    * n_distance_computations  — pairwise distance evaluations
    * n, d, k                  — dataset / parameter shape

Usage:
    ./tools/plot_bench.py results.json [--output prefix]

If `--output prefix` is supplied, plots are written as
`{prefix}.recall_vs_n.png` / `.time_vs_n.png` /
`.memory_vs_n.png`. Otherwise they are shown interactively
via `plt.show()`.

Dependencies (install with `pip install --user ...`):
    matplotlib  — plotting
    (json + argparse + pathlib are standard library)

Why this lives in tools/ rather than benchmarks/:
The C++ benchmark target produces JSON; rendering is a separate
concern that should not pull a Python interpreter into the
default developer build. CI installs matplotlib only on the docs
job; ordinary contributors never touch this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Field names emitted by `annotate()` in benchmarks/bench_brute_force.cpp.
# Centralised here so renaming a counter is a single-edit change in the
# Python side too.
FIELD_N        = "n"
FIELD_D        = "d"
FIELD_K        = "k"
FIELD_TIME_MS  = "real_time"
FIELD_RECALL   = "recall_at_k"
FIELD_MEMORY   = "peak_memory_mb"
FIELD_DISTANCE = "n_distance_computations"


def _load_runs(path: Path) -> list[dict[str, Any]]:
    """Read Google-Benchmark JSON and return only the per-iteration
    rows. Rows of `run_type == "aggregate"` (mean / stddev when the
    bench is run with `--benchmark_repetitions=...`) are dropped so
    every plot point corresponds to exactly one wall-clock measurement.
    """
    with path.open("r") as f:
        doc = json.load(f)
    benchmarks = doc.get("benchmarks", [])
    return [b for b in benchmarks if b.get("run_type") == "iteration"]


def _grouped_by_d(runs: list[dict[str, Any]]) -> dict[float, list[dict[str, Any]]]:
    """Bucket runs by the `d` (dimensionality) field. The bench harness
    produces an outer-product over (n, d), so plotting "metric vs n"
    only makes sense within a single `d` bucket.
    """
    buckets: dict[float, list[dict[str, Any]]] = {}
    for r in runs:
        d = r.get(FIELD_D)
        if d is None:
            continue
        buckets.setdefault(d, []).append(r)
    for d, rs in buckets.items():
        rs.sort(key=lambda r: r.get(FIELD_N, 0.0))
    return buckets


def _plot_metric_vs_n(buckets: dict[float, list[dict[str, Any]]],
                      field: str,
                      ylabel: str,
                      title: str,
                      out: Path | None,
                      log_y: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for d, rs in sorted(buckets.items()):
        xs = [r.get(FIELD_N, 0.0) for r in rs]
        ys = [r.get(field, 0.0)   for r in rs]
        ax.plot(xs, ys, marker="o", label=f"d={int(d)}")
    ax.set_xlabel("n (points)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if out is None:
        plt.show()
    else:
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render knng bench JSON to Matplotlib plots.")
    parser.add_argument("input", type=Path,
                        help="Path to a Google-Benchmark JSON file "
                             "(produced by --benchmark_format=json).")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output filename prefix. If omitted, plots "
                             "are shown interactively.")
    args = parser.parse_args(argv)

    runs = _load_runs(args.input)
    if not runs:
        print(f"no iteration rows found in {args.input}", file=sys.stderr)
        return 2

    buckets = _grouped_by_d(runs)
    if not buckets:
        print("no rows with a `d` field; refusing to plot",
              file=sys.stderr)
        return 2

    plots = [
        (FIELD_RECALL, "recall@k",         "Recall@k vs n",
         (str(args.output) + ".recall_vs_n.png") if args.output else None,
         False),
        (FIELD_TIME_MS, "wall time (ms)", "Wall time vs n",
         (str(args.output) + ".time_vs_n.png") if args.output else None,
         True),
        (FIELD_MEMORY, "peak memory (MB)", "Peak memory vs n",
         (str(args.output) + ".memory_vs_n.png") if args.output else None,
         False),
    ]
    for field, ylabel, title, outpath, log_y in plots:
        _plot_metric_vs_n(buckets, field, ylabel, title,
                          Path(outpath) if outpath else None, log_y=log_y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
