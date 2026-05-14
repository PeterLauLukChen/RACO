#!/usr/bin/env python3
"""Compute per-run mean/variance for selected RACO metrics from train logs.

It parses the Python-dict metric snapshots written to `*-train.out` by
`script/run_and_eval.sh`, extracts the requested metrics, and reports
per-run statistics.

Examples:

```bash
python3 script/calc_raco_metric_stats.py --log-dir logs/raco-diag
```

```bash
python3 script/calc_raco_metric_stats.py \
  --input logs/raco-diag/unclip-wq0.8_wv0.2_lr1e-5_c0.4_clipTrue_lnFalse-train.out
```

```bash
python3 script/calc_raco_metric_stats.py \
  --log-dir logs/raco-diag \
  --format csv
```
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "raco/g_quality_norm",
    "raco/g_verbosity_norm",
    "raco/cos_quality_verbosity",
    "raco/norm_ratio_quality_over_verbosity",
    "raco/norm_ratio_verbosity_over_quality",
    "raco/norm_ratio_larger_over_smaller",
]

SAFE_EVAL_NAMES = {
    "nan": math.nan,
    "inf": math.inf,
    "Infinity": math.inf,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        nargs="+",
        help="One or more explicit *-train.out files to summarize.",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        help="Directory to scan for *-train.out files.",
    )
    ap.add_argument(
        "--glob",
        default="*-train.out",
        help="Glob pattern used with --log-dir. Default: *-train.out",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to summarize. Default: the six M=2 diagnostic metrics.",
    )
    ap.add_argument(
        "--sample-variance",
        action="store_true",
        help="Use sample variance (divide by n-1). Default is population variance (divide by n).",
    )
    ap.add_argument(
        "--format",
        choices=["json", "csv", "pretty"],
        default="pretty",
        help="Output format. Default: pretty",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the output. If omitted, prints to stdout.",
    )
    return ap.parse_args()


def parse_logged_dict(line: str) -> dict[str, Any]:
    try:
        payload = ast.literal_eval(line)
    except (ValueError, SyntaxError):
        payload = eval(line, {"__builtins__": {}}, SAFE_EVAL_NAMES)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload, got {type(payload).__name__}")
    return payload


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(p) for p in args.input]
    if args.log_dir:
        return sorted(args.log_dir.glob(args.glob))
    return []


def compute_variance(xs: list[float], *, sample: bool) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    if sample:
        if n == 1:
            return float("nan")
        denom = n - 1
    else:
        denom = n
    mu = sum(xs) / n
    return sum((x - mu) ** 2 for x in xs) / denom


def summarize_file(path: Path, metrics: list[str], *, sample_variance: bool) -> dict[str, Any]:
    values = {metric: [] for metric in metrics}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue

            try:
                payload = parse_logged_dict(line)
            except Exception:
                continue

            if "loss" not in payload or "eval_loss" in payload:
                continue

            if not all(metric in payload for metric in metrics):
                continue

            for metric in metrics:
                values[metric].append(float(payload[metric]))

    num_points = len(next(iter(values.values()))) if values else 0
    summary: dict[str, Any] = {
        "path": str(path),
        "num_points": num_points,
        "metrics": {},
    }

    for metric, arr in values.items():
        if not arr:
            summary["metrics"][metric] = {
                "mean": float("nan"),
                "variance": float("nan"),
            }
            continue

        mu = sum(arr) / len(arr)
        summary["metrics"][metric] = {
            "mean": mu,
            "variance": compute_variance(arr, sample=sample_variance),
        }

    return summary


def render_pretty(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    parts: list[str] = []
    for row in rows:
        parts.append(f"{row['path']} (n={row['num_points']})")
        for metric in metrics:
            stats = row["metrics"][metric]
            parts.append(
                f"  {metric}: mean={stats['mean']:.6f}, variance={stats['variance']:.6f}"
            )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_csv(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    out_lines: list[str] = []
    header = ["path", "num_points", "metric", "mean", "variance"]
    out_lines.append(",".join(header))
    for row in rows:
        path = row["path"]
        n = row["num_points"]
        for metric in metrics:
            stats = row["metrics"][metric]
            record = [
                path,
                str(n),
                metric,
                str(stats["mean"]),
                str(stats["variance"]),
            ]
            out_lines.append(",".join(record))
    return "\n".join(out_lines) + "\n"


def main() -> int:
    args = parse_args()
    inputs = discover_inputs(args)
    if not inputs:
        print("error: no input files found", file=sys.stderr)
        return 1

    missing = [path for path in inputs if not path.exists()]
    if missing:
        for path in missing:
            print(f"error: missing file: {path}", file=sys.stderr)
        return 1

    rows = [
        summarize_file(path, args.metrics, sample_variance=bool(args.sample_variance))
        for path in inputs
    ]

    if args.format == "json":
        rendered = json.dumps(rows, indent=2, sort_keys=False) + "\n"
    elif args.format == "csv":
        rendered = render_csv(rows, args.metrics)
    else:
        rendered = render_pretty(rows, args.metrics)

    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
