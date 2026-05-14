#!/usr/bin/env python3
"""Generate vLLM summaries then immediately score them with the local reward model.

This is a thin orchestration layer around:
  - eval/vllm_generate.py (generation)
  - script/score_reward_summarization.py (reward scoring)

Typical usage:
  python3 eval/vllm_generate_and_score.py \
    --reward_model_dir /path/to/reward_model \
    -- \
    --start_server --server_model_path ... --served_model_name ... \
    --host 127.0.0.1 --port 5000 --model ... \
    --input_parquet ... --prompt_column prompt --concurrency 32

Everything after `--` is passed through to eval/vllm_generate.py.

You can also skip generation and only score an existing generated jsonl:
  python3 eval/vllm_generate_and_score.py \
    --skip_generate --generated_jsonl /path/to/generated.jsonl \
    --reward_model_dir /path/to/reward_model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScoreConfig:
    quality_model_dir: str
    faithful_model_dir: str
    quality_mode: str
    batch_size: int
    max_length: int
    device: str
    fp16: bool


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _repo_root() -> Path:
    # This file lives in <repo>/eval/.
    return Path(__file__).resolve().parents[1]


def _default_paths() -> Tuple[Path, Path, Path]:
    root = _repo_root()
    return (
        root / "eval" / "vllm_generate.py",
        root / "script" / "score_reward_summarization.py",
        root / "script" / "score_faithful_summarization.py",
    )


def _run_generation(cmd: List[str]) -> Tuple[str, str]:
    """Run vllm_generate.py and capture stdout while streaming stderr.

    Note: when --start_server is used, vLLM server logs may be emitted to stderr.
    We stream stderr in real-time so the user can see progress, while capturing
    stdout for parsing the output path.
    """
    # Stream stderr to console in real-time; capture stdout for parsing
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=None)
    return p.stdout, ""


def _extract_generated_jsonl(stdout: str, stderr: str) -> str:
    """Extract the generated JSONL path printed by vllm_generate.py.

    vllm_generate.py prints the output path on stdout, but when it starts a server,
    vLLM logs can interleave. We therefore:
      - strip ANSI
      - search for tokens ending with .jsonl
      - prefer ones that exist on disk
      - take the last matching path
    """

    def candidates(text: str) -> List[str]:
        text = _strip_ansi(text)
        out: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # If the entire line looks like a path.
            if line.endswith(".jsonl"):
                out.append(line)
                continue
            # Otherwise find any token/path-like substring ending in .jsonl
            for m in re.finditer(r"/[^\s]+\.jsonl", line):
                out.append(m.group(0))
        return out

    c = candidates(stdout) + candidates(stderr)
    if not c:
        dbg = (stdout + "\n" + stderr).splitlines()[-50:]
        raise RuntimeError(
            "Could not find a generated .jsonl path in vllm_generate output. "
            "Last lines were:\n" + "\n".join(dbg)
        )

    # Prefer existing files.
    existing = [p for p in c if Path(p).exists()]
    if existing:
        return existing[-1]

    return c[-1]


def _parse_metrics_from_scorer_stdout(stdout: str) -> Dict[str, Any]:
    """Parse the scorer's stdout which prints `key value` lines."""
    out: Dict[str, Any] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        k, v = parts
        if k == "rows":
            try:
                out[k] = int(v)
                continue
            except Exception:
                pass
        if k.endswith("_reward") or k.startswith("mean_") or k.startswith("avg_"):
            try:
                out[k] = float(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    """Return (mean, std). Std is population std (ddof=0)."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, math.sqrt(var)


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="Generate summaries using vLLM and immediately score them with the local reward model."
    )

    ap.add_argument(
        "--vllm_generate_py",
        default=None,
        help="Path to eval/vllm_generate.py (default: auto-detect from repo).",
    )
    ap.add_argument(
        "--quality_scorer_py",
        default=None,
        help="Path to script/score_reward_summarization.py (default: auto-detect from repo).",
    )
    ap.add_argument(
        "--faithful_scorer_py",
        default=None,
        help="Path to script/score_faithful_summarization.py (default: auto-detect from repo).",
    )

    ap.add_argument("--skip_generate", action="store_true", help="Skip generation and only score --generated_jsonl")
    ap.add_argument(
        "--generated_jsonl",
        default=None,
        help="Path to the generated JSONL (required if --skip_generate).",
    )

    # Back-compat: --reward_model_dir maps to quality judge.
    ap.add_argument("--reward_model_dir", default=None, help="(deprecated) Local directory for the quality reward model")
    ap.add_argument("--quality_model_dir", default=None, help="Local directory for the quality judge model (GPT-2 reward)")
    ap.add_argument("--faithful_model_dir", required=True, help="Local directory for the faithful judge model")
    # Quality scorer formatting mode.
    # Back-compat: --score_mode maps to quality mode.
    ap.add_argument(
        "--score_mode",
        choices=["output_only", "input_output"],
        default=None,
        help="(deprecated) Use --quality_score_mode instead.",
    )
    ap.add_argument(
        "--quality_score_mode",
        choices=["output_only", "input_output"],
        default="output_only",
        help="How to format text for the quality judge (default: output_only).",
    )
    ap.add_argument("--score_batch_size", type=int, default=64)
    ap.add_argument("--score_max_length", type=int, default=1024)
    ap.add_argument("--score_device", default="cuda")
    ap.add_argument("--score_fp16", action="store_true")

    ap.add_argument(
        "--scored_jsonl",
        default=None,
        help="Where to write scored JSONL. Default: <generated>.scored.jsonl",
    )
    ap.add_argument(
        "--metrics_out",
        default=None,
        help="Optional path to write a JSON metrics report. Default: <scored>.metrics.json",
    )

    # Everything after `--` (or any unknown args) is passed to vllm_generate.py.
    args, gen_args = ap.parse_known_args(argv)

    vllm_generate_py, quality_scorer_py, faithful_scorer_py = _default_paths()
    if args.vllm_generate_py is not None:
        vllm_generate_py = Path(args.vllm_generate_py)
    if args.quality_scorer_py is not None:
        quality_scorer_py = Path(args.quality_scorer_py)
    if args.faithful_scorer_py is not None:
        faithful_scorer_py = Path(args.faithful_scorer_py)

    quality_model_dir = args.quality_model_dir or args.reward_model_dir
    if not quality_model_dir:
        raise SystemExit("--quality_model_dir is required (or provide deprecated --reward_model_dir)")

    if args.skip_generate:
        if not args.generated_jsonl:
            raise SystemExit("--generated_jsonl is required when --skip_generate is set")
        generated_jsonl = args.generated_jsonl
    else:
        if gen_args and gen_args[0] == "--":
            gen_args = gen_args[1:]

        cmd = [sys.executable, str(vllm_generate_py), *gen_args]
        print("[vllm_generate_and_score] running generation:")
        print(shlex.join(cmd))
        gen_stdout, gen_stderr = _run_generation(cmd)
        generated_jsonl = _extract_generated_jsonl(gen_stdout, gen_stderr)

    # Derive scored output paths
    if args.scored_jsonl is not None:
        scored_jsonl = args.scored_jsonl
    else:
        scored_jsonl = generated_jsonl[:-6] + ".scored.jsonl" if generated_jsonl.endswith(".jsonl") else generated_jsonl + ".scored.jsonl"

    metrics_out = args.metrics_out
    if metrics_out is None:
        metrics_out = scored_jsonl[:-6] + ".metrics.json" if scored_jsonl.endswith(".jsonl") else scored_jsonl + ".metrics.json"

    quality_mode = args.quality_score_mode

    if args.score_mode is not None:
        quality_mode = args.score_mode

    # Safety: the GPT-2 summarization reward model is designed to score a (summary, passage) pair.
    # If the caller accidentally requests output_only, override to input_output to avoid invalid scoring.
    # (This keeps old scripts working even if they pass --quality_score_mode output_only.)
    qdir = str(quality_model_dir).lower()
    if quality_mode == "output_only" and ("gpt2_reward_summarization" in qdir or "reward_summarization" in qdir):
        print(
            "[new_score] Warning: quality_score_mode=output_only would score summaries without the source passage. "
            "Overriding to quality_score_mode=input_output for the GPT-2 summarization reward model."
        )
        quality_mode = "input_output"

    score_cfg = ScoreConfig(
        quality_model_dir=quality_model_dir,
        faithful_model_dir=args.faithful_model_dir,
        quality_mode=quality_mode,
        batch_size=int(args.score_batch_size),
        max_length=int(args.score_max_length),
        device=str(args.score_device),
        fp16=bool(args.score_fp16),
    )

    # Score with both judges into temp files, then merge.
    quality_scored_jsonl = scored_jsonl[:-6] + ".quality.jsonl" if scored_jsonl.endswith(".jsonl") else scored_jsonl + ".quality.jsonl"
    faithful_scored_jsonl = scored_jsonl[:-6] + ".faithful.jsonl" if scored_jsonl.endswith(".jsonl") else scored_jsonl + ".faithful.jsonl"

    quality_cmd = [
        sys.executable,
        str(quality_scorer_py),
        "--jsonl",
        generated_jsonl,
        "--model_dir",
        score_cfg.quality_model_dir,
        "--out_jsonl",
        quality_scored_jsonl,
        "--mode",
        score_cfg.quality_mode,
        "--batch_size",
        str(score_cfg.batch_size),
        "--max_length",
        str(score_cfg.max_length),
        "--device",
        score_cfg.device,
    ]
    if score_cfg.fp16:
        quality_cmd.append("--fp16")

    faithful_cmd = [
        sys.executable,
        str(faithful_scorer_py),
        "--jsonl",
        generated_jsonl,
        "--model_dir",
        score_cfg.faithful_model_dir,
        "--out_jsonl",
        faithful_scored_jsonl,
        "--batch_size",
        str(score_cfg.batch_size),
        "--max_length",
        str(score_cfg.max_length),
        "--device",
        score_cfg.device,
    ]
    if score_cfg.fp16:
        faithful_cmd.append("--fp16")

    print("[new_score] running quality scoring:")
    print(" ".join(quality_cmd))
    p_q = subprocess.run(quality_cmd, check=True, text=True, stdout=subprocess.PIPE)
    quality_stdout = p_q.stdout
    sys.stdout.write(quality_stdout)

    print("[new_score] running faithful scoring:")
    print(" ".join(faithful_cmd))
    p_f = subprocess.run(faithful_cmd, check=True, text=True, stdout=subprocess.PIPE)
    faithful_stdout = p_f.stdout
    sys.stdout.write(faithful_stdout)

    # Merge per-row outputs into `scored_jsonl`.
    Path(scored_jsonl).parent.mkdir(parents=True, exist_ok=True)
    q_f = open(quality_scored_jsonl, "r", encoding="utf-8")
    f_f = open(faithful_scored_jsonl, "r", encoding="utf-8")
    with q_f, f_f, open(scored_jsonl, "w", encoding="utf-8") as out_f:
        rows = 0
        quality_vals: List[float] = []
        faithful_vals: List[float] = []
        for q_line, f_line in zip(q_f, f_f):
            q_obj = json.loads(q_line)
            f_obj = json.loads(f_line)
            # Keep base fields from quality output (includes `reward`).
            merged = q_obj
            merged["quality_reward"] = float(q_obj.get("reward"))
            if "faithful_reward" in f_obj:
                merged["faithful_reward"] = float(f_obj["faithful_reward"])
            if "faithful_logit" in f_obj:
                merged["faithful_logit"] = float(f_obj["faithful_logit"])
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            if merged.get("quality_reward") is not None:
                quality_vals.append(float(merged["quality_reward"]))
            if merged.get("faithful_reward") is not None:
                faithful_vals.append(float(merged["faithful_reward"]))
            rows += 1

    quality_mean, quality_std = _mean_std(quality_vals)
    faithful_mean, faithful_std = _mean_std(faithful_vals)

    print("quality_score_mean", quality_mean)
    print("quality_score_std", quality_std)
    print("faithful_score_mean", faithful_mean)
    print("faithful_score_std", faithful_std)

    # Metrics (prefix to avoid collisions).
    metrics = {}
    q_metrics = _parse_metrics_from_scorer_stdout(quality_stdout)
    f_metrics = _parse_metrics_from_scorer_stdout(faithful_stdout)
    metrics.update({f"quality_{k}": v for k, v in q_metrics.items()})
    metrics.update({f"faithful_{k}": v for k, v in f_metrics.items()})
    metrics.update(
        {
            "quality_score_mean": quality_mean,
            "quality_score_std": quality_std,
            "faithful_score_mean": faithful_mean,
            "faithful_score_std": faithful_std,
        }
    )
    metrics.update(
        {
            "generated_jsonl": generated_jsonl,
            "scored_jsonl": scored_jsonl,
            "quality_scored_jsonl": quality_scored_jsonl,
            "faithful_scored_jsonl": faithful_scored_jsonl,
            "quality_model_dir": score_cfg.quality_model_dir,
            "faithful_model_dir": score_cfg.faithful_model_dir,
        }
    )

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("generated_jsonl", generated_jsonl)
    print("scored_jsonl", scored_jsonl)
    print("metrics_out", metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
