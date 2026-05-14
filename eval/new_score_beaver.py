#!/usr/bin/env python3
"""Generate with vLLM then score with local Beaver reward + cost models.

This mirrors `eval/vllm_generate_and_score.py` / `eval/new_score.py`, but uses:
  - eval/vllm_generate.py
  - script/score_beaver_reward_cost.py

Typical usage:
  python3 eval/new_score_beaver.py \
    --reward_model_dir /path/to/beaver-reward-model \
    --cost_model_dir /path/to/beaver-cost-model \
    -- \
    --start_server --server_model_path ... --served_model_name ... \
    --host 127.0.0.1 --port 8000 --model ... \
    --input_parquet ... --prompt_column prompt --concurrency 32

Skip generation and only score an existing generated jsonl:
  python3 eval/new_score_beaver.py \
    --skip_generate --generated_jsonl /path/to/generated.jsonl \
    --reward_model_dir ... --cost_model_dir ...
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
    reward_model_dir: str
    cost_model_dir: str
    batch_size: int
    max_length: int
    device: str
    fp16: bool
    device_map: Optional[str]


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> Tuple[Path, Path]:
    root = _repo_root()
    return root / "eval" / "vllm_generate.py", root / "script" / "score_beaver_reward_cost.py"


def _run_generation(cmd: List[str]) -> Tuple[str, str]:
    p = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=None)
    return p.stdout, ""


def _extract_generated_jsonl(stdout: str, stderr: str) -> str:
    def candidates(text: str) -> List[str]:
        text = _strip_ansi(text)
        out: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.endswith(".jsonl"):
                out.append(line)
                continue
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

    existing = [p for p in c if Path(p).exists()]
    if existing:
        return existing[-1]
    return c[-1]


def _parse_metrics_from_scorer_stdout(stdout: str) -> Dict[str, Any]:
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
        # float-ish keys
        if any(
            k.endswith(suf)
            for suf in (
                "_mean",
                "_std",
                "_median",
                "_p10",
                "_p90",
                "_reward",
                "_cost",
            )
        ) or k.startswith("avg_"):
            try:
                out[k] = float(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, math.sqrt(var)


def _scan_means_from_scored_jsonl(path: str) -> Dict[str, float]:
    rewards: List[float] = []
    costs: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "beaver_reward" in obj:
                rewards.append(float(obj["beaver_reward"]))
            if "beaver_cost" in obj:
                costs.append(float(obj["beaver_cost"]))
    r_mean, r_std = _mean_std(rewards)
    c_mean, c_std = _mean_std(costs)
    return {
        "beaver_reward_mean": r_mean,
        "beaver_reward_std": r_std,
        "beaver_cost_mean": c_mean,
        "beaver_cost_std": c_std,
        "rows_scanned": float(len(rewards)) if rewards else float("nan"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(description="Generate using vLLM and score with Beaver reward+cost.")
    ap.add_argument("--vllm_generate_py", default=None, help="Path to eval/vllm_generate.py (default: auto-detect).")
    ap.add_argument(
        "--scorer_py", default=None, help="Path to script/score_beaver_reward_cost.py (default: auto-detect)."
    )

    ap.add_argument("--skip_generate", action="store_true", help="Skip generation and only score --generated_jsonl")
    ap.add_argument("--generated_jsonl", default=None, help="Path to generated JSONL (required if --skip_generate)")

    ap.add_argument("--reward_model_dir", required=True, help="Local dir: beaver reward model")
    ap.add_argument("--cost_model_dir", required=True, help="Local dir: beaver cost model")
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--score_max_length", type=int, default=1024)
    ap.add_argument("--score_device", default="cuda")
    ap.add_argument("--score_fp16", action="store_true")
    ap.add_argument("--score_device_map", default=None, help="Optional HF device_map (e.g. 'auto')")

    ap.add_argument("--scored_jsonl", default=None, help="Where to write scored JSONL (default: <generated>.scored.jsonl)")
    ap.add_argument("--metrics_out", default=None, help="Where to write metrics JSON (default: <scored>.metrics.json)")

    args, gen_args = ap.parse_known_args(argv)

    vllm_generate_py, scorer_py = _default_paths()
    if args.vllm_generate_py is not None:
        vllm_generate_py = Path(args.vllm_generate_py)
    if args.scorer_py is not None:
        scorer_py = Path(args.scorer_py)

    if args.skip_generate:
        if not args.generated_jsonl:
            raise SystemExit("--generated_jsonl is required when --skip_generate is set")
        generated_jsonl = args.generated_jsonl
    else:
        if gen_args and gen_args[0] == "--":
            gen_args = gen_args[1:]
        cmd = [sys.executable, str(vllm_generate_py), *gen_args]
        print("[new_score_beaver] running generation:")
        print(shlex.join(cmd))
        gen_stdout, gen_stderr = _run_generation(cmd)
        generated_jsonl = _extract_generated_jsonl(gen_stdout, gen_stderr)

    if args.scored_jsonl is not None:
        scored_jsonl = args.scored_jsonl
    else:
        scored_jsonl = (
            generated_jsonl[:-6] + ".scored.jsonl" if generated_jsonl.endswith(".jsonl") else generated_jsonl + ".scored.jsonl"
        )

    metrics_out = args.metrics_out
    if metrics_out is None:
        metrics_out = scored_jsonl[:-6] + ".metrics.json" if scored_jsonl.endswith(".jsonl") else scored_jsonl + ".metrics.json"

    score_cfg = ScoreConfig(
        reward_model_dir=str(args.reward_model_dir),
        cost_model_dir=str(args.cost_model_dir),
        batch_size=int(args.score_batch_size),
        max_length=int(args.score_max_length),
        device=str(args.score_device),
        fp16=bool(args.score_fp16),
        device_map=(str(args.score_device_map) if args.score_device_map else None),
    )

    score_cmd = [
        sys.executable,
        str(scorer_py),
        "--jsonl",
        generated_jsonl,
        "--reward_model_dir",
        score_cfg.reward_model_dir,
        "--cost_model_dir",
        score_cfg.cost_model_dir,
        "--out_jsonl",
        scored_jsonl,
        "--batch_size",
        str(score_cfg.batch_size),
        "--max_length",
        str(score_cfg.max_length),
        "--device",
        score_cfg.device,
    ]
    if score_cfg.fp16:
        score_cmd.append("--fp16")
    if score_cfg.device_map is not None:
        score_cmd += ["--device_map", score_cfg.device_map]

    print("[new_score_beaver] running scoring:")
    print(" ".join(score_cmd))
    p = subprocess.run(score_cmd, check=True, text=True, stdout=subprocess.PIPE)
    scorer_stdout = p.stdout
    sys.stdout.write(scorer_stdout)

    metrics = _parse_metrics_from_scorer_stdout(scorer_stdout)
    # Recompute mean/std from JSONL as a sanity check (and as a stable key set).
    try:
        metrics.update(_scan_means_from_scored_jsonl(scored_jsonl))
    except Exception as e:
        metrics["scan_error"] = str(e)

    metrics.update(
        {
            "generated_jsonl": generated_jsonl,
            "scored_jsonl": scored_jsonl,
            "reward_model_dir": score_cfg.reward_model_dir,
            "cost_model_dir": score_cfg.cost_model_dir,
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


