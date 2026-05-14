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
    mode: str
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


def _default_paths() -> Tuple[Path, Path]:
    root = _repo_root()
    return root / "eval" / "vllm_generate.py", root / "script" / "score_reward_summarization.py"


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
        "--scorer_py",
        default=None,
        help="Path to script/score_reward_summarization.py (default: auto-detect from repo).",
    )

    ap.add_argument("--skip_generate", action="store_true", help="Skip generation and only score --generated_jsonl")
    ap.add_argument(
        "--generated_jsonl",
        default=None,
        help="Path to the generated JSONL (required if --skip_generate).",
    )

    ap.add_argument("--reward_model_dir", required=True, help="Local directory for the reward model")
    ap.add_argument(
        "--score_mode",
        choices=["output_only", "input_output"],
        default="input_output",
        help="How to format text for scoring (default: input_output).",
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

    score_cfg = ScoreConfig(
        reward_model_dir=args.reward_model_dir,
        mode=args.score_mode,
        batch_size=int(args.score_batch_size),
        max_length=int(args.score_max_length),
        device=str(args.score_device),
        fp16=bool(args.score_fp16),
    )

    score_cmd = [
        sys.executable,
        str(scorer_py),
        "--jsonl",
        generated_jsonl,
        "--model_dir",
        score_cfg.reward_model_dir,
        "--out_jsonl",
        scored_jsonl,
        "--mode",
        score_cfg.mode,
        "--batch_size",
        str(score_cfg.batch_size),
        "--max_length",
        str(score_cfg.max_length),
        "--device",
        score_cfg.device,
    ]
    if score_cfg.fp16:
        score_cmd.append("--fp16")

    print("[vllm_generate_and_score] running scoring:")
    print(" ".join(score_cmd))

    p = subprocess.run(score_cmd, check=True, text=True, stdout=subprocess.PIPE)
    scorer_stdout = p.stdout
    sys.stdout.write(scorer_stdout)

    metrics = _parse_metrics_from_scorer_stdout(scorer_stdout)
    metrics.update({"generated_jsonl": generated_jsonl, "scored_jsonl": scored_jsonl})

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("generated_jsonl", generated_jsonl)
    print("scored_jsonl", scored_jsonl)
    print("metrics_out", metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
