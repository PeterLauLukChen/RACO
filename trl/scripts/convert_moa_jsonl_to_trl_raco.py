#!/usr/bin/env python3
"""
Convert MOA-style JSONL into TRL DPO/RACO preference JSONL.

Input example keys (from your dataset):
  - "Summary_A": list[{"role": "...", "content": "..."}]  (conversation)
  - "Summary_B": list[{"role": "...", "content": "..."}]
  - "Quality": label indicating whether A or B is preferred for quality
  - "Verbosity": label indicating whether A or B is preferred for verbosity

Output (one JSON object per line):
  - "chosen": Summary_A conversation (kept as list-of-messages)
  - "rejected": Summary_B conversation
  - "raco_s_quality": +1 if Quality prefers A over B, -1 if prefers B, 0 if tie/unknown
  - "raco_s_verbosity": +1 if Verbosity prefers A over B, -1 if prefers B, 0 if tie/unknown

Notes:
  - TRL will run maybe_extract_prompt() to extract the shared user prompt from chosen/rejected.
  - We keep the conversational structure so TRL can apply the tokenizer chat template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _to_pref_sign(v: Any) -> float:
    """
    Map various label encodings to a preference sign over (A vs B).
      +1 => A preferred
      -1 => B preferred
       0 => tie/unknown
    """
    if v is None:
        return 0.0

    # numeric encodings
    if isinstance(v, (int, float)):
        if v > 0:
            return 1.0
        if v < 0:
            return -1.0
        return 0.0

    if isinstance(v, bool):
        # ambiguous; treat True as A, False as B
        return 1.0 if v else -1.0

    # strings
    s = str(v).strip().lower()
    if s in {"a", "summary_a", "1", "chosen", "left", "first"}:
        return 1.0
    if s in {"b", "summary_b", "2", "rejected", "right", "second"}:
        return -1.0
    if s in {"tie", "equal", "same", "0", "none", "unknown", "n/a"}:
        return 0.0

    # try to parse something like "A>B" or "B > A"
    if "a" in s and "b" in s:
        if "a>b" in s or "a > b" in s:
            return 1.0
        if "b>a" in s or "b > a" in s:
            return -1.0

    return 0.0


def _validate_conversation(x: Any, key: str) -> list[dict[str, str]]:
    if not isinstance(x, list) or not x:
        raise ValueError(f"{key} must be a non-empty list of messages, got: {type(x).__name__}")
    for m in x:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"{key} must be list of {{role, content}} dicts; bad message: {m!r}")
    return x  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to MOA JSONL (with Summary_A/Summary_B/Quality/Verbosity).")
    ap.add_argument("--output", required=True, help="Path to write TRL preference JSONL.")
    ap.add_argument(
        "--swap_if_quality_prefers_b",
        action="store_true",
        help="If set, swap chosen/rejected so that chosen is ALWAYS the quality-preferred summary. "
        "This is optional; if you enable it, you must also flip raco_s_* accordingly.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_bad = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                a = _validate_conversation(obj["Summary_A"], "Summary_A")
                b = _validate_conversation(obj["Summary_B"], "Summary_B")
                s_q = _to_pref_sign(obj.get("Quality"))
                s_v = _to_pref_sign(obj.get("Verbosity"))

                chosen = a
                rejected = b

                if args.swap_if_quality_prefers_b and s_q < 0:
                    chosen, rejected = rejected, chosen
                    s_q = -s_q
                    s_v = -s_v

                out = {
                    "chosen": chosen,
                    "rejected": rejected,
                    "raco_s_quality": float(s_q),
                    "raco_s_verbosity": float(s_v),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1
            except Exception:
                n_bad += 1
                continue

    print(
        json.dumps(
            {"input_lines": n_in, "output_lines": n_out, "skipped_bad_lines": n_bad, "output": str(out_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


