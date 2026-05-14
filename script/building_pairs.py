#!/usr/bin/env python3
"""
Build vLLM generation prompts directly from the same `pairs` dataset used for TRL training.

Why:
  - TRL DPO training uses MOA-style `Summary_A`/`Summary_B` list-of-messages.
  - TRL then extracts the *shared prompt* (typically the user message) and uses the tokenizer chat_template.
  - For evaluation, we want prompts that match those training prompts exactly (at the message level).

This script writes a Parquet file with:
  - `prompt`   (string): the user content (same as training user message content)
  - `messages` (list<struct{role,content}>): exact chat messages to feed vLLM Chat Completions

Then you can run:
  python3 eval/vllm_generate.py --input_parquet <out.parquet> --prompt_column prompt ...

The `messages` column is kept for users who adapt generation to pass native chat
messages directly.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple


def _extract_user_prompt_from_moa_row(row: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract the shared user prompt message from a MOA row with Summary_A/Summary_B.
    Returns (prompt_string, messages_list).
    """
    a = row.get("Summary_A")
    b = row.get("Summary_B")
    if not isinstance(a, list) or not a:
        raise ValueError("Row missing Summary_A list")
    if not isinstance(b, list) or not b:
        raise ValueError("Row missing Summary_B list")

    # Most MOA rows are [user, assistant]. We require first message to be user and (ideally) identical across A/B.
    ma0 = a[0]
    mb0 = b[0]
    if not (isinstance(ma0, dict) and isinstance(mb0, dict)):
        raise ValueError("Summary_A[0]/Summary_B[0] must be dict messages")
    if ma0.get("role") != "user" or mb0.get("role") != "user":
        raise ValueError("Expected first message role=='user' in both Summary_A and Summary_B")
    ua = ma0.get("content")
    ub = mb0.get("content")
    if not isinstance(ua, str) or not isinstance(ub, str):
        raise ValueError("User content must be string")
    if ua != ub:
        # Still proceed (some datasets might differ slightly); prefer A's user prompt.
        pass
    prompt = ua
    messages = [{"role": "user", "content": prompt}]
    return prompt, messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_pairs_parquet",
        required=True,
        help="Path to pairs parquet (e.g. model/dataset/pairs/val-00000-of-00001.parquet).",
    )
    ap.add_argument(
        "--output_prompts_parquet",
        required=True,
        help="Path to write prompts parquet (will contain columns: prompt, messages).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    args = ap.parse_args()

    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(args.input_pairs_parquet)
    rows = table.to_pylist()
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    prompts: List[str] = []
    messages: List[List[Dict[str, str]]] = []
    bad = 0
    for r in rows:
        try:
            p, m = _extract_user_prompt_from_moa_row(r)
            prompts.append(p)
            messages.append(m)
        except Exception:
            bad += 1
            continue

    out_table = pa.table(
        {
            "prompt": pa.array(prompts, type=pa.string()),
            # list<struct<role:string, content:string>>
            "messages": pa.array(messages),
        }
    )
    pq.write_table(out_table, args.output_prompts_parquet)
    print(
        {
            "input": args.input_pairs_parquet,
            "output": args.output_prompts_parquet,
            "written": len(prompts),
            "skipped_bad": bad,
        }
    )


if __name__ == "__main__":
    main()

