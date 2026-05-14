#!/usr/bin/env python3
"""
Build prompts that are *exactly* what TRL DPO preprocessing feeds into the model.

TRL (for conversational preference data) effectively:
  1) extracts the shared prompt messages (usually the user message)
  2) applies tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=False)

This script produces a parquet with:
  - trl_prompt: string  (the rendered chat template prompt, ending with assistant generation marker)
  - raw_user_prompt: string (the user content, for debugging)

You can then use the `trl_prompt` column with a completion-style generator, or
adapt `eval/vllm_generate.py` if you want to serve these rendered prompts
instead of using its default chat-completions flow.

Also supports building from an existing prompts parquet that has a string column (default: `prompt`).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple


def _extract_user_content_from_moa_row(row: Dict[str, Any]) -> str:
    a = row.get("Summary_A")
    b = row.get("Summary_B")
    if not isinstance(a, list) or not a:
        raise ValueError("Row missing Summary_A list")
    if not isinstance(b, list) or not b:
        raise ValueError("Row missing Summary_B list")
    ma0 = a[0]
    mb0 = b[0]
    if not (isinstance(ma0, dict) and isinstance(mb0, dict)):
        raise ValueError("Summary_A[0]/Summary_B[0] must be dict messages")
    if ma0.get("role") != "user" or mb0.get("role") != "user":
        raise ValueError("Expected first message role=='user' in both Summary_A and Summary_B")
    ua = ma0.get("content")
    if not isinstance(ua, str):
        raise ValueError("User content must be string")
    return ua


def _render_trl_prompt(tokenizer, user_content: str) -> str:
    # TRL: apply_chat_template on prompt messages; since last role is user => add_generation_prompt=True
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True, help="Model/tokenizer path used in TRL training.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_pairs_parquet", default=None, help="Pairs parquet (Summary_A/Summary_B).")
    src.add_argument(
        "--input_prompts_parquet",
        default=None,
        help="Prompts parquet with a string column (e.g. model/dataset/prompts/test1-*.parquet).",
    )
    ap.add_argument(
        "--prompts_column",
        default="prompt",
        help="Column to read from --input_prompts_parquet (default: prompt).",
    )
    ap.add_argument("--output_parquet", required=True, help="Output parquet path (trl_prompt, raw_user_prompt).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    raw_user: List[str] = []
    trl_prompts: List[str] = []
    bad = 0

    if args.input_pairs_parquet:
        table = pq.read_table(args.input_pairs_parquet)
        rows = table.to_pylist()
        if args.limit is not None:
            rows = rows[: int(args.limit)]
        for r in rows:
            try:
                u = _extract_user_content_from_moa_row(r)
                raw_user.append(u)
                trl_prompts.append(_render_trl_prompt(tokenizer, u))
            except Exception:
                bad += 1
                continue
    else:
        table = pq.read_table(args.input_prompts_parquet, columns=[args.prompts_column])
        col = table[args.prompts_column].to_pylist()
        if args.limit is not None:
            col = col[: int(args.limit)]
        for p in col:
            try:
                if not isinstance(p, str):
                    p = str(p)
                raw_user.append(p)
                trl_prompts.append(_render_trl_prompt(tokenizer, p))
            except Exception:
                bad += 1
                continue

    out_table = pa.table(
        {
            "raw_user_prompt": pa.array(raw_user, type=pa.string()),
            "trl_prompt": pa.array(trl_prompts, type=pa.string()),
        }
    )
    pq.write_table(out_table, args.output_parquet)
    print(
        {
            "output": args.output_parquet,
            "written": len(trl_prompts),
            "skipped_bad": bad,
            "source": "pairs" if args.input_pairs_parquet else "prompts",
        }
    )


if __name__ == "__main__":
    main()

