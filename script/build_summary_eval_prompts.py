#!/usr/bin/env python3
"""
Build a deduplicated Parquet prompt file for summary evaluation from preference JSONL.

Input rows are expected to look like:
  {
    "chosen":   [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}],
    "rejected": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
  }

Output Parquet contains:
  - prompt: string

The prompt is the shared user message content, which matches the exact Reddit
summarization prompt format used by the training/eval pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _common_prompt_messages(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt: List[Dict[str, Any]] = []
    for ma, mb in zip(a, b):
        if ma != mb:
            break
        prompt.append(ma)
    return prompt


def _last_user_text(messages: List[Dict[str, Any]], key: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    raise ValueError(f"{key} does not contain a user message with string content")


def extract_prompt(obj: Dict[str, Any]) -> str:
    chosen = obj.get("chosen")
    rejected = obj.get("rejected")
    if not isinstance(chosen, list) or not chosen:
        raise ValueError("Missing or invalid `chosen` conversation")
    if not isinstance(rejected, list) or not rejected:
        raise ValueError("Missing or invalid `rejected` conversation")

    prompt_messages = _common_prompt_messages(chosen, rejected)
    if not prompt_messages:
        raise ValueError("Could not find a shared prompt prefix between chosen and rejected")
    return _last_user_text(prompt_messages, "shared prompt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input preference JSONL.")
    ap.add_argument("--out_parquet", required=True, help="Output parquet with a single `prompt` column.")
    args = ap.parse_args()

    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    prompts = OrderedDict()
    total_rows = 0
    skipped_bad = 0

    for obj in iter_jsonl(args.jsonl):
        total_rows += 1
        try:
            prompt = extract_prompt(obj)
            prompts.setdefault(prompt, None)
        except Exception:
            skipped_bad += 1

    table = pa.table({"prompt": pa.array(list(prompts.keys()), type=pa.string())})
    pq.write_table(table, args.out_parquet)

    print(
        {
            "input_jsonl": args.jsonl,
            "output_parquet": args.out_parquet,
            "total_rows": total_rows,
            "unique_prompts": len(prompts),
            "skipped_bad": skipped_bad,
        }
    )


if __name__ == "__main__":
    main()
