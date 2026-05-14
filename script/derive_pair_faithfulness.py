#!/usr/bin/env python3
"""Derive pairwise faithfulness labels for preference JSONL.

This follows the exact scoring format used by `script/score_faithful_summarization.py`:
  - summary is passed as `text`
  - source passage is passed as `text_pair`
  - the source passage is extracted by stripping the fixed summarization prompt

Input JSONL is expected to contain preference rows such as:
  {
    "chosen":   [{"role": "user", ...}, {"role": "assistant", ...}],
    "rejected": [{"role": "user", ...}, {"role": "assistant", ...}]
  }

Output preserves all original keys and adds:
  - chosen_faithful_logit
  - chosen_faithful_reward
  - rejected_faithful_logit
  - rejected_faithful_reward
  - raco_s_faithfulness   (+1 if chosen is more faithful, -1 if rejected is more faithful, 0 if tie)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm

# Some envs are missing the built-in `_lzma` module, and importing torchvision
# can cascade into an lzma import. We don't need torchvision for text-only BART
# scoring, so hide it from Transformers package discovery for this script.
_orig_find_spec = importlib.util.find_spec


def _find_spec_without_torchvision(name: str, *args, **kwargs):
    if name == "torchvision":
        return None
    return _orig_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_torchvision

from transformers import AutoModelForSequenceClassification, AutoTokenizer


_DEFAULT_SYSTEM_MARKER = "\n\nSubreddit:"
_DEFAULT_TRAILING_INSTR = "Please directly output your summarization."


@dataclass
class PairRow:
    prompt: str
    chosen_summary: str
    rejected_summary: str
    raw: Dict[str, Any]


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def strip_system_prompt(inp: str) -> str:
    """Strip the fixed system prompt and trailing instruction, leaving only the passage."""
    if not isinstance(inp, str):
        return ""
    s = inp
    i = s.find(_DEFAULT_SYSTEM_MARKER)
    if i != -1:
        s = s[i + 2 :]  # keep "Subreddit:" line
    j = s.rfind(_DEFAULT_TRAILING_INSTR)
    if j != -1:
        s = s[:j].rstrip()
    return s.strip()


def _common_prompt_messages(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt: List[Dict[str, Any]] = []
    for ma, mb in zip(a, b):
        if ma != mb:
            break
        prompt.append(ma)
    return prompt


def _last_assistant_text(messages: List[Dict[str, Any]], key: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            return msg["content"]
    raise ValueError(f"{key} does not contain an assistant message with string content")


def _last_user_text(messages: List[Dict[str, Any]], key: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    raise ValueError(f"{key} does not contain a user message with string content")


def _extract_pair_row(obj: Dict[str, Any]) -> PairRow:
    chosen = obj.get("chosen")
    rejected = obj.get("rejected")
    if not isinstance(chosen, list) or not chosen:
        raise ValueError("Missing or invalid `chosen` conversation")
    if not isinstance(rejected, list) or not rejected:
        raise ValueError("Missing or invalid `rejected` conversation")

    prompt_messages = _common_prompt_messages(chosen, rejected)
    if not prompt_messages:
        raise ValueError("Could not find a shared prompt prefix between chosen and rejected")

    return PairRow(
        prompt=_last_user_text(prompt_messages, "shared prompt"),
        chosen_summary=_last_assistant_text(chosen, "chosen"),
        rejected_summary=_last_assistant_text(rejected, "rejected"),
        raw=obj,
    )


@torch.inference_mode()
def score_pairs(
    summaries: List[str],
    passages: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    fp16: bool,
):
    enc = tokenizer(
        text=summaries,
        text_pair=passages,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    if fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**enc)
    else:
        out = model(**enc)

    logits = out.logits
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    faithful_logit = logits[:, 1]
    faithful_prob = torch.softmax(logits, dim=1)[:, 1]
    return faithful_logit.detach().float().cpu().tolist(), faithful_prob.detach().float().cpu().tolist()


def _sign_from_scores(chosen_score: float, rejected_score: float, tie_epsilon: float) -> float:
    diff = float(chosen_score) - float(rejected_score)
    if abs(diff) <= tie_epsilon:
        return 0.0
    return 1.0 if diff > 0.0 else -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input preference JSONL with chosen/rejected conversations.")
    ap.add_argument("--model_dir", required=True, help="Local faithful-detector model directory.")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with faithfulness labels added.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument(
        "--tie_epsilon",
        type=float,
        default=0.0,
        help="Treat score differences with abs(diff) <= tie_epsilon as ties.",
    )
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True, num_labels=2)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    sign_counts = {-1.0: 0, 0.0: 0, 1.0: 0}
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        batch_rows: List[PairRow] = []
        batch_summaries: List[str] = []
        batch_passages: List[str] = []

        def flush() -> None:
            if not batch_rows:
                return

            logits, probs = score_pairs(
                batch_summaries, batch_passages, tokenizer, model, device, args.max_length, args.fp16
            )
            for i, row in enumerate(batch_rows):
                chosen_logit = float(logits[2 * i])
                chosen_reward = float(probs[2 * i])
                rejected_logit = float(logits[2 * i + 1])
                rejected_reward = float(probs[2 * i + 1])
                sign = _sign_from_scores(chosen_reward, rejected_reward, args.tie_epsilon)

                row.raw["chosen_faithful_logit"] = chosen_logit
                row.raw["chosen_faithful_reward"] = chosen_reward
                row.raw["rejected_faithful_logit"] = rejected_logit
                row.raw["rejected_faithful_reward"] = rejected_reward
                row.raw["raco_s_faithfulness"] = sign
                out_f.write(json.dumps(row.raw, ensure_ascii=False) + "\n")

                chosen_scores.append(chosen_reward)
                rejected_scores.append(rejected_reward)
                sign_counts[sign] += 1

            batch_rows.clear()
            batch_summaries.clear()
            batch_passages.clear()

        for obj in tqdm(iter_jsonl(args.jsonl), desc="DerivingFaithfulness", unit="row"):
            row = _extract_pair_row(obj)
            passage = strip_system_prompt(row.prompt)

            batch_rows.append(row)
            batch_summaries.extend([row.chosen_summary, row.rejected_summary])
            batch_passages.extend([passage, passage])

            if len(batch_rows) >= args.batch_size:
                flush()

        flush()

    n = len(chosen_scores)
    deltas = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    mean_chosen = sum(chosen_scores) / max(n, 1)
    mean_rejected = sum(rejected_scores) / max(n, 1)
    mean_delta = sum(deltas) / max(n, 1)
    delta_sorted = sorted(deltas)
    median_delta = delta_sorted[n // 2] if n else float("nan")
    p10_delta = delta_sorted[int(math.floor(0.10 * (n - 1)))] if n else float("nan")
    p90_delta = delta_sorted[int(math.floor(0.90 * (n - 1)))] if n else float("nan")

    print("rows", n)
    print("device", device)
    print("mean_chosen_faithful_reward", mean_chosen)
    print("mean_rejected_faithful_reward", mean_rejected)
    print("mean_delta_faithful_reward", mean_delta)
    print("median_delta_faithful_reward", median_delta)
    print("p10_delta_faithful_reward", p10_delta)
    print("p90_delta_faithful_reward", p90_delta)
    print("count_raco_s_faithfulness_-1", sign_counts[-1.0])
    print("count_raco_s_faithfulness_0", sign_counts[0.0])
    print("count_raco_s_faithfulness_1", sign_counts[1.0])
    print("scored_jsonl", args.out_jsonl)


if __name__ == "__main__":
    main()
