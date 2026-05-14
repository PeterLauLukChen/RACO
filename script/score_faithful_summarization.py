#!/usr/bin/env python3
"""Score summary faithfulness with a local faithful-detector model (BART).

Expected JSONL schema:
  {"input": <prompt+passage>, "output": <generated summary>}

This model requires both the original passage and the generated summary.
We strip the system prompt from `input` and keep only the passage.

We write:
  - faithful_reward: P(FAITHFUL)
  - faithful_logit: logit for the FAITHFUL class (index 1)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Row:
    input: str
    output: str
    raw: Dict[str, Any]


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_DEFAULT_SYSTEM_MARKER = "\n\nSubreddit:"
_DEFAULT_TRAILING_INSTR = "Please directly output your summarization."


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    # Avoid the common transformers warning by explicitly aligning num_labels to 2 for this model.
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True, num_labels=2)

    if tokenizer.pad_token is None:
        # BART usually has a pad token, but keep this safe for local variants.
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    scores: List[float] = []
    n = 0
    sum_score = 0.0

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        batch_rows: List[Row] = []
        batch_summaries: List[str] = []
        batch_passages: List[str] = []

        for obj in tqdm(iter_jsonl(args.jsonl), desc="ScoringFaithful", unit="row"):
            if "output" not in obj:
                raise KeyError(f"Missing 'output' in row keys={list(obj.keys())}")
            row = Row(input=obj.get("input", ""), output=obj["output"], raw=obj)
            passage = strip_system_prompt(row.input)
            batch_rows.append(row)
            batch_summaries.append(row.output)
            batch_passages.append(passage)

            if len(batch_rows) >= args.batch_size:
                logits, probs = score_pairs(
                    batch_summaries, batch_passages, tokenizer, model, device, args.max_length, args.fp16
                )
                for r, lg, pr in zip(batch_rows, logits, probs):
                    r.raw["faithful_logit"] = float(lg)
                    r.raw["faithful_reward"] = float(pr)
                    out_f.write(json.dumps(r.raw, ensure_ascii=False) + "\n")
                    scores.append(float(pr))
                    n += 1
                    sum_score += float(pr)
                batch_rows.clear(); batch_summaries.clear(); batch_passages.clear()

        if batch_rows:
            logits, probs = score_pairs(
                batch_summaries, batch_passages, tokenizer, model, device, args.max_length, args.fp16
            )
            for r, lg, pr in zip(batch_rows, logits, probs):
                r.raw["faithful_logit"] = float(lg)
                r.raw["faithful_reward"] = float(pr)
                out_f.write(json.dumps(r.raw, ensure_ascii=False) + "\n")
                scores.append(float(pr))
                n += 1
                sum_score += float(pr)

    scores_sorted = sorted(scores)
    mean = sum_score / max(n, 1)
    median = scores_sorted[n // 2] if n else float("nan")
    p10 = scores_sorted[int(math.floor(0.10 * (n - 1)))] if n else float("nan")
    p90 = scores_sorted[int(math.floor(0.90 * (n - 1)))] if n else float("nan")

    print("rows", n)
    print("device", device)
    print("mean_faithful_reward", mean)
    print("median_faithful_reward", median)
    print("p10_faithful_reward", p10)
    print("p90_faithful_reward", p90)
    print("scored_jsonl", args.out_jsonl)


if __name__ == "__main__":
    main()



