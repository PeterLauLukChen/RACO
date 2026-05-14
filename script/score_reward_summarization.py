#!/usr/bin/env python3
"""Score summaries with a local GPT-2 reward model.

Expected JSONL schema (your file):
  {"input": <prompt/post>, "output": <model summary>}

We compute a scalar reward per row using GPT2ForSequenceClassification logits.
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

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
_CHAT_SYSTEM_RE = re.compile(r"<\|im_start\|>system[\s\S]*?<\|im_end\|>", flags=re.MULTILINE)


def strip_system_prompt(inp: str) -> str:
    """
    Strip the system prompt / instruction boilerplate from `input`, leaving only the raw passage.

    Supports both:
      - plain-text prompts containing a "Subreddit:" block (our RedditSummary prompts)
      - chat templates containing an explicit system block (<|im_start|>system ... <|im_end|>)
    """
    if not isinstance(inp, str):
        return ""
    s = inp

    # Remove explicit chat system blocks if present.
    if "<|im_start|>system" in s:
        s = _CHAT_SYSTEM_RE.sub("", s)

    # For our Reddit prompts, keep from "Subreddit:" onward (drops the leading system instruction sentence).
    i = s.find(_DEFAULT_SYSTEM_MARKER)
    if i != -1:
        s = s[i + 2 :]  # keep "Subreddit:" line

    # Drop trailing instruction line if present.
    j = s.rfind(_DEFAULT_TRAILING_INSTR)
    if j != -1:
        s = s[:j].rstrip()

    return s.strip()


def build_text(row: Row, mode: str, *, bos_token: str | None) -> str:
    if mode == "output_only":
        return row.output
    if mode == "input_output":
        # Match prior (working) GPT-2 summarization reward formatting:
        #   summary + BOS + passage
        # Also strip the system prompt so the judge sees only the raw post/passage.
        passage = strip_system_prompt(row.input)
        bos = (bos_token or "").strip()
        if bos:
            return f"{row.output} {bos} {passage}".strip()
        return f"{row.output} {passage}".strip()
    raise ValueError(f"Unknown --mode: {mode}")


@torch.inference_mode()
def score_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    fp16: bool,
):
    enc = tokenizer(
        texts,
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
    # Common cases:
    # - regression-style: (B, 1)
    # - classification-style: (B, C)
    if logits.ndim != 2:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        rewards = logits.squeeze(1)
    else:
        # fallback: expected value under softmax over class indices
        probs = torch.softmax(logits, dim=1)
        idx = torch.arange(logits.shape[1], device=logits.device, dtype=probs.dtype)
        rewards = (probs * idx).sum(dim=1)

    return rewards.detach().float().cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--mode", choices=["output_only", "input_output"], default="input_output")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Force offline/local load: no accidental network calls.
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True)

    # GPT-2 needs an explicit pad token for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    # Stream rows, score in batches.
    scores: List[float] = []
    n = 0
    sum_score = 0.0
    sum_resp_chars = 0
    sum_resp_tokens = 0

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        batch_rows: List[Row] = []
        batch_texts: List[str] = []
        batch_outputs: List[str] = []

        for obj in tqdm(iter_jsonl(args.jsonl), desc="Scoring", unit="row"):
            if "output" not in obj:
                raise KeyError(f"Missing 'output' in row keys={list(obj.keys())}")
            row = Row(input=obj.get("input", ""), output=obj["output"], raw=obj)
            txt = build_text(row, args.mode, bos_token=tokenizer.bos_token)

            batch_rows.append(row)
            batch_texts.append(txt)
            batch_outputs.append(row.output)

            if len(batch_texts) >= args.batch_size:
                batch_scores = score_texts(batch_texts, tokenizer, model, device, args.max_length, args.fp16)
                out_enc = tokenizer(batch_outputs, add_special_tokens=False)
                out_lens = [len(x) for x in out_enc["input_ids"]]
                for r, s in zip(batch_rows, batch_scores):
                    r.raw["reward"] = float(s)
                    out_f.write(json.dumps(r.raw, ensure_ascii=False) + "\n")
                    scores.append(float(s))
                    n += 1
                    sum_score += float(s)
                for out_txt, out_len in zip(batch_outputs, out_lens):
                    sum_resp_chars += len(out_txt)
                    sum_resp_tokens += int(out_len)
                batch_rows.clear(); batch_texts.clear(); batch_outputs.clear()

        if batch_texts:
            batch_scores = score_texts(batch_texts, tokenizer, model, device, args.max_length, args.fp16)
            out_enc = tokenizer(batch_outputs, add_special_tokens=False)
            out_lens = [len(x) for x in out_enc["input_ids"]]
            for r, s in zip(batch_rows, batch_scores):
                r.raw["reward"] = float(s)
                out_f.write(json.dumps(r.raw, ensure_ascii=False) + "\n")
                scores.append(float(s))
                n += 1
                sum_score += float(s)

            for out_txt, out_len in zip(batch_outputs, out_lens):
                sum_resp_chars += len(out_txt)
                sum_resp_tokens += int(out_len)

    scores_sorted = sorted(scores)
    mean = sum_score / max(n, 1)
    median = scores_sorted[n // 2] if n else float("nan")
    p10 = scores_sorted[int(math.floor(0.10 * (n - 1)))] if n else float("nan")
    p90 = scores_sorted[int(math.floor(0.90 * (n - 1)))] if n else float("nan")
    avg_resp_chars = (sum_resp_chars / n) if n else float("nan")
    avg_resp_tokens = (sum_resp_tokens / n) if n else float("nan")

    print("rows", n)
    print("mode", args.mode)
    print("device", device)
    print("mean_reward", mean)
    print("avg_response_chars", avg_resp_chars)
    print("avg_response_tokens", avg_resp_tokens)
    print("median_reward", median)
    print("p10_reward", p10)
    print("p90_reward", p90)
    print("scored_jsonl", args.out_jsonl)


if __name__ == "__main__":
    main()
