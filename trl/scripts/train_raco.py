#!/usr/bin/env python3
"""
Minimal training entrypoint for TRL DPOTrainer with RACO/CGrad enabled.

This expects a JSONL dataset produced by:
  scripts/convert_moa_jsonl_to_trl_raco.py

I.e. each row contains conversational `chosen` / `rejected` (list-of-messages) and optional:
  - raco_s_quality, raco_s_verbosity  (floats in {-1,0,+1})
  - raco_s_faithfulness               (float in {-1,0,+1}) for 3-objective runs
"""

from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer


def _to_pref_sign(v: Any) -> float:
    """
    Map various label encodings to a preference sign over (A vs B).
      +1 => A preferred
      -1 => B preferred
       0 => tie/unknown
    """
    if v is None:
        return 0.0

    if isinstance(v, bool):
        # ambiguous; treat True as A, False as B
        return 1.0 if v else -1.0

    # numeric encodings
    if isinstance(v, (int, float)):
        if v > 0:
            return 1.0
        if v < 0:
            return -1.0
        return 0.0

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


def _ensure_trl_pref_schema(ds, *, name: str):
    """
    Ensure dataset has TRL preference columns:
      - chosen: list-of-messages
      - rejected: list-of-messages
      - raco_s_quality / raco_s_verbosity: floats in {-1,0,+1} (optional but recommended for RACO)
      - raco_s_faithfulness: float in {-1,0,+1} (required for 3-objective runs)

    Accepts raw MOA format with Summary_A/Summary_B/Quality/Verbosity.
    """
    cols = set(ds.column_names)
    if {"chosen", "rejected"}.issubset(cols):
        return ds

    if {"Summary_A", "Summary_B"}.issubset(cols):
        print(f"Converting {name} dataset from MOA format (Summary_A/Summary_B) -> TRL preference format (chosen/rejected).")

        def _map(ex):
            return {
                "chosen": ex["Summary_A"],
                "rejected": ex["Summary_B"],
                "raco_s_quality": float(_to_pref_sign(ex.get("Quality"))),
                "raco_s_verbosity": float(_to_pref_sign(ex.get("Verbosity"))),
            }

        return ds.map(_map, remove_columns=ds.column_names)

    raise ValueError(
        f"Unsupported {name} dataset schema. Expected either columns including "
        f"['chosen','rejected'] or ['Summary_A','Summary_B'], got columns: {ds.column_names}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--dataset_path", required=True, help="JSONL path (train split).")
    ap.add_argument(
        "--val_dataset_path",
        default=None,
        help="Optional validation dataset path (JSON/JSONL). If provided, periodic eval can be enabled.",
    )
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_prompt_length", type=int, default=1536)
    ap.add_argument("--max_completion_length", type=int, default=512)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1, help="If >0, overrides num_train_epochs.")
    # NOTE: argparse help strings use %-formatting internally; escape literal '%' as '%%'.
    ap.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup ratio (e.g. 0.1 for 10%%).")
    ap.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps (overrides warmup_ratio if >0).",
    )
    ap.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="LR scheduler type (e.g. linear, cosine).",
    )

    ap.add_argument("--bf16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument(
        "--fp16",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=False,
        help="Use fp16 mixed precision. Ignored if --bf16 is enabled. Requires CUDA.",
    )
    ap.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Comma-separated list of reporters (e.g. 'wandb') or 'none'.",
    )
    ap.add_argument("--run_name", type=str, default=None, help="Training run name (used by W&B/other reporters).")
    ap.add_argument("--wandb_project", type=str, default=None, help="Optional W&B project (sets WANDB_PROJECT).")
    ap.add_argument("--wandb_entity", type=str, default=None, help="Optional W&B entity/team (sets WANDB_ENTITY).")
    ap.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy when --val_dataset_path is provided. Defaults to 'steps'.",
    )
    ap.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N update steps (when eval_strategy=steps).")

    # Baseline mode
    ap.add_argument(
        "--mode",
        type=str,
        default="raco",
        choices=["raco", "amopo"],
        help="Training baseline to run: 'raco' (current RACO+CAGrad) or 'amopo' (static AMoPO Eq.9 baseline).",
    )

    # RACO flags
    ap.add_argument("--raco", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument(
        "--raco_num_objectives",
        type=int,
        default=None,
        choices=[2, 3],
        help="Expected number of RACO objectives. If omitted, inferred from --raco_weights.",
    )
    ap.add_argument("--raco_weights", type=str, default="0.8,0.2")
    ap.add_argument("--raco_c", type=float, default=0.4)
    ap.add_argument("--raco_use_cagrad", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--raco_clip_lambda", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    ap.add_argument(
        "--length_normalized",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=False,
        help="If True, use length-normalized (avg logp) delta for RACO. If False, keep the original unnormalized RACO.",
    )
    ap.add_argument(
        "--raco_verbosity_prefer_shorter",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=True,
    )

    args = ap.parse_args()

    # Precision selection: be resilient to misconfigured nodes (no CUDA visible) and
    # mismatched precision requests. Transformers validates bf16/fp16 at args init time,
    # so we must sanitize here before constructing DPOConfig/TrainingArguments.
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print(f"Warning: torch.cuda.is_available() failed ({type(e).__name__}: {e}); treating CUDA as unavailable.")
        cuda_available = False

    bf16_supported = False
    if cuda_available:
        try:
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception:
            bf16_supported = False

    if args.bf16 and not bf16_supported:
        if cuda_available:
            print("Warning: --bf16 requested but bf16 is not supported; falling back to fp16.")
            args.bf16 = False
            if not args.fp16:
                args.fp16 = True
        else:
            print("Warning: --bf16 requested but CUDA is not available; falling back to fp32.")
            args.bf16 = False

    if args.fp16 and not cuda_available:
        print("Warning: --fp16 requested but CUDA is not available; falling back to fp32.")
        args.fp16 = False

    w = [float(x) for x in args.raco_weights.split(",")]
    if len(w) not in {2, 3}:
        raise ValueError("--raco_weights must be two or three comma-separated floats, e.g. 0.8,0.2 or 0.5,0.3,0.2")
    num_objectives = int(args.raco_num_objectives) if args.raco_num_objectives is not None else len(w)
    if len(w) != num_objectives:
        raise ValueError(
            f"--raco_num_objectives={num_objectives} but received {len(w)} weights from --raco_weights={args.raco_weights!r}"
        )

    # Mode selects baseline; keep --raco for backward compatibility but let --mode win.
    mode = str(args.mode).strip().lower()
    if mode not in {"raco", "amopo"}:
        raise ValueError(f"Unsupported --mode={args.mode!r}; expected 'raco' or 'amopo'.")
    raco_enabled = (mode == "raco")
    if bool(args.raco) != raco_enabled:
        print(f"Warning: --mode={mode} overrides --raco={args.raco}; using raco={raco_enabled}.")

    # Tokenizer
    # Some tokenizers (notably Mistral-family) require `fix_mistral_regex=True` to avoid a known regex issue.
    # Not all Transformers versions expose this flag, so do a best-effort load.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # Processor (for multimodal models like Gemma3-MM). vLLM expects processor files such as
    # `preprocessor_config.json` to exist in the checkpoint directory.
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    except Exception as e:
        print(f"Warning: could not load AutoProcessor for {args.model_name_or_path!r}: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Some base model tokenizers (e.g. non-instruct variants) don't ship a chat_template.
    # Our dataset is conversational (list-of-messages), and TRL's preprocessing calls
    # tokenizer.apply_chat_template(...). Provide a minimal fallback template for smoke runs.
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )

    # W&B configuration (sbatch-safe: provide WANDB_API_KEY via environment, not here).
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = str(args.wandb_entity)

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    )

    # NOTE: RACO uses torch.autograd.grad(), which is incompatible with re-entrant checkpointing.
    # We set non-reentrant checkpointing via TrainingArguments when possible. (Trainer enables it consistently.)
    # We also *try* to set it on the model directly as a best-effort.
    gc_requested = bool(args.gradient_checkpointing)
    gc_supported_nonreentrant = False
    if gc_requested and hasattr(model, "gradient_checkpointing_enable"):
        for call in (
            lambda: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}),
            lambda: model.gradient_checkpointing_enable(use_reentrant=False),
        ):
            try:
                call()
                gc_supported_nonreentrant = True
                break
            except TypeError:
                continue
            except Exception:
                continue

    ds = load_dataset("json", data_files={"train": args.dataset_path})["train"]
    eval_ds = None
    val_path = args.val_dataset_path
    if not val_path:
        # MODPO-style convenience: if a sibling validation file exists next to train.jsonl, pick it up automatically.
        base_dir = os.path.dirname(os.path.abspath(args.dataset_path))
        for name in ("validation.jsonl", "validation.json", "val.jsonl", "val.json"):
            cand = os.path.join(base_dir, name)
            if os.path.exists(cand):
                val_path = cand
                print(f"Auto-detected validation dataset: {val_path}")
                break
    if val_path:
        eval_ds = load_dataset("json", data_files={"validation": val_path})["validation"]

    # Accept either already-converted TRL preference JSONL or raw MOA JSONL for both train/val.
    ds = _ensure_trl_pref_schema(ds, name="train")
    if eval_ds is not None:
        eval_ds = _ensure_trl_pref_schema(eval_ds, name="eval")
    if num_objectives == 3:
        required_cols = {"raco_s_quality", "raco_s_verbosity", "raco_s_faithfulness"}
        missing_train = sorted(required_cols - set(ds.column_names))
        if missing_train:
            raise ValueError(
                f"3-objective RACO requires train dataset columns {sorted(required_cols)}, missing: {missing_train}"
            )
        if eval_ds is not None:
            missing_eval = sorted(required_cols - set(eval_ds.column_names))
            if missing_eval:
                raise ValueError(
                    f"3-objective RACO requires eval dataset columns {sorted(required_cols)}, missing: {missing_eval}"
                )

    # Transformers renamed some TrainingArguments fields across versions:
    # - evaluation_strategy -> eval_strategy
    # Use the actual signature to stay compatible with the user's environment.
    sig = inspect.signature(DPOConfig.__init__)
    report_to = []
    if str(args.report_to).strip().lower() not in {"", "none", "no", "null"}:
        report_to = [s.strip() for s in str(args.report_to).split(",") if s.strip()]
    kwargs = dict(
        output_dir=args.output_dir,
        # main training knobs
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=float(args.warmup_ratio),
        # sequence lengths
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # precision
        bf16=args.bf16,
        fp16=args.fp16,
        # IMPORTANT: keep raco signs in the batch
        remove_unused_columns=False,
        # reporting
        report_to=report_to,
        # baseline mode
        mode=mode,
        # RACO
        raco=raco_enabled,
        raco_weights=w,
        raco_c=args.raco_c,
        raco_use_cagrad=args.raco_use_cagrad,
        raco_verbosity_prefer_shorter=args.raco_verbosity_prefer_shorter,
    )
    if "raco_length_normalized" in sig.parameters:
        kwargs["raco_length_normalized"] = bool(args.length_normalized)
    if "raco_clip_lambda" in sig.parameters:
        kwargs["raco_clip_lambda"] = bool(args.raco_clip_lambda)
    # Warmup steps and scheduler type are optional across Transformers versions; set only if supported.
    if int(args.warmup_steps) > 0 and "warmup_steps" in sig.parameters:
        kwargs["warmup_steps"] = int(args.warmup_steps)
    if "lr_scheduler_type" in sig.parameters and args.lr_scheduler_type:
        kwargs["lr_scheduler_type"] = str(args.lr_scheduler_type)
    if args.run_name is not None and "run_name" in sig.parameters:
        kwargs["run_name"] = str(args.run_name)
    # Evaluation: enable periodic validation if a val dataset is provided.
    eval_strategy = args.eval_strategy if eval_ds is not None else "no"
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = eval_strategy
    if eval_strategy == "steps" and eval_ds is not None and "eval_steps" in sig.parameters:
        kwargs["eval_steps"] = int(args.eval_steps)

    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "no"

    # Gradient checkpointing: prefer configuring via TrainingArguments so Trainer uses it correctly.
    # If we cannot set non-reentrant mode, disable GC for RACO to avoid runtime error.
    if "gradient_checkpointing" in sig.parameters:
        if gc_requested and args.raco:
            # Enable only if we can force non-reentrant; otherwise disable.
            if "gradient_checkpointing_kwargs" in sig.parameters:
                kwargs["gradient_checkpointing"] = True
                kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
            elif gc_supported_nonreentrant:
                kwargs["gradient_checkpointing"] = True
            else:
                print(
                    "Warning: cannot force non-reentrant gradient checkpointing in this Transformers version; "
                    "disabling gradient checkpointing for RACO."
                )
                kwargs["gradient_checkpointing"] = False
        else:
            kwargs["gradient_checkpointing"] = bool(gc_requested)

    train_args = DPOConfig(**kwargs)

    if is_dataclass(train_args):
        print("DPOConfig:", {k: v for k, v in asdict(train_args).items() if k.startswith("raco")})
    else:
        print("DPOConfig: (non-dataclass)", {k: getattr(train_args, k) for k in dir(train_args) if k.startswith("raco")})

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    # Ensure tokenizer + processor artifacts are saved alongside the model weights.
    # This is required by vLLM for some model families (e.g., Gemma3-MM) even for text-only usage.
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"Warning: failed to save tokenizer to {args.output_dir!r}: {e}")
    if processor is not None:
        try:
            processor.save_pretrained(args.output_dir)
        except Exception as e:
            print(f"Warning: failed to save processor to {args.output_dir!r}: {e}")


if __name__ == "__main__":
    main()
