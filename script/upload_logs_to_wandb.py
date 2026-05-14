#!/usr/bin/env python3
"""Upload RACO train/eval logs to Weights & Biases.

Set your W&B key in the environment before uploading:

```bash
export WANDB_API_KEY="..."
```

This script understands the log files produced by `script/run_and_eval.sh`:
- `*-train.out`: repeated `DPOConfig: {...}` lines plus Python-dict metric snapshots
- `*-eval.out`: stage headers, shell commands, and `key value` metric lines

Examples:

```bash
python3 script/upload_logs_to_wandb.py \
  --input logs/unclip-m3_wq0.1_wv0.1_wf0.8_lr1e-5_c0.4_clipTrue_lnFalse-train.out \
  --project my-wandb-project
```

```bash
python3 script/upload_logs_to_wandb.py \
  --log-dir logs \
  --project my-wandb-project \
  --group raco-sweep
```

Use `--dry-run` to inspect what would be uploaded without importing `wandb`.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRAIN_SUFFIX = "-train.out"
EVAL_SUFFIX = "-eval.out"
KV_RE = re.compile(r"^(?P<key>[A-Za-z0-9_./-]+)\s+(?P<value>.+)$")
SAFE_EVAL_NAMES = {
    "nan": math.nan,
    "inf": math.inf,
    "Infinity": math.inf,
}


@dataclass
class ParsedTrainLog:
    config: Dict[str, Any]
    train_events: List[Tuple[int, Dict[str, Any]]]
    eval_events: List[Tuple[int, Dict[str, Any]]]
    ordered_events: List[Tuple[int, Dict[str, Any]]]
    final_summary: Dict[str, Any]


@dataclass
class ParsedEvalLog:
    metrics: Dict[str, Any]
    commands: List[str]
    stage_headers: List[str]


@dataclass
class RunBundle:
    run_name: str
    train_log: Optional[Path]
    eval_log: Optional[Path]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        help="Single-run mode: path to a -train.out file, -eval.out file, or a run prefix without suffix.",
    )
    ap.add_argument(
        "--log-dir",
        help="Batch mode: import every discovered run under this directory.",
    )
    ap.add_argument(
        "--project",
        help="W&B project name. Required unless --dry-run is set.",
    )
    ap.add_argument("--entity", help="Optional W&B entity/team.")
    ap.add_argument("--group", help="Optional W&B run group.")
    ap.add_argument("--job-type", default="log_import")
    ap.add_argument("--name", help="Optional W&B run name override in single-run mode.")
    ap.add_argument(
        "--metric-set",
        choices=["all", "eval_p"],
        default="all",
        help="Choose which train-log metrics to upload.",
    )
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--attach-logs", action="store_true", help="Upload the raw log files as a W&B artifact.")
    ap.add_argument("--dry-run", action="store_true", help="Parse and print a summary without uploading.")
    ap.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="In batch mode, upload runs even if only train or eval logs are present.",
    )
    args = ap.parse_args()

    if not args.input and not args.log_dir:
        ap.error("one of --input or --log-dir is required")
    if args.input and args.log_dir:
        ap.error("use either --input or --log-dir, not both")
    if not args.dry_run and not args.project:
        ap.error("--project is required unless --dry-run is set")
    if args.name and not args.input:
        ap.error("--name can only be used with --input")
    return args


def maybe_number(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"nan", "+nan", "-nan"}:
        return math.nan
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def sanitize_for_wandb(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_for_wandb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_wandb(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_wandb(v) for v in value]
    return value


def parse_logged_dict(line: str) -> Dict[str, Any]:
    try:
        payload = ast.literal_eval(line)
    except (ValueError, SyntaxError):
        payload = eval(line, {"__builtins__": {}}, SAFE_EVAL_NAMES)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload, got {type(payload).__name__}")
    return payload


def parse_filename_metadata(run_name: str) -> Dict[str, Any]:
    tokens = run_name.split("_")
    if not tokens:
        return {"run_basename": run_name}

    metadata: Dict[str, Any] = {"run_basename": run_name, "variant": tokens[0]}
    field_map = {
        "wq": "weight_quality",
        "wv": "weight_verbosity",
        "wf": "weight_faithfulness",
        "lr": "learning_rate",
        "clip": "clip_lambda",
        "ln": "length_normalized",
        "c": "raco_c",
    }

    for token in tokens[1:]:
        matched = False
        for prefix, key in sorted(field_map.items(), key=lambda item: len(item[0]), reverse=True):
            if token.startswith(prefix):
                metadata[key] = maybe_number(token[len(prefix) :])
                matched = True
                break
        if not matched:
            metadata.setdefault("extra_tokens", []).append(token)

    return metadata


def parse_train_log(path: Path) -> ParsedTrainLog:
    config: Dict[str, Any] = {}
    train_events: List[Tuple[int, Dict[str, Any]]] = []
    eval_events: List[Tuple[int, Dict[str, Any]]] = []
    ordered_events: List[Tuple[int, Dict[str, Any]]] = []
    final_summary: Dict[str, Any] = {}
    step = 0

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("DPOConfig:"):
                if not config:
                    _, payload = line.split(":", 1)
                    config = parse_logged_dict(payload.strip())
                continue

            if not (line.startswith("{") and line.endswith("}")):
                continue

            payload = parse_logged_dict(line)
            payload = sanitize_for_wandb(payload)
            if "loss" in payload and "eval_loss" not in payload:
                step += 1
                train_events.append((step, payload))
                ordered_events.append((step, payload))
            elif "eval_loss" in payload:
                eval_events.append((step, payload))
                ordered_events.append((step, payload))
            elif "train_runtime" in payload:
                final_summary = payload

    return ParsedTrainLog(
        config=config,
        train_events=train_events,
        eval_events=eval_events,
        ordered_events=ordered_events,
        final_summary=final_summary,
    )


def make_default_run_name(filename_metadata: Dict[str, Any], fallback: str) -> str:
    wq = filename_metadata.get("weight_quality")
    wv = filename_metadata.get("weight_verbosity")
    wf = filename_metadata.get("weight_faithfulness")
    if wq is None or wv is None or wf is None:
        return fallback
    return f"{wq}-{wv}-{wf}-RACO-clip"


def filter_logged_events(
    events: List[Tuple[int, Dict[str, Any]]],
    metric_set: str,
) -> List[Tuple[int, Dict[str, Any]]]:
    if metric_set == "all":
        return events

    filtered: List[Tuple[int, Dict[str, Any]]] = []
    for step, metrics in events:
        kept = {key: value for key, value in metrics.items() if key.startswith("eval_p/")}
        if kept:
            filtered.append((step, kept))
    return filtered


def parse_eval_log(path: Path) -> ParsedEvalLog:
    metrics: Dict[str, Any] = {}
    commands: List[str] = []
    stage_headers: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith(":"):
                stage_headers.append(line)
                continue

            if " python " in f" {line} " or line.startswith("/") or line.startswith("python "):
                commands.append(line)
                continue

            match = KV_RE.match(line)
            if match:
                key = match.group("key")
                value = maybe_number(match.group("value"))
                metrics[key] = value

    return ParsedEvalLog(metrics=metrics, commands=commands, stage_headers=stage_headers)


def resolve_bundle_from_member(path: Path) -> RunBundle:
    path = path.resolve()
    name = path.name
    if name.endswith(TRAIN_SUFFIX):
        prefix = path.with_name(name[: -len(TRAIN_SUFFIX)])
    elif name.endswith(EVAL_SUFFIX):
        prefix = path.with_name(name[: -len(EVAL_SUFFIX)])
    else:
        raise ValueError(f"Unsupported input file: {path}")
    return resolve_bundle_from_prefix(prefix)


def resolve_bundle_from_prefix(prefix: Path) -> RunBundle:
    prefix = prefix.resolve()
    train_log = prefix.with_name(prefix.name + TRAIN_SUFFIX)
    eval_log = prefix.with_name(prefix.name + EVAL_SUFFIX)
    return RunBundle(
        run_name=prefix.name,
        train_log=train_log if train_log.exists() else None,
        eval_log=eval_log if eval_log.exists() else None,
    )


def resolve_single_input(raw_input: str) -> RunBundle:
    path = Path(raw_input)
    if path.exists():
        if path.is_dir():
            raise ValueError(f"{path} is a directory; use --log-dir instead")
        return resolve_bundle_from_member(path)
    return resolve_bundle_from_prefix(path)


def discover_bundles(log_dir: Path, allow_missing_pairs: bool) -> List[RunBundle]:
    prefixes: Dict[str, RunBundle] = {}
    for child in sorted(log_dir.iterdir()):
        if child.name.endswith(TRAIN_SUFFIX):
            bundle = resolve_bundle_from_member(child)
        elif child.name.endswith(EVAL_SUFFIX):
            bundle = resolve_bundle_from_member(child)
        else:
            continue
        prefixes[bundle.run_name] = bundle

    bundles = []
    for bundle in prefixes.values():
        if bundle.train_log or bundle.eval_log:
            if allow_missing_pairs or (bundle.train_log and bundle.eval_log):
                bundles.append(bundle)
    return bundles


def build_wandb_payload(bundle: RunBundle, metric_set: str) -> Dict[str, Any]:
    filename_metadata = parse_filename_metadata(bundle.run_name)
    train = parse_train_log(bundle.train_log) if bundle.train_log else None
    post_eval = parse_eval_log(bundle.eval_log) if bundle.eval_log else None
    default_wandb_name = make_default_run_name(filename_metadata, bundle.run_name)

    payload: Dict[str, Any] = {
        "run_name": bundle.run_name,
        "wandb_name": default_wandb_name,
        "filename_metadata": filename_metadata,
        "paths": {
            "train_log": str(bundle.train_log) if bundle.train_log else None,
            "eval_log": str(bundle.eval_log) if bundle.eval_log else None,
        },
    }

    if train:
        filtered_events = filter_logged_events(train.ordered_events, metric_set)
        payload["dpo_config"] = train.config
        payload["train_event_count"] = len(train.train_events)
        payload["in_train_eval_event_count"] = len(train.eval_events)
        payload["logged_event_count"] = len(filtered_events)
        payload["ordered_train_events"] = filtered_events
        payload["final_train_summary"] = train.final_summary
        payload["train_events"] = train.train_events
        payload["eval_events"] = train.eval_events
    else:
        payload["dpo_config"] = {}
        payload["train_event_count"] = 0
        payload["in_train_eval_event_count"] = 0
        payload["logged_event_count"] = 0
        payload["ordered_train_events"] = []
        payload["final_train_summary"] = {}
        payload["train_events"] = []
        payload["eval_events"] = []

    if post_eval:
        payload["post_eval_metrics"] = post_eval.metrics
        payload["post_eval_commands"] = post_eval.commands
        payload["post_eval_stage_headers"] = post_eval.stage_headers
    else:
        payload["post_eval_metrics"] = {}
        payload["post_eval_commands"] = []
        payload["post_eval_stage_headers"] = []

    return sanitize_for_wandb(payload)


def preview_payload(payload: Dict[str, Any]) -> str:
    preview = {
        "run_name": payload["run_name"],
        "wandb_name": payload["wandb_name"],
        "filename_metadata": payload["filename_metadata"],
        "paths": payload["paths"],
        "logged_event_count": payload["logged_event_count"],
        "dpo_config": payload["dpo_config"],
        "train_event_count": payload["train_event_count"],
        "in_train_eval_event_count": payload["in_train_eval_event_count"],
        "final_train_summary": payload["final_train_summary"],
        "post_eval_metrics": payload["post_eval_metrics"],
        "post_eval_commands": payload["post_eval_commands"],
    }
    return json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False)


def upload_payload(
    *,
    wandb_module: Any,
    payload: Dict[str, Any],
    project: str,
    entity: Optional[str],
    group: Optional[str],
    job_type: str,
    tags: List[str],
    name_override: Optional[str],
    attach_logs: bool,
    metric_set: str,
) -> None:
    config = {
        "filename_metadata": payload["filename_metadata"],
        "dpo_config": payload["dpo_config"],
        "source_paths": payload["paths"],
        "post_eval_commands": payload["post_eval_commands"],
        "post_eval_stage_headers": payload["post_eval_stage_headers"],
    }
    run = wandb_module.init(
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        name=name_override or payload["wandb_name"],
        tags=list(tags),
        config=sanitize_for_wandb(config),
        reinit="finish_previous",
    )

    try:
        for step, metrics in payload["ordered_train_events"]:
            wandb_module.log(metrics, step=step)

        final_step = payload["train_event_count"]
        if payload["final_train_summary"] and metric_set == "all":
            wandb_module.log(payload["final_train_summary"], step=final_step)

        if payload["post_eval_metrics"] and metric_set == "all":
            prefixed = {f"post_eval/{k}": v for k, v in payload["post_eval_metrics"].items()}
            wandb_module.log(prefixed, step=final_step)

        summary = {
            "source/train_event_count": payload["train_event_count"],
            "source/in_train_eval_event_count": payload["in_train_eval_event_count"],
            "source/logged_event_count": payload["logged_event_count"],
            "source/metric_set": metric_set,
        }
        if metric_set == "all":
            for key, value in payload["final_train_summary"].items():
                summary[f"train_summary/{key}"] = value
            for key, value in payload["post_eval_metrics"].items():
                summary[f"post_eval/{key}"] = value
        run.summary.update(sanitize_for_wandb(summary))

        if attach_logs:
            artifact = wandb_module.Artifact(f"{payload['run_name']}-logs", type="logs")
            train_log = payload["paths"].get("train_log")
            eval_log = payload["paths"].get("eval_log")
            if train_log:
                artifact.add_file(train_log, name=Path(train_log).name)
            if eval_log:
                artifact.add_file(eval_log, name=Path(eval_log).name)
            run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> int:
    args = parse_args()

    try:
        if args.input:
            bundles = [resolve_single_input(args.input)]
        else:
            bundles = discover_bundles(Path(args.log_dir).resolve(), args.allow_missing_pairs)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not bundles:
        print("error: no matching runs found", file=sys.stderr)
        return 2

    payloads = []
    for bundle in bundles:
        if not bundle.train_log and not bundle.eval_log:
            continue
        payloads.append(build_wandb_payload(bundle, args.metric_set))

    if args.dry_run:
        for payload in payloads:
            print(preview_payload(payload))
        return 0

    try:
        import wandb  # type: ignore
    except ImportError:
        print(
            "error: wandb is not installed in the current environment. Install it and rerun.",
            file=sys.stderr,
        )
        return 2

    for index, payload in enumerate(payloads):
        name_override = args.name if index == 0 and len(payloads) == 1 else None
        upload_payload(
            wandb_module=wandb,
            payload=payload,
            project=args.project,
            entity=args.entity,
            group=args.group,
            job_type=args.job_type,
            tags=args.tags,
            name_override=name_override,
            attach_logs=args.attach_logs,
            metric_set=args.metric_set,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
