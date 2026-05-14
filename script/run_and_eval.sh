#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Override these with environment variables for your machine.
TRL_DIR="${TRL_DIR:-${ROOT_DIR}/trl}"
TRL_VENV="${TRL_VENV:-${ROOT_DIR}/trl/env/bin/activate}"
EVAL_VENV="${EVAL_VENV:-${ROOT_DIR}/eval/env/bin/activate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT_DIR}/script/multi_gpu.yaml}"

# Model and dataset paths. The defaults match the README examples.
BASE_MODEL="${BASE_MODEL:-}"
TRAIN_DATASET="${TRAIN_DATASET:-${ROOT_DIR}/data/reddit-summary/train.jsonl}"
VAL_DATASET="${VAL_DATASET:-${ROOT_DIR}/data/reddit-summary/val.jsonl}"
TRAIN_DATASET_M3="${TRAIN_DATASET_M3:-${ROOT_DIR}/data/reddit-summary/train-m3.jsonl}"
VAL_DATASET_M3="${VAL_DATASET_M3:-${ROOT_DIR}/data/reddit-summary/val-m3.jsonl}"

# Number of RACO objectives:
# - M=2 keeps the existing quality+verbosity setup on train.jsonl/val.jsonl
# - M=3 switches to quality+verbosity+faithfulness on train-m3.jsonl/val-m3.jsonl
M="${M:-2}"
if [[ "$M" != "2" && "$M" != "3" ]]; then
  echo "M must be 2 or 3, got: $M" >&2
  exit 1
fi

ACTIVE_TRAIN_DATASET="$TRAIN_DATASET"
ACTIVE_VAL_DATASET="$VAL_DATASET"
if [[ "$M" == "3" ]]; then
  ACTIVE_TRAIN_DATASET="$TRAIN_DATASET_M3"
  ACTIVE_VAL_DATASET="$VAL_DATASET_M3"
fi

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing required setting: ${name}. Set it in the environment or edit this script." >&2
    exit 1
  fi
}

require_path() {
  local name="$1"
  local path="$2"
  require_value "$name" "$path"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path for ${name}: ${path}" >&2
    exit 1
  fi
}

# Safety alignment judge paths.
BEAVER_REWARD_MODEL_DIR="${BEAVER_REWARD_MODEL_DIR:-}"
BEAVER_COST_MODEL_DIR="${BEAVER_COST_MODEL_DIR:-}"
BEAVER_EVAL_PROMPTS="${BEAVER_EVAL_PROMPTS:-}"

# Summary alignment judge paths.
SUMMARY_QUALITY_MODEL_DIR="${SUMMARY_QUALITY_MODEL_DIR:-}"
SUMMARY_FAITHFUL_MODEL_DIR="${SUMMARY_FAITHFUL_MODEL_DIR:-}"
EVAL_PROMPTS="${EVAL_PROMPTS:-${ROOT_DIR}/data/reddit-summary/val.prompts.dedup.parquet}"

# Eval mode:
# - summary: uses eval/new_score.py (quality + faithful judges on RedditSummary prompts)
# - beaver:  uses eval/new_score_beaver.py (beaver reward + cost judges on BeaverTails prompts)
EVAL_MODE="${EVAL_MODE:-summary}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/raco-diag}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${ROOT_DIR}/eval/output}"

mkdir -p "$LOG_DIR" "$EVAL_OUTPUT_DIR" "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

require_path "TRL_DIR" "$TRL_DIR"
require_path "TRL_VENV" "$TRL_VENV"
require_path "ACCELERATE_CONFIG" "$ACCELERATE_CONFIG"
require_value "BASE_MODEL" "$BASE_MODEL"
require_path "ACTIVE_TRAIN_DATASET" "$ACTIVE_TRAIN_DATASET"
require_path "ACTIVE_VAL_DATASET" "$ACTIVE_VAL_DATASET"

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  require_path "EVAL_VENV" "$EVAL_VENV"
  if [[ "$EVAL_MODE" == "beaver" ]]; then
    require_path "BEAVER_REWARD_MODEL_DIR" "$BEAVER_REWARD_MODEL_DIR"
    require_path "BEAVER_COST_MODEL_DIR" "$BEAVER_COST_MODEL_DIR"
    require_path "BEAVER_EVAL_PROMPTS" "$BEAVER_EVAL_PROMPTS"
  elif [[ "$EVAL_MODE" == "summary" ]]; then
    require_path "SUMMARY_QUALITY_MODEL_DIR" "$SUMMARY_QUALITY_MODEL_DIR"
    require_path "SUMMARY_FAITHFUL_MODEL_DIR" "$SUMMARY_FAITHFUL_MODEL_DIR"
    require_path "EVAL_PROMPTS" "$EVAL_PROMPTS"
  else
    echo "EVAL_MODE must be 'summary' or 'beaver', got: $EVAL_MODE" >&2
    exit 1
  fi
fi

# Training defaults (edit if needed)
MAX_LENGTH=2048
TRAIN_BS=2
GRAD_ACCUM=4
WARMUP_RATIO="0.1"
SCHEDULER="cosine"

# RACO defaults (edit if needed)
# NOTE: this is now a sweep variable (see DEFAULT_CS / CS_OVERRIDE below).
RACO_C="0.4"
RACO_USE_CAGRAD="True" # Whether activate CAGrad, set it as false would be DPO LW
RACO_CLIP_LAMBDA="False" # Whether activate CAGrad-Clip
LENGTH_NORMALIZED="False"

# Default sweep weights + LRs + Cs + clip lambdas
DEFAULT_WEIGHTS_M2=("0.8,0.2")
#DEFAULT_WEIGHTS_M2=("0.35,0.65" "0.65,0.35")
DEFAULT_WEIGHTS_M3=("0.3333333333,0.3333333333,0.3333333334")
DEFAULT_WEIGHTS=("${DEFAULT_WEIGHTS_M2[@]}")
if [[ "$M" == "3" ]]; then
  DEFAULT_WEIGHTS=("${DEFAULT_WEIGHTS_M3[@]}")
fi
DEFAULT_LRS=("1e-5")
# Default Cs (override via env: CS_OVERRIDE="0.2 0.4 0.6")
DEFAULT_CS=("0.4")
# Default clip lambdas (override via env: CLIP_LAMBDAS_OVERRIDE="True False")
DEFAULT_CLIP_LAMBDAS=("$RACO_CLIP_LAMBDA")

to_tag() {
  # Make a string safe for filenames (keep it readable).
  echo "$1" | tr '/ :,' '____'
}

parse_weights() {
  # Input:
  # - M=2: "wq,wv" or "wq"
  # - M=3: "wq,wv,wf"
  # Output: prints the objective weights space-separated
  local spec="$1"
  if [[ "$M" == "3" ]]; then
    python3 - "$spec" <<'PY'
import sys

spec = sys.argv[1]
parts = [p.strip() for p in spec.split(",") if p.strip()]
if len(parts) != 3:
    raise SystemExit("For M=3, each weight spec must be three comma-separated floats, e.g. 0.5,0.3,0.2")

vals = [float(p) for p in parts]
print(*[f"{v:.10g}" for v in vals])
PY
  else
    if [[ "$spec" == *","* ]]; then
      local wq="${spec%%,*}"
      local wv="${spec#*,}"
      echo "$wq" "$wv"
    else
      local wq="$spec"
      # wv := 1 - wq (bash arithmetic doesn't do floats; use python3)
      local wv
      wv="$(python3 - <<PY
wq=float("$wq")
print(f"{1.0-wq:.10g}")
PY
)"
      echo "$wq" "$wv"
    fi
  fi
}

run_one() {
  local spec="$1"
  local lr="$2"
  local raco_c="$3"
  local clip_lambda="$4"
  local port="$5"

  local weights=()
  read -r -a weights < <(parse_weights "$spec")

  local weights_csv
  weights_csv="$(IFS=,; echo "${weights[*]}")"

  local tag
  local run_name
  if [[ "$M" == "3" ]]; then
    local wq="${weights[0]}"
    local wv="${weights[1]}"
    local wf="${weights[2]}"
    tag="$(to_tag "m3_wq${wq}_wv${wv}_wf${wf}_lr${lr}_c${raco_c}_clip${clip_lambda}_ln${LENGTH_NORMALIZED}")"
    run_name="raco/m=3,wq=${wq},wv=${wv},wf=${wf},lr=${lr},ln=${LENGTH_NORMALIZED}"
  else
    local wq="${weights[0]}"
    local wv="${weights[1]}"
    tag="$(to_tag "wq${wq}_wv${wv}_lr${lr}_c${raco_c}_clip${clip_lambda}_ln${LENGTH_NORMALIZED}")"
    run_name="raco/wq=${wq},wv=${wv},lr=${lr},ln=${LENGTH_NORMALIZED}"
  fi

  local output_dir="${OUTPUT_ROOT}/raco-${tag}"
  local train_out="${LOG_DIR}/unclip-${tag}-train.out"
  local train_err="${LOG_DIR}/unclip-${tag}-train.err"
  local eval_out="${LOG_DIR}/unclip-${tag}-eval.out"
  local eval_err="${LOG_DIR}/unclip-${tag}-eval.err"

  echo "==> Starting run ${tag}"
  echo "    train stdout: ${train_out}"
  echo "    train stderr: ${train_err}"
  echo "    eval stdout:  ${eval_out}"
  echo "    eval stderr:  ${eval_err}"

  # Train (blocking). Run in a subshell so venv activation can't leak into later runs.
  (
    cd "$TRL_DIR"
    source "$TRL_VENV"

    # NOTE: NOHUP_TRAIN=1 helps runs survive an SSH disconnect by making the *training*
    # process ignore SIGHUP. Still recommended: run the whole sweep under tmux/screen.
    train_cmd=(env PYTHONPATH=. accelerate launch
      --main_process_port 0
      --config_file "$ACCELERATE_CONFIG"
      --num_processes "$NUM_PROCESSES"
      scripts/train_raco.py
      --mode raco
      --model_name_or_path "$BASE_MODEL"
      --dataset_path "$ACTIVE_TRAIN_DATASET"
      --val_dataset_path "$ACTIVE_VAL_DATASET"
      --output_dir "$output_dir"
      --max_length "$MAX_LENGTH"
      --per_device_train_batch_size "$TRAIN_BS"
      --gradient_accumulation_steps "$GRAD_ACCUM"
      --learning_rate "$lr"
      --logging_steps 1
      --bf16 True
      --raco True
      --raco_num_objectives "$M"
      --raco_weights "$weights_csv"
      --raco_c "$raco_c"
      --raco_use_cagrad "$RACO_USE_CAGRAD"
      --per_device_eval_batch_size 8
      --eval_strategy steps
      --eval_steps 100
      --report_to none
      --run_name "$run_name"
      --warmup_ratio "$WARMUP_RATIO"
      --length_normalized "$LENGTH_NORMALIZED"
      --raco_clip_lambda "$clip_lambda"
      --lr_scheduler_type "$SCHEDULER"
    )

    if [[ "${NOHUP_TRAIN:-0}" == "1" ]]; then
      nohup "${train_cmd[@]}" >"$train_out" 2>"$train_err" < /dev/null
    else
      if ! "${train_cmd[@]}" >"$train_out" 2>"$train_err"; then
        echo "Training failed for ${tag}. Tail of ${train_err}:"
        tail -n 80 "$train_err" || true
        return 1
      fi
    fi
  )

  if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
    echo "SKIP_EVAL=1 set; skipping eval for ${output_dir}"
    return
  fi

  # Eval (blocking).
  (
    source "$EVAL_VENV"
    if [[ "$EVAL_MODE" == "beaver" ]]; then
      # Beaver workflow: generate on BeaverTails prompts, then score reward+cost.
      #
      # NOTE: vllm_generate.py expects `--model` to be the *served model name*.
      # In this repo we use the model path as the served model name for simplicity.
      # Stop tokens:
      # - Qwen3:  "<|im_end|>"
      # - Gemma3: "<end_of_turn>"
      # - Llama3: "<|eot_id|>"
      python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../eval/new_score_beaver.py" \
        --reward_model_dir "$BEAVER_REWARD_MODEL_DIR" \
        --cost_model_dir "$BEAVER_COST_MODEL_DIR" \
        --score_device cuda \
        --score_fp16 \
        --score_batch_size 8 \
        --score_max_length 4096 \
        -- \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --start_server \
        --server_model_path "$output_dir" \
        --served_model_name "$output_dir" \
        --server_bind_host 0.0.0.0 \
        --server_ready_host 127.0.0.1 \
        --host 127.0.0.1 \
        --port "$port" \
        --tp 8 \
        --server_max_model_len 4096 \
        --model "$output_dir" \
        --input_parquet "$BEAVER_EVAL_PROMPTS" \
        --prompt_column prompt \
        --concurrency 64 \
        --max_tokens 2048 \
        --temperature 0.6 \
        --stop "<|im_end|>" \
        >"$eval_out" 2>"$eval_err"
    else
      # Summarization workflow: generate on RedditSummary prompts, then score quality+faithful.
      # Stop tokens:
      # - Qwen3:  "<|im_end|>"
      # - Gemma3: "<end_of_turn>"
      # - Llama3: "<|eot_id|>"
      if ! python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../eval/new_score.py" \
        --quality_model_dir "$SUMMARY_QUALITY_MODEL_DIR" \
        --faithful_model_dir "$SUMMARY_FAITHFUL_MODEL_DIR" \
        --quality_score_mode input_output \
        --score_device cuda \
        --score_fp16 \
        --score_batch_size 128 \
        --score_max_length 1024 \
        -- \
        --output_dir "$EVAL_OUTPUT_DIR" \
        --start_server \
        --server_model_path "$output_dir" \
        --served_model_name "$output_dir" \
        --server_bind_host 0.0.0.0 \
        --server_ready_host 127.0.0.1 \
        --host 127.0.0.1 \
        --port "$port" \
        --tp 8 \
        --server_max_model_len 3072 \
        --model "$output_dir" \
        --input_parquet "$EVAL_PROMPTS" \
        --prompt_column prompt \
        --concurrency 64 \
        --max_tokens 512 \
        --temperature 0.6 \
        --stop "<|im_end|>" \
        >"$eval_out" 2>"$eval_err"; then
        echo "Evaluation failed for ${tag}. Tail of ${eval_err}:"
        tail -n 80 "$eval_err" || true
        return 1
      fi
    fi
  )
}

# Weights from CLI (optional)
WEIGHTS=("$@")
if [ "${#WEIGHTS[@]}" -eq 0 ]; then
  WEIGHTS=("${DEFAULT_WEIGHTS[@]}")
fi

# LRs from env override (optional)
LRS=("${DEFAULT_LRS[@]}")
if [[ -n "${LRS_OVERRIDE:-}" ]]; then
  read -r -a LRS <<<"${LRS_OVERRIDE}"
fi

# Cs from env override (optional)
CS=("${DEFAULT_CS[@]}")
if [[ -n "${CS_OVERRIDE:-}" ]]; then
  read -r -a CS <<<"${CS_OVERRIDE}"
fi

# Clip lambdas from env override (optional)
CLIP_LAMBDAS=("${DEFAULT_CLIP_LAMBDAS[@]}")
if [[ -n "${CLIP_LAMBDAS_OVERRIDE:-}" ]]; then
  read -r -a CLIP_LAMBDAS <<<"${CLIP_LAMBDAS_OVERRIDE}"
fi

# Start ports at 8200 to avoid collisions with other scripts using 8000+.
port=8200
for raco_c in "${CS[@]}"; do
  for lr in "${LRS[@]}"; do
    for clip_lambda in "${CLIP_LAMBDAS[@]}"; do
      for spec in "${WEIGHTS[@]}"; do
        run_one "$spec" "$lr" "$raco_c" "$clip_lambda" "$port"
        port=$((port + 1))
      done
    done
  done
done
