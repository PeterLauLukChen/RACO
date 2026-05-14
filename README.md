# [ICML 2026] Reward-free Alignment for Conflicting Objectives

RACO is a **reward-free alignment fine-tuning method for multi-objective preference optimization**.

This repository contains:

- `trl/`: a TRL-based training stack with RACO/CAGrad support added to `DPOTrainer`.
- `script/`: data-preparation, training, scoring, and log utilities.
- `eval/`: vLLM generation and local-judge evaluation scripts.
- `trl/data/`: small prompt/eval assets used by the included evaluation scripts.

The code supports 2-objective and 3-objective summarization alignment, plus safety-alignment evaluation.

## Setup

This repo uses two Python environments:

- `trl/env/` for training
- `eval/env/` for generation and evaluation

Use Python 3.10 or newer. Python 3.11 is a good default when available.

### Training Environment

```bash
cd trl
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Evaluation Environment

```bash
cd eval
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Data

### 2-Objective Summary Data

For the 2-objective Reddit summarization setup, use:

[RACOo/RedditSummary-Alignment](https://huggingface.co/datasets/RACOo/RedditSummary-Alignment)

The training script accepts either:

- TRL preference format with `chosen`, `rejected`, `raco_s_quality`, `raco_s_verbosity`
- MOA-style format with `Summary_A`, `Summary_B`, `Quality`, `Verbosity`

If you download the dataset through `datasets`, export the splits to local JSONL files before training:

```bash
python - <<'PY'
from pathlib import Path

from datasets import load_dataset

ds = load_dataset("RACOo/RedditSummary-Alignment")
print(ds)

Path("data/reddit-summary").mkdir(parents=True, exist_ok=True)
ds["train"].to_json("data/reddit-summary/train.jsonl", orient="records", lines=True)
ds["validation"].to_json("data/reddit-summary/val.jsonl", orient="records", lines=True)
PY
```

If the dataset exposes different split names, print `ds` first and replace `train` / `validation` accordingly.

### 3-Objective Summary Data

The 3-objective setup adds faithfulness as the third objective:

- quality
- verbosity
- faithfulness

Start from the 2-objective preference JSONL, then use the faithfulness judge from Appendix C.2 of the paper:

- [CogComp/bart-faithful-summary-detector](https://huggingface.co/CogComp/bart-faithful-summary-detector)

The paper's Appendix C.2 also uses the GPT-2 summary quality judge:

- [Tristan/gpt2_reward_summarization](https://huggingface.co/Tristan/gpt2_reward_summarization)

To construct the 3-objective training files, score both summaries in each pair with the BART faithfulness judge:

```bash
source eval/env/bin/activate

python script/derive_pair_faithfulness.py \
  --jsonl data/reddit-summary/train.jsonl \
  --model_dir /path/to/bart-faithful-summary-detector \
  --out_jsonl data/reddit-summary/train-m3.jsonl \
  --batch_size 64 \
  --max_length 1024 \
  --device cuda \
  --fp16

python script/derive_pair_faithfulness.py \
  --jsonl data/reddit-summary/val.jsonl \
  --model_dir /path/to/bart-faithful-summary-detector \
  --out_jsonl data/reddit-summary/val-m3.jsonl \
  --batch_size 64 \
  --max_length 1024 \
  --device cuda \
  --fp16
```

This preserves the original pairwise rows and adds:

- `chosen_faithful_logit`
- `chosen_faithful_reward`
- `rejected_faithful_logit`
- `rejected_faithful_reward`
- `raco_s_faithfulness`

`raco_s_faithfulness` is `+1` when the chosen response is more faithful, `-1` when the rejected response is more faithful, and `0` for ties.

### Build Summary Eval Prompts

For summarization evaluation, build a deduplicated prompt parquet from the validation preference file:

```bash
python script/build_summary_eval_prompts.py \
  --jsonl data/reddit-summary/val.jsonl \
  --out_parquet data/reddit-summary/val.prompts.dedup.parquet
```

Use this parquet as `EVAL_PROMPTS` in `script/run_and_eval.sh`.

## Training

The main training entrypoint is:

```bash
trl/scripts/train_raco.py
```

It is usually launched through Accelerate from inside the `trl/` directory.

### 2-Objective Training

```bash
cd trl
source env/bin/activate

accelerate launch \
  --config_file ../script/multi_gpu.yaml \
  --num_processes 8 \
  scripts/train_raco.py \
  --mode raco \
  --model_name_or_path /path/to/base-model \
  --dataset_path ../data/reddit-summary/train.jsonl \
  --val_dataset_path ../data/reddit-summary/val.jsonl \
  --output_dir /path/to/output/raco-wq0.8-wv0.2 \
  --max_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --bf16 True \
  --raco True \
  --raco_num_objectives 2 \
  --raco_weights 0.8,0.2 \
  --raco_c 0.4 \
  --raco_use_cagrad True \
  --raco_clip_lambda True \
  --report_to none
```

### 3-Objective Training

Use the `train-m3.jsonl` / `val-m3.jsonl` files produced by `derive_pair_faithfulness.py`:

```bash
cd trl
source env/bin/activate

accelerate launch \
  --config_file ../script/multi_gpu.yaml \
  --num_processes 8 \
  scripts/train_raco.py \
  --mode raco \
  --model_name_or_path /path/to/base-model \
  --dataset_path ../data/reddit-summary/train-m3.jsonl \
  --val_dataset_path ../data/reddit-summary/val-m3.jsonl \
  --output_dir /path/to/output/raco-m3 \
  --max_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --bf16 True \
  --raco True \
  --raco_num_objectives 3 \
  --raco_weights 0.3333333333,0.3333333333,0.3333333334 \
  --raco_c 0.4 \
  --raco_use_cagrad True \
  --raco_clip_lambda True \
  --report_to none
```

For DPO-LW-style weighted training, keep the same data and weights but set:

```bash
--raco_use_cagrad False
```

For the AMoPO baseline, use:

```bash
--mode amopo --raco False
```

## End-To-End Sweep Script

`script/run_and_eval.sh` runs training and then evaluation. Before using it, either edit the defaults at the top of the file or set these environment variables:

- `TRL_DIR`
- `TRL_VENV`
- `EVAL_VENV`
- `BASE_MODEL`
- `TRAIN_DATASET`
- `VAL_DATASET`
- `TRAIN_DATASET_M3`
- `VAL_DATASET_M3`
- `OUTPUT_ROOT`
- `LOG_DIR`
- `EVAL_OUTPUT_DIR`
- summary judge paths:
  - `SUMMARY_QUALITY_MODEL_DIR`
  - `SUMMARY_FAITHFUL_MODEL_DIR`
  - `EVAL_PROMPTS`
- safety judge paths:
  - `BEAVER_REWARD_MODEL_DIR`
  - `BEAVER_COST_MODEL_DIR`
  - `BEAVER_EVAL_PROMPTS`

Run a 2-objective summary sweep:

```bash
EVAL_MODE=summary M=2 bash script/run_and_eval.sh "0.8,0.2"
```

Run a 3-objective summary sweep:

```bash
EVAL_MODE=summary M=3 bash script/run_and_eval.sh "0.3333333333,0.3333333333,0.3333333334"
```

Override sweep values:

```bash
LRS_OVERRIDE="1e-5 2e-5" \
CS_OVERRIDE="0.25 0.4" \
CLIP_LAMBDAS_OVERRIDE="True False" \
EVAL_MODE=summary \
M=3 \
bash script/run_and_eval.sh "0.5,0.25,0.25"
```

Skip evaluation and only train:

```bash
SKIP_EVAL=1 M=2 bash script/run_and_eval.sh "0.8,0.2"
```

## Evaluation

### Summary Evaluation

`eval/new_score.py` generates summaries with vLLM and scores them with:

- quality judge: `Tristan/gpt2_reward_summarization`
- faithfulness judge: `CogComp/bart-faithful-summary-detector`

`run_and_eval.sh` calls this automatically when `EVAL_MODE=summary`.

Model-family stop tokens should match the base model:

- Qwen3: `<|im_end|>`
- Gemma3: `<end_of_turn>`
- Llama3: `<|eot_id|>`

The stop token is currently set inside `script/run_and_eval.sh` in the vLLM generation command.

### Safety Evaluation

For safety alignment, `eval/new_score_beaver.py` uses Beaver reward/cost judges for scalar scoring.

For pairwise LLM-as-a-judge safety evaluation, run:

```bash
source eval/env/bin/activate
export OPENAI_API_KEY="..."

python script/eval_gpt5.py \
  --red_corner_model_path /path/to/base-model \
  --blue_corner_model_path /path/to/challenger-model \
  --tp 8 \
  --port 8200 \
  --server_max_model_len 3072 \
  --gen_max_tokens 1024 \
  --write_summary \
  --judge_concurrency 8
```

By default, this uses prompts from `trl/data/gpt5-eval.json` and writes results under `eval/output/`.

## Paper Defaults

### Beta

- DPO-LW and RACO do not include a length term in the objective, so we use fixed `beta=0.2` across setups.
- AMoPO includes a length term and typically requires tuning. We sweep `beta` in `[0.1, 1.5]` with step `0.1`.

Best `beta` from our sweeps:

- Qwen3-4B-Base: `beta=0.8`
- Qwen3-4B-Instruct: `beta=0.4`
- gemma-3-4B-pt: `beta=0.7`
- gemma-3-4B-it: `beta=0.8`
- Llama3-8B-Instruct: `beta=0.1`

### Learning Rate

Following SimPO (Meng et al., 2024), we sweep learning rates in `[1e-7, 5e-5]`.

Best learning rates from our sweeps:

- Qwen3-4B-Base: AMoPO `1e-5`; RACO/DPO-LW `2.75e-5`
- Qwen3-4B-Instruct: AMoPO `1e-5`; RACO/DPO-LW `2e-5`
- gemma-3-4B-pt: AMoPO `8e-6`; RACO/DPO-LW `1.5e-5`
- gemma-3-4B-it: AMoPO `5e-6`; RACO/DPO-LW `1.5e-5`
- Llama3-8B-Instruct: AMoPO `5e-7`; RACO/DPO-LW `7e-7`

Since the main difference between RACO and DPO-LW is how conflicted gradients are resolved, we generally observe similar learning-rate optima for the two methods.

## References

- Paper: [Reward-free Alignment for Conflicting Objectives](https://arxiv.org/abs/2602.02495)
- 2-objective Reddit summary dataset: [RACOo/RedditSummary-Alignment](https://huggingface.co/datasets/RACOo/RedditSummary-Alignment)
- Summary quality judge: [Tristan/gpt2_reward_summarization](https://huggingface.co/Tristan/gpt2_reward_summarization)
- Summary faithfulness judge: [CogComp/bart-faithful-summary-detector](https://huggingface.co/CogComp/bart-faithful-summary-detector)
