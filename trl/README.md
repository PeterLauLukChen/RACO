# RACO TRL Backend

This directory contains the TRL-based training backend used by the top-level RACO repository. It is vendored from Hugging Face TRL and kept intentionally narrow for the RACO camera-ready code release.

RACO-specific entry points:

- `scripts/train_raco.py`: launches RACO, AMOPO, and related preference-optimization runs.
- `scripts/convert_moa_jsonl_to_trl_raco.py`: converts pairwise multi-objective JSONL data into the TRL/RACO dataset format.
- `trl/trainer/dpo_config.py`: adds RACO and AMOPO configuration fields.
- `trl/trainer/dpo_trainer.py`: implements the RACO training logic on top of DPO-style preference batches.

See the repository root `README.md` for setup, dataset construction, training, and evaluation commands.

## License

The vendored TRL code is distributed under the Apache-2.0 License. See `LICENSE`.
