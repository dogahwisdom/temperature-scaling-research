# Experiment Documentation

This directory contains the scripts used to produce the stored per-seed results in
`results/raw/` and the aggregated summaries in `results/tables/`.

## Files

- `vision/train_resnet.py`:
  Vision pipeline for ResNet-18/50/101 on CIFAR-10 with CIFAR-10H soft-label evaluation.
- `language/train_bert.py`:
  Language pipeline for DistilBERT/BERT-base/BERT-large, with separate runs for
  `SNLI` and `MNLI`, evaluated on `ChaosNLI-S` and `ChaosNLI-M` respectively.
- `run_all.py`:
  Orchestrates all model/dataset/seed runs and skips already computed outputs.
- `aggregate.py`:
  Aggregates raw per-seed JSON files into mean/std summaries.

## Output Artifacts

- Raw run files: `results/raw/results_*.json` (27 files, 9 vision + 18 language).
- Aggregated summaries:
  - `results/tables/final_results.json`
  - `results/tables/final_results_summary.txt`

## Protocol Implemented

For each model and seed (`42`, `123`, `456`):

1. Train/fine-tune model on hard labels.
2. Fit `T_star_hard` on hard-label validation NLL.
3. Fit `T_star_soft` on first half of soft-label test set (oracle).
4. Evaluate on held-out second half and save:
   - accuracy
   - uncalibrated ECE/Brier
   - TS-hard ECE/Brier
   - TS-soft ECE/Brier
   - gap = `ts_hard_bs_soft - ts_soft_bs_soft`

## Important Clarification: SNLI vs MNLI

The language script runs **separate fine-tuning paths**:

- `dataset_type=SNLI` -> evaluate on `ChaosNLI-S`
- `dataset_type=MNLI` -> evaluate on `ChaosNLI-M`

This is matched-domain training/evaluation by split, not SNLI-only training for both.

## Reproduction Command

```bash
python experiments/run_all.py
python experiments/aggregate.py
```

## Notes

- The provided artifact set contains per-seed metric outputs and aggregated summaries.
- Model checkpoints are not saved by the current scripts; only metric JSON outputs are persisted.
