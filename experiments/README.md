# Experiments

Scripts that produce the per-seed metrics in `results/raw/` (and optional
re-runs under `results/raw_v2/`).

## Pipelines

| Script | Role |
|--------|------|
| `vision/train_resnet.py` | ResNet-18/50/101 on CIFAR-10 + CIFAR-10H soft-label eval |
| `language/train_bert.py` | DistilBERT / BERT-base / BERT-large on SNLI or MNLI → ChaosNLI |
| `language/finetune_bert_large_snli.py` | Independent BERT-large SNLI checkpoint from `bert-large-uncased` |
| `utils/calibration.py` | Shared temperature scaling + multiclass isotonic regression |
| `run_all.py` | Orchestrate all configs; supports `--results_dir` |
| `aggregate.py` | Mean/std tables, including TS vs isotonic gaps |

## Protocol

For each model and seed (`42`, `123`, `456`):

1. Train / adapt on hard labels.
2. Fit `T_hard` on hard-label validation NLL.
3. Fit `T_oracle` on the first half of the soft-label test set (Brier).
4. Fit hard and soft isotonic calibrators on the same splits.
5. Evaluate on the held-out second half; save metrics JSON and logits (`.npz`).

Vision default: 30 epochs. Language default: 1 epoch on a 10{,}000-example subset,
starting from a domain-matched checkpoint. BERT-large SNLI uses
`checkpoints/bert-large-uncased-snli-independent/` (built by
`finetune_bert_large_snli.py`), not an MNLI checkpoint.

## Matched-domain language splits

- `dataset_type=SNLI` → evaluate on ChaosNLI-S  
- `dataset_type=MNLI` → evaluate on ChaosNLI-M  

## Reproduction

```bash
python experiments/run_all.py --results_dir results/raw
python experiments/aggregate.py --results_dir results/raw

# Re-run without overwriting verified raw files
python experiments/run_all.py --results_dir results/raw_v2
python experiments/aggregate.py --results_dir results/raw_v2 --out_prefix final_results_v2
```

Checkpoints under `checkpoints/` and logits under `results/logits/` are local
artifacts (gitignored) and can be regenerated from the scripts above.
