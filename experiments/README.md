# Experiment Documentation

## Vision Experiments (CIFAR-10H + ResNet)

This directory contains training and evaluation code for vision experiments using ResNet models on CIFAR-10H.

### Files

- `train_resnet.py`: Training loop for ResNet-18/50/101 on CIFAR-10 hard labels
- `evaluate.py`: Calibration evaluation (hard vs. oracle soft-label TS)
- `config.yaml`: Hyperparameters for training
- `README.md`: Detailed instructions

### Quick Start

```bash
python train_resnet.py --model resnet18 --seed 42
python evaluate.py --model resnet18 --seed 42
```

### Models

- ResNet-18 (11M parameters)
- ResNet-50 (25M parameters)  
- ResNet-101 (44M parameters)

### Hyperparameters

See `config.yaml` for full details:
- Epochs: 100
- Learning rate: Cosine annealing from 0.1
- Batch size: 128
- Optimizer: SGD with momentum

### Output

Results saved to:
- `../results/tables/vision_results.csv` - Calibration metrics
- `../results/figures/resnet_gap_by_scale.pdf` - Visualization

---

## Language Experiments (ChaosNLI + BERT)

This directory contains fine-tuning and evaluation code for language experiments using BERT models on ChaosNLI.

### Files

- `train_bert.py`: Fine-tuning loop for BERT variants on SNLI/MNLI hard labels
- `evaluate.py`: Calibration evaluation (hard vs. oracle soft-label TS)
- `config.yaml`: Hyperparameters for fine-tuning
- `README.md`: Detailed instructions

### Quick Start

```bash
python train_bert.py --model bert-base-uncased --dataset snli --seed 42
python evaluate.py --model bert-base-uncased --dataset snli --seed 42
```

### Models

- DistilBERT-base-uncased (66M parameters)
- BERT-base-uncased (110M parameters)
- BERT-large-uncased (340M parameters)

### Datasets

- `snli`: SNLI training set, ChaosNLI-SNLI evaluation
- `mnli`: MNLI training set, ChaosNLI-MNLI evaluation

### Hyperparameters

See `config.yaml` for full details:
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 32
- Optimizer: AdamW

### Output

Results saved to:
- `../results/tables/language_results.csv` - Calibration metrics
- `../results/figures/bert_gap_by_scale.pdf` - Visualization

---

## Utilities

This directory contains shared utilities for both vision and language experiments.

### Files

- `calibration.py`: Temperature scaling implementation
- `metrics.py`: ECE and other calibration metrics
- `brier_score.py`: Soft-label Brier Score computation
- `plotting.py`: Figure generation utilities

### Key Functions

#### Calibration

```python
from utils.calibration import fit_temperature, apply_temperature

# Fit T on validation set (hard labels)
T_hard = fit_temperature(logits_val, labels_val, method='nll')

# Fit T on test set (soft labels) - oracle
T_oracle = fit_temperature(logits_test, soft_labels_test, method='brier')

# Apply to predictions
probs_hard = apply_temperature(logits, T_hard)
probs_oracle = apply_temperature(logits, T_oracle)
```

#### Metrics

```python
from utils.metrics import expected_calibration_error, brier_score

# ECE against hard labels
ece = expected_calibration_error(probs, hard_labels)

# Brier Score against soft labels
bs = brier_score(probs, soft_labels)

# Calibration gap
gap = brier_score(probs_hard, soft_labels) - brier_score(probs_oracle, soft_labels)
```

#### Plotting

```python
from utils.plotting import plot_calibration_curve, plot_gap_by_scale

plot_calibration_curve(probs, labels, title="ResNet-18 Calibration")
plot_gap_by_scale(gaps_by_model, title="Gap Growth with Model Scale")
```

---

## Experiment Workflow

For each model and seed:

1. **Training**
   - Load training set (hard labels)
   - Train on hard-label objective (cross-entropy)
   - Save model weights

2. **Calibration**
   - Load validation set (hard labels)
   - Fit T_hard by minimizing NLL
   - Load test set (soft labels)
   - Fit T_oracle by minimizing Brier Score
   - Save temperatures

3. **Evaluation**
   - Load test set (held-out half)
   - Evaluate with T_hard → BS_hard, ECE_hard
   - Evaluate with T_oracle → BS_oracle, ECE_oracle
   - Compute gap: BS_hard - BS_oracle

4. **Results**
   - Record metrics for all 3 seeds
   - Compute mean ± std
   - Generate plots

---

## Expected Results

### Vision (CIFAR-10H)
- Mean gap: ~0.003
- Scale-dependent growth confirmed
- Gaps: ResNet-18 (0.002) → ResNet-50 (0.003) → ResNet-101 (0.005)

### Language (ChaosNLI)
- Mean gap: ~0.079 (26× larger than vision)
- Mostly scale-dependent
- Gaps: DistilBERT (0.045) → BERT-base (0.079) → BERT-large (0.134)
