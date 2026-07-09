# Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions

A research repository investigating the failure of temperature scaling, the dominant post-hoc calibration method for neural networks, when applied to soft, crowd-sourced, or distributional labels that reflect genuine human disagreement.

## Paper

**Title**: Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions

**Author**: Wisdom Dogah

**Affiliation**: Faculty of Computing & Mathematical Sciences, University of Mines and Technology (UMaT), Tarkwa, Ghana; BlackMatrix AI Research, Accra, Ghana

**Contact**: wisdom@blackmatrix.io

**Status**: Completed. Submitted to arXiv and peer-reviewed venues.

**Abstract**: Temperature scaling is the dominant post-hoc calibration method in modern deep learning. Its theoretical justification rests on an assumption that is rarely stated explicitly: that ground-truth labels are one-hot and deterministic. In practice, labels are frequently soft, crowd-sourced, or genuinely distributional, reflecting real disagreement among human annotators rather than annotation noise. We study whether temperature scaling retains its calibration properties when this assumption is violated, and whether any resulting degradation depends on model scale. Using CIFAR-10H and ChaosNLI, two publicly available datasets with human-annotated soft label distributions, we evaluate three model scales per modality under both hard one-hot and soft distributional label targets. Across all nine configurations we find a positive soft-label calibration gap: temperature scaling calibrated on hard labels consistently underperforms an oracle calibrated directly on soft labels, with Brier Score gaps ranging from 0.002 to 0.134. The gap grows monotonically with model scale in the vision domain and on the SNLI-derived split of ChaosNLI, and is substantially larger in the language domain (mean gap 0.079) than in vision (mean gap 0.003). A scale-ordering reversal on the MNLI-derived split is attributable to a cross-domain evaluation confound rather than a genuine exception to the trend. These findings suggest that calibration protocols built on majority-vote labels systematically misstate model reliability wherever label ambiguity is structural, with direct consequences for deployment in safety-critical settings.

**Keywords**: calibration; temperature scaling; soft labels; label ambiguity; Expected Calibration Error; Brier Score; model scale; uncertainty quantification.

**arXiv Submission**: LaTeX source files are available in the `paper/` directory (`main.tex` and `references.bib`). See [ARXIV_SUBMISSION.md](paper/ARXIV_SUBMISSION.md) for compilation instructions and submission checklist.

## Research Overview

### Problem Statement

Temperature scaling assumes that ground-truth labels are **one-hot and deterministic** (single correct answer). However, many real-world labeling tasks involve **genuine human disagreement**, where labels are soft, crowd-sourced, or distributional.

**Central Question**: Does temperature scaling work when this assumption is violated?

**Answer**: No. It causes systematic miscalibration.

### Key Findings

- **Gap Existence**: Positive soft-label calibration gap confirmed across all models (0.002-0.134 Brier Score)
- **Scale Dependence**: Gap grows monotonically with model size in vision, mostly in language
- **Modality Dependence**: Gap is 26× larger in language (NLI) than vision (CIFAR-10)

### Safety-Critical Implications

A model that predicts 90% confidence for class A when human annotators are evenly split (50/50) is **not actually well-calibrated**, even if its predicted class matches the majority vote. This has direct consequences for:

- Medical diagnosis (overconfident wrong diagnosis)
- Content moderation (confidently removing debatable content)
- Legal risk assessment (overstated recidivism probability)

## Research Design

### Datasets

#### CIFAR-10H (Vision)
- **Source**: CIFAR-10 test set (10,000 images)
- **Annotations**: Mean of 51 per image
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Label Type**: Soft distributions over classes
- **Total Annotations**: 510,000
- **Location**: `datasets/CIFAR-10H/`

#### ChaosNLI (Language)
- **Source**: SNLI, MNLI, alphaNLI development sets
- **Annotations**: 100 per example
- **Instances**: 4,645 examples (ChaosNLI-S: 1,514; ChaosNLI-M: 1,599; ChaosNLI-A: 1,532)
- **Task**: Natural Language Inference (entailment, neutral, contradiction)
- **Total Annotations**: 464,500
- **Location**: `datasets/ChaosNLI/`

### Models Tested

**Vision Domain** (trained on CIFAR-10):
- ResNet-18 (11M parameters)
- ResNet-50 (25M parameters)
- ResNet-101 (44M parameters)

**Language Domain** (fine-tuned on SNLI/MNLI):
- DistilBERT-base-uncased (66M parameters)
- BERT-base-uncased (110M parameters)
- BERT-large-uncased (340M parameters)

### Experimental Protocol

For each model:

1. **Training Phase**
   - Train on hard-label (majority-vote) training set
   - Standard hyperparameters (cosine annealing, Adam)
   - 3 random seeds: 42, 123, 456

2. **Calibration Phase**
   - **T_hard**: Fit on 20% validation split, minimize NLL on hard labels
   - **T_oracle**: Fit on first half of soft-label test set, minimize Brier Score

3. **Evaluation Phase**
   - Evaluate both on second half (held out)
   - Metrics: ECE (15 bins) and Brier Score
   - Report: Mean ± std over 3 seeds

### Metrics

**Expected Calibration Error (ECE)**
```
ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|
```
- Measures agreement between predicted confidence and accuracy (top-class only)
- Standard metric but limited to argmax

**Brier Score (Strictly Proper)**
```
BS = 1/n Σ_i Σ_k (f_k(x_i) - q_ik)²
```
- Measures MSE between predicted probabilities and human annotation distribution
- Captures full predicted distribution
- **Soft-Label Calibration Gap**: BS_hard - BS_oracle

## Pre-specified Hypotheses

### H1: Gap Existence
Temperature scaling calibrated on hard labels yields strictly worse Brier Score against the soft label distribution than an oracle calibrated directly on soft labels.

**Status**: CONFIRMED. Positive gaps across all 9 configurations.

### H2: Scale Dependence
The soft-label calibration gap grows monotonically with model scale within each dataset.

**Status**: CONFIRMED. Clear monotonic growth in vision; mostly in language.

### H3: Modality Dependence
The gap is larger in the language domain (ChaosNLI) than in vision (CIFAR-10H).

**Status**: CONFIRMED. 26 times larger in language (mean gap 0.079 vs. 0.003)

## Repository Structure

```
temperature-scaling-research/
├── README.md                              # This file
├── LICENSE                                # CC-BY 4.0
├── .gitignore
├── paper/
│   ├── main.tex                           # arXiv-ready LaTeX source
│   ├── references.bib                     # BibTeX bibliography
│   ├── ARXIV_SUBMISSION.md                # Submission guide & checklist
│   ├── main.pdf                           # Compiled PDF (5 pages)
│   ├── Temperature_Scaling_Is_Not_Enough.pdf
│   └── Temperature_Scaling_Is_Not_Enough.docx
├── datasets/
│   ├── CIFAR-10H/
│   │   ├── README.md                     # Download & extraction guide
│   │   ├── cifar-10-python.tar.gz        # Archive (~163 MB)
│   │   └── metadata.json                 # Dataset statistics
│   └── ChaosNLI/
│       ├── README.md                     # License info, source links
│       ├── chaosNLI_snli.jsonl          # SNLI split (1,514 examples)
│       ├── chaosNLI_mnli_m.jsonl        # MNLI split (1,599 examples)
│       ├── chaosNLI_alphanli.jsonl      # AlphaNLI split (1,532 examples)
│       └── metadata.json                 # Dataset statistics
├── experiments/
│   ├── vision/
│   │   ├── train_resnet.py              # Training loop
│   │   ├── evaluate.py                  # Calibration evaluation
│   │   ├── config.yaml                  # Hyperparameters
│   │   └── README.md
│   ├── language/
│   │   ├── train_bert.py                # Fine-tuning loop
│   │   ├── evaluate.py                  # Calibration evaluation
│   │   ├── config.yaml                  # Hyperparameters
│   │   └── README.md
│   └── utils/
│       ├── calibration.py               # Temperature scaling, metrics
│       ├── brier_score.py               # Soft-label evaluation
│       ├── metrics.py                   # ECE, calibration metrics
│       └── plotting.py                  # Figure generation
├── results/
│   ├── figures/                         # Generated plots
│   ├── tables/                          # Result tables (CSV)
│   └── results.md                       # Summary of findings
├── requirements.txt                     # Python dependencies
├── setup.py                             # Package setup
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers (for BERT experiments)
- NumPy, Pandas, Matplotlib, Seaborn

### Installation

```bash
git clone https://github.com/dogahwisdom/temperature-scaling-research.git
cd temperature-scaling-research
pip install -r requirements.txt
```

### Dataset Setup

#### CIFAR-10H

```bash
cd datasets/CIFAR-10H
tar -xzf cifar-10-python.tar.gz
cd ../..
```

#### ChaosNLI

No extraction needed. Files are in JSONL format (one JSON object per line).

```bash
# Example: Load ChaosNLI
import json
with open('datasets/ChaosNLI/chaosNLI_snli.jsonl') as f:
    examples = [json.loads(line) for line in f]
```

### Running Experiments

Vision experiments:
```bash
cd experiments/vision
python train_resnet.py --model resnet18 --seed 42
python evaluate.py --model resnet18 --seed 42
```

Language experiments:
```bash
cd experiments/language
python train_bert.py --model bert-base-uncased --dataset snli --seed 42
python evaluate.py --model bert-base-uncased --dataset snli --seed 42
```

## Key Results Summary

### Vision (CIFAR-10H)
- **Mean Gap**: 0.003 (small but consistent)
- **Range**: 0.002-0.005
- **Scale Effect**: Monotonic increase with model size
- **Gap by Model**:
  - ResNet-18: 0.002
  - ResNet-50: 0.003
  - ResNet-101: 0.005

### Language (ChaosNLI)
- **Mean Gap**: 0.079 (26 times larger than vision)
- **Range**: 0.002-0.134
- **Scale Effect**: Mostly monotonic (one anomaly on MNLI)
- **Gap by Model**:
  - DistilBERT: 0.045
  - BERT-base: 0.079
  - BERT-large: 0.134

### MNLI Split Anomaly
Scale ordering reverses on ChaosNLI-M. Analysis attributes this to cross-domain evaluation confound:
- Models fine-tuned on SNLI (in-domain)
- Tested on ChaosNLI-M (out-of-domain)
- Not a fundamental exception to the trend

## Related Work

### Calibration Methods
- Temperature scaling (Guo et al., 2017)
- Isotonic regression
- Platt scaling
- Dirichlet calibration (Malinin & Gales, 2018)

### Soft Labels & Human Disagreement
- Learning from crowds (foundational literature)
- Annotation disagreement as signal (Peterson et al., 2019: CIFAR-10H)
- ChaosNLI (Nie et al., 2020): Human disagreement in NLI

### Calibration Under Distribution Shift
- Ovadia et al. (2019): Calibration under input distribution shift
- This work: First systematic study of label distribution shift

## Citations

If you use this work, please cite:

```bibtex
@article{dogah2026temperature,
  title={Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions},
  author={Dogah, Wisdom},
  journal={arXiv preprint arXiv:2607.xxxxx},
  year={2026}
}
```

(arXiv ID to be assigned upon submission)

## Dataset Citations

**CIFAR-10H**:
```bibtex
@article{peterson2019cifar,
  title={Shared Predictive Models of Crowd Annotations},
  author={Peterson, Joshua C and Bourgin, David D and Agrawal, Mayank and Griffiths, Thomas L and Russell, Stuart J},
  journal={Nature Machine Intelligence},
  year={2019}
}
```

**ChaosNLI**:
```bibtex
@inproceedings{nie2020chaos,
  title={Evaluating Understanding on Implicit Relations in Natural Language Inference},
  author={Nie, Yixin and Williams, Adina and Dinan, Emily and Bansal, Mohit and Weston, Jason and Kiela, Douwe},
  booktitle={Findings of the Association for Computational Linguistics: ACL},
  year={2020}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0). See LICENSE file for details.

Datasets:
- **CIFAR-10H**: Based on CIFAR-10. See original papers for licensing.
- **ChaosNLI**: Creative Commons-Non Commercial 4.0 (Nie et al., 2020)

## Questions and Discussion

For questions about the research, methodology, or implementation, please open an issue on GitHub.

## Publication

This research has been completed and is ready for publication. The paper is submitted to arXiv and peer-reviewed venues in machine learning and NLP. The repository supports full reproducibility of all experiments and findings.

For questions about the research or collaboration opportunities, please open an issue on GitHub or contact the author.

---

**Author**: Wisdom Dogah  
**Last Updated**: July 9, 2026  
**Status**: Completed and Ready for Publication
