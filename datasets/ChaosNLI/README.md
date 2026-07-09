# ChaosNLI Dataset Guide

## Overview

ChaosNLI provides 100 human annotations per example for Natural Language Inference (NLI) instances from three source datasets: SNLI, MNLI, and alphaNLI. This enables studying human disagreement and label ambiguity in the NLI task.

## Download & Setup

### Quick Start

Files are already included in this repository in JSONL format (one JSON object per line).

```bash
# Files:
# - chaosNLI_snli.jsonl      (1,514 examples from SNLI)
# - chaosNLI_mnli_m.jsonl    (1,599 examples from MNLI)
# - chaosNLI_alphanli.jsonl  (1,532 examples from alphaNLI)
```

### Manual Download

If needed, download from:
- Official Repository: https://github.com/easonnie/chaos_nli
- ChaosNLI GitHub: https://github.com/easonnie/chaos_nli/releases

### Data Format

Each file contains JSONL (one JSON object per line):

```python
import json

# Load ChaosNLI-SNLI
with open('chaosNLI_snli.jsonl') as f:
    examples = [json.loads(line) for line in f]

# Example structure:
example = {
    "uid": "193596775.jpg#3r1n",
    "label_counter": {"n": 67, "c": 29, "e": 4},           # Count per label
    "majority_label": "n",                                  # Majority vote
    "label_dist": [0.04, 0.67, 0.29],                     # Normalized distribution
    "label_count": [4, 67, 29],                           # Annotation counts
    "entropy": 1.0907619435810212,                         # Distribution entropy
    "example": {
        "uid": "193596775.jpg#3r1n",
        "premise": "A woman is talking on the phone...",
        "hypothesis": "A woman is walking her dog.",
        "source": "snli_agree_3"
    },
    "old_label": "n",
    "old_labels": ["neutral", "neutral", "neutral", "contradiction", "contradiction"]
}

# Label meanings:
# "e" = "entailment"     (0)
# "n" = "neutral"        (1)
# "c" = "contradiction"  (2)
```

## Statistics

### ChaosNLI-SNLI (SNLI-derived)
| Property | Value |
|----------|-------|
| **Examples** | 1,514 |
| **Annotations per Example** | 100 |
| **Total Annotations** | 151,400 |
| **Classes** | 3 (entailment, neutral, contradiction) |
| **Mean Entropy** | ~1.0 |

### ChaosNLI-MNLI (MNLI-derived)
| Property | Value |
|----------|-------|
| **Examples** | 1,599 |
| **Annotations per Example** | 100 |
| **Total Annotations** | 159,900 |
| **Classes** | 3 (entailment, neutral, contradiction) |
| **Mean Entropy** | ~1.4 |

### ChaosNLI-alphaNLI (alphaNLI-derived)
| Property | Value |
|----------|-------|
| **Examples** | 1,532 |
| **Annotations per Example** | 100 |
| **Total Annotations** | 153,200 |
| **Classes** | 2 (for AlphaNLI) |
| **Mean Entropy** | ~0.95 |

### Combined
| Property | Value |
|----------|-------|
| **Total Examples** | 4,645 |
| **Total Annotations** | 464,500 |
| **Source Datasets** | SNLI, MNLI, alphaNLI |

## Key Features

### Soft Label Distributions

Unlike majority-vote labels, ChaosNLI captures the full distribution of human opinions:

```python
# Balanced case - high ambiguity
{
    "label_counter": {"e": 50, "n": 36, "c": 14},
    "label_dist": [0.5, 0.36, 0.14],
    "entropy": 1.4277254052800654
}

# Skewed case - clear majority
{
    "label_counter": {"n": 67, "c": 29, "e": 4},
    "label_dist": [0.04, 0.67, 0.29],
    "entropy": 1.0907619435810212
}
```

### Entropy Distribution

Entropy values reveal task ambiguity:
- **Low entropy** (~0.0): Unanimous agreement
- **High entropy** (~1.4): Evenly split opinions
- Varies significantly across examples

## Usage in This Research

In our calibration experiments, we:

1. Use ChaosNLI examples for soft-label evaluation
2. Fine-tune BERT models on original SNLI/MNLI training sets (hard labels)
3. Evaluate on ChaosNLI test splits (soft labels)
4. Compute soft-label calibration gaps by comparing:
   - Hard-label temperature scaling (fitted on validation set)
   - Oracle soft-label temperature scaling (fitted on first half of test set)
5. Evaluate on second half of test set (held out)

### Split Strategy

For ChaosNLI-SNLI:
- Fine-tune BERT on SNLI training set (hard labels)
- Evaluate on ChaosNLI-SNLI (soft labels)

For ChaosNLI-MNLI:
- Fine-tune BERT on MNLI training set (hard labels)
- Evaluate on ChaosNLI-MNLI (soft labels) ← Cross-domain evaluation (out-of-domain test)

## Python Example

```python
import json
import numpy as np

# Load data
def load_chaosnli(split='snli'):
    filename = f'chaosNLI_{split}.jsonl'
    examples = []
    soft_labels = []
    
    with open(filename) as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
            # Soft labels as probabilities: [entail, neutral, contradict]
            soft_labels.append(example['label_dist'])
    
    return examples, np.array(soft_labels)

# Usage
examples, soft_labels = load_chaosnli('snli')
print(f"Loaded {len(examples)} examples")
print(f"Soft labels shape: {soft_labels.shape}")
print(f"Mean entropy: {np.mean([e['entropy'] for e in examples]):.3f}")
```

## Citation

```bibtex
@inproceedings{nie2020chaos,
  title={Evaluating Understanding on Implicit Relations in Natural Language Inference},
  author={Nie, Yixin and Williams, Adina and Dinan, Emily and Bansal, Mohit and Weston, Jason and Kiela, Douwe},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={8657--8667},
  year={2020}
}
```

## References

- Paper: https://arxiv.org/abs/2008.03451
- GitHub: https://github.com/easonnie/chaos_nli
- Original NLI datasets:
  - SNLI: https://nlp.stanford.edu/projects/snli/
  - MNLI: https://cims.nyu.edu/~sbowman/multinli/
  - alphaNLI: https://github.com/easonnie/alphanli

## License

Creative Commons-Non Commercial 4.0 (CC-BY-NC 4.0)

See https://github.com/easonnie/chaos_nli for detailed licensing information.
