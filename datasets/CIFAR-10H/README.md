# CIFAR-10H Dataset Guide

## Overview

CIFAR-10H provides soft label distributions for all 10,000 test images from the CIFAR-10 dataset. Each image was annotated by approximately 51 human annotators, yielding per-image probability distributions over the 10 classes.

## Download & Setup

### Quick Start

```bash
# Extract the archive
tar -xzf cifar-10-python.tar.gz

# You'll get:
# cifar-10-batches-py/
# ├── batches.meta          # Class names
# ├── data_batch_1-5        # Training data (50,000 images)
# ├── test_batch            # Test data (10,000 images with soft labels)
# └── readme.html
```

### Manual Download

If you don't have the archive, download from:
- Official CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10H (soft labels): https://github.com/jcpeterson/cifar-10h

### Data Format

Each batch file is a Python pickle containing a dictionary:

```python
import pickle

with open('cifar-10-batches-py/test_batch', 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    
# Keys:
# - b'data': numpy array (N, 3072) - images as flattened 32×32 RGB
# - b'labels': list - majority-vote hard labels
# - b'fine_labels': list - fine-grained labels (if available)

# For CIFAR-10H soft labels, additional data structures:
# - soft_labels: numpy array (N, 10) - soft label distributions
# - label_distributions: numpy array (N, 10) - normalized annotation counts
```

### Image Reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt

# Load test batch
with open('cifar-10-batches-py/test_batch', 'rb') as f:
    data = pickle.load(f, encoding='bytes')

images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Display first image
plt.imshow(images[0].astype('uint8'))
plt.title(f"Image 0")
plt.show()
```

## Statistics

| Property | Value |
|----------|-------|
| **Total Images** | 10,000 (test set) |
| **Image Size** | 32×32 pixels, RGB |
| **Classes** | 10 |
| **Annotations per Image** | ~51 (mean) |
| **Total Annotations** | 510,000 |
| **Soft Label Distribution** | Probability vector (sums to 1) |

## Classes

```
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
```

## Usage in This Research

In our calibration experiments, we:

1. Use CIFAR-10 training set for model training
2. Use CIFAR-10H test set for soft-label evaluation
3. Compute soft-label calibration gaps by comparing:
   - Hard-label temperature scaling (fitted on validation set)
   - Oracle soft-label temperature scaling (fitted on first half of test set)
4. Evaluate on second half of test set (held out)

## Citation

```bibtex
@article{peterson2019cifar,
  title={Shared Predictive Models of Crowd Annotations},
  author={Peterson, Joshua C and Bourgin, David D and Agrawal, Mayank and Griffiths, Thomas L and Russell, Stuart J},
  journal={Nature Machine Intelligence},
  year={2019}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex},
  year={2009}
}
```

## References

- Paper: https://arxiv.org/abs/1911.01969
- Dataset: https://github.com/jcpeterson/cifar-10h
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

## License

CIFAR-10H soft label annotations: See Peterson et al. (2019) for licensing information.
