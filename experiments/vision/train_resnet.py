import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
import json
import os
import argparse
import sys
from pathlib import Path

# Define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "datasets" / "CIFAR-10H"
RESULTS_ROOT = REPO_ROOT / "results" / "raw"

def get_logits_and_labels(model, loader, device):
    """Extract raw logits and hard integer labels."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits).astype(np.float64), np.concatenate(all_labels)

def nll_loss_hard(T, logits, labels):
    """NLL for fitting T against hard one-hot labels."""
    scaled = logits / T[0]
    probs  = scipy_softmax(scaled, axis=1)
    probs  = np.clip(probs, 1e-9, 1.0)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

def brier_loss_soft(T, logits, soft_targets):
    """Brier Score for fitting T against soft label distributions."""
    scaled = logits / T[0]
    probs  = scipy_softmax(scaled, axis=1)
    return np.mean(np.sum((probs - soft_targets) ** 2, axis=1))

def fit_temperature_hard(logits, labels):
    """Fit T* by minimising NLL against hard labels."""
    result = minimize(nll_loss_hard, x0=[1.0], args=(logits, labels),
                      method='L-BFGS-B', bounds=[(0.01, 20.0)],
                      options={'maxiter': 100})
    return float(result.x[0])

def fit_temperature_soft(logits, soft_targets):
    """Fit T* by minimising Brier Score against soft label distributions."""
    result = minimize(brier_loss_soft, x0=[1.0], args=(logits, soft_targets),
                      method='L-BFGS-B', bounds=[(0.01, 20.0)],
                      options={'maxiter': 100})
    return float(result.x[0])

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)

def compute_brier_soft(probs, soft_targets):
    """Brier Score against full soft label distribution. Lower is better."""
    return float(np.mean(np.sum((probs - soft_targets) ** 2, axis=1)))

def compute_accuracy(probs, labels):
    return float((probs.argmax(axis=1) == labels).mean())

def run_vision_experiment(model_size, seed, epochs=100, batch_size=128):
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Download/load datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=str(REPO_ROOT / 'data'), train=True,
                                             download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root=str(REPO_ROOT / 'data'), train=False,
                                             download=True, transform=transform_test)

    # Load CIFAR-10H soft labels
    probs_path = DATA_ROOT / 'cifar10h-probs.npy'
    alt_probs_path = DATA_ROOT / 'cifar10h-probs.npy.1'
    if not probs_path.exists():
        # Check if we have the file with .1 suffix or download it
        if alt_probs_path.exists():
            os.rename(alt_probs_path, probs_path)
        else:
            raise FileNotFoundError(f"cifar10h-probs.npy not found at {probs_path}")
    
    soft_labels_vision = np.load(probs_path)

    # Split training: 80% train, 20% val
    n_val   = int(0.2 * len(trainset))
    n_train = len(trainset) - n_val
    train_sub, val_sub = random_split(trainset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_sub,  batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_sub,    batch_size=256,        shuffle=False, num_workers=2)
    test_loader  = DataLoader(testset,    batch_size=256,        shuffle=False, num_workers=2)

    # Build model
    if model_size == 'resnet18':
        model = models.resnet18(weights=None, num_classes=10)
    elif model_size == 'resnet50':
        model = models.resnet50(weights=None, num_classes=10)
    elif model_size == 'resnet101':
        model = models.resnet101(weights=None, num_classes=10)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Adjust for CIFAR-10 (32x32 images, not 224x224)
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\n==========================================")
    print(f"TRAINING {model_size} | Seed {seed} | Epochs {epochs}")
    print(f"==========================================")

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out  = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    print("Training complete. Extracting logits...")

    # Extract logits
    val_logits,  val_labels  = get_logits_and_labels(model, val_loader,  DEVICE)
    test_logits, test_labels = get_logits_and_labels(model, test_loader, DEVICE)

    # First 5000 for oracle calibration, last 5000 for evaluation
    n_half        = 5000
    oracle_logits = test_logits[:n_half]
    oracle_soft   = soft_labels_vision[:n_half]
    eval_logits   = test_logits[n_half:]
    eval_hard     = test_labels[n_half:]
    eval_soft     = soft_labels_vision[n_half:]

    # Fit temperatures
    T_hard = fit_temperature_hard(val_logits, val_labels)
    T_soft = fit_temperature_soft(oracle_logits, oracle_soft)
    print(f"T_star (hard): {T_hard:.4f}  |  T_star (soft, oracle): {T_soft:.4f}")

    # Compute probabilities under each condition
    probs_uncal   = scipy_softmax(eval_logits,             axis=1)
    probs_ts_hard = scipy_softmax(eval_logits / T_hard,    axis=1)
    probs_ts_soft = scipy_softmax(eval_logits / T_soft,    axis=1)

    # Compute all metrics
    results = {
        'model':            model_size,
        'seed':             seed,
        'T_star_hard':      T_hard,
        'T_star_soft':      T_soft,
        'test_accuracy':    compute_accuracy(probs_uncal, eval_hard),
        'uncal_ece':        compute_ece(probs_uncal,    eval_hard),
        'ts_hard_ece':      compute_ece(probs_ts_hard,  eval_hard),
        'ts_soft_ece':      compute_ece(probs_ts_soft,  eval_hard),
        'uncal_bs_soft':    compute_brier_soft(probs_uncal,    eval_soft),
        'ts_hard_bs_soft':  compute_brier_soft(probs_ts_hard,  eval_soft),
        'ts_soft_bs_soft':  compute_brier_soft(probs_ts_soft,  eval_soft),
    }
    results['gap'] = results['ts_hard_bs_soft'] - results['ts_soft_bs_soft']

    print("\n===== RESULTS =====")
    for k, v in results.items():
        print(f"  {k:25s}: {v:.6f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    # Save results to file
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    fname = RESULTS_ROOT / f"results_{model_size}_seed{seed}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {fname}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vision experiments.")
    parser.add_argument('--model_size', type=str, choices=['resnet18', 'resnet50', 'resnet101', 'all'], default='all',
                        help="Model size or 'all'")
    parser.add_argument('--seed', type=str, default='all', help="Seed (e.g. 42, 123, 456) or 'all'")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    models_to_run = [args.model_size] if args.model_size != 'all' else ['resnet18', 'resnet50', 'resnet101']
    
    if args.seed == 'all':
        seeds_to_run = [42, 123, 456]
    else:
        seeds_to_run = [int(args.seed)]

    for m in models_to_run:
        for s in seeds_to_run:
            try:
                run_vision_experiment(m, s, epochs=args.epochs)
            except Exception as e:
                print(f"FAILED: model={m}, seed={s}. Error: {str(e)}", file=sys.stderr)
