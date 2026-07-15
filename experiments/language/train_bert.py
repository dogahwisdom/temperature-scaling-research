import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import json
import sys
from pathlib import Path

# Local repo has a datasets/ data folder that shadows HuggingFace `datasets`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path = [
    p for p in sys.path
    if p not in ("", ".") and Path(p).resolve() != _REPO_ROOT.resolve()
]
sys.path.insert(0, str(_EXPERIMENTS))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from scipy.special import softmax as scipy_softmax
from utils.calibration import (
    fit_temperature_hard,
    fit_temperature_soft,
    compute_ece,
    compute_brier_soft,
    compute_accuracy,
    MulticlassIsotonicCalibrator,
)

# Define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
REPO_ROOT = _REPO_ROOT
CHAOSNLI_ROOT = REPO_ROOT / "datasets" / "ChaosNLI"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "raw"
LOGITS_ROOT = REPO_ROOT / "results" / "logits"
INDEPENDENT_SNLI_CKPT = REPO_ROOT / "checkpoints" / "bert-large-uncased-snli-independent"

# Label map
label_map = {'e': 0, 'n': 1, 'c': 2}

def load_chaosnli(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def to_soft_label(entry):
    counts = entry['label_counter']
    total  = sum(counts.values())
    dist   = np.zeros(3)
    for label, count in counts.items():
        key = label[0].lower()   # 'entailment' -> 'e'
        if key in label_map:
            dist[label_map[key]] = count / total
    assert abs(dist.sum() - 1.0) < 1e-6, f"Distribution does not sum to 1: {dist}"
    return dist

def to_hard_label(entry):
    counts  = entry['label_counter']
    top_key = max(counts, key=counts.get)
    key     = top_key[0].lower()
    return label_map[key]

def get_nli_logits(model, premises, hypotheses, tokenizer, device, batch_size=64, max_length=128):
    """Extract logits for a list of premise/hypothesis pairs."""
    model.eval()
    all_logits = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hypotheses[i:i+batch_size]
        enc = tokenizer(batch_p, batch_h, truncation=True,
                        padding='max_length', max_length=max_length,
                        return_tensors='pt')
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out = model(input_ids=enc['input_ids'].to(device),
                            attention_mask=enc['attention_mask'].to(device))
        all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits).astype(np.float64)

def save_logits_bundle(prefix, val_logits, val_labels, oracle_logits, oracle_soft,
                       eval_logits, eval_hard, eval_soft):
    """Persist raw logits and labels for post-hoc calibration reuse."""
    LOGITS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LOGITS_ROOT / f"{prefix}.npz",
        val_logits=val_logits,
        val_labels=val_labels,
        oracle_logits=oracle_logits,
        oracle_soft=oracle_soft,
        eval_logits=eval_logits,
        eval_hard=eval_hard,
        eval_soft=eval_soft,
    )
    print(f"Saved logits bundle to {LOGITS_ROOT / f'{prefix}.npz'}")

model_mapping = {
    ('distilbert-base-uncased', 'SNLI'): 'kweinmeister/distilbert-snli',
    ('bert-base-uncased', 'SNLI'): 'textattack/bert-base-uncased-SNLI',
    # Independent SNLI checkpoint fine-tuned from base bert-large-uncased (NOT MNLI).
    ('bert-large-uncased', 'SNLI'): str(INDEPENDENT_SNLI_CKPT),
    ('distilbert-base-uncased', 'MNLI'): 'textattack/distilbert-base-uncased-MNLI',
    ('bert-base-uncased', 'MNLI'): 'textattack/bert-base-uncased-MNLI',
    ('bert-large-uncased', 'MNLI'): 'yoshitomo-matsubara/bert-large-uncased-mnli',
}

def run_language_experiment(model_name, dataset_type, seed, epochs=1, results_root=None):
    # dataset_type must be either 'SNLI' or 'MNLI'
    results_root = Path(results_root) if results_root else DEFAULT_RESULTS_ROOT
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model_id = model_mapping[(model_name, dataset_type)]
    if model_name == 'bert-large-uncased' and dataset_type == 'SNLI':
        if not Path(model_id).exists():
            raise FileNotFoundError(
                f"Independent SNLI checkpoint not found at {model_id}. "
                f"Run: python experiments/language/finetune_bert_large_snli.py"
            )

    print(f"\n==========================================")
    print(f"TRAINING {model_name} on {dataset_type} | Seed {seed} | Epochs {epochs}")
    print(f"Using Hugging Face Model ID: {model_id}")
    print(f"Results dir: {results_root}")
    print(f"==========================================")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure batch size, gradient accumulation, and max sequence length
    batch_size = 32
    grad_accum_steps = 1
    eval_batch_size = 64
    max_length = 128
    if 'large' in model_name:
        batch_size = 2
        grad_accum_steps = 16
        eval_batch_size = 16
        max_length = 80
        print(f"Large model detected. Using batch size = 2, gradient accumulation = 16, eval batch size = 16, max_length = 80")

    # Load datasets
    if dataset_type == 'SNLI':
        train_raw = load_dataset('snli', split='train')
        val_raw   = load_dataset('snli', split='validation')
        # Filter out label -1
        train_raw = train_raw.filter(lambda x: x['label'] != -1)
        val_raw   = val_raw.filter(lambda x: x['label'] != -1)
    elif dataset_type == 'MNLI':
        # Use GLUE MNLI
        train_raw = load_dataset('glue', 'mnli', split='train')
        val_raw   = load_dataset('glue', 'mnli', split='validation_matched')
        # Filter out label -1
        train_raw = train_raw.filter(lambda x: x['label'] != -1)
        val_raw   = val_raw.filter(lambda x: x['label'] != -1)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # OPTIMIZATION: Shuffle and select a subset of 10,000 training examples
    print("Selecting 10,000 training examples subset...")
    train_raw = train_raw.shuffle(seed=seed).select(range(10000))

    # Tokenize function
    def tokenize_fn(batch):
        return tokenizer(batch['premise'], batch['hypothesis'],
                         truncation=True, padding='max_length', max_length=max_length)

    print("Tokenizing training data...")
    train_tok = train_raw.map(tokenize_fn, batched=True, batch_size=1000)
    train_tok.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print("Tokenizing validation data...")
    val_tok = val_raw.map(tokenize_fn, batched=True, batch_size=1000)
    val_tok.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_tok, batch_size=256, shuffle=False)

    # Build model from pre-trained ID
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id).to(DEVICE)
    if 'large' in model_name:
        # Freeze embeddings
        if hasattr(model, 'bert'):
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(20):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            print("Froze embeddings and first 20 encoder layers of BERT-large.")
        elif hasattr(model, 'roberta'):
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
            for i in range(20):
                for param in model.roberta.encoder.layer[i].parameters():
                    param.requires_grad = False
            print("Froze embeddings and first 20 encoder layers of RoBERTa-large.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
    
    # Use Mixed Precision training (AMP)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            with torch.cuda.amp.autocast():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / grad_accum_steps

            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            if (step + 1) % 2000 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")
        
        print(f"Epoch {epoch+1}/{epochs} complete. Average Loss: {total_loss/len(train_loader):.4f}")

    print("Training complete. Extracting validation logits...")
    
    # Extract validation logits for hard T* calibration
    val_premises = val_raw['premise']
    val_hypotheses = val_raw['hypothesis']
    val_labels = np.array(val_raw['label'])
    
    val_logits = get_nli_logits(model, val_premises, val_hypotheses, tokenizer, DEVICE, batch_size=eval_batch_size, max_length=max_length)

    # Extract ChaosNLI evaluation logits
    if dataset_type == 'SNLI':
        chaos_data = load_chaosnli(str(CHAOSNLI_ROOT / 'chaosNLI_snli.jsonl'))
        eval_dataset_name = 'ChaosNLI-S'
    else:
        chaos_data = load_chaosnli(str(CHAOSNLI_ROOT / 'chaosNLI_mnli_m.jsonl'))
        eval_dataset_name = 'ChaosNLI-M'

    chaos_soft = np.array([to_soft_label(e) for e in chaos_data])
    chaos_hard = np.array([to_hard_label(e) for e in chaos_data])
    chaos_premises = [e['example']['premise'] for e in chaos_data]
    chaos_hypotheses = [e['example']['hypothesis'] for e in chaos_data]

    print(f"Extracting {eval_dataset_name} logits...")
    chaos_logits = get_nli_logits(model, chaos_premises, chaos_hypotheses, tokenizer, DEVICE, batch_size=eval_batch_size, max_length=max_length)

    # Fit temperatures
    T_hard = fit_temperature_hard(val_logits, val_labels)

    # For soft oracle: use first half of ChaosNLI for fitting
    n_half       = len(chaos_logits) // 2
    oracle_logits = chaos_logits[:n_half]
    oracle_soft   = chaos_soft[:n_half]
    T_soft        = fit_temperature_soft(oracle_logits, oracle_soft)

    print(f"T_star (hard): {T_hard:.4f}  |  T_star (soft, oracle): {T_soft:.4f}")

    # Evaluate on second half of ChaosNLI
    eval_logits = chaos_logits[n_half:]
    eval_hard   = chaos_hard[n_half:]
    eval_soft   = chaos_soft[n_half:]

    # Persist logits for post-hoc analysis / re-fitting
    model_short = model_name.replace('/', '_')
    eval_slug = eval_dataset_name.lower().replace('-', '_')
    logits_prefix = f"{model_short}_{eval_slug}_seed{seed}"
    save_logits_bundle(
        logits_prefix,
        val_logits, val_labels,
        oracle_logits, oracle_soft,
        eval_logits, eval_hard, eval_soft,
    )

    probs_uncal   = scipy_softmax(eval_logits,             axis=1)
    probs_ts_hard = scipy_softmax(eval_logits / T_hard,    axis=1)
    probs_ts_soft = scipy_softmax(eval_logits / T_soft,    axis=1)

    # Isotonic regression baselines (hard-val and soft-oracle), same splits as TS
    iso_hard = MulticlassIsotonicCalibrator().fit_hard(val_logits, val_labels)
    iso_soft = MulticlassIsotonicCalibrator().fit_soft(oracle_logits, oracle_soft)
    probs_iso_hard = iso_hard.predict_proba(eval_logits)
    probs_iso_soft = iso_soft.predict_proba(eval_logits)

    results = {
        'model':            model_name,
        'dataset':          eval_dataset_name,
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
        # Isotonic regression fields (additive; existing TS fields unchanged)
        'iso_hard_ece':     compute_ece(probs_iso_hard, eval_hard),
        'iso_soft_ece':     compute_ece(probs_iso_soft, eval_hard),
        'iso_hard_bs_soft': compute_brier_soft(probs_iso_hard, eval_soft),
        'iso_soft_bs_soft': compute_brier_soft(probs_iso_soft, eval_soft),
    }
    results['gap'] = results['ts_hard_bs_soft'] - results['ts_soft_bs_soft']
    results['iso_gap'] = results['iso_hard_bs_soft'] - results['iso_soft_bs_soft']

    print(f"\n===== {eval_dataset_name} RESULTS =====")
    for k, v in results.items():
        print(f"  {k:25s}: {v:.6f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    # Save results to file
    results_root.mkdir(parents=True, exist_ok=True)
    fname = results_root / f"results_{model_short}_{eval_slug}_seed{seed}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {fname}")

    # Clean up model to free GPU memory
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Model weight checkpoints from experiment runs are not retained (disk);
    # logits bundles under results/logits/ are saved instead.
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run language experiments.")
    parser.add_argument('--model_name', type=str, choices=['distilbert-base-uncased', 'bert-base-uncased', 'bert-large-uncased', 'all'], default='all')
    parser.add_argument('--dataset_type', type=str, choices=['SNLI', 'MNLI', 'all'], default='all')
    parser.add_argument('--seed', type=str, default='all', help="Seed (e.g. 42, 123, 456) or 'all'")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Directory for result JSON files (default: results/raw)")
    args = parser.parse_args()

    models_to_run = [args.model_name] if args.model_name != 'all' else ['distilbert-base-uncased', 'bert-base-uncased', 'bert-large-uncased']
    datasets_to_run = [args.dataset_type] if args.dataset_type != 'all' else ['SNLI', 'MNLI']
    
    if args.seed == 'all':
        seeds_to_run = [42, 123, 456]
    else:
        seeds_to_run = [int(args.seed)]

    results_root = args.results_dir or str(DEFAULT_RESULTS_ROOT)

    for m in models_to_run:
        for d in datasets_to_run:
            for s in seeds_to_run:
                try:
                    run_language_experiment(m, d, s, epochs=args.epochs, results_root=results_root)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"FAILED: model={m}, dataset={d}, seed={s}. Error: {str(e)}", file=sys.stderr)
