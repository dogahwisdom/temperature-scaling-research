"""Fine-tune an independent bert-large-uncased SNLI checkpoint from the base model.

No public HF bert-large-uncased SNLI checkpoint exists that is not MNLI-derived
(or is cased / sentence-embedding / undocumented). This script starts from
`bert-large-uncased` and applies the same recipe used in train_bert.py:
10,000-example subset, 1 epoch, lr 2e-5, AdamW, with the large-model freezing
and micro-batch settings.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
from pathlib import Path

# Local repo has a datasets/ data folder that shadows HuggingFace `datasets`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [
    p for p in sys.path
    if p not in ("", ".") and Path(p).resolve() != _REPO_ROOT.resolve()
]

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = _REPO_ROOT
DEFAULT_OUT = REPO_ROOT / "checkpoints" / "bert-large-uncased-snli-independent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    seed = args.seed
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    base_id = "bert-large-uncased"
    print(f"Fine-tuning {base_id} on SNLI -> {out_dir}")
    print(f"Device: {DEVICE} | seed={seed} | epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=3
    ).to(DEVICE)

    # Same large-model freezing as train_bert.py
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(20):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
    print("Froze embeddings and first 20 encoder layers.")

    train_raw = load_dataset("snli", split="train")
    train_raw = train_raw.filter(lambda x: x["label"] != -1)
    train_raw = train_raw.shuffle(seed=seed).select(range(10000))

    max_length = 80
    batch_size = 2
    grad_accum_steps = 16

    def tokenize_fn(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_tok = train_raw.map(tokenize_fn, batched=True, batch_size=1000)
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            with torch.cuda.amp.autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * grad_accum_steps
            if (step + 1) % 500 == 0:
                print(
                    f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                    f"Loss: {total_loss/(step+1):.4f}"
                )
        print(
            f"Epoch {epoch+1}/{args.epochs} complete. "
            f"Average Loss: {total_loss/len(train_loader):.4f}"
        )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "base_model": base_id,
        "dataset": "SNLI",
        "train_subset_size": 10000,
        "epochs": args.epochs,
        "lr": 2e-5,
        "optimizer": "AdamW",
        "seed": seed,
        "frozen_layers": "embeddings + encoder.layers[0:20]",
        "note": (
            "Independent SNLI fine-tune from bert-large-uncased (NOT from any "
            "MNLI checkpoint). Created to replace the MNLI-pretrained confound "
            "in ('bert-large-uncased', 'SNLI') model_mapping."
        ),
    }
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved independent SNLI checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
