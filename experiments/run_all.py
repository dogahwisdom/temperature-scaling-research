import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
python_bin = sys.executable

# Vision runs
vision_models = ['resnet18', 'resnet50', 'resnet101']
vision_seeds = [42, 123, 456]

# Language runs
language_models = ['distilbert-base-uncased', 'bert-base-uncased', 'bert-large-uncased']
language_datasets = ['SNLI', 'MNLI']
language_seeds = [42, 123, 456]

print("Starting Master Experiment Script...")

# 1. Vision Experiments
for m in vision_models:
    for s in vision_seeds:
        expected_file = REPO_ROOT / "results" / "raw" / f"results_{m}_seed{s}.json"
        if os.path.exists(expected_file):
            print(f"Skipping vision model={m}, seed={s} (file {expected_file} already exists)")
            continue
        
        cmd = [python_bin, str(REPO_ROOT / "experiments" / "vision" / "train_resnet.py"), "--model_size", m, "--seed", str(s), "--epochs", "30"]
        print(f"\nRunning: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"ERROR: Vision run failed for model={m}, seed={s}", file=sys.stderr)

# 2. Language Experiments
for m in language_models:
    for d in language_datasets:
        for s in language_seeds:
            model_short = m.replace('/', '_')
            eval_name = 'chaosnli_s' if d == 'SNLI' else 'chaosnli_m'
            expected_file = REPO_ROOT / "results" / "raw" / f"results_{model_short}_{eval_name}_seed{s}.json"
            if os.path.exists(expected_file):
                print(f"Skipping language model={m}, dataset={d}, seed={s} (file {expected_file} already exists)")
                continue

            cmd = [python_bin, str(REPO_ROOT / "experiments" / "language" / "train_bert.py"), "--model_name", m, "--dataset_type", d, "--seed", str(s), "--epochs", "1"]
            print(f"\nRunning: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"ERROR: Language run failed for model={m}, dataset={d}, seed={s}", file=sys.stderr)

# 3. Aggregate results
print("\nRunning aggregation...")
subprocess.run([python_bin, str(REPO_ROOT / "experiments" / "aggregate.py")])
print("Master Experiment Run Complete!")
