import os
import subprocess
import sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
python_bin = sys.executable

# Vision runs
vision_models = ['resnet18', 'resnet50', 'resnet101']
vision_seeds = [42, 123, 456]

# Language runs
language_models = ['distilbert-base-uncased', 'bert-base-uncased', 'bert-large-uncased']
language_datasets = ['SNLI', 'MNLI']
language_seeds = [42, 123, 456]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=str(REPO_ROOT / 'results' / 'raw'))
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip runs whose result JSON already exists in results_dir')
    parser.add_argument('--vision_only', action='store_true')
    parser.add_argument('--language_only', action='store_true')
    parser.add_argument('--epochs_vision', type=int, default=30)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Master Experiment Script...")
    print(f"Results dir: {results_dir}")

    # 1. Vision Experiments
    if not args.language_only:
        for m in vision_models:
            for s in vision_seeds:
                expected_file = results_dir / f"results_{m}_seed{s}.json"
                if args.skip_existing and expected_file.exists():
                    print(f"Skipping vision model={m}, seed={s} (file {expected_file} already exists)")
                    continue

                cmd = [
                    python_bin, str(REPO_ROOT / "experiments" / "vision" / "train_resnet.py"),
                    "--model_size", m, "--seed", str(s), "--epochs", str(args.epochs_vision),
                    "--results_dir", str(results_dir),
                ]
                print(f"\nRunning: {' '.join(cmd)}")
                res = subprocess.run(cmd)
                if res.returncode != 0:
                    print(f"ERROR: Vision run failed for model={m}, seed={s}", file=sys.stderr)

    # 2. Language Experiments
    if not args.vision_only:
        for m in language_models:
            for d in language_datasets:
                for s in language_seeds:
                    model_short = m.replace('/', '_')
                    eval_name = 'chaosnli_s' if d == 'SNLI' else 'chaosnli_m'
                    expected_file = results_dir / f"results_{model_short}_{eval_name}_seed{s}.json"
                    if args.skip_existing and expected_file.exists():
                        print(f"Skipping language model={m}, dataset={d}, seed={s} (file {expected_file} already exists)")
                        continue

                    cmd = [
                        python_bin, str(REPO_ROOT / "experiments" / "language" / "train_bert.py"),
                        "--model_name", m, "--dataset_type", d, "--seed", str(s), "--epochs", "1",
                        "--results_dir", str(results_dir),
                    ]
                    print(f"\nRunning: {' '.join(cmd)}")
                    res = subprocess.run(cmd)
                    if res.returncode != 0:
                        print(f"ERROR: Language run failed for model={m}, dataset={d}, seed={s}", file=sys.stderr)

    # 3. Aggregate results
    print("\nRunning aggregation...")
    out_prefix = "final_results_v2" if "raw_v2" in str(results_dir) else "final_results"
    subprocess.run([
        python_bin, str(REPO_ROOT / "experiments" / "aggregate.py"),
        "--results_dir", str(results_dir),
        "--out_prefix", out_prefix,
    ])
    print("Master Experiment Run Complete!")

if __name__ == "__main__":
    main()
