import json
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_DIR = REPO_ROOT / "results" / "raw"
TABLES_DIR = REPO_ROOT / "results" / "tables"

def aggregate_results():
    all_files = glob.glob(str(RAW_RESULTS_DIR / 'results_*.json'))
    all_results = []
    for f in all_files:
        try:
            with open(f) as fp:
                all_results.append(json.load(fp))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Total result files found: {len(all_results)}")

    # Group by model and dataset
    grouped = defaultdict(list)
    for r in all_results:
        # For vision models, dataset is 'CIFAR-10H' (not explicitly in results sometimes, so default to 'CIFAR-10H')
        dataset = r.get('dataset', 'CIFAR-10H')
        key = (r['model'], dataset)
        grouped[key].append(r)

    metrics_to_report = [
        'test_accuracy',
        'uncal_ece', 'ts_hard_ece', 'ts_soft_ece',
        'uncal_bs_soft', 'ts_hard_bs_soft', 'ts_soft_bs_soft',
        'gap', 'T_star_hard', 'T_star_soft'
    ]

    print("\n===== AGGREGATED RESULTS (mean +/- std) =====\n")
    
    final_table = {}
    summary_text = ""

    for key in sorted(grouped.keys()):
        model, dataset = key
        runs = grouped[key]
        print(f"MODEL: {model} | DATASET: {dataset} | RUNS: {len(runs)}")
        summary_text += f"MODEL: {model} | DATASET: {dataset} | RUNS: {len(runs)}\n"
        
        row = {'model': model, 'dataset': dataset, 'n_runs': len(runs)}
        
        for m in metrics_to_report:
            vals = [r[m] for r in runs if m in r]
            if vals:
                mean_v = np.mean(vals)
                std_v  = np.std(vals)
                row[f'{m}_mean'] = float(mean_v)
                row[f'{m}_std']  = float(std_v)
                print(f"  {m:25s}: {mean_v:.4f} +/- {std_v:.4f}")
                summary_text += f"  {m:17s}: {mean_v:.4f} +/- {std_v:.4f}\n"
            else:
                print(f"  {m:25s}: N/A")
                summary_text += f"  {m:17s}: N/A\n"
        print()
        summary_text += "\n"
        final_table[f"{model}_{dataset}"] = row

    # Save final aggregated table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / 'final_results.json', 'w') as f:
        json.dump(final_table, f, indent=2)
    print("All aggregated results saved to results/tables/final_results.json")

    # Save summary text file
    with open(TABLES_DIR / 'final_results_summary.txt', 'w') as f:
        f.write(summary_text)
    print("Plain text summary saved to results/tables/final_results_summary.txt")

if __name__ == "__main__":
    aggregate_results()
