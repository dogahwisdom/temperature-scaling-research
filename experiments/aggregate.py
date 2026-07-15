import json
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_RESULTS_DIR = REPO_ROOT / "results" / "raw"
TABLES_DIR = REPO_ROOT / "results" / "tables"

def aggregate_results(raw_results_dir=None, out_prefix="final_results"):
    raw_results_dir = Path(raw_results_dir) if raw_results_dir else DEFAULT_RAW_RESULTS_DIR
    all_files = glob.glob(str(raw_results_dir / 'results_*.json'))
    all_results = []
    for f in all_files:
        try:
            with open(f) as fp:
                all_results.append(json.load(fp))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Total result files found: {len(all_results)} in {raw_results_dir}")

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
        'gap', 'T_star_hard', 'T_star_soft',
        # Isotonic regression (Task 2)
        'iso_hard_ece', 'iso_soft_ece',
        'iso_hard_bs_soft', 'iso_soft_bs_soft',
        'iso_gap',
    ]

    print("\n===== AGGREGATED RESULTS (mean +/- std) =====\n")
    
    final_table = {}
    summary_text = ""
    gap_comparison_rows = []

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

        if 'gap_mean' in row or 'iso_gap_mean' in row:
            gap_comparison_rows.append({
                'model': model,
                'dataset': dataset,
                'ts_gap_mean': row.get('gap_mean'),
                'ts_gap_std': row.get('gap_std'),
                'iso_gap_mean': row.get('iso_gap_mean'),
                'iso_gap_std': row.get('iso_gap_std'),
            })

    # Save final aggregated table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = TABLES_DIR / f'{out_prefix}.json'
    out_txt = TABLES_DIR / f'{out_prefix}_summary.txt'
    out_gap = TABLES_DIR / f'{out_prefix}_ts_vs_iso_gap.json'
    out_gap_txt = TABLES_DIR / f'{out_prefix}_ts_vs_iso_gap.txt'

    with open(out_json, 'w') as f:
        json.dump(final_table, f, indent=2)
    print(f"All aggregated results saved to {out_json}")

    with open(out_txt, 'w') as f:
        f.write(summary_text)
    print(f"Plain text summary saved to {out_txt}")

    with open(out_gap, 'w') as f:
        json.dump(gap_comparison_rows, f, indent=2)

    gap_lines = ["model\tdataset\tts_gap_mean\tts_gap_std\tiso_gap_mean\tiso_gap_std\n"]
    for r in gap_comparison_rows:
        gap_lines.append(
            f"{r['model']}\t{r['dataset']}\t"
            f"{r['ts_gap_mean'] if r['ts_gap_mean'] is not None else 'N/A'}\t"
            f"{r['ts_gap_std'] if r['ts_gap_std'] is not None else 'N/A'}\t"
            f"{r['iso_gap_mean'] if r['iso_gap_mean'] is not None else 'N/A'}\t"
            f"{r['iso_gap_std'] if r['iso_gap_std'] is not None else 'N/A'}\n"
        )
    with open(out_gap_txt, 'w') as f:
        f.writelines(gap_lines)
    print(f"TS vs isotonic gap table saved to {out_gap} and {out_gap_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory of per-seed result JSON files')
    parser.add_argument('--out_prefix', type=str, default='final_results',
                        help='Output filename prefix under results/tables/')
    args = parser.parse_args()
    aggregate_results(raw_results_dir=args.results_dir, out_prefix=args.out_prefix)
