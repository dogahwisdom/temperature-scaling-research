# Temperature Scaling Is Not Enough

**Calibration Gaps Under Human Label Distributions**

Author: **Wisdom Dogah**  
Affiliations: University of Mines and Technology (UMaT), Tarkwa, Ghana; BlackMatrix AI Research, Accra, Ghana  
Contact: [wisdom@blackmatrix.io](mailto:wisdom@blackmatrix.io)  
Code: [github.com/dogahwisdom/temperature-scaling-research](https://github.com/dogahwisdom/temperature-scaling-research)

## Abstract (short)

Temperature scaling assumes one-hot, deterministic labels. Soft, crowd-sourced labels often violate that assumption. Across CIFAR-10H and ChaosNLI (nine model configurations), hard-label temperature scaling leaves a positive soft-label Brier gap versus a soft-label oracle (about 0.002 to 0.134). The gap is much larger in language than in vision, and the same qualitative pattern holds under multiclass isotonic regression.

## Paper (arXiv-ready)

| File | Description |
|------|-------------|
| [`paper/main.tex`](paper/main.tex) | Manuscript source |
| [`paper/references.bib`](paper/references.bib) | Bibliography |
| [`paper/main.pdf`](paper/Temperature_Scaling_Is_Not_Enough.pdf) | Compiled PDF |
| [`paper/Temperature_Scaling_arXiv_source.tar.gz`](paper/Temperature_Scaling_arXiv_source.tar.gz) | Upload this to arXiv |
| [`paper/ARXIV_SUBMISSION.md`](paper/ARXIV_SUBMISSION.md) | Submission checklist |

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Repository layout

```
temperature-scaling-research/
├── paper/                 # LaTeX manuscript + arXiv tarball
├── experiments/
│   ├── vision/train_resnet.py
│   ├── language/train_bert.py
│   ├── language/finetune_bert_large_snli.py
│   ├── utils/calibration.py
│   ├── aggregate.py
│   └── run_all.py
├── datasets/              # CIFAR-10H soft labels + ChaosNLI JSONL
├── results/
│   ├── raw/               # Verified per-seed JSON (Table 1 / language baselines)
│   └── tables/            # Aggregated summaries
├── requirements.txt
└── LICENSE                # CC-BY 4.0
```

## Setup

```bash
git clone https://github.com/dogahwisdom/temperature-scaling-research.git
cd temperature-scaling-research
pip install -r requirements.txt
```

Place `cifar10h-probs.npy` under `datasets/CIFAR-10H/` (see `datasets/CIFAR-10H/README.md`). ChaosNLI JSONL files are already in `datasets/ChaosNLI/`.

## Reproduce experiments

```bash
# Single vision run
python experiments/vision/train_resnet.py --model_size resnet18 --seed 42 --epochs 30

# Single language run
python experiments/language/train_bert.py --model_name bert-base-uncased --dataset_type SNLI --seed 42 --epochs 1

# Optional: build independent BERT-large SNLI checkpoint (not MNLI-derived)
python experiments/language/finetune_bert_large_snli.py

# Full sweep (writes under results/raw by default)
python experiments/run_all.py
python experiments/aggregate.py
```

Use `--results_dir results/raw_v2` to avoid overwriting verified `results/raw/` files. Logits are saved under `results/logits/` when present.

## Key results (verified means over seeds 42, 123, 456)

**Vision (CIFAR-10H)** — soft-label gap ≈ 0.002 / 0.003 / 0.003 for ResNet-18 / 50 / 101.

**Language (ChaosNLI)** — soft-label gap ≈ 0.045–0.134; mean language gap ≈ 0.079 (≈30× vision). ChaosNLI-S is monotonic in scale; ChaosNLI-M ordering is inconclusive (near-chance accuracy). BERT-large / ChaosNLI-S uses an independent SNLI checkpoint (`T_hard ≈ 0.980`).

**Isotonic regression** — positive soft-label gaps in all nine configurations (see `results/tables/final_results_v2_ts_vs_iso_gap.txt`).

## Citation

```bibtex
@article{dogah2026temperature,
  title   = {Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions},
  author  = {Dogah, Wisdom},
  year    = {2026},
  journal = {arXiv preprint},
  note    = {arXiv ID to be assigned}
}
```

## License

Creative Commons Attribution 4.0 International (CC-BY 4.0). See [`LICENSE`](LICENSE).

Dataset licenses: follow CIFAR-10H / ChaosNLI source terms (ChaosNLI is CC-BY-NC 4.0).

---

**Author:** Wisdom Dogah  
**Last updated:** July 2026
