# arXiv Submission Guide

**Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions**

## Upload these files

Primary (recommended): upload the source tarball

- `Temperature_Scaling_arXiv_source.tar.gz`

Or upload individually from `arxiv_bundle/`:

- `main.tex`
- `references.bib`

Optional: attach `main.pdf` / `Temperature_Scaling_Is_Not_Enough.pdf` for visual check.

## Compile locally

```bash
cd paper/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Metadata for arXiv

| Field | Value |
|-------|-------|
| Title | Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions |
| Author | Wisdom Dogah |
| Affiliation | University of Mines and Technology (UMaT), Tarkwa, Ghana; BlackMatrix AI Research, Accra, Ghana |
| Email | wisdom@blackmatrix.io |
| Primary category | **cs.LG** (Machine Learning) |
| Secondary / cross-list | **stat.ML**; optionally **cs.CL** and/or **cs.CV** |
| About cs.AI | You *can* add **cs.AI** as a cross-list, but arXiv defines cs.AI as AI *excluding* Machine Learning (which has its own category). Moderators often reclassify ML calibration papers to **cs.LG**. Prefer **cs.LG** as primary. |
| Comments | Source, experiment scripts, and per-seed metrics: https://github.com/dogahwisdom/temperature-scaling-research |

## Integrity checklist (pre-submit)

- [x] Vision Table 1 numbers match verified `results/raw/` (ResNet-18 seed 42 recalibrated from cached logits)
- [x] Language Table 2 uses corrected BERT-large SNLI from independent checkpoint (in `results/raw/`)
- [x] Methods match released code (30 vision epochs; 10k/1-epoch language adaptation)
- [x] ChaosNLI-M described as matched-domain (not SNLI-to-MNLI mismatch)
- [x] Isotonic baseline included (Table 3); no fabricated gap reversals
- [x] Bibliography entries are real papers with DOI/arXiv where available (19 refs)
- [x] No em dashes in manuscript source or PDF text
- [x] Traxia-style professional article layout (geometry, natbib, hyperref)
- [ ] You have reviewed the PDF page-by-page before clicking Submit

## Archive

Older Word/PDF drafts are in `paper/archive/`. The working manuscript is `main.tex` + `main.pdf`.
