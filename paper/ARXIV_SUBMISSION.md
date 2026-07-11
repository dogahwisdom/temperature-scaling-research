# arXiv Submission Guide

This directory contains the arXiv-ready LaTeX source files for the paper:

**Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions**

## Files

- `main.tex` - Main LaTeX document (11 KB)
- `references.bib` - BibTeX bibliography with 17 verified references (5.4 KB)
- `main.pdf` - Compiled PDF output (5 pages, 166 KB)

## Compiling the Paper

### Requirements

```bash
# On Ubuntu/Debian
sudo apt-get install texlive-full

# On macOS
brew install --cask mactex

# Verify installation
pdflatex --version
```

### Compilation

```bash
# Navigate to paper directory
cd paper/

# Compile once (basic)
pdflatex main.tex

# Compile with bibliography (recommended)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Clean up build artifacts
rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz
```

### Using Makefile (optional)

Create a `Makefile` in the paper directory:

```makefile
.PHONY: build clean

build:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz

.DEFAULT_GOAL := build
```

Then compile with: `make`

## arXiv Submission Checklist

### Before Submission

- [x] LaTeX source files present (`main.tex`, `references.bib`)
- [x] All references verified and real (17 papers, 100% confirmed)
- [x] PDF compiles without errors
- [x] Email updated to wisdom@blackmatrix.io
- [x] No Copilot or AI attribution in source code
- [x] Professional formatting (no emojis, plain text)
- [x] Experiment code for vision/language pipelines is present in repo
- [x] Per-seed metrics files backing reported table values are present
- [ ] Checkpoints/logs exist for reproducibility audit
- [ ] End-to-end rerun from clean checkout has been validated

### Submission Steps

1. Go to https://arxiv.org/submit
2. Create account or log in
3. Select category: **cs.LG** (Machine Learning)
4. Secondary category: **stat.ML** (Statistics - Machine Learning)
5. Upload files:
   - `main.tex`
   - `references.bib`
   - `main.pdf` (optional but recommended)
6. Fill metadata:
   - **Title**: Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions
   - **Authors**: Wisdom Dogah
   - **Affiliations**: University of Mines and Technology (UMaT), Tarkwa, Ghana; BlackMatrix AI Research, Accra, Ghana
   - **Email**: wisdom@blackmatrix.io
   - **Abstract**: Copy from `main.tex` (lines 18-31)
   - **Keywords**: calibration, temperature scaling, soft labels, label ambiguity, Expected Calibration Error, Brier Score, model scale, uncertainty quantification
7. Comments field:
   - "Manuscript sources, experiment scripts, and per-seed metric outputs are available at: https://github.com/dogahwisdom/temperature-scaling-research"
8. Review and submit

### After Acceptance

- arXiv will assign a paper ID (format: 2607.xxxxx)
- Update `references.bib` line 7 with actual arXiv ID:
  ```bibtex
  journal={arXiv preprint arXiv:2607.xxxxx}
  ```
- Update GitHub repository README with arXiv ID
- Share on social media and academic networks

## Paper Metadata

| Field | Value |
|-------|-------|
| Title | Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions |
| Author | Wisdom Dogah |
| Email | wisdom@blackmatrix.io |
| Affiliation 1 | University of Mines and Technology (UMaT), Tarkwa, Ghana |
| Affiliation 2 | BlackMatrix AI Research, Accra, Ghana |
| Date | July 2026 |
| Pages | 5 |
| Keywords | calibration, temperature scaling, soft labels, label ambiguity, ECE, Brier Score, model scale, uncertainty quantification |
| arXiv Category | cs.LG (Machine Learning), stat.ML (Secondary) |
| Funding | No institutional funding; experiments on freely available compute |

## Citation Format

BibTeX:
```bibtex
@article{Dogah2026,
  author = {Dogah, Wisdom},
  year = {2026},
  title = {Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions},
  journal = {arXiv preprint arXiv:2607.xxxxx}
}
```

APA:
```
Dogah, W. (2026). Temperature scaling is not enough: Calibration gaps under human label distributions. arXiv preprint arXiv:2607.xxxxx.
```

## Related Resources

- **GitHub Repository**: https://github.com/dogahwisdom/temperature-scaling-research
- **Datasets**: CIFAR-10H and ChaosNLI (see `datasets/` directory)
- **Experiments**: Runnable scripts in `experiments/` with raw outputs in `results/raw/`
- **Original Manuscript**: Temperature_Scaling_Is_Not_Enough.pdf (PDF version)

## Support

For questions about the paper or submission:
- Email: wisdom@blackmatrix.io
- GitHub Issues: https://github.com/dogahwisdom/temperature-scaling-research/issues
