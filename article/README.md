# LaTeX article draft

This folder contains the first scientific article draft for the MAELIA sensitivity-analysis work.

Main files:

- `main.tex`: article source.
- `references.bib`: BibTeX references.

Suggested compilation from this folder, on a machine with LaTeX installed:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The article uses figures stored in `../figs/`, `../analysis/terrainSA_results/`, and `../analysis/decision_tree_thresholds/`.
