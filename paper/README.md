# Paper

`precision_patches.tex` — the arXiv manuscript for this project.

Self-contained, single file, standard packages only (no external `.bib`,
no custom figures). Compiles on arXiv/Overleaf out of the box.

## Build

```bash
pdflatex precision_patches.tex
pdflatex precision_patches.tex   # second pass resolves \ref / \cite
```

or with latexmk:

```bash
latexmk -pdf precision_patches.tex
```

## Numbers

Every figure in the paper is drawn from the JSON result files in `../results/`.
Key sources:

| claim | file |
|---|---|
| task-sensitivity atlas (0.246 signal) | `atlas_qwen0.5b.json` |
| GSM8K accuracy frontier | `gsm8k_frontier.json` |
| pb3 vs fp16 (2.13×, p=1.000) | `ppl_confirm.json` |
| patches vs mixed precision | `allocation_showdown.json`, `full_surface.json` |
| PPL vs accuracy discrepancy | `gsm8k_check.json` + `full_surface.json` |
| scale (flat curve) | `structure_{1.5b_stream,7b}.json` |
| bitmap A/B + seed variance | `bitmap_ab.json` |
