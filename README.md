# ef_gain — selfquant: task-conditional quantization patches

Can a shared, aggressively-quantized base model plus small **"precision
patches"** (selected weight groups restored at higher precision) beat ordinary
uniform quantization? This repo is the full investigation — code, experiments,
and an honest write-up including the parts that failed and the claims that were
retracted.

**Short answer:** the patch mechanism is real and measurable, but uniform
quantization remains the better use of a bit on absolute quality. Patches win
only by filling the sub-integer-bit gaps uniform cannot occupy (3.25→5.25
bits/weight), and most of the practical benefit comes from a cheap
mixed-precision base rather than the patches themselves.

Run on Qwen2.5-0.5B / 1.5B / 7B-Instruct on a single 6 GB laptop GPU.

## Start here

- **[`results/report/SUMMARY.md`](results/report/SUMMARY.md)** — master index:
  what holds up, what was retracted, and why.
- **[`results/report/gsm8k_findings.md`](results/report/gsm8k_findings.md)** —
  the two most important results: the parity claim confirmed on real GSM8K
  accuracy, and the finding that perplexity understates quantization damage
  ~10× vs. downstream accuracy.
- **[`results/report/RUNNING_NOTES.md`](results/report/RUNNING_NOTES.md)** —
  the final Pareto frontier across 2.25→5.25 bits, in both perplexity and
  accuracy, plus calibration-seed variance.

## Headline findings

1. **Usable compression floor is ~3.25 bits/weight.** Below it every config is
   random on GSM8K; a 2-bit base is too destroyed for any affordable patch to
   rescue.
2. **The robust win is mixed precision** — one extra bit on attention (~12% of
   a full-surface model). Cheap, holds on accuracy, dominates every efficiency
   ranking.
3. **Patches add a small, real increment** — a mixed 3-bit base + 3-bit patch
   ties uniform 4-bit at slightly lower memory on *both* perplexity (p=0.406)
   and GSM8K accuracy (23.0% vs 23.5%). Confirmed twice.
4. **What made the design work:** storing corrections at 3–4 bits instead of
   fp16 (2.13× better, p=1.000); a closed-form activation-space correction that
   reuses the quantizer's own Hessian; and a dense bitmap mask encoding.
5. **What didn't:** solve/quantize iteration, residual-aware re-selection,
   attention-weighted budget allocation, and scaling past 0.5B — all rejected
   with evidence. See SUMMARY.md §2 for four retracted claims and why.

## Layout

```
src/selfquant/
  quant/       rtn.py, gptq.py (own GPTQ w/ candidate-aware hold-out), streaming.py (>VRAM models)
  patch/       residual.py (closed-form solve + OBS scoring), varbit.py (N-bit patches, bitmap), apply/build/compose
  sensitivity/ activation stats, Fisher, AWQ/OBS group scores
  analysis/    overlap atlas, proxy validation
  data/        longform.py (2048-token calibration + Hessian conditioning report)
scripts/       one per experiment (build_*, run_atlas, *_pareto, gsm8k_*, full_surface, ...)
results/       *.json experiment outputs (kept); report/*.md findings (kept)
tests/         65 tests
```

## Reproduce

```bash
pip install -e .
python scripts/build_longform_calibration.py     # regenerate calibration (needs HF datasets)
python scripts/full_surface.py                   # full-surface base + patch comparison
python scripts/gsm8k_frontier.py                 # accuracy across the frontier
python -m pytest tests/ -q                        # 65 tests
```

Large artifacts (calibration `.pt`, patch `.safetensors`, raw logs) are
gitignored; the build scripts above regenerate them.

## Caveats (quantified in-repo)

- Perplexity understates quantization damage ~10× vs. GSM8K accuracy — rankings
  hold across metrics, but absolute "gap closed" figures are optimistic.
- Calibration-seed variance is up to 3.35% (math); single-seed differences
  smaller than that are unproven.
- Fake quantization throughout (dequantized values); memory and quality are
  measured, no wall-clock speedup is claimed.
