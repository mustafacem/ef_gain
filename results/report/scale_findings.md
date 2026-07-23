# Scale test (7B) — and a correction to the "scale-gated" conclusion

## What was claimed before

Report §6 concluded that patches were **scale-gated**: uniform quantization was
2.4–3.0× more byte-efficient at 0.5B but only 1.3–2.1× at 1.5B, so the gap
looked like it was closing with model size and might cross over at 7B.

## What the 7B run shows

It does not cross over, and the trend I read was largely an artifact.

**Matched-depth comparison** (mid-depth `o_proj`, 3-bit base, k=5%-equivalent
budget, best patch precision at each scale):

| model | layer | patch %/bit | uniform %/bit | uniform advantage |
|---|---|---|---|---|
| 0.5B | layer12 / 24 | 75.2 | 78.8 | **1.05×** |
| 1.5B | layer14 / 28 | 81.2 | 78.6 | **0.97×** (patch wins) |
| 7B   | layer14 / 28 | 70.0 | 78.6 | **1.12×** |

Flat. No trend. And **layer-to-layer variation within one model exceeds the
variation between models** — at 1.5B the three probes span 0.97× to 1.38×,
wider than the 1.05→1.12 spread across a 14× difference in model size.

## Why the earlier trend was misleading

The 0.5B and 1.5B numbers in §6 were not measuring the same thing. 0.5B used
**fp16 patches**; the 1.5B figure quoted the **hybrid** (sparse + low-rank).
Different patch designs at different scales, compared as if the only variable
were scale.

## What actually mattered: patch precision, not scale

Holding scale fixed and varying only patch precision:

| model | layer | fp16 patch | best patch | improvement |
|---|---|---|---|---|
| 1.5B | layer14.o_proj | 27.7% | 51.8% (3-bit) | **1.9×** |
| 7B   | layer14.o_proj | 24.5% | 44.6% (4-bit) | **1.8×** |

Storing the correction at 3–4 bits instead of fp16 is worth ~1.9×, at every
scale tested. That single change moves patches from a 2.4–3.0× deficit to
roughly parity (0.97–1.23×) — far more than scaling the model from 0.5B to 7B
did.

The optimum is a genuine interior one: **2-bit is worse than 3-bit**, sharply
so on `o_proj` (1.5% vs 17.4% error removed at 0.5B). My earlier extrapolation
from the monotonic 4 < 6 < 8 < 16 trend — "lower is always better" — was wrong.

## PPL confirmation (run after the above)

The precision finding and the parity claim were both established in
activation-space error. Re-tested on real perplexity, 0.5B, all four tasks,
patching both `o_proj` and `down_proj`, at **matched effective bits**
(patch configs 4.075 bits/weight, uniform-4 at 4.250 — the patches are the
*cheaper* option):

| task | full | 3-bit base | +fp16 patch | +3-bit patch | uniform 4-bit | gap closed: pb3 / pb16 |
|---|---|---|---|---|---|---|
| code | 3.483 | 3.920 | 3.853 | 3.816 | 3.573 | 23.8% / 15.3% |
| math | 3.203 | 3.353 | 3.322 | **3.239** | 3.213 | **76.2%** / 20.7% |
| knowledge | 5.618 | 5.940 | 5.846 | 5.787 | 5.660 | 47.4% / 29.0% |
| chat | 21.708 | 25.161 | 24.364 | 23.768 | 22.464 | 40.4% / 23.1% |

**Confirmed — the precision result transfers.** 3-bit patches beat fp16
patches at **p = 1.000 on every task**, closing 46.9% of the quantization gap
versus 22.0%, a **2.13× improvement**. That matches the 1.8–2.1× the
activation-space diagnostic predicted, so the proxy is validated as a
design-ranking tool.

**Not confirmed — parity with uniform.** Uniform 4-bit still wins on 3 of 4
tasks (p = 0.000). The exception is **math**, where the 3-bit patch is
statistically indistinguishable from uniform (p = 0.138, 3.239 vs 3.213)
while using **less memory** (4.075 vs 4.250 bits/weight). One task at parity
below the memory cost is a real result; four tasks would have been a claim.

The discrepancy with the activation-space "parity" number is explained and was
predicted: that figure came from `o_proj` probes, where patches do reach
0.97–1.12× of uniform. This run patches the whole model, and `down_proj` — far
larger in parameter count and around 1.44× on the same metric — dominates the
average. **The parity claim should be scoped to attention layers, not stated
model-wide.**

## Honest status

- **Patch precision is the real lever, and it is confirmed in perplexity:**
  storing corrections at 3 bits instead of fp16 is worth **2.13×** (p = 1.000,
  all four tasks). This holds at 0.5B, 1.5B and 7B.
- **Scale is not a lever.** The uniform-vs-patch curve is flat across a 14×
  size range once patch design is held constant (1.05× / 0.97× / 1.12× at
  matched depth). The earlier "scale-gated" reading compared fp16 patches at
  0.5B against a hybrid at 1.5B — different designs, not different scales.
- **Parity holds on attention layers and on one task, not model-wide.** With
  the whole model patched, uniform 4-bit still wins on 3 of 4 tasks; math
  reaches statistical parity at lower memory. `down_proj` (~1.44×) drags the
  average that `o_proj` (0.97–1.12×) alone would win.
- The obvious next experiment follows directly: **patch attention only, or
  allocate the budget asymmetrically toward `o_proj`**, since that is where
  the method is competitive and where uniform's advantage is smallest.

## Reproduce

```
SQ_MODEL=Qwen/Qwen2.5-7B-Instruct SQ_OUT=results/structure_7b.json \
  SQ_CALIB_SEQ=16 python scripts/structure_streaming.py
python scripts/patchbit_diagnostic.py          # patch-precision optimum
```

Raw inputs consolidated in `results/report/scale_curve_inputs.json`.

## Cost note

The closed-form solve is O(b³) in per-row patch width, and lower patch
precision buys more coverage at fixed bytes — so a 2-bit patch costs ~12× more
to *build* than an fp16 one (measured 2.7 vs 0.3 min per 48-layer pass). This
is a one-off construction cost, not a serving cost, but it dominates sweeps and
is why the precision question was settled with the activation-space diagnostic
rather than a full PPL grid.
