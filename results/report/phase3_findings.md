# Phase 3 — three ideas gated, one promoted, full-surface re-baseline

## What changed vs the current implementation

| change | status | effect |
|---|---|---|
| **F. Bitmap mask encoding** | **promoted** | 12/12 probes, median **1.088×**. Free. |
| C. Solve/quantize iteration | rejected | 1.000× (0/12 probes) at matched bytes |
| E. Residual-aware re-selection | rejected | 1.005× (1/12 probes) |
| **Composite: mixed base + patch** | **confirmed** | +11 to +29pp over a uniform base at equal budget |
| Full-surface quantization | re-baselined | 358M weights (all 7 linear types) vs 124M before |

## 1. Idea F — the one that worked, and why the failures produced it

Ideas C and E were rejected **exactly as the gate was designed to catch**:
both trade coverage for fidelity, and the patch-precision sweep had already
established that coverage dominates. C's failure was pre-registered in the
script docstring before running — two 3-bit rounds is ~6-bit fidelity on half
the coverage, and pb6 already lost to pb3. Measured: rounds=2 removes 25.2%
vs rounds=1's 35.6% at matched bytes.

That failure pointed at the fix. If coverage is what matters, buy it cheaper.
Per patched group at pb3:

```
values 48 B  +  qparams 4 B  +  explicit (row,group) index 8 B  =  60 B
                                └── 13.3% of every group, paid per group
```

A dense **bitmap** over all groups costs `total_groups/8` bytes *flat*,
regardless of how many are selected. It wins above ~1.5% coverage; these
patches run at 22%. Same budget now buys **25.1% coverage instead of 22.0%**
(1.143×) — pure accounting, no new computation, no approximation.

Unanimous across all four layer shapes probed (o_proj, down_proj, gate_proj,
q_proj). It is now the default in `groups_for_budget(..., bitmap=True)` /
`patch_bytes(..., total_groups=)`.

## 2. Full-surface re-baseline (these numbers supersede all earlier ones)

All seven linear types quantized — 358M weights, vs the 124M (35%) of every
prior experiment. Earlier figures let 65% of weights ride along in fp16, which
inflated quality and undercut every bits/weight claim.

Gap from uniform-3-bit to fp16 recovered, mean of 4 tasks:

| config | bits/w | gap closed | **%/extra bit** |
|---|---|---|---|
| **mixed attn@4 + mlp@3** | 3.373 | 41.2% | **334.7** ← most efficient by far |
| mix43 + pb3 patch @1% | 3.538 | 50.2% | 174.3 |
| u3 + pb3 patch @1% | 3.415 | 21.0% | 127.0 |
| mix43 + pb3 patch @2% | 3.703 | 55.9% | 123.4 |
| **uniform 4-bit** | 4.250 | **89.3%** | 89.3 ← best absolute |
| mix43 + pb3 patch @5% | 4.198 | 67.3% | 71.0 |
| uniform 5-bit | 5.250 | 98.6% | 49.3 |

## 3. The composite hypothesis is confirmed

Mixed-precision base and patches address orthogonal structure (across layer
types vs within layers), and they stack:

| budget | uniform-3 base | mixed base | gain |
|---|---|---|---|
| 1% | 21.0% | **50.2%** | +29.3pp |
| 2% | 34.1% | **55.9%** | +21.8pp |
| 5% | 55.9% | **67.3%** | +11.4pp |

`mix43+pb3@2%` reaches 55.9% at **3.703 bits** — the same quality
`u3+pb3@5%` needs **4.075 bits** for. The mixed base saves ~0.37 bits/weight
outright.

## 4. Uniform 4-bit still wins on absolute quality

89.3% at 4.250 bits vs the best composite's 67.3% at 4.198 bits, p = 0.000 on
all four tasks. Nothing at or below uniform-4's cost beats it.

**But the efficiency picture inverted on the full surface.** On attention-only
weights (~12% of the full surface, vs 15.6% of the old partial surface),
upgrading attention by one bit costs just 0.123 bits/weight and recovers 41.2%
of the gap — **334.7 %/bit, nearly 4× uniform 4-bit's 89.3 %/bit**. The four
most byte-efficient configurations all involve mixed precision, small patches,
or both; uniform 4-bit ranks fifth.

So the honest split is:
- **Maximum quality at ~4.25 bits → uniform 4-bit.** Unbeaten.
- **Maximum quality per bit → mixed precision, optionally + a 1–2% patch.**
  Dominant, by a wide margin.

## 5. Reproduce

```
python scripts/idea_gate.py      # gates C / E / F -> results/idea_gate.json
python scripts/full_surface.py   # -> results/full_surface.json
python -m pytest tests/ -q       # 65 tests
```

## 6. Honest caveats

- Perplexity only; no downstream accuracy.
- Single seed; bootstrap covers eval noise, not calibration/selection variance.
- Full-surface numbers are **not** comparable to anything in SUMMARY.md §3 —
  different weight surface, deliberately.
- The bitmap gain was measured in activation space (12/12 probes) and is baked
  into the full-surface patch configs, but was not isolated in a PPL A/B.
