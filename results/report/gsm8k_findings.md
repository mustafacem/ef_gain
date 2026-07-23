# GSM8K downstream accuracy — the parity claim, and a warning about perplexity

200 GSM8K test problems, 4-shot, greedy, exact match. Full surface (358M
weights quantized). Answer extraction spot-checked by hand before the run.

## 1. The parity claim survives on real accuracy

| config | bits/w | GSM8K acc |
|---|---|---|
| fp16 | 16.000 | 34.5% |
| uniform 3-bit | 3.250 | 8.5% |
| mixed attn@4 + mlp@3 | 3.373 | 12.5% |
| **mix43 + pb3 patch @5%** | **4.198** | **23.0%** |
| uniform 4-bit | 4.250 | 23.5% |

The composite is **statistically tied with uniform 4-bit** (paired bootstrap
p = 0.406) while using **less memory** — 4.198 vs 4.250 bits/weight. It beats
the mixed base alone at p = 0.998 and the 3-bit base at p = 1.000.

This was the single most important missing number: the "tied at lower memory"
result previously rested only on perplexity. It now holds on task accuracy.

Caveat kept from before the run: at ~30% accuracy with n=200 this test
resolves differences of roughly ±13%. A 0.5pp gap is far inside that, so the
honest statement is "indistinguishable at this sample size", not "identical".

## 2. Perplexity badly understates quantization damage

Same configs, same task, both metrics:

| config | PPL vs fp16 | GSM8K acc vs fp16 |
|---|---|---|
| uniform 4-bit | **+2.8%** | **−31.9%** |
| mix43 + pb3 @5% | +10.6% | −33.3% |
| mixed attn@4 + mlp@3 | +19.7% | −63.8% |
| uniform 3-bit | +31.3% | **−75.4%** |

A 2.8% perplexity increase costs a **third of GSM8K accuracy**. A 31%
perplexity increase costs **three quarters** of it.

This is the most consequential finding here and it cuts against the project's
own reporting. **Every "gap closed" percentage in SUMMARY.md, phase3_findings.md
and allocation_findings.md is perplexity-based and therefore optimistic** —
not wrong as a ranking (the ordering of configs matches on both metrics), but
badly wrong as a measure of how much quality survives quantization.

Concretely: uniform 4-bit "closes 89.3% of the gap" on perplexity while losing
**32% of downstream accuracy**. Those two statements describe the same model.

## 3. What this changes

- The **ranking** conclusions stand: config order is identical under both
  metrics, so comparisons between designs remain valid.
- The **absolute** conclusions do not: 3-bit quantization is far more
  destructive than the perplexity tables implied (8.5% vs 34.5% on GSM8K),
  and the sub-4-bit band is a harsher place than it looked.
- The composite's value is *better* supported than before, not worse — it
  matches uniform 4-bit on accuracy at lower cost, which was the claim.

## Reproduce

```
python scripts/gsm8k_check.py      # -> results/gsm8k_check.json
```
