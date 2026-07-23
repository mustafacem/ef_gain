# Learned patches break the restoration ceiling (idea #4)

Every patch in the main study only *restored* information already in the fp16
model: the closed-form solve reconstructs each layer's output locally. That is
a hard ceiling. This experiment tests whether a patch **trained end-to-end**
against the fp16 teacher does better -- it can compensate for quantization
error that propagates *across* layers, which local restoration cannot.

## Setup

- Base: mixed attention@4 / MLP@3, full surface, frozen.
- Patch surface: `mlp.down_proj` only (where the atlas found task-structure;
  also what fits in 6 GB for training).
- Selection: OBS top-k, 5% of down_proj groups (Fisher/task-loss scoring was
  tested and found *worse* -- consistent with the study's "scoring is not the
  bottleneck").
- Learned patch: a sparse trainable residual on the selected groups, trained
  with KL(fp16 teacher || student) for 4 epochs, LR 3e-4, grad-clip 1.0, then
  quantized to 3 bits. Matched to the solved patch at the same groups/bytes.
- 0.5B model, 24 calibration sequences, 160 tokens.

## Result (perplexity, matched 3.43 bits/weight)

| task | base (3.37b) | solved patch | **learned patch (3-bit)** | uniform-4 (4.25b) | fp16 |
|---|---|---|---|---|---|
| math | 5.85 | 5.97 | **5.11** | 4.73 | 4.49 |
| code | 8.02 | 7.87 | **7.67** | 5.47 | 4.94 |

Derived:

| task | learned vs solved | learned vs base | % of base→uniform gap closed |
|---|---|---|---|
| math | **+0.86 ppl** | +0.74 ppl | **66%** |
| code | +0.20 ppl | +0.35 ppl | 14% |

## Findings

1. **Learning breaks the restoration ceiling.** The learned patch beats the
   solved/restored patch on *both* tasks -- decisively on math (+0.86), modestly
   on code (+0.20). On math the solved patch actually *hurt* (5.97 > base 5.85)
   while the learned patch helped a lot (5.11). This is the first clearly
   positive patch result in the project.

2. **The benefit is strongly task-dependent.** Math (66% gap closed) vs code
   (14%). Math is also where task-conditioning was strongest in the main study,
   so the two findings are consistent: patches pay most where a task's
   structure departs most from the generic.

3. **Quantizing the learned delta to 3 bits is free.** learned_3bit vs
   learned_fp16 differ by <0.02 ppl on both tasks. The concern that quantizing
   a continuously-trained delta would destroy it is refuted.

4. **But learned patches still do not beat uniform-4 on absolute quality.**
   math 5.11 vs 4.73, code 7.67 vs 5.47. They close much more of the gap than
   restoration, at lower memory (3.43 vs 4.25 b/w), but do not reach uniform's
   quality. The frontier verdict of the main study stands; what changes is that
   *learned* patches are a materially better point on the sub-integer-bit
   frontier than *restored* ones.

## Honest caveats

- Two tasks, single seed, 0.5B, down_proj-only, short sequences (6 GB limit).
- This changes the problem setting: training-free PTQ -> a QLoRA-adjacent
  trained residual. The comparison to *uniform quantization* remains fair
  (matched bits), but "no new training" is no longer a property of the method.
- The initial-KL diagnostic (0.96-0.99, matching the base's fp16 divergence)
  confirms the training mechanism is correct; the earlier 83-ppl blow-up was a
  too-high learning rate (2e-3), not a bug.

## Reproduce

```
SQ_TASKS=math SQ_EPOCHS=4 SQ_TRAIN_SEQ=24 SQ_TRAIN_TOK=160 \
  SQ_COVERAGE=0.05 SQ_LR=3e-4 python scripts/learned_patch.py
# code: same with SQ_TASKS=code SQ_OUT=results/learned_patch_code.json
```

---

## UPDATE: the perplexity win is a MIRAGE (GSM8K reversal)

Pushing the learned patch to convergence (128 seqs, 12 epochs) drove math to
**perplexity parity with uniform-4 at 19% lower memory**: learned 4.77 @ 3.43
b/w vs uniform-4 4.73 @ 4.25 b/w. On perplexity, the breakthrough looked real.

**GSM8K accuracy destroys it:**

| config (math) | bits/w | perplexity | GSM8K acc |
|---|---|---|---|
| base (mixed) | 3.37 | 5.51 | 12.5% |
| learned patch | 3.43 | **4.77** | **8.5%** |
| uniform-4 | 4.25 | 4.73 | 23.5% |
| fp16 | - | 4.49 | 34.5% |

The learned patch scores **8.5%** GSM8K -- **below the unpatched base (12.5%)**
and far below uniform-4 (23.5%), p=0.000. KL distillation lowered perplexity by
matching the teacher's output distribution on calibration text while destroying
the reasoning GSM8K needs. **Training harder for lower perplexity made accuracy
worse.**

Harness validated: base=12.5% and uniform-4=23.5% match prior independent
measurements exactly, so 8.5% is trustworthy.

**Conclusion:** learned patches are NOT a win. They are the sharpest instance of
this project's central finding -- perplexity did not merely understate damage,
it pointed the opposite way. Code never even reached perplexity parity (7.37 vs
5.47), confirming the effect is also task-dependent (only lightly-damaged bases
reach perplexity parity, and even that parity is hollow on accuracy).

The KL-only objective is the likely culprit; a task-loss-aligned objective is
untested and might behave differently. But as tested, the learned patch fails
harder than the restored one on the metric that matters.
