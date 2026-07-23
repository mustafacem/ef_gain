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
