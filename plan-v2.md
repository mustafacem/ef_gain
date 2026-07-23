# Precision Patches — Task-Conditional Quantization with a Shared Frozen Base

**Version 2 — detailed execution plan.** Supersedes `plan.md`.

---

## 0. Thesis

One aggressively-quantized base model (2–3 bit, uniform, frozen) + small swappable
**precision patches**: the top-k weight groups restored to fp16, where k-selection is
computed from *task-specific* calibration data. Zero new parameters — a patch only
"un-loses" information the full-precision model already had.

**Positioning (the one-line pitch):** TAQ (Nov 2025) is the "full fine-tune" of
task-aware quantization — layer-level bit allocation, one static model per task.
We are the **LoRA of quantization** — group-level, one shared base, N cheap swappable
patches, exact composability story.

**What's genuinely new vs. prior art:**

| Work | Granularity | Task-aware? | Artifact |
|---|---|---|---|
| GPTQ / AWQ | group | no (generic calib) | 1 static model |
| SpQR / OWQ / SqueezeLLM | outlier weights | no | 1 static model (mixed precision baked in) |
| TAQ (2025) | **layer** | yes | 1 static model **per task** |
| Cross-calibration (Hessian) | group | partially | 1 static model |
| **Ours** | group | yes | **1 base + N swappable patches** |

---

## 1. Research questions, hypotheses, decision thresholds

**RQ1 (science, go/no-go).** How task-dependent is group-level quantization
sensitivity? Measured as overlap of top-k sensitive group sets across tasks,
**normalized against the within-task noise ceiling** (see §4.4 — this control is
missing from every prior discussion of this question and from plan v1).

**RQ2 (method).** Does a task-matched patch beat, at equal memory:
(a) a mismatched patch, (b) a task-agnostic patch, (c) a random patch (placebo),
(d) uniform quantization at matched average bits?

**RQ3 (scaling).** How does the effect move with patch budget k, base bit-width,
and model size?

**RQ4 (composition).** Can two tasks' patches be merged (union of groups under a
shared budget) and serve both tasks at once with graceful degradation?

**Hypotheses with numbers attached:**

- H1: within-task split-half overlap of top-1% groups ≥ 0.7 Jaccard (i.e., the
  sensitivity signal is stable, not calibration noise).
- H2: cross-task overlap is at least 15 Jaccard points *below* within-task overlap
  for at least one task pair (i.e., there is exploitable task signal).
- H3: own-task patch beats task-agnostic patch at equal k by ≥ 2 points (or ≥ 5%
  relative) on at least 2 of 4 tasks, at the 2-bit base.
- H4: attention layers' sensitive sets are more task-universal than MLP layers'
  (mechanistic sub-finding; free to check once the atlas exists).

**Go/no-go rule after Phase 1:**
`signal = mean(within-task overlap) − mean(cross-task overlap)`, computed at
k = 1% on the calibrated sensitivity metric that survives proxy validation (§5-E1).

- signal ≥ 0.15 → full project as planned.
- 0.05 ≤ signal < 0.15 → proceed, but shift emphasis to the composition/systems
  story (even modest specialization is valuable if patches are 20 MB).
- signal < 0.05 → pivot: the paper becomes *"Quantization-sensitive weights are
  task-universal: a controlled study"* — a clean negative result that explains
  why AWQ-style generic calibration generalizes. Still publishable; the atlas,
  the noise-ceiling methodology, and the proxy-validation experiment carry it.

---

## 2. Formal setup

Model weights partitioned into groups of 128 consecutive input-dim weights per
output row — **deliberately identical to the quantization group size**, so each
patch group carries its own scale/zero-point and restoration never touches a
neighbor's quantization parameters.

- Group set G, |G| ≈ 12M for a 1.5B model (1.5e9 / 128).
- Task-t sensitivity map: S_t : G → ℝ (per-group score, §4.2).
- Patch: P_t(k) = top-k groups of G by S_t, stored as {(tensor_name, group_idx) → fp16[128]}.
- Serving: dequantize base, scatter fp16 blocks over patched groups. Swap = scatter
  of ~30 MB → milliseconds.

**Memory arithmetic (1.5B model, 2-bit base, group-wise scales at fp16 per 128):**
- Base: 2 bits + 16/128 scale + 16/128 zero ≈ 2.25 effective bits → ~420 MB.
- 1% patch: 15M weights × 2 B = 30 MB + indices (120k group ids × 6 B ≈ 0.7 MB).
- Effective average bits with patch: 2.25 + 0.01 × (16 − 2.25) ≈ **2.39 bits**.
- Fair uniform baseline for comparison: ~2.4-bit uniform (interpolate 2/3-bit GPTQ
  results, or use a 2.5-bit method); also always report the 3-bit uniform point.

---

## 3. The coupling problem (new in v2 — this can silently kill the results)

GPTQ quantizes columns sequentially and **compensates**: after quantizing column j,
it updates not-yet-quantized columns to absorb column j's error. Consequence: if you
later restore group g to its *original* fp16 values, the rest of the matrix still
carries compensation for g's quantization error — restoration can help less than
expected or even hurt. Plan v1 ignored this entirely.

Three designs, all of which we implement (it's ~50 lines of difference):

- **A. Naive-restore (baseline design):** GPTQ base, patch = original fp16 W for
  selected groups. Coupling error present. Cheapest; measure how bad it actually is.
- **B. Candidate-aware base (recommended primary):** before quantizing, fix a
  *candidate universe* C = top-5% groups by max-over-tasks sensitivity (plus the
  task-agnostic top-5%). Run GPTQ with C's columns held out of error propagation
  (SpQR-style), then quantize C's groups with plain RTN (no compensation flows
  from them). The base still works standalone (C is quantized too, just slightly
  worse), and **patch restoration is exact** for any patch ⊆ C. One base serves
  all patches. Cost: base quality dips slightly vs. pure GPTQ — measure it.
- **C. RTN base:** round-to-nearest with group scales; zero coupling by
  construction. Scientifically cleanest, but RTN at 2-bit is usually catastrophic;
  use as the 3-bit-base sanity configuration.

**Micro-experiment (do in week 2, half a day):** restore 100 random groups in a
design-A base, compare measured Δloss vs. the no-coupling prediction. If naive
restore loses < 10% of the predicted benefit, design A is fine and the story
simplifies. Otherwise design B is the paper's method and gets a paragraph.

---

## 4. Method components

### 4.1 Base quantization

- Library: `GPTQModel` (maintained fork of AutoGPTQ) or a ~300-line own GPTQ
  implementation (preferable — we need to hook error propagation for design B,
  and owning the loop makes that trivial).
- Configs: 2-bit and 3-bit, group size 128, asymmetric, generic calibration
  (C4/RedPajama sample, 128 seqs × 2048 tokens).
- **Fake-quant throughout the research phase:** weights stored as dequantized fp16
  holding quantized values. No custom kernels; every result is about accuracy and
  memory accounting, and swap latency is measured on the scatter op. Real mixed
  kernels (SpQR-style CSR outliers) are future work — say so honestly in the
  writeup.

### 4.2 Sensitivity metrics (three, cross-validated)

For group g with weights w_g, quantized to ŵ_g, calibration set D_t:

1. **AWQ-style saliency (no gradients):**
   `S_awq(g) = Σ_{ij∈g} |W_ij| · E_{D_t}[|X_j|]` — cheap, activation-aware,
   but not quantization-error-aware.
2. **OBD/Fisher quantization score (primary candidate):**
   `S_fisher(g) = Σ_{ij∈g} E_{D_t}[(∂L/∂W_ij)²] · (W_ij − Ŵ_ij)²`
   — expected task-loss increase from quantizing exactly this group (diagonal
   second-order Taylor). This is the theoretically right objective: it weights the
   Fisher by the *actual damage* quantization does to each weight.
3. **KL-teacher variant (no labels needed):** same as (2) but L = KL(p_fp16 ‖ p_model)
   on task calibration text — measures divergence from the full-precision model
   rather than next-token loss. Robust when calibration data is unlabeled/noisy.

Implementation: one backward pass per calibration batch with per-weight grad²
accumulation (store accumulator in fp32 on CPU, layer at a time via hooks;
~6 GB for 1.5B, streamed). All three metrics from the same runs where possible.

### 4.3 Patch construction, format, swap

- Budgets: k ∈ {0.1%, 0.5%, 1%, 2%} of weights.
- Selection: global top-k across all linear layers (not per-layer quotas) —
  but *log* the per-layer allocation; it is itself an atlas result.
- Format: one `safetensors` file per patch — a flat fp16 tensor of blocks +
  int32 index tensor + JSON metadata (base hash, metric, task, k, calib seed).
  The base hash check prevents applying a patch to the wrong base — cheap
  insurance, and it makes the "swappable artifact" claim concrete.
- Swap: `Tensor.index_copy_` per layer on GPU; report median/p99 latency over
  100 swaps, patch load-from-disk time separately.

### 4.4 Overlap methodology (the RQ1 machinery)

Metrics, all reported:
- Jaccard of top-k sets, k swept over {0.1, 0.5, 1, 2, 5}% (curve, not one number).
- Spearman rank correlation over the full score vectors (threshold-free).
- Per-layer-type breakdown (attention Q/K/V/O vs. MLP up/gate/down, by depth).

**Controls (the part that makes this rigorous):**
- **Noise ceiling:** split each task's calibration set in half, compute S_t twice,
  measure within-task overlap. Cross-task overlap is only meaningful relative to
  this ceiling. (If within-task Jaccard is 0.75 and cross-task is 0.70, there is
  almost no task signal even though 0.70 "looks high".)
- **Random floor:** expected Jaccard of two random top-k sets (≈ k for small k).
- **Calibration-size sweep:** S_t at 32/64/128/256 sequences → how much data does
  a stable patch need? (Directly practical: it prices "make your own patch".)

Deliverable: the **task-sensitivity atlas** — overlap matrices + per-layer heatmaps
+ ceiling/floor bars. Standalone contribution regardless of RQ2's outcome.

---

## 5. Experiments

### E1 — Proxy validation (week 2; before trusting any sensitivity map)

The Taylor scores are proxies. Validate: sample ~200 groups spanning the score
range, for each restore it alone in the quantized model (or quantize it alone in
the fp16 model), measure actual Δloss on held-out task data; report Spearman(actual,
predicted) per metric. **The metric with the best correlation becomes the paper's
primary; the others become ablations.** If all three correlate < 0.3, stop and
debug before Phase 1 conclusions — this experiment is the guard rail that keeps
the atlas from being an artifact of a bad proxy.

### E2 — Overlap atlas (weeks 2–3) ← go/no-go gate of §1

All four tasks × validated metric(s) × {2-bit, 3-bit} error terms, with noise
ceiling and random floor. Apply the decision rule.

### E3 — Main results grid (weeks 6–8)

Per task × per base bit-width, evaluate:

| Config | What it isolates |
|---|---|
| fp16 model | ceiling |
| base only | floor |
| base + **own** patch | the method |
| base + each mismatched patch | task-specificity (RQ2a) |
| base + task-agnostic patch (generic calib, same k) | conditioning vs. generic outlier restoration (RQ2b — **the comparison reviewers will demand**) |
| base + random patch (same k) | placebo: "is it just any fp16 mass?" (RQ2c — missing from v1) |
| base + magnitude-top-k patch | "do you even need activations/gradients?" |
| uniform quant @ matched avg bits | practical value (RQ2d) |

Statistics: 3 calibration seeds per patch (report mean ± sd of the *selection*
pipeline, not just eval noise); paired bootstrap on eval sets for significance;
all configs share identical eval seeds/prompts.

### E4 — Scaling & ablations (weeks 8–10)

- Model scale: best config repeated on 3B; **one 7B–8B run in the cloud for the
  headline table** (see metric-collapse note in §6 — the 7B numbers may be the
  only benchmark-accuracy numbers that are cleanly interpretable at 2-bit).
- Base bit-width sweep: 2 vs 3 (where is the sweet spot? hypothesis: patches
  matter most where the base is most damaged, until the base is *too* damaged).
- Group size: 64 / 128 / 256.
- Design A vs B (coupling, §3).
- Patch budget sweep → Pareto curves: task metric vs. effective average bits,
  with uniform-quant points on the same axes.

### E5 — Patch composition (weeks 9–10)

Merge patches for tasks (A,B): union of groups; where budgets collide, keep
groups by max normalized score under total budget k_A + k_B (and under
min(k_A,k_B) for the hard version). Evaluate both tasks under the merged patch
vs. own-patch. Nobody has done adapter-composition for quantization — even a
modest result here is a novel section, and it is the experiment that cements the
LoRA analogy.

---

## 6. Evaluation protocol

**Tasks and data (calibration strictly from train splits, eval from test):**

| Task | Calibration (~128 × 2048 tok) | Eval |
|---|---|---|
| Code | The Stack (dedup) / MBPP-train prompts+solutions | HumanEval+, MBPP (pass@1, greedy) |
| Math | GSM8K-train or MetaMathQA | GSM8K test (8-shot, strict extraction) |
| Knowledge | NQ-train / Wikipedia | TriviaQA (0-shot), MMLU subset |
| Chat/general | UltraChat / C4 sample | WikiText-2 ppl + IFEval |

**The metric-collapse problem (new in v2):** at 2-bit, a 1–1.5B model may score
~0 on GSM8K/HumanEval, making all configs indistinguishable at the benchmark
level. Mitigations, in order:
1. **Primary continuous metric at small scale: ΔPPL on held-out task text and
   KL(fp16 ‖ patched) on task data.** Sensitive, low-variance, differentiates
   configs even when accuracy is floored.
2. Benchmark accuracies reported where above floor (3-bit base, and 3B/7B models).
3. If 1B@2-bit is floored everywhere, the headline grid moves to 3B@2-bit and
   1B@3-bit — decide by end of week 4, not in week 9.

**Hygiene:** verify calibration/eval disjointness with n-gram overlap check;
fixed lm-eval-harness version + task hashes committed to the repo; every result
row traceable to a config file.

---

## 7. Engineering plan

### Repo layout

```
selfquant/
├── pyproject.toml            # torch, transformers, safetensors, lm-eval, datasets
├── configs/                  # YAML per experiment; every run is a config
├── src/selfquant/
│   ├── quant/
│   │   ├── rtn.py            # group-wise RTN (design C)
│   │   ├── gptq.py           # own GPTQ loop, ~300 lines, fake-quant out
│   │   └── candidate_aware.py# design B: hold-out columns from compensation
│   ├── sensitivity/
│   │   ├── activations.py    # E[|X|] stats via forward hooks
│   │   ├── fisher.py         # grad² accumulation, layer-streamed to CPU
│   │   └── scores.py         # S_awq / S_fisher / S_kl → per-group score files
│   ├── patch/
│   │   ├── build.py          # top-k selection → .safetensors patch
│   │   ├── apply.py          # scatter/unscatter, base-hash check, latency timer
│   │   └── compose.py        # E5 merging
│   ├── analysis/
│   │   ├── overlap.py        # Jaccard/Spearman, noise ceiling, atlas plots
│   │   └── proxy_check.py    # E1
│   └── eval/
│       ├── run_lm_eval.py    # harness wrapper, pinned tasks
│       └── ppl_kl.py         # continuous metrics
├── scripts/                  # one CLI per pipeline stage
└── results/                  # parquet per run + plots; append-only
```

### Hardware & compute budget (rough, fp16 fake-quant)

| Item | Cost |
|---|---|
| Sensitivity map, 1.5B, one task | ~1–2 GPU·h (fwd+bwd over 128×2048, grad² hooks) |
| GPTQ quantization, 1.5B | ~0.5–1 GPU·h |
| E1 proxy validation | ~10 GPU·h (200 single-group evals, cheap fwd-only) |
| Full E3 grid, 1.5B (2 bases × ~8 configs × 4 tasks) | ~80–120 GPU·h |
| 3B repeats | ~2× a 1.5B pass |
| One 7B headline run (cloud A100) | ~40–60 GPU·h |

Everything except the 7B run fits a single 24 GB GPU (1.5B fp16 fake-quant +
grads streamed = fine; 3B needs gradient accumulation care but fits for
sensitivity since only grad² per layer is kept).

**Models:** Qwen2.5-1.5B-Instruct (primary), Llama-3.2-1B-Instruct (second
architecture — every headline claim shown on both), Qwen2.5-3B or Llama-3.2-3B
(scaling), one 7–8B (headline). Base models, not just instruct, for the ppl/KL
metrics if instruct chat templates add noise.

---

## 8. Timeline (12 weeks)

| Week | Milestone (each ends with a written artifact) |
|---|---|
| 1 | Repo scaffold; RTN + GPTQ working (ppl sanity vs. published numbers); calib/eval datasets built + disjointness check |
| 2 | Sensitivity pipeline (all 3 metrics); **E1 proxy validation memo**; §3 coupling micro-experiment memo |
| 3 | **E2 atlas + go/no-go memo** (overlap matrices, ceiling/floor, per-layer heatmaps) |
| 4 | Patch build/apply/swap + latency numbers; design A-vs-B decision; metric-floor check → freeze headline scale |
| 5 | Task-agnostic, random, magnitude baseline patches; uniform-bit baselines |
| 6–7 | E3 main grid at primary scale, 3 seeds |
| 8 | E3 on second architecture; start 3B |
| 9 | E4 ablations (bit-width, group size, budget Pareto); E5 composition |
| 10 | 7B cloud run; atlas finalized as figures |
| 11 | Writeup draft; re-run arXiv sweep for mid-project scoops |
| 12 | Polish, repo cleanup, preprint + workshop submission |

Hard rule: weeks 1–3 spend zero effort on the patch *system* — the go/no-go
science comes first (patch machinery in week 4 only matters if E2 passes).

---

## 9. Risks

1. **Task-conditioning doesn't beat generic outlier restoration** (AWQ explicitly
   claims generic salient-weight protection generalizes). Live possibility; caught
   at week 3 by E2 + the decision rule; negative-result paper pre-scoped in §1.
2. **Coupling (§3) erases patch benefit on GPTQ bases.** Caught week 2; design B
   is the escape hatch and is already specced.
3. **Metric collapse at 2-bit small models.** Mitigated in §6; decision point
   week 4, not week 9.
4. **Proxy scores don't track real loss change.** E1 catches this before any
   conclusion depends on it.
5. **Scooped mid-project.** Monthly arXiv sweep ("task-aware quantization",
   "mixed-precision adapter", "swappable quantization"); the atlas + noise-ceiling
   methodology + composition experiment survive a scoop of the core framing.
6. **Eval noise swamps 1–2 point effects.** 3 selection seeds + paired bootstrap +
   continuous ΔPPL/KL metrics; never claim a win that isn't significant under the
   bootstrap.

---

## 10. Deliverables & paper skeleton

- **Repo**: pipeline + configs to reproduce every table from scratch.
- **Atlas**: overlap matrices, per-layer heatmaps, ceiling/floor analysis.
- **Patch zoo**: base(s) + 4 task patches per budget, in the safetensors format —
  the demo is "load base, swap coding↔medical patch in <1 s, watch the benchmark move".
- **Paper** (workshop length; conference = add 7B+more tasks):
  1. Intro: quantization as a static artifact → make it task-conditional & swappable.
  2. The atlas (RQ1) — with the noise-ceiling methodology as a contribution in itself.
  3. Method: candidate-aware base + patches + the coupling analysis.
  4. Results: main grid, Pareto, scaling.
  5. Composition (RQ4).
  6. Positioning vs. TAQ / SpQR / cross-calibration.

---

## 11. Start here (week 1, day by day)

1. **Day 1–2:** scaffold repo; implement group-wise RTN; sanity: Qwen2.5-1.5B
   RTN-4bit ppl on WikiText-2 within ~0.5 of published.
2. **Day 3–4:** GPTQ loop (own implementation, hookable for design B); sanity vs.
   GPTQModel output at 4-bit and 3-bit.
3. **Day 5:** build all four calibration/eval dataset pairs + disjointness check;
   commit dataset manifests (hashes) to the repo.
4. **Day 6–7:** activation-stat and grad² hooks; produce the first sensitivity map
   (code task) and eyeball the per-layer distribution — you'll know within an hour
   of plotting whether the pipeline is sane (outlier structure should be visible
   and stable across two calibration halves).
