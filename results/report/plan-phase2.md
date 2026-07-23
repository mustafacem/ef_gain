# Plan: Validate on corrected data, then rebuild patches as variable-bit

## Context

Four passes tested whether a shared low-bit base plus swappable task patches
can beat ordinary quantization. Two problems now block any conclusion:

**1. Every H-dependent result rests on a singular Hessian.** Calibration used
128-token truncations, giving **0.8–1.6 activation samples per Hessian
dimension** for `down_proj` (4864 dims). For code and knowledge, `samples/dim
< 1.0` — H was *rank-deficient*, so damping carried structural load. This
undermines the closed-form solve, OBS scores, and the GPTQ base itself
(passes 3, 3b, and the structure diagnostic). The atlas (§2) is safe: AWQ
scores are per-dimension means, well estimated.

Critically, **code had the worst conditioning (0.90) and was the only task
where an own-task patch beat a pooled one** — the finding the project's
remaining value rests on. It is the most likely artifact and is untested.

`results/calibration_longform/` (built, verified) fixes this: 2048-token
sequences, codeparrot for code calibration, **26–54 samples/dim**, eval grown
from ~5k to 37–98k tokens, leakage ≤0.04%.

**2. The patch design pays fp16 prices for a rounding problem.** Patches
upgrade selected groups from 3-bit straight to fp16 (+12.75 bits/weight). The
measured bit-scaling law (~79% of error removed per added bit, matching
Δ²/12 theory) implies a 3→6-bit upgrade captures ~98% of the benefit at
**~1/4 the storage**. This is the same order as the gap by which uniform
quantization was beating patches, so it may reverse the core verdict.

A third gap: no experiment compares patches against the obvious cheap
alternative — **a base calibrated on the task's own data**, which needs no
patch machinery and is standard practice.

Outcome sought: trustworthy numbers on whether task patches beat (a) a
universal patch, (b) uniform bit-width, and (c) task-specific calibration —
then a variable-bit redesign measured against those.

## Phase 1 — Unblock and validate

### 1a. Fix the OOM in `scripts/lowbit_corrected.py`

Cause: Qwen's 151936 vocab × 2048 tokens × fp32 logits = 1.24 GB, atop a 2 GB
fp32 model.

- Hessian collection calls `model.model(ids)` instead of `model(ids)`,
  skipping `lm_head` entirely — hooks are on inner layers, logits are never
  needed. Saves ~1.2 GB per forward.
- Eval loss computed in token chunks: run `model.model(...)` for hidden
  states, then apply `lm_head` over slices of ~256 tokens, accumulating NLL
  and freeing each logit slice. Add as `chunked_nll()` in the script.
- Use **bfloat16** here, not fp32. The bf16 confound only mattered at 8-bit;
  at 2–4 bit, quantization error exceeds bf16 error by orders of magnitude.
  Keep fp32 solely for `highbit_experiment.py`.

### 1b. Run the corrected comparison

Bases **2, 3, 4-bit** (2-bit included per decision), each vs. its uniform
reference one bit higher. Per task, per budget k ∈ {1, 2, 5}%:

| config | question it answers |
|---|---|
| full precision | ceiling |
| `{b}-bit` base | floor |
| `+ own` patch | does task conditioning help |
| `+ pooled` patch | **the control — universal repair at identical budget** |
| `+ random` patch | placebo |
| uniform `{b+1}`-bit | does patching beat just adding a bit |
| **base calibrated on task data** | **does patching beat cheap task calibration** |

The last row is new and decisive for the code story. Build it by passing the
task's own Hessian to `gptq_quantize` rather than the pooled one — no new
code, just a different `H` argument.

Report `own_beats_pooled`, `own_beats_uniform`, `own_beats_taskcalib` via the
existing paired bootstrap. Log samples/dim alongside every result.

**Gate:** if `own_beats_pooled` collapses for code on well-conditioned H, the
honest conclusion is that task-conditioning never worked and the exception was
an artifact. Record that plainly; it changes what Phase 2 is for (a better
*universal* corrector rather than a task-conditional one) but not whether it
is worth building.

## Phase 2 — Variable-bit patches

New module `src/selfquant/patch/varbit.py`. Reuse
`compute_group_qparams` / `quantize_dequantize_group` from
[rtn.py](src/selfquant/quant/rtn.py) — the quantization math already exists.

- `quantize_residual(w, q, mask, group_size, patch_bits)` — quantize the
  residual `w − q` to `patch_bits` on masked groups only, returning the
  dequantized delta plus its own scales/zeros.
- `varbit_patch_bytes(...)` — honest accounting: `patch_bits` per weight
  **plus** scale/zero per patched group plus indices. The fp16 patch is the
  `patch_bits=16` special case, so old and new are directly comparable.

Sweep `patch_bits ∈ {4, 5, 6, 8, 16}` against fp16 patches at **matched total
bytes** — the fair question is not "does 6-bit beat fp16 at equal k" but
"at equal storage, is it better to patch more groups at lower precision?"
Expect the optimum well below 16.

Also evaluate the closed-form solve composed with variable-bit storage: solve
for the optimal correction, then quantize *that* to `patch_bits`, rather than
quantizing the raw residual. `solve_residual_layer` already returns a dense
delta, so this is a one-line composition.

## Phase 3 — Report

Update the artifact at
`https://claude.ai/code/artifact/8c5ed678-0006-42d3-891f-1c8a3b301cf5`
(pass `url=`, same file path). Must include:

- A conditioning caveat marking which earlier conclusions were computed on a
  singular H, and which of them survived recomputation.
- The corrected code-exception verdict, whichever way it lands.
- The variable-bit result and any reversal of the "uniform wins" conclusion.
- A note that activation-space error disagreed with PPL about task variation
  (2.4× uniform uniformly vs. a 0.66–3.1× range), so proxy-based conclusions
  are downgraded relative to measured ones.

## Files

- `scripts/lowbit_corrected.py` — OOM fix, 2-bit, task-calibrated baseline, pooled/random controls
- `src/selfquant/patch/varbit.py` — new
- `tests/test_varbit.py` — new
- `src/selfquant/data/longform.py` — already built, no change
- `results/calibration_longform/` — already built, no change

## Verification

1. `python -m pytest tests/ -q` — 44 existing tests must stay green; add
   varbit tests asserting (a) `patch_bits=16` reproduces the fp16 path,
   (b) reconstruction error decreases monotonically in `patch_bits`,
   (c) byte accounting matches a hand-computed figure.
2. Watch peak VRAM during the corrected run (`nvidia-smi`), confirming the
   chunked-loss fix holds it under ~5 GB.
3. Sanity-check that 2-bit is not floored: if `2-bit` PPL is not finite or is
   wildly off (>3× full precision), report it as uninformative rather than
   drawing conclusions from it.
4. Confirm every reported comparison is at matched effective bits/weight, and
   print that column in the results table.
