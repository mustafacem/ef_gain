# selfquant — findings summary

Testing whether a shared low-bit base plus small swappable **task patches**
(selected weight groups restored at higher precision) can beat ordinary
quantization. Six passes on Qwen2.5 at 0.5B / 1.5B / 7B, on a 6 GB laptop GPU.

**Verdict: the mechanism is real and beats every null hypothesis tested, but
uniform quantization is still the better use of a bit at these scales.**

---

## 1. What holds up

| Claim | Evidence | Status |
|---|---|---|
| Weight sensitivity is genuinely task-dependent | signal 0.246 vs 0.15 threshold, against a split-half noise ceiling of 0.970 | **holds** |
| GPTQ compensation makes naive fp16 restoration *harmful* | patching helped 3/16 configs on a GPTQ base; 16/16 on an uncoupled RTN base | **holds** |
| A closed-form activation-space correction fixes that | `δ[S] = (H_SS)⁻¹(Hr)[S]`, reusing the quantizer's own Hessian; 20/48 configs beat base | **holds** |
| **Patch precision matters far more than anything else** | 3-bit patches beat fp16 patches **2.13×** (46.9% vs 22.0% gap closed), **p = 1.000 all 4 tasks** | **holds** |
| Patches are *not* just per-layer-type mixed precision | 46.9% vs 24.4% at matched bits, **p = 1.000 all 4 tasks** | **holds** |
| Uniform quantization still wins overall | 84.5% vs 46.9% for 0.175 more bits/weight; p = 0.000 on 3/4 tasks | **holds** |

## 2. What was retracted

| Retracted claim | Why |
|---|---|
| "Code is the task where patches pay" | Artifact of a **singular Hessian** — code had 0.90 samples/dim (rank-deficient). Vanished at 13.5 samples/dim. |
| "The approach is scale-gated; the gap halves by 1.5B" | Compared **fp16 patches at 0.5B against a hybrid at 1.5B** — different designs, not different scales. The curve is flat: 1.05× / 0.97× / 1.12× at 0.5B/1.5B/7B, matched depth. |
| "Lower patch precision is always better" | Extrapolated from a monotonic 4<6<8<16 trend. **2-bit is worse than 3-bit** — the optimum is interior. |
| "Task-conditioning carries little weight" (pass 3) | Asserted from a grid with **no cross-task controls at all**; the tier comparison also had a 2× budget confound. |

## 3. The numbers that matter

Gap from 3-bit base to full precision, recovered (0.5B, mean of 4 tasks):

| config | bits/w | gap closed | per extra bit |
|---|---|---|---|
| mixed `o_proj`@4 + `down_proj`@3 | 3.406 | 20.1% | **129.2** ← most efficient marginal spend |
| patch pb3 (proportional top-k) | 4.075 | 46.9% | 56.9 |
| **uniform 4-bit** | 4.250 | **84.5%** | 84.5 ← best absolute recovery |

- **Task-dependence is real but narrow.** It shows up cleanly only at 3-bit with
  small budgets. On **math**, patches recover 76.2% and are statistically tied
  with uniform (p = 0.138) at *lower* memory — the one config where the idea wins.
- **8-bit is essentially lossless** (+0.03–0.10% PPL) — there is nothing to patch
  above ~6 bits.
- **Uniform bit-scaling law:** each added bit removes ~79% of remaining error,
  matching Δ²/12 theory (predicts 75%). Stable across all scales and layers.

## 4. Where the files are

**Findings**
- `scale_findings.md` — 7B scale test, the flat curve, PPL confirmation of pb3
- `allocation_findings.md` — patches vs mixed precision, allocation sweep

**Key raw results**
- `../ppl_confirm.json` — pb3 vs fp16 vs uniform, PPL, 4 tasks
- `../allocation_showdown.json` — patches vs mixed precision at matched bits
- `../structure_7b.json`, `../structure_1.5b_stream.json` — scale curve
- `../patchbit_diagnostic.json` — patch-precision optimum
- `../lowbit_corrected.json` — corrected-Hessian task-conditioning grid
- `../atlas_qwen0.5b.json` — task-sensitivity atlas
- `scale_curve_inputs.json` — consolidated scale-curve inputs

**Reusable code** (58 tests passing)
- `src/selfquant/patch/residual.py` — closed-form correction + OBS scoring
- `src/selfquant/patch/varbit.py` — N-bit patch storage and byte accounting
- `src/selfquant/quant/streaming.py` — block-streaming for models > VRAM
  (7B at 0.59 GB peak); sequential GPTQ formulation
- `src/selfquant/data/longform.py` — 2048-token calibration, conditioning report

## 5. Methodological lessons worth carrying forward

1. **Check Hessian conditioning before trusting anything that inverts H.**
   `samples/dim < 1` means H is singular and damping is doing structural work.
   This invalidated the project's headline positive result.
2. **Always report bits/weight.** Three separate confounds in this project came
   from comparing configs at unequal cost.
3. **Run the null hypothesis you're most afraid of.** "Mixed precision does the
   same thing more simply" was the strongest threat and went untested for weeks.
   It turned out to be refutable — but only because it was finally run.
4. **The activation-space proxy is a reliable *ranking* tool** (it picked the
   pb3 optimum that PPL independently confirmed, and predicted the 2.13×), but
   not a reliable *magnitude* tool — it overstated parity because probe layers
   were attention-only while the model is 84% MLP.
5. **Cheap diagnostics beat expensive grids.** The precision question was settled
   in ~2 minutes of linear algebra after a 5-hour PPL sweep failed to finish.

## 6. Open threads

- Patches remain untested against uniform at **budgets below k=1%**, where the
  storage story is most attractive.
- `q/k/v`, `gate`, `up` projections were never patched or quantized.
- No downstream task accuracy (HumanEval/GSM8K) — perplexity only.
- Single seed throughout; bootstrap covers eval noise only.
