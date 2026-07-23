"""Phase-1 gate: do ideas C and E survive a matched-BYTES test?

Both ideas look good in isolation and could still be worthless, for the same
reason fp16 patches were: they buy quality by spending bytes that could have
bought coverage instead. The pb sweep already established that at a fixed
budget, patching many groups coarsely beats patching few groups finely --
so any idea that trades coverage for fidelity starts at a disadvantage and
has to prove itself.

  C. solve/quantize iteration -- `rounds` rounds store `rounds` quantized
     increments over the same groups, so 2 rounds at k groups costs the same
     as 1 round at 2k groups. Two 3-bit rounds is roughly 6-bit fidelity on
     half the coverage, and pb6 already lost to pb3. Prediction: C loses.

  E. residual-aware re-selection -- spend half the budget, solve, re-score the
     remaining groups against the partially corrected residual, then spend the
     rest. Costs nothing extra in bytes (same total groups), only compute, so
     it only has to beat one-shot top-k on quality.

Promotion rule (set before looking): keep an idea only if it improves
activation-space error removed by >= 5% relative, at matched bytes, on a
majority of probe layers.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import (
    groups_for_budget,
    iterate_solve_quantize,
    patch_bytes,
    quantize_delta,
)
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/idea_gate.json")
DATA_DIR = "results/calibration_longform"
TASKS = ("code", "math", "knowledge", "chat")
GROUP_SIZE = 128
BASE_BITS = 3
PATCH_BITS = 3
BUDGET_FRAC = 0.05
N_CALIB_SEQ = 16
# probe one of each shape family that the full surface will contain
PROBE_MODULES = ("self_attn.o_proj", "mlp.down_proj", "mlp.gate_proj", "self_attn.q_proj")
PROBE_FRACS = (0.25, 0.5, 0.75)


def dev_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def err(R, H):
    return float(torch.einsum("oi,ij,oj->", R, H, R).item())


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)
    n_blocks = len(model.model.layers)
    depths = sorted({min(n_blocks - 1, int(f * n_blocks)) for f in PROBE_FRACS})

    mods = {}
    for d in depths:
        for mt in PROBE_MODULES:
            m = model.model.layers[d]
            for p in mt.split("."):
                m = getattr(m, p)
            if m.in_features % GROUP_SIZE == 0:
                mods[f"layer{d}.{mt}"] = m
    print(f"probes: {len(mods)} layers across depths {depths}", flush=True)

    seqs = torch.cat(
        [torch.load(f"{DATA_DIR}/{t}.pt")["calib"][: max(1, N_CALIB_SEQ // len(TASKS))]
         for t in TASKS], 0
    )
    hs = {n: GPTQHessian(m.in_features, device=dev) for n, m in mods.items()}
    handles = []
    for n, m in mods.items():
        def mk(h, inf):
            def hook(mod, args):
                h.update(args[0].reshape(-1, inf).detach())
            return hook
        handles.append(m.register_forward_pre_hook(mk(hs[n], m.in_features)))
    with torch.no_grad():
        for i in range(seqs.shape[0]):
            model.model(seqs[i : i + 1].to(dev))
    for h in handles:
        h.remove()

    weights = {n: m.weight.data.float().cpu() for n, m in mods.items()}
    H_all = {n: h.H.cpu() for n, h in hs.items()}
    del mods, model, hs
    if dev == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for name, W in sorted(weights.items()):
        H = H_all[name]
        out_f, in_f = W.shape
        n_groups_layer = (out_f * in_f) // GROUP_SIZE
        Q, _, _ = gptq_quantize(W, H, bits=BASE_BITS, group_size=GROUP_SIZE)
        Wg, Qg, Hg = W.to(dev), Q.to(dev), H.to(dev).float()
        R = Wg - Qg
        E0 = err(R, Hg)
        obs = score_obs_groups(Wg, Qg, Hg, GROUP_SIZE)
        budget = patch_bytes(int(BUDGET_FRAC * n_groups_layer), GROUP_SIZE, 16)
        rec = {}
        print(f"\n=== {name} [{out_f}x{in_f}] base err {E0:.3e} ===", flush=True)

        # --- baseline + idea C at MATCHED BYTES -------------------------
        for rounds in (1, 2, 3):
            ng = groups_for_budget(
                n_groups_layer, GROUP_SIZE, PATCH_BITS, budget, rounds=rounds
            )
            if ng < 1:
                continue
            mask = top_k_group_mask(obs, ng / n_groups_layer)
            d = iterate_solve_quantize(
                Wg, Qg, Hg, mask, GROUP_SIZE, PATCH_BITS, rounds=rounds
            )
            fr = 1 - err(R - d, Hg) / E0
            rec[f"C_rounds{rounds}"] = {"coverage": ng / n_groups_layer, "frac_removed": fr}
            print(f"  C rounds={rounds}: cov {100*ng/n_groups_layer:5.1f}% removes {100*fr:5.1f}%")
            del d
            if dev == "cuda":
                torch.cuda.empty_cache()

        # --- idea F: bitmap mask encoding buys coverage for free --------
        ng_bm = groups_for_budget(
            n_groups_layer, GROUP_SIZE, PATCH_BITS, budget, bitmap=True
        )
        mask_bm = top_k_group_mask(obs, ng_bm / n_groups_layer)
        d_bm = quantize_delta(
            solve_residual_layer(Wg, Qg, Hg, mask_bm, GROUP_SIZE),
            mask_bm, GROUP_SIZE, PATCH_BITS,
        )
        fr_f = 1 - err(R - d_bm, Hg) / E0
        rec["F_bitmap"] = {"coverage": ng_bm / n_groups_layer, "frac_removed": fr_f}
        print(f"  F bitmap   : cov {100*ng_bm/n_groups_layer:5.1f}% removes {100*fr_f:5.1f}%")
        del d_bm
        if dev == "cuda":
            torch.cuda.empty_cache()

        # --- idea E: half-budget, re-score, half again ------------------
        ng_full = groups_for_budget(n_groups_layer, GROUP_SIZE, PATCH_BITS, budget)
        half = ng_full // 2
        m1 = top_k_group_mask(obs, half / n_groups_layer)
        d1 = quantize_delta(
            solve_residual_layer(Wg, Qg, Hg, m1, GROUP_SIZE), m1, GROUP_SIZE, PATCH_BITS
        )
        # re-score against the partially corrected base, then forbid re-picking
        obs2 = score_obs_groups(Wg, Qg + d1, Hg, GROUP_SIZE)
        obs2 = obs2.masked_fill(m1.to(obs2.device), float("-inf"))
        m2 = top_k_group_mask(obs2, (ng_full - half) / n_groups_layer)
        d2 = quantize_delta(
            solve_residual_layer(Wg, Qg + d1, Hg, m2, GROUP_SIZE), m2, GROUP_SIZE, PATCH_BITS
        )
        fr_e = 1 - err(R - d1 - d2, Hg) / E0
        rec["E_reselect"] = {
            "coverage": (int(m1.sum()) + int(m2.sum())) / n_groups_layer,
            "frac_removed": fr_e,
        }
        print(f"  E reselect : cov {100*(int(m1.sum())+int(m2.sum()))/n_groups_layer:5.1f}% "
              f"removes {100*fr_e:5.1f}%")

        base_fr = rec["C_rounds1"]["frac_removed"]
        rec["_baseline_frac_removed"] = base_fr
        rec["_C_best_rel"] = max(
            rec[k]["frac_removed"] for k in rec if k.startswith("C_")
        ) / base_fr
        rec["_E_rel"] = fr_e / base_fr
        rec["_F_rel"] = fr_f / base_fr
        print(f"  -> vs one-shot baseline ({100*base_fr:.1f}%): "
              f"C best {rec['_C_best_rel']:.3f}x, E {rec['_E_rel']:.3f}x, "
              f"F {rec['_F_rel']:.3f}x")
        out[name] = rec
        del Wg, Qg, Hg, R, d1, d2
        if dev == "cuda":
            torch.cuda.empty_cache()

    # verdicts
    import statistics as st
    c_rels = [v["_C_best_rel"] for v in out.values()]
    e_rels = [v["_E_rel"] for v in out.values()]
    f_rels = [v["_F_rel"] for v in out.values()]
    c_win = sum(1 for r in c_rels if r >= 1.05)
    e_win = sum(1 for r in e_rels if r >= 1.05)
    f_win = sum(1 for r in f_rels if r >= 1.05)
    n = len(out)
    print(f"\n=== VERDICT (promote if >=1.05x on a majority of {n} probes) ===")
    print(f"  C iteration : median {st.median(c_rels):.3f}x, {c_win}/{n} probes >=1.05  "
          f"-> {'PROMOTE' if c_win > n/2 else 'REJECT'}")
    print(f"  E reselect  : median {st.median(e_rels):.3f}x, {e_win}/{n} probes >=1.05  "
          f"-> {'PROMOTE' if e_win > n/2 else 'REJECT'}")
    print(f"  F bitmap    : median {st.median(f_rels):.3f}x, {f_win}/{n} probes >=1.05  "
          f"-> {'PROMOTE' if f_win > n/2 else 'REJECT'}")
    out["_verdict"] = {
        "F_median_rel": st.median(f_rels), "F_wins": f_win,
        "F_promoted": f_win > n / 2,
        "C_median_rel": st.median(c_rels), "C_wins": c_win,
        "E_median_rel": st.median(e_rels), "E_wins": e_win, "n_probes": n,
        "C_promoted": c_win > n / 2, "E_promoted": e_win > n / 2,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
