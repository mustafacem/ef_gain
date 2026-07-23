"""Which patch precision minimises error per byte? (no model eval)

The full PPL sweep over patch precisions is expensive for a reason that is
itself a finding: the closed-form solve is O(b^3) in the per-row patched
width, and lower precision buys MORE coverage at fixed bytes, so a 2-bit
patch costs ~8x more to build than an fp16 one (measured: 2.7 vs 0.3 min
per 48-layer pass).

This answers the same ranking question directly in the objective the patches
optimise -- activation-space error tr((R-D) H (R-D)^T) -- on a few probe
layers, with no forward passes beyond Hessian collection. That is the same
trick that settled the sparse-vs-low-rank question earlier, and it runs in
seconds per configuration instead of minutes.

Caveat worth stating: activation-space error previously disagreed with PPL
about how much task-to-task VARIATION there is. It is used here only to rank
patch precisions within one method on one layer, which is the comparison it
is most trustworthy for -- but the winner should still be confirmed by a
single PPL run before being taken to a larger model.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS
from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/patchbit_diagnostic.json")
DATA_DIR = "results/calibration_longform"
GROUP_SIZE = 128
BASE_BITS = (3,)
PATCH_BITS = (2, 3, 4, 6, 8, 16)
BUDGET_FRACS = (0.01, 0.05)
PROBE_FRACS = (0.25, 0.5, 0.75)
N_CALIB_SEQ = 16


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
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt, low_cpu_mem_usage=True).to(dev)
    n_blocks = len(model.model.layers)

    probes = []
    for f in PROBE_FRACS:
        d = min(n_blocks - 1, int(f * n_blocks))
        probes.append((f"layer{d}.mlp.down_proj", d, "mlp.down_proj"))
    mid = min(n_blocks - 1, n_blocks // 2)
    probes.append((f"layer{mid}.self_attn.o_proj", mid, "self_attn.o_proj"))

    mods = {}
    for name, depth, mtype in probes:
        m = model.model.layers[depth]
        for p in mtype.split("."):
            m = getattr(m, p)
        mods[name] = m
    print(f"model={MODEL_ID} probes={[p[0] for p in probes]}")

    seqs = torch.cat(
        [torch.load(f"{DATA_DIR}/{t}.pt")["calib"][: N_CALIB_SEQ // len(TASKS)] for t in TASKS], 0
    )
    print(f"calibration {seqs.shape[0]} seqs x {seqs.shape[1]} tok")

    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in mods.items()}
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
    del mods, model
    if dev == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for name, W in weights.items():
        H = hs[name].H
        out_f, in_f = W.shape
        n_groups_layer = (out_f * in_f) // GROUP_SIZE
        print(f"\n=== {name} [{out_f} x {in_f}] ===")
        out[name] = {}

        for bits in BASE_BITS:
            Q, _, _ = gptq_quantize(W, H, bits=bits, group_size=GROUP_SIZE)
            Wg, Qg, Hg = W.to(dev), Q.to(dev), H.to(dev).float()
            R = Wg - Qg
            E0 = err(R, Hg)
            obs = score_obs_groups(Wg, Qg, Hg, GROUP_SIZE)  # independent of patch_bits
            print(f"  {bits}-bit base error {E0:.4e}")

            for bf in BUDGET_FRACS:
                budget = patch_bytes(int(bf * n_groups_layer), GROUP_SIZE, 16)
                rec = {}
                for pb in PATCH_BITS:
                    ng = groups_for_budget(n_groups_layer, GROUP_SIZE, pb, budget)
                    mask = top_k_group_mask(obs, ng / n_groups_layer)
                    solved = solve_residual_layer(Wg, Qg, Hg, mask, GROUP_SIZE)
                    # store the solved correction at pb precision -- this is
                    # the step the sweep is about
                    d = quantize_delta(solved.cpu(), mask.cpu(), GROUP_SIZE, pb).to(dev)
                    e = err(R - d, Hg)
                    rec[f"pb{pb}"] = {
                        "coverage": ng / n_groups_layer,
                        "frac_removed": 1 - e / E0,
                    }
                    print(f"    bf{int(bf*100)}% pb={pb:2d}: cov {100*ng/n_groups_layer:5.1f}%"
                          f"  removes {100*(1-e/E0):5.1f}%")
                    del solved, d
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                best = max(rec, key=lambda k: rec[k]["frac_removed"])
                print(f"    -> best {best}")
                out[name][f"{bits}bit_bf{int(bf*100)}"] = rec
            del Wg, Qg, Hg, R
            if dev == "cuda":
                torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
