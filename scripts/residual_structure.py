"""Why does sparse correction plateau? Decompose the quantization residual.

Every correction so far has been SPARSE: pick some groups, fix them. That is
the right tool when damage is concentrated in outliers. But uniform 3-bit
under-precision damages *every* weight a little, and a sparse corrector
cannot represent dense damage no matter how good its mask is. If that is
what is happening, better masks (OBS over AWQ) and better values (solve over
restore) both hit the same ceiling -- which is exactly the pattern observed.

This script measures the ceiling directly, per correction family, at matched
byte budgets. No training, no model forward beyond collecting H: it is all
linear algebra on the residual R = W - Q(W) and the Hessian H.

Objective (same activation-space error the patches optimise):
    E(M) = tr((R - M) H (R - M)^T)

Families compared at equal bytes:
  sparse-k   top-k groups by OBS score, optimally solved
  rank-r     best rank-r approximation in the H metric.
             With H = L L^T, tr(M H M^T) = ||M L||_F^2, so the optimum is
             the truncated SVD of R L, mapped back by B = C L^-1.
  hybrid     rank-r first, then sparse-k on what remains

Reading the output: if rank-r removes far more error per byte than sparse-k,
the sparse-only design is the binding constraint and a low-rank term is the
fix. If sparse wins, the residual really is outlier-dominated and the
plateau lies elsewhere.

FINDING (Qwen2.5-0.5B, results/residual_structure.json): neither. Sparse
edges out low-rank on MLP (21.0% vs 17.1% at k=5%), low-rank edges out
sparse on attention o_proj (26.3% vs 22.7%), and the hybrid never clearly
wins. Decisively, the removable fraction is near-invariant to base
bit-width (2-bit 24.8%, 3-bit 21.0%, 4-bit 20.5%) -- the signature of a
residual with no exploitable structure left. GPTQ's compensation has
already absorbed the correlated part; what remains is close to i.i.d.
rounding noise, which is neither sparse nor low-rank.

Consequence, in bits: a patch removes 26-71% of the error per bit/weight
spent (degrading as the budget grows), whereas simply adding one bit
removes ~79% (theory: error ~ Delta^2/12, so halving the step is a 4x
squared-error reduction = 75%). Uniform precision is ~2.4x more
byte-efficient and the gap widens with investment -- which is the whole
reason no patched configuration beat uniform 4-bit.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS
from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GROUP_SIZE = 128
DATA_DIR = "results/calibration_data"
MAX_TOKENS = 128
# Probe depths are fractions of model depth so the same script can profile
# models of different sizes at comparable relative positions.
PROBE_FRACS = (0.25, 0.5, 0.75)
OUT_PATH = os.environ.get("SQ_OUT", "results/residual_structure.json")
BITS = (2, 3, 4, 8)
K_FRACS = (0.01, 0.02, 0.05)


def dev_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def load_texts(p, n):
    o = []
    with open(p) as f:
        for line in f:
            o.append(json.loads(line)["text"])
            if len(o) >= n:
                break
    return o


def err(R, H):
    """tr(R H R^T) -- activation-space reconstruction error."""
    return float(torch.einsum("oi,ij,oj->", R, H, R).item())


def lowrank_in_H_metric(R, H, r, jitter=1e-4):
    """Best rank-r M minimising tr((R-M) H (R-M)^T)."""
    n = H.shape[0]
    Hd = H + jitter * torch.diagonal(H).mean() * torch.eye(n, device=H.device)
    L = torch.linalg.cholesky(Hd)
    Rp = R @ L  # [out, in]
    U, S, Vh = torch.linalg.svd(Rp, full_matrices=False)
    Rp_r = (U[:, :r] * S[:r]) @ Vh[:r]
    # map back: M L = Rp_r  =>  M = Rp_r L^-1
    M = torch.linalg.solve_triangular(L, Rp_r, upper=False, left=False)
    return M


def sparse_bytes(out_f, in_f, k):
    n_groups = in_f // GROUP_SIZE
    n_sel = round(k * out_f * n_groups)
    return n_sel * GROUP_SIZE * 2 + n_sel * 8  # fp16 values + (row, group) idx


def lowrank_bytes(out_f, in_f, r):
    return (out_f * r + r * in_f) * 2


def rank_for_bytes(out_f, in_f, target_bytes):
    return max(1, int(target_bytes / (2 * (out_f + in_f))))


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)

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
        for part in mtype.split("."):
            m = getattr(m, part)
        mods[name] = m
    print(f"model={MODEL_ID} blocks={n_blocks} probes={[p[0] for p in probes]}")

    pooled = []
    for t in TASKS:
        pooled += load_texts(f"{DATA_DIR}/{t}_calib.jsonl", 16)

    print("collecting Hessians ...")
    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in mods.items()}
    handles = []
    for n, m in mods.items():
        def mk(h, inf):
            def hook(mod, args):
                h.update(args[0].reshape(-1, inf).detach())
            return hook
        handles.append(m.register_forward_pre_hook(mk(hs[n], m.in_features)))
    with torch.no_grad():
        for t in pooled:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
            model(ids)
    for h in handles:
        h.remove()

    # Everything past this point is linear algebra on W and H -- no more
    # forward passes -- so the model is dead weight on the GPU. Freeing it
    # hands the whole device to the solves, which is what makes larger
    # models profilable on a small card at all.
    weights = {name: m.weight.data.float().cpu() for name, m in mods.items()}
    del mods, model
    if dev == "cuda":
        torch.cuda.empty_cache()
        free_b, _ = torch.cuda.mem_get_info()
        print(f"freed model; {free_b/1e9:.2f} GB free for solves")

    out = {}
    for name, W in weights.items():
        H = hs[name].H
        out_f, in_f = W.shape
        print(f"\n=== {name}  [{out_f} x {in_f}] ===")
        out[name] = {}

        for bits in BITS:
            Q, _, _ = gptq_quantize(W, H, bits=bits, group_size=GROUP_SIZE)
            Wg, Qg, Hg = W.to(dev), Q.to(dev), H.to(dev).float()
            R = Wg - Qg
            E0 = err(R, Hg)
            obs = score_obs_groups(Wg, Qg, Hg, GROUP_SIZE)
            print(f"  {bits}-bit  baseline activation error = {E0:.4e}")
            rec = {}

            for k in K_FRACS:
                b_sparse = sparse_bytes(out_f, in_f, k)
                mask = top_k_group_mask(obs, k)
                d = solve_residual_layer(Wg, Qg, Hg, mask, GROUP_SIZE)
                e_sp = err(R - d, Hg)

                r = rank_for_bytes(out_f, in_f, b_sparse)
                M = lowrank_in_H_metric(R, Hg, r)
                e_lr = err(R - M, Hg)

                # hybrid at the SAME total budget: half bytes each
                r_h = max(1, rank_for_bytes(out_f, in_f, b_sparse // 2))
                Mh = lowrank_in_H_metric(R, Hg, r_h)
                Rh = R - Mh
                obs_h = score_obs_groups(Wg, Qg + Mh, Hg, GROUP_SIZE)
                mask_h = top_k_group_mask(obs_h, k / 2)
                dh = solve_residual_layer(Wg, Qg + Mh, Hg, mask_h, GROUP_SIZE)
                e_hy = err(Rh - dh, Hg)

                rec[f"k{int(k*100)}"] = {
                    "bytes": b_sparse,
                    "rank_equiv": r,
                    "err_base": E0,
                    "sparse_frac_removed": 1 - e_sp / E0,
                    "lowrank_frac_removed": 1 - e_lr / E0,
                    "hybrid_frac_removed": 1 - e_hy / E0,
                }
                print(
                    f"    k={int(k*100)}% ({b_sparse/1e3:.0f} KB, == rank {r}):"
                    f"  sparse removes {100*(1-e_sp/E0):5.1f}%"
                    f"  | low-rank {100*(1-e_lr/E0):5.1f}%"
                    f"  | hybrid {100*(1-e_hy/E0):5.1f}%"
                )
            out[name][f"{bits}bit"] = rec
            del Wg, Qg, Hg, R
            if dev == "cuda":
                torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
