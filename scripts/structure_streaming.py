"""Structural diagnostic for models too large to hold in VRAM.

results/residual_structure*.json established, at 0.5B and 1.5B, how much of
the quantization residual a patch can remove per byte -- and that uniform
precision is 2.4-3.0x more byte-efficient at 0.5B but only 1.3-2.1x at 1.5B.
Whether that trend continues is the single open question the project ends on,
and it needs a 7B point.

The 0.5B/1.5B script moved the whole model to the GPU, which 7B (15.2 GB
against 4.7 GB free) cannot do. This version collects probe-layer Hessians
through selfquant.quant.streaming instead, so only one transformer block is
resident at a time, then frees the model entirely before the linear algebra.

Probes default to o_proj: at 7B its Hessian is 3584^2 x 4 B = 51 MB, versus
1.44 GB for down_proj, and a full Cholesky/SVD of the latter would not fit
alongside the working set. Attention is also where the 1.5B run showed the
gap narrowing most (1.3x), so it is the most informative place to look.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import gptq_quantize
from selfquant.quant.streaming import stream_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/structure_7b.json")
DATA_DIR = "results/calibration_longform"
TASKS = ("code", "math", "knowledge", "chat")
GROUP_SIZE = 128
MODULE_TYPES = tuple(os.environ.get("SQ_MODULES", "self_attn.o_proj").split(","))
PROBE_FRACS = (0.25, 0.5, 0.75)
BITS = tuple(int(x) for x in os.environ.get("SQ_BITS", "3,4").split(","))
PATCH_BITS = tuple(int(x) for x in os.environ.get("SQ_PATCH_BITS", "3,4,16").split(","))
BUDGET_FRACS = (0.05,)
N_CALIB_SEQ = int(os.environ.get("SQ_CALIB_SEQ", "24"))


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


def lowrank_in_H_metric(R, H, r, jitter=1e-4):
    n = H.shape[0]
    Hd = H + jitter * torch.diagonal(H).mean() * torch.eye(n, device=H.device)
    L = torch.linalg.cholesky(Hd)
    U, S, Vh = torch.linalg.svd(R @ L, full_matrices=False)
    return torch.linalg.solve_triangular(L, (U[:, :r] * S[:r]) @ Vh[:r], upper=False, left=False)


def rank_for_bytes(out_f, in_f, target_bytes):
    return max(1, int(target_bytes / (2 * (out_f + in_f))))


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"model={MODEL_ID} device={dev} modules={MODULE_TYPES}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    n_blocks = len(model.model.layers)
    probe_depths = {min(n_blocks - 1, int(f * n_blocks)) for f in PROBE_FRACS}
    print(f"blocks={n_blocks} probe depths={sorted(probe_depths)}")

    seqs = torch.cat(
        [torch.load(f"{DATA_DIR}/{t}.pt")["calib"][: max(1, N_CALIB_SEQ // len(TASKS))]
         for t in TASKS], 0
    )
    in_f_guess = getattr(model.model.layers[0].self_attn.o_proj, "in_features", 1)
    print(f"calibration {seqs.shape[0]} seqs x {seqs.shape[1]} tok = {seqs.numel()} tokens "
          f"({seqs.numel()/in_f_guess:.1f} samples/dim for o_proj)")

    # Keep only the probe layers' weights and Hessians; everything else is
    # streamed through and discarded.
    keep: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def qf(name, w, H):
        depth = int(name.split(".")[0].replace("layer", ""))
        if depth in probe_depths:
            keep[name] = (w, H.clone())
        return None

    print("streaming Hessian collection ...", flush=True)
    stream_quantize(model, seqs, MODULE_TYPES, dev, qf)

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
        print(f"model freed; {torch.cuda.mem_get_info()[0]/1e9:.2f} GB free for linear algebra")

    out = {}
    for name, (W, H) in sorted(keep.items()):
        out_f, in_f = W.shape
        n_groups_layer = (out_f * in_f) // GROUP_SIZE
        print(f"\n=== {name} [{out_f} x {in_f}] ===", flush=True)
        out[name] = {}

        for bits in BITS:
            Q, _, _ = gptq_quantize(W, H, bits=bits, group_size=GROUP_SIZE)
            Wg, Qg, Hg = W.to(dev), Q.to(dev), H.to(dev).float()
            R = Wg - Qg
            E0 = err(R, Hg)
            obs = score_obs_groups(Wg, Qg, Hg, GROUP_SIZE)
            print(f"  {bits}-bit base error {E0:.4e}")
            rec = {}

            for bf in BUDGET_FRACS:
                budget = patch_bytes(int(bf * n_groups_layer), GROUP_SIZE, 16)
                for pb in PATCH_BITS:
                    ng = groups_for_budget(n_groups_layer, GROUP_SIZE, pb, budget)
                    mask = top_k_group_mask(obs, ng / n_groups_layer)
                    solved = solve_residual_layer(Wg, Qg, Hg, mask, GROUP_SIZE)
                    d = quantize_delta(solved.cpu(), mask.cpu(), GROUP_SIZE, pb).to(dev)
                    e = err(R - d, Hg)
                    rec[f"sparse_pb{pb}_bf{int(bf*100)}"] = {
                        "coverage": ng / n_groups_layer, "frac_removed": 1 - e / E0,
                    }
                    print(f"    pb={pb:2d} cov {100*ng/n_groups_layer:5.1f}% removes {100*(1-e/E0):5.1f}%")
                    del solved, d
                    if dev == "cuda":
                        torch.cuda.empty_cache()

                r = rank_for_bytes(out_f, in_f, budget)
                try:
                    M = lowrank_in_H_metric(R, Hg, r)
                    e_lr = err(R - M, Hg)
                    rec[f"lowrank_r{r}_bf{int(bf*100)}"] = {"frac_removed": 1 - e_lr / E0}
                    print(f"    low-rank r={r} removes {100*(1-e_lr/E0):5.1f}%")
                    del M
                except torch.OutOfMemoryError:
                    print(f"    low-rank r={r}: OOM, skipped")
                if dev == "cuda":
                    torch.cuda.empty_cache()

            # the comparison the scale question turns on
            e_next = None
            if bits + 1 in BITS:
                Qn, _, _ = gptq_quantize(W, H, bits=bits + 1, group_size=GROUP_SIZE)
                e_next = err(Wg - Qn.to(dev), Hg)
                rec["uniform_next_bit_frac_removed"] = 1 - e_next / E0
                print(f"    +1 bit uniform removes {100*(1-e_next/E0):5.1f}%")
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
