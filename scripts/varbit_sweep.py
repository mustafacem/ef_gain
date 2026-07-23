"""Is fp16 the wrong precision to store a patch in?

Earlier passes always upgraded selected groups to fp16. Against a 3-bit base
that is +12.75 bits per patched weight, while the measured bit-scaling law
says each added bit removes ~79% of remaining error -- so 6 bits should
capture nearly everything for a quarter of the storage. Uniform quantization
was beating fp16 patches by roughly that same factor, which makes patch
precision a prime suspect for the earlier negative results.

The comparison is at MATCHED BYTES, not matched k. Comparing "6-bit at k=5%"
against "fp16 at k=5%" would just be comparing two different storage budgets
and the cheaper one would win trivially. The real question is:

    given a fixed byte budget, is it better to patch FEW groups precisely
    or MANY groups coarsely?

groups_for_budget() converts each budget into the number of groups that
precision can afford, so every row of the sweep costs the same.

Corrections are the closed-form activation-space optimum (solve_residual_layer)
and are then quantized to patch_bits -- quantizing the solved correction
rather than the raw residual keeps the optimality the solve bought.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_longform"
TASKS = ("code", "math", "knowledge", "chat")
BASE_BITS = tuple(int(x) for x in os.environ.get("SQ_BASE_BITS","2,3").split(","))
PATCH_BITS = tuple(int(x) for x in os.environ.get("SQ_PATCH_BITS","4,6,8,16").split(","))
# Byte budgets chosen so the fp16 row lands near the k values used earlier
# (k=1%, 2%, 5% of a ~124M-weight patchable set).
BUDGET_FRACS = tuple(float(x) for x in os.environ.get("SQ_BUDGETS","0.01,0.02,0.05").split(","))
N_CALIB_SEQ = 32
N_EVAL_SEQ = 16
LOGIT_CHUNK = 256
N_BOOT = 4000


def dev_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def target_layers(model):
    ls = {}
    for d in range(len(model.model.layers)):
        blk = model.model.layers[d]
        for mt in MODULE_TYPES:
            m = blk
            for part in mt.split("."):
                m = getattr(m, part)
            if m.in_features % GROUP_SIZE == 0:
                ls[f"layer{d}.{mt}"] = m
    return ls


def accum_H(model, layers, seqs, dev):
    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in layers.items()}
    hd = []
    for n, m in layers.items():
        def mk(h, inf):
            def hook(mod, args):
                h.update(args[0].reshape(-1, inf).detach())
            return hook
        hd.append(m.register_forward_pre_hook(mk(hs[n], m.in_features)))
    with torch.no_grad():
        for i in range(seqs.shape[0]):
            model.model(seqs[i : i + 1].to(dev))
    for h in hd:
        h.remove()
    return {n: h.H for n, h in hs.items()}


def chunked_nll(model, seqs, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(seqs.shape[0]):
            ids = seqs[i : i + 1].to(dev)
            hidden = model.model(ids).last_hidden_state[0]
            tgt, src = ids[0, 1:], hidden[:-1]
            total, n = 0.0, src.shape[0]
            for s in range(0, n, LOGIT_CHUNK):
                e = min(s + LOGIT_CHUNK, n)
                lg = model.lm_head(src[s:e]).float()
                total += torch.nn.functional.cross_entropy(lg, tgt[s:e], reduction="sum").item()
                del lg
            out.append((total, n))
            del hidden, src
    return out


def ppl(pt):
    return math.exp(sum(a for a, _ in pt) / sum(b for _, b in pt))


def boot(a, b, seed=0):
    rng = random.Random(seed)
    idx = range(len(a))
    w = 0
    for _ in range(N_BOOT):
        s = [rng.choice(idx) for _ in idx]
        if sum(a[i][0] for i in s) < sum(b[i][0] for i in s):
            w += 1
    return w / N_BOOT


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)
    layers = target_layers(model)
    order = sorted(layers)
    w_orig = {n: m.weight.data.float().cpu() for n, m in layers.items()}
    n_weights = sum(v.numel() for v in w_orig.values())
    total_groups = n_weights // GROUP_SIZE
    print(f"device={dev} layers={len(layers)} weights={n_weights/1e6:.1f}M groups={total_groups}")

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled_seqs = torch.cat([data[t]["calib"][: N_CALIB_SEQ // len(TASKS)] for t in TASKS], 0)
    print("pooled Hessian ...")
    H_pool = accum_H(model, layers, pooled_seqs, dev)

    print("bases ...")
    bases = {}
    for b in sorted(set(BASE_BITS) | {x + 1 for x in BASE_BITS}):
        bases[b] = {}
        for n in order:
            bases[b][n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE)
        print(f"  {b}-bit")

    print("task Hessians ...")
    H_task = {t: accum_H(model, layers, data[t]["calib"][:N_CALIB_SEQ], dev) for t in TASKS}

    # OBS scores depend on (weights, base, Hessian) but NOT on patch_bits or
    # on the budget -- only the threshold applied to them does. Recomputing
    # them per patch_bits was pure waste (a Cholesky per group per layer,
    # repeated once per precision in the sweep).
    _obs_cache: dict = {}

    def obs_scores(base, H_src, cache_key):
        if cache_key not in _obs_cache:
            sc = {}
            for n in order:
                Hg = H_src[n].to(dev)
                sc[n] = score_obs_groups(w_orig[n].to(dev), base[n].to(dev), Hg, GROUP_SIZE)
                del Hg
            _obs_cache[cache_key] = sc
            if dev == "cuda":
                torch.cuda.empty_cache()
        return _obs_cache[cache_key]

    def build(base, H_src, n_groups, patch_bits, cache_key):
        """Patch exactly n_groups groups, storing the solved correction at
        patch_bits precision.

        Note the cost profile: the closed-form solve is O(b^3) in the
        per-row patched width b, and lower patch_bits buys MORE coverage at
        the same byte budget -- so a 2-bit patch (30% coverage) costs ~12x
        more to construct than an fp16 one (5%). That is a one-off build
        cost, not a serving cost, but it dominates this sweep.
        """
        obs = obs_scores(base, H_src, cache_key)
        flat = torch.cat([obs[n].flatten() for n in order])
        k_frac = n_groups / flat.numel()
        mflat = top_k_group_mask(flat, k_frac)
        wd, i = {}, 0
        for n in order:
            sh = obs[n].shape
            c = sh[0] * sh[1]
            mk_ = mflat[i : i + c].reshape(sh)
            i += c
            Hg = H_src[n].to(dev)
            solved = solve_residual_layer(
                w_orig[n].to(dev), base[n].to(dev), Hg, mk_, GROUP_SIZE
            )
            del Hg
            wd[n] = base[n] + quantize_delta(solved.cpu(), mk_.cpu(), GROUP_SIZE, patch_bits)
        if dev == "cuda":
            torch.cuda.empty_cache()
        return wd, int(mflat.sum().item())

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"n_weights": n_weights, "grid": {}, "sig": {}, "budgets": {}}

    for t in TASKS:
        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache = {}, {}
        cache["full"] = chunked_nll(model, ev, dev)
        row["full_precision"] = ppl(cache["full"])

        for b in BASE_BITS:
            s = set_w(bases[b]); cache[f"{b}bit"] = chunked_nll(model, ev, dev); unset(s)
            row[f"{b}bit"] = ppl(cache[f"{b}bit"])
            s = set_w(bases[b + 1]); cache[f"uniform{b+1}bit"] = chunked_nll(model, ev, dev); unset(s)
            row[f"uniform{b+1}bit"] = ppl(cache[f"uniform{b+1}bit"])

            for bf in BUDGET_FRACS:
                # Budget defined by what an fp16 patch at this k would cost;
                # every patch_bits row then gets the SAME number of bytes.
                fp16_groups = int(bf * total_groups)
                budget_bytes = patch_bytes(fp16_groups, GROUP_SIZE, 16)
                results["budgets"][f"{b}bit_bf{int(bf*100)}"] = budget_bytes

                for pb in PATCH_BITS:
                    ng = groups_for_budget(total_groups, GROUP_SIZE, pb, budget_bytes)
                    wd, used = build(bases[b], H_task[t], ng, pb, cache_key=(b, t))
                    key = f"{b}bit|pb{pb}|bf{int(bf*100)}"
                    s = set_w(wd); cache[key] = chunked_nll(model, ev, dev); unset(s)
                    row[key] = ppl(cache[key])
                    row[f"{key}|groups"] = used
                    del wd
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                print(f"  {t} {b}bit bf={int(bf*100)}% done")

        sig = {}
        for b in BASE_BITS:
            for bf in BUDGET_FRACS:
                tag = f"{b}bit|bf{int(bf*100)}"
                fp16 = cache[f"{b}bit|pb16|bf{int(bf*100)}"]
                for pb in PATCH_BITS:
                    if pb == 16:
                        continue
                    sig[f"{tag}|pb{pb}_beats_fp16"] = boot(cache[f"{b}bit|pb{pb}|bf{int(bf*100)}"], fp16)
                best_pb = min(PATCH_BITS, key=lambda p: row[f"{b}bit|pb{p}|bf{int(bf*100)}"])
                sig[f"{tag}|best_pb"] = best_pb
                sig[f"{tag}|best_beats_uniform{b+1}"] = boot(
                    cache[f"{b}bit|pb{best_pb}|bf{int(bf*100)}"], cache[f"uniform{b+1}bit"]
                )
        results["grid"][t] = row
        results["sig"][t] = sig

        print(f"\n=== {t} ===")
        for kk in sorted(row):
            if kk.endswith("|groups"):
                continue
            print(f"  {kk:26s} {row[kk]:.5f}")
        for kk, v in sorted(sig.items()):
            print(f"    {kk:34s} {v}")

    os.makedirs("results", exist_ok=True)
    with open(os.environ.get("SQ_OUT","results/varbit_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {os.environ.get('SQ_OUT','results/varbit_sweep.json')}")


if __name__ == "__main__":
    main()
