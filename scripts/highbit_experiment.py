"""8-bit base vs. full precision vs. 8-bit + patch.

Everything so far probed 2-4 bit, where quantization damage is large. This
probes the opposite end: at 8-bit the model is nearly lossless, so the
question is whether a patch can close the small remaining gap to full
precision more cheaply than simply storing more bits.

The structural finding from results/residual_structure.json predicts the
answer: the fraction of residual error a patch can remove is near-invariant
to base precision (~20-25% at 2, 3 and 4 bit). If that invariance extends to
8-bit, a patch removes ~20% of an already-negligible error, and should be
undetectable in perplexity.

IMPORTANT -- dtype. Earlier passes ran the model in bfloat16, which carries
only 8 mantissa bits. Comparing an 8-bit quantized weight against a bf16
"reference" would be confounded: the reference is itself lossy at about the
precision under test. This script runs in float32 so the reference is
genuinely full precision, and so that 8-bit quantization error is not
swamped by compute-dtype error.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS
from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_data"
MAX_TOKENS = 128
N_EVAL = 100          # more texts than earlier passes: the effects here are tiny
BITS = (4, 6, 8)
PATCH_BITS = 8
K_FRACS = (0.01, 0.02, 0.05)
N_BOOT = 4000


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


def per_text_nll(model, tok, texts, dev):
    model.eval()
    o = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
            if ids.shape[1] < 2:
                continue
            r = model(ids, labels=ids)
            n = ids.shape[1] - 1
            o.append((r.loss.item() * n, n))
    return o


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


def eff_bits(bits, k=0.0):
    """bits + fp16 scale/zero per group, plus fp16 patch cells."""
    base = bits + 2 * 16 / GROUP_SIZE
    return base + k * (16 - base)


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # float32, deliberately: see module docstring.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev)
    print(f"device={dev} dtype=float32")

    layers = target_layers(model)
    order = sorted(layers)
    w_orig = {n: m.weight.data.float().cpu() for n, m in layers.items()}
    n_w = sum(v.numel() for v in w_orig.values())
    print(f"layers={len(layers)} weights={n_w/1e6:.1f}M")

    pooled = []
    for t in TASKS:
        pooled += load_texts(f"{DATA_DIR}/{t}_calib.jsonl", 16)

    print("pooled Hessian ...")
    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in layers.items()}
    handles = []
    for n, m in layers.items():
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
    H = {n: hs[n].H for n in order}

    print("quantizing bases ...")
    bases = {}
    for b in BITS:
        bases[b] = {}
        for n in order:
            bases[b][n], _, _ = gptq_quantize(w_orig[n], H[n], bits=b, group_size=GROUP_SIZE)
        print(f"  {b}-bit done")

    # patches on the 8-bit base, using the best method from pass 3:
    # OBS mask + closed-form activation-space solve.
    print(f"scoring + solving patches on the {PATCH_BITS}-bit base ...")
    obs = {}
    for n in order:
        Hg = H[n].to(dev)
        obs[n] = score_obs_groups(w_orig[n].to(dev), bases[PATCH_BITS][n].to(dev), Hg, GROUP_SIZE)
        del Hg
    if dev == "cuda":
        torch.cuda.empty_cache()

    flat = torch.cat([obs[n].flatten() for n in order])
    patched = {}
    for k in K_FRACS:
        mflat = top_k_group_mask(flat, k)
        wd, i = {}, 0
        for n in order:
            sh = obs[n].shape
            c = sh[0] * sh[1]
            mk_ = mflat[i : i + c].reshape(sh)
            i += c
            Hg = H[n].to(dev)
            d = solve_residual_layer(
                w_orig[n].to(dev), bases[PATCH_BITS][n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
            wd[n] = bases[PATCH_BITS][n] + d
        patched[k] = wd
        print(f"  k={int(k*100)}% done")
        if dev == "cuda":
            torch.cuda.empty_cache()

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"eff_bits": {}, "ppl": {}, "sig": {}}
    results["eff_bits"]["fp32_ref"] = 32.0
    results["eff_bits"]["fp16_store"] = 16.0
    for b in BITS:
        results["eff_bits"][f"{b}bit"] = eff_bits(b)
    for k in K_FRACS:
        results["eff_bits"][f"{PATCH_BITS}bit+k{int(k*100)}"] = eff_bits(PATCH_BITS, k)

    for task in TASKS:
        ev = load_texts(f"{DATA_DIR}/{task}_eval.jsonl", N_EVAL)
        row, cache = {}, {}

        cache["full"] = per_text_nll(model, tok, ev, dev)
        row["full_precision"] = ppl(cache["full"])

        for b in BITS:
            s = set_w(bases[b])
            cache[f"{b}bit"] = per_text_nll(model, tok, ev, dev)
            row[f"{b}bit"] = ppl(cache[f"{b}bit"])
            unset(s)

        for k in K_FRACS:
            key = f"{PATCH_BITS}bit+k{int(k*100)}"
            s = set_w(patched[k])
            cache[key] = per_text_nll(model, tok, ev, dev)
            row[key] = ppl(cache[key])
            unset(s)

        sig = {}
        for k in K_FRACS:
            key = f"{PATCH_BITS}bit+k{int(k*100)}"
            sig[f"{key}_beats_{PATCH_BITS}bit"] = boot(cache[key], cache[f"{PATCH_BITS}bit"])
            sig[f"{key}_beats_full"] = boot(cache[key], cache["full"])
        sig[f"{PATCH_BITS}bit_beats_full"] = boot(cache[f"{PATCH_BITS}bit"], cache["full"])
        results["ppl"][task] = row
        results["sig"][task] = sig

        print(f"\n=== {task} ===")
        for kk in ["full_precision"] + [f"{b}bit" for b in BITS] + [f"{PATCH_BITS}bit+k{int(k*100)}" for k in K_FRACS]:
            print(f"  {kk:18s} {row[kk]:.5f}")
        for kk, v in sig.items():
            print(f"    {kk:34s} p={v:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/highbit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/highbit_results.json")


if __name__ == "__main__":
    main()
