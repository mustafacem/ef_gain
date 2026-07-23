"""Improvement pass on the negative E3 result (plan-v2 sections 3, 5-E4).

The first patch grid found task-matched patches barely beating random and
losing to a plain uniform-4-bit base. This script tests the three most
likely causes at once:

  1. COUPLING (plan-v2 section 3). GPTQ compensates for each group's
     quantization error by adjusting *later* columns. Restoring a group's
     original fp16 values afterwards does not undo that compensation, so a
     patch can fight its own base. RTN has no compensation at all, so
     restoration is exact by construction. Running the identical patch
     protocol on both bases isolates coupling as a cause.
  2. BUDGET. Only k=1% was tested. Sweeps 1/2/5/10%.
  3. COVERAGE. Only 10 of ~168 linears were patchable. Uses every
     o_proj + down_proj in all 24 blocks (48 layers).

Also reports paired-bootstrap significance over eval texts, so near-ties
(which several of the first-pass "wins" were) are labelled as such rather
than read as results.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS
from selfquant.patch.apply import apply_patch_layer
from selfquant.patch.build import build_patch_layer, patch_layer_bytes
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.quant.rtn import rtn_quantize
from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.scores import score_awq, top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_data"
N_CALIB_SCORE = 64
N_HESS_CALIB = 48
N_PPL_EVAL = 40
MAX_TOKENS = 128
BASE_BITS = 3
UNIFORM_BITS = 4
K_FRACS = (0.01, 0.02, 0.05, 0.10)
N_BOOTSTRAP = 2000


def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def load_jsonl_texts(path, n):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if len(out) >= n:
                break
    return out


def all_target_layers(model):
    layers = {}
    n_blocks = len(model.model.layers)
    for d in range(n_blocks):
        block = model.model.layers[d]
        for mtype in MODULE_TYPES:
            mod = block
            for part in mtype.split("."):
                mod = getattr(mod, part)
            if mod.in_features % GROUP_SIZE == 0:
                layers[f"layer{d}.{mtype}"] = mod
    return layers


def per_text_nll(model, tok, texts, device):
    """Per-text (sum_nll, n_tokens) so PPL and a paired bootstrap can both
    be computed from one pass."""
    model.eval()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            res = model(ids, labels=ids)
            n = ids.shape[1] - 1
            out.append((res.loss.item() * n, n))
    return out


def ppl_from(per_text):
    tot_nll = sum(a for a, _ in per_text)
    tot_tok = sum(b for _, b in per_text)
    return math.exp(tot_nll / tot_tok)


def bootstrap_win_prob(per_text_a, per_text_b, n_boot=N_BOOTSTRAP, seed=0):
    """P(config A has lower PPL than config B) under resampling of eval
    texts. ~0.5 means indistinguishable; >0.95 is a real win."""
    rng = random.Random(seed)
    n = len(per_text_a)
    wins = 0
    idxs = range(n)
    for _ in range(n_boot):
        sample = [rng.choice(idxs) for _ in idxs]
        nll_a = sum(per_text_a[i][0] for i in sample)
        nll_b = sum(per_text_b[i][0] for i in sample)
        if nll_a < nll_b:
            wins += 1
    return wins / n_boot


def set_layers(layers, wdict):
    saved = {}
    for name, mod in layers.items():
        saved[name] = mod.weight.data.clone()
        mod.weight.data = wdict[name].to(mod.weight.dtype).to(mod.weight.device)
    return saved


def restore_layers(layers, saved):
    for name, mod in layers.items():
        mod.weight.data = saved[name]


def main():
    device = get_device()
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)

    layers = all_target_layers(model)
    layer_order = sorted(layers.keys())
    w_orig = {n: m.weight.data.float().cpu() for n, m in layers.items()}
    n_weights = sum(w.numel() for w in w_orig.values())
    print(f"patchable layers: {len(layers)}  ({n_weights/1e6:.1f}M weights)")

    # ---------- bases ----------
    print("\naccumulating Hessians for GPTQ ...")
    generic_texts = load_jsonl_texts(f"{DATA_DIR}/chat_calib.jsonl", N_HESS_CALIB)
    hess = {n: GPTQHessian(m.in_features, device="cpu") for n, m in layers.items()}

    def mk_hook(h, inf):
        def hook(module, args):
            h.update(args[0].reshape(-1, inf).detach())
        return hook

    handles = [m.register_forward_pre_hook(mk_hook(hess[n], m.in_features)) for n, m in layers.items()]
    with torch.no_grad():
        for t in generic_texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
            model(ids)
    for h in handles:
        h.remove()

    print(f"building bases: GPTQ-{BASE_BITS}b (coupled), RTN-{BASE_BITS}b (uncoupled), GPTQ-{UNIFORM_BITS}b (uniform ref) ...")
    base_gptq, base_rtn, base_uniform4 = {}, {}, {}
    for i, name in enumerate(layer_order):
        w = w_orig[name]
        base_gptq[name], _, _ = gptq_quantize(w, hess[name].H, bits=BASE_BITS, group_size=GROUP_SIZE)
        base_rtn[name], _, _ = rtn_quantize(w, bits=BASE_BITS, group_size=GROUP_SIZE)
        base_uniform4[name], _, _ = gptq_quantize(w, hess[name].H, bits=UNIFORM_BITS, group_size=GROUP_SIZE)
        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{len(layer_order)} layers quantized")

    # ---------- sensitivity scores ----------
    def awq_scores(texts):
        stats = {n: ActivationAbsMean(m) for n, m in layers.items()}
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
                model(ids)
        sc = {}
        for name in layer_order:
            w = w_orig[name]
            act = stats[name].result().cpu()
            if act.shape[0] != w.shape[1]:
                act = torch.nn.functional.pad(act, (0, w.shape[1] - act.shape[0]))
            sc[name] = score_awq(w, act, GROUP_SIZE)
            stats[name].remove()
        return sc

    print("\nscoring per-task sensitivity ...")
    task_scores = {}
    for task in TASKS:
        task_scores[task] = awq_scores(load_jsonl_texts(f"{DATA_DIR}/{task}_calib.jsonl", N_CALIB_SCORE))
        print(f"  {task}: done")

    shapes = {n: task_scores[TASKS[0]][n].shape for n in layer_order}

    def flat(sc):
        return torch.cat([sc[n].flatten() for n in layer_order])

    def unflat(mask_flat):
        out, i = {}, 0
        for n in layer_order:
            s = shapes[n]
            cnt = s[0] * s[1]
            out[n] = mask_flat[i : i + cnt].reshape(s)
            i += cnt
        return out

    def patch_set(sc, k):
        masks = unflat(top_k_group_mask(flat(sc), k))
        return {n: build_patch_layer(w_orig[n], masks[n], GROUP_SIZE) for n in layer_order}

    torch.manual_seed(0)
    rand_scores = {n: torch.rand_like(task_scores[TASKS[0]][n]) for n in layer_order}

    # ---------- evaluation ----------
    results = {"n_layers": len(layers), "n_weights": n_weights, "grid": {}, "significance": {}}
    overhead = 16.0 * 2 / GROUP_SIZE

    for task in TASKS:
        eval_texts = load_jsonl_texts(f"{DATA_DIR}/{task}_eval.jsonl", N_PPL_EVAL)
        print(f"\n=== task: {task} ===")
        row = {}
        pt_cache = {}

        pt = per_text_nll(model, tok, eval_texts, device)
        row["fp16"] = ppl_from(pt); pt_cache["fp16"] = pt

        for base_label, base in (("gptq3", base_gptq), ("rtn3", base_rtn)):
            saved = set_layers(layers, base)
            pt = per_text_nll(model, tok, eval_texts, device)
            restore_layers(layers, saved)
            row[f"base_{base_label}"] = ppl_from(pt); pt_cache[f"base_{base_label}"] = pt

        saved = set_layers(layers, base_uniform4)
        pt = per_text_nll(model, tok, eval_texts, device)
        restore_layers(layers, saved)
        row["uniform_4bit"] = ppl_from(pt); pt_cache["uniform_4bit"] = pt

        for k in K_FRACS:
            own_p = patch_set(task_scores[task], k)
            rnd_p = patch_set(rand_scores, k)
            for base_label, base in (("gptq3", base_gptq), ("rtn3", base_rtn)):
                for sel_label, ps in (("own", own_p), ("random", rnd_p)):
                    patched = {n: apply_patch_layer(base[n], ps[n]) for n in layer_order}
                    saved = set_layers(layers, patched)
                    pt = per_text_nll(model, tok, eval_texts, device)
                    restore_layers(layers, saved)
                    key = f"{base_label}+{sel_label}_k{int(k*100)}"
                    row[key] = ppl_from(pt); pt_cache[key] = pt
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  k={int(k*100)}% done")

        # paired-bootstrap: own-patch vs random-patch, per base, per budget
        sig = {}
        for k in K_FRACS:
            for base_label in ("gptq3", "rtn3"):
                a = f"{base_label}+own_k{int(k*100)}"
                b = f"{base_label}+random_k{int(k*100)}"
                sig[f"{a}_beats_{b}"] = bootstrap_win_prob(pt_cache[a], pt_cache[b])
            # own patch vs its own unpatched base (does patching help at all?)
            for base_label in ("gptq3", "rtn3"):
                a = f"{base_label}+own_k{int(k*100)}"
                sig[f"{a}_beats_base"] = bootstrap_win_prob(pt_cache[a], pt_cache[f"base_{base_label}"])
        results["significance"][task] = sig
        results["grid"][task] = row

        for key in sorted(row):
            print(f"    {key:26s} {row[key]:.4f}")

    # ---------- economics ----------
    patch_mb = {}
    for k in K_FRACS:
        ps = patch_set(task_scores[TASKS[0]], k)
        patch_mb[f"k{int(k*100)}"] = sum(patch_layer_bytes(p) for p in ps.values()) / 1e6
    results["patch_sizes_mb"] = patch_mb
    results["memory"] = {
        "n_weights": n_weights,
        "fp16_mb": n_weights * 16 / 8 / 1e6,
        "base3_mb": n_weights * (BASE_BITS + overhead) / 8 / 1e6,
        "uniform4_mb": n_weights * (UNIFORM_BITS + overhead) / 8 / 1e6,
        "eff_bits_per_k": {f"k{int(k*100)}": (BASE_BITS + overhead) + k * (16 - (BASE_BITS + overhead)) for k in K_FRACS},
    }

    os.makedirs("results", exist_ok=True)
    with open("results/improved_patch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/improved_patch_results.json")


if __name__ == "__main__":
    main()
