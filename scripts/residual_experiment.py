"""Pass 3: learned residual corrections instead of fp16 restoration.

Isolates three factors that the earlier passes conflated, so each can be
attributed separately rather than as one bundled change:

  VALUES  restore (delta = w - q, the old method)  vs  solve (closed-form
          activation-space optimum for the same mask)
  MASK    AWQ saliency (|W| * E|X|)  vs  OBS score (exact activation-space
          error reduction from optimally correcting the group)
  BASE    GPTQ 3-bit (compensated/coupled)  vs  RTN 3-bit (uncoupled)

Prediction being tested: `solve` should rescue the GPTQ base, because it
optimises against the base that actually exists rather than assuming the
fp16 weight is the right target — coupling is then not a problem to avoid
but a term the solve absorbs.

Also runs a two-tier hierarchy (shared correction fit on pooled data, task
correction fit on what remains) to measure how much of the benefit is
universal quantization repair versus genuinely task-specific.

All Hessians are task-specific: the correction targets each task's own
activation distribution, which is the point of an activation-space
objective.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS
from selfquant.patch.residual import (
    restoration_delta,
    score_obs_groups,
    solve_residual_layer,
)
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.quant.rtn import rtn_quantize
from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.scores import score_awq, top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_data"
N_CALIB = 64
N_EVAL = 40
MAX_TOKENS = 128
BASE_BITS = 3
UNIFORM_BITS = 4
K_FRACS = (0.01, 0.02, 0.05)
N_BOOT = 2000


def device_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def load_texts(path, n):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if len(out) >= n:
                break
    return out


def target_layers(model):
    layers = {}
    for d in range(len(model.model.layers)):
        blk = model.model.layers[d]
        for mt in MODULE_TYPES:
            m = blk
            for p in mt.split("."):
                m = getattr(m, p)
            if m.in_features % GROUP_SIZE == 0:
                layers[f"layer{d}.{mt}"] = m
    return layers


def accumulate_hessians(model, tok, layers, texts, device):
    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in layers.items()}
    handles = []
    for n, m in layers.items():
        inf = m.in_features

        def mk(h, inf):
            def hook(mod, args):
                h.update(args[0].reshape(-1, inf).detach())
            return hook

        handles.append(m.register_forward_pre_hook(mk(hs[n], inf)))
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
            model(ids)
    for h in handles:
        h.remove()
    return {n: h.H for n, h in hs.items()}


def per_text_nll(model, tok, texts, device):
    model.eval()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            r = model(ids, labels=ids)
            n = ids.shape[1] - 1
            out.append((r.loss.item() * n, n))
    return out


def ppl(pt):
    return math.exp(sum(a for a, _ in pt) / sum(b for _, b in pt))


def boot_win(a, b, seed=0, n=N_BOOT):
    rng = random.Random(seed)
    idx = range(len(a))
    w = 0
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        if sum(a[i][0] for i in s) < sum(b[i][0] for i in s):
            w += 1
    return w / n


def set_w(layers, wd):
    saved = {}
    for n, m in layers.items():
        saved[n] = m.weight.data.clone()
        m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
    return saved


def restore_w(layers, saved):
    for n, m in layers.items():
        m.weight.data = saved[n]


def global_mask(scores, layer_order, k):
    flat = torch.cat([scores[n].flatten() for n in layer_order])
    mflat = top_k_group_mask(flat, k)
    out, i = {}, 0
    for n in layer_order:
        s = scores[n].shape
        c = s[0] * s[1]
        out[n] = mflat[i : i + c].reshape(s)
        i += c
    return out


def main():
    dev = device_of()
    print(f"device: {dev}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)

    layers = target_layers(model)
    order = sorted(layers)
    w_orig = {n: m.weight.data.float().cpu() for n, m in layers.items()}
    n_w = sum(v.numel() for v in w_orig.values())
    print(f"layers: {len(layers)}  weights: {n_w/1e6:.1f}M")

    # ---- pooled Hessian (generic) for the bases and the shared tier ----
    pooled_texts = []
    for t in TASKS:
        pooled_texts += load_texts(f"{DATA_DIR}/{t}_calib.jsonl", 16)
    print("\naccumulating pooled Hessian ...")
    H_pool = accumulate_hessians(model, tok, layers, pooled_texts, dev)

    print(f"building bases (GPTQ-{BASE_BITS}b, RTN-{BASE_BITS}b, GPTQ-{UNIFORM_BITS}b) ...")
    bases = {"gptq3": {}, "rtn3": {}}
    uni4 = {}
    for i, n in enumerate(order):
        w = w_orig[n]
        bases["gptq3"][n], _, _ = gptq_quantize(w, H_pool[n], bits=BASE_BITS, group_size=GROUP_SIZE)
        bases["rtn3"][n], _, _ = rtn_quantize(w, bits=BASE_BITS, group_size=GROUP_SIZE)
        uni4[n], _, _ = gptq_quantize(w, H_pool[n], bits=UNIFORM_BITS, group_size=GROUP_SIZE)
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(order)}")

    # ---- per-task Hessians + AWQ activation stats ----
    H_task, awq_act = {}, {}
    for task in TASKS:
        txt = load_texts(f"{DATA_DIR}/{task}_calib.jsonl", N_CALIB)
        print(f"accumulating Hessian for {task} ...")
        H_task[task] = accumulate_hessians(model, tok, layers, txt, dev)
        st = {n: ActivationAbsMean(m) for n, m in layers.items()}
        with torch.no_grad():
            for t in txt:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
                model(ids)
        awq_act[task] = {n: st[n].result().cpu() for n in order}
        for n in order:
            st[n].remove()

    def build_wd(base, mask, variant, H_src):
        """Assemble patched weights for every layer under one config."""
        wd = {}
        for n in order:
            w, q = w_orig[n], base[n]
            if variant == "restore":
                d = restoration_delta(w, q, mask[n], GROUP_SIZE)
            else:
                Hg = H_src[n].to(dev)
                d = solve_residual_layer(w.to(dev), q.to(dev), Hg, mask[n], GROUP_SIZE).cpu()
                del Hg
            wd[n] = q + d
        if dev == "cuda":
            torch.cuda.empty_cache()
        return wd

    results = {"n_layers": len(layers), "n_weights": n_w, "grid": {}, "sig": {}}

    for task in TASKS:
        print(f"\n=== {task} ===")
        ev = load_texts(f"{DATA_DIR}/{task}_eval.jsonl", N_EVAL)
        row, cache = {}, {}

        def record(key, wd):
            saved = set_w(layers, wd)
            pt = per_text_nll(model, tok, ev, dev)
            restore_w(layers, saved)
            row[key] = ppl(pt)
            cache[key] = pt

        pt = per_text_nll(model, tok, ev, dev)
        row["fp16"] = ppl(pt); cache["fp16"] = pt
        record("uniform_4bit", uni4)
        for bl in ("gptq3", "rtn3"):
            record(f"base_{bl}", bases[bl])

        # scores: AWQ (base-independent) and OBS (per base, uses task H)
        awq_sc = {}
        for n in order:
            a = awq_act[task][n]
            if a.shape[0] != w_orig[n].shape[1]:
                a = torch.nn.functional.pad(a, (0, w_orig[n].shape[1] - a.shape[0]))
            awq_sc[n] = score_awq(w_orig[n], a, GROUP_SIZE)

        obs_sc = {}
        for bl in ("gptq3", "rtn3"):
            obs_sc[bl] = {}
            for n in order:
                Hg = H_task[task][n].to(dev)
                obs_sc[bl][n] = score_obs_groups(w_orig[n].to(dev), bases[bl][n].to(dev), Hg, GROUP_SIZE)
                del Hg
            if dev == "cuda":
                torch.cuda.empty_cache()

        for k in K_FRACS:
            m_awq = global_mask(awq_sc, order, k)
            for bl in ("gptq3", "rtn3"):
                m_obs = global_mask(obs_sc[bl], order, k)
                for mask_name, mk in (("awq", m_awq), ("obs", m_obs)):
                    for variant in ("restore", "solve"):
                        key = f"{bl}|{mask_name}|{variant}|k{int(k*100)}"
                        record(key, build_wd(bases[bl], mk, variant, H_task[task]))
            print(f"  k={int(k*100)}% done")

        # ---- two-tier: shared correction, then task correction on the rest ----
        for k in (0.01, 0.02):
            obs_pool = {}
            for n in order:
                Hg = H_pool[n].to(dev)
                obs_pool[n] = score_obs_groups(w_orig[n].to(dev), bases["gptq3"][n].to(dev), Hg, GROUP_SIZE)
                del Hg
            m_shared = global_mask(obs_pool, order, k)
            shared_wd = build_wd(bases["gptq3"], m_shared, "solve", H_pool)
            record(f"tier|shared_only|k{int(k*100)}", shared_wd)

            # task tier scored and solved against the shared-corrected base
            obs2 = {}
            for n in order:
                Hg = H_task[task][n].to(dev)
                obs2[n] = score_obs_groups(w_orig[n].to(dev), shared_wd[n].to(dev), Hg, GROUP_SIZE)
                del Hg
            m_task2 = global_mask(obs2, order, k)
            record(f"tier|shared+task|k{int(k*100)}", build_wd(shared_wd, m_task2, "solve", H_task[task]))
            print(f"  tier k={int(k*100)}% done")

        # significance vs. the strongest simple alternative
        sig = {}
        for key in list(row):
            if "|" in key:
                sig[f"{key}_beats_uniform4"] = boot_win(cache[key], cache["uniform_4bit"])
                bl = key.split("|")[0]
                if bl in ("gptq3", "rtn3"):
                    sig[f"{key}_beats_base"] = boot_win(cache[key], cache[f"base_{bl}"])
        results["sig"][task] = sig
        results["grid"][task] = row
        for kk in sorted(row):
            print(f"    {kk:32s} {row[kk]:.4f}")
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/residual_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/residual_results.json")


if __name__ == "__main__":
    main()
