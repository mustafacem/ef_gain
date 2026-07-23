"""Pass 3b: the cross-task controls that pass 3 omitted.

Pass 3 established that learned corrections beat restoration and that they
rescue the GPTQ base — but every configuration in it was task-matched, so
it could not say anything about task-*specificity*. This script supplies
the missing controls, using the improved method.

A patch here is fully self-contained, as it would be in deployment: its
mask AND its correction values are both derived from one task's
calibration data. Patch_S is then evaluated on every task's eval set,
producing a full 4x4 matrix. Diagonal = matched, off-diagonal = mismatched.

Controls:
  random  - random mask, values solved against pooled data
  pooled  - the "shared/universal" patch (pooled mask + pooled values),
            i.e. task-agnostic repair at the same budget

The comparison that matters is diagonal vs. its own row/column
off-diagonals at IDENTICAL budget, and diagonal vs. pooled at identical
budget. Pass 3's tier experiment failed the second of these by giving
shared+task twice the budget of shared_only.
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
K_FRACS = (0.01, 0.02, 0.05)
MASK_KINDS = ("awq", "obs")
N_BOOT = 2000


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
            for p in mt.split("."):
                m = getattr(m, p)
            if m.in_features % GROUP_SIZE == 0:
                ls[f"layer{d}.{mt}"] = m
    return ls


def accum_H(model, tok, layers, texts, dev):
    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in layers.items()}
    hd = []
    for n, m in layers.items():
        def mk(h, inf):
            def hook(mod, args):
                h.update(args[0].reshape(-1, inf).detach())
            return hook
        hd.append(m.register_forward_pre_hook(mk(hs[n], m.in_features)))
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
            model(ids)
    for h in hd:
        h.remove()
    return {n: h.H for n, h in hs.items()}


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


def gmask(scores, order, k):
    flat = torch.cat([scores[n].flatten() for n in order])
    mf = top_k_group_mask(flat, k)
    out, i = {}, 0
    for n in order:
        sh = scores[n].shape
        c = sh[0] * sh[1]
        out[n] = mf[i : i + c].reshape(sh)
        i += c
    return out


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)
    layers = target_layers(model)
    order = sorted(layers)
    w_orig = {n: m.weight.data.float().cpu() for n, m in layers.items()}
    print(f"device={dev} layers={len(layers)}")

    pooled_texts = []
    for t in TASKS:
        pooled_texts += load_texts(f"{DATA_DIR}/{t}_calib.jsonl", 16)
    print("pooled Hessian ...")
    H_pool = accum_H(model, tok, layers, pooled_texts, dev)

    print("building GPTQ 3-bit base ...")
    base = {}
    for n in order:
        base[n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=BASE_BITS, group_size=GROUP_SIZE)

    # per-task Hessians and AWQ activation stats
    H_task, awq_act = {}, {}
    for task in TASKS:
        txt = load_texts(f"{DATA_DIR}/{task}_calib.jsonl", N_CALIB)
        print(f"Hessian + act stats: {task}")
        H_task[task] = accum_H(model, tok, layers, txt, dev)
        st = {n: ActivationAbsMean(m) for n, m in layers.items()}
        with torch.no_grad():
            for t in txt:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
                model(ids)
        awq_act[task] = {n: st[n].result().cpu() for n in order}
        for n in order:
            st[n].remove()

    # ---- scores per source ----
    def awq_scores(src):
        sc = {}
        for n in order:
            a = awq_act[src][n]
            if a.shape[0] != w_orig[n].shape[1]:
                a = torch.nn.functional.pad(a, (0, w_orig[n].shape[1] - a.shape[0]))
            sc[n] = score_awq(w_orig[n], a, GROUP_SIZE)
        return sc

    def obs_scores(H):
        sc = {}
        for n in order:
            Hg = H[n].to(dev)
            sc[n] = score_obs_groups(w_orig[n].to(dev), base[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        return sc

    print("\nscoring all patch sources ...")
    scores = {}
    for task in TASKS:
        scores[("awq", task)] = awq_scores(task)
        scores[("obs", task)] = obs_scores(H_task[task])
    # pooled AWQ: average the per-task activation stats (same pooled data)
    pooled_awq = {}
    for n in order:
        acc = None
        for task in TASKS:
            a = awq_act[task][n]
            acc = a.clone() if acc is None else acc + a
        a = acc / len(TASKS)
        if a.shape[0] != w_orig[n].shape[1]:
            a = torch.nn.functional.pad(a, (0, w_orig[n].shape[1] - a.shape[0]))
        pooled_awq[n] = score_awq(w_orig[n], a, GROUP_SIZE)
    scores[("awq", "pooled")] = pooled_awq
    scores[("obs", "pooled")] = obs_scores(H_pool)
    torch.manual_seed(0)
    rnd = {n: torch.rand_like(scores[("awq", TASKS[0])][n]) for n in order}
    scores[("awq", "random")] = rnd
    scores[("obs", "random")] = rnd

    def build(src, kind, k):
        """Self-contained patch: mask AND values both from source `src`."""
        H_src = H_pool if src in ("pooled", "random") else H_task[src]
        mk = gmask(scores[(kind, src)], order, k)
        wd = {}
        for n in order:
            Hg = H_src[n].to(dev)
            d = solve_residual_layer(w_orig[n].to(dev), base[n].to(dev), Hg, mk[n], GROUP_SIZE).cpu()
            del Hg
            wd[n] = base[n] + d
        if dev == "cuda":
            torch.cuda.empty_cache()
        return wd

    SOURCES = list(TASKS) + ["pooled", "random"]
    results = {"matrix": {}, "sig": {}, "base": {}, "fp16": {}}

    eval_sets = {t: load_texts(f"{DATA_DIR}/{t}_eval.jsonl", N_EVAL) for t in TASKS}
    saved0 = None
    for t in TASKS:
        results["fp16"][t] = ppl(per_text_nll(model, tok, eval_sets[t], dev))

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    s = set_w(base)
    for t in TASKS:
        results["base"][t] = ppl(per_text_nll(model, tok, eval_sets[t], dev))
    unset(s)

    cache = {}
    for kind in MASK_KINDS:
        for k in K_FRACS:
            for src in SOURCES:
                wd = build(src, kind, k)
                s = set_w(wd)
                for t in TASKS:
                    pt = per_text_nll(model, tok, eval_sets[t], dev)
                    key = f"{kind}|k{int(k*100)}|patch={src}|eval={t}"
                    results["matrix"][key] = ppl(pt)
                    cache[key] = pt
                unset(s)
                del wd
                if dev == "cuda":
                    torch.cuda.empty_cache()
            print(f"  {kind} k={int(k*100)}% done")

    # significance: matched vs each control, at identical budget
    for kind in MASK_KINDS:
        for k in K_FRACS:
            kk = int(k * 100)
            for t in TASKS:
                own = cache[f"{kind}|k{kk}|patch={t}|eval={t}"]
                for ctrl in [x for x in SOURCES if x != t]:
                    c = cache[f"{kind}|k{kk}|patch={ctrl}|eval={t}"]
                    results["sig"][f"{kind}|k{kk}|{t}: own_beats_{ctrl}"] = boot(own, c)

    os.makedirs("results", exist_ok=True)
    with open("results/crosstask_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/crosstask_results.json")


if __name__ == "__main__":
    main()
