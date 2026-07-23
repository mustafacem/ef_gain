"""Is the code result real, or calibration/eval leakage?

Code is the only task where the task-matched patch beats a pooled patch
(bootstrap p=1.00, consistently). It is also the only task with meaningful
calibration/eval overlap: 3.0% of eval 8-grams also appear in calibration
text, versus 0.054% / 0.059% / 0.008% for math / knowledge / chat. MBPP's
train and test splits share a lot of boilerplate.

If the code patch is partly fitting n-grams that recur in its eval set,
the "task-conditioning works for code" conclusion is an artifact. This
rebuilds the comparison on a decontaminated eval set: every eval text
sharing any 8-gram with any calibration text is dropped.

Reports own-vs-pooled on the original and cleaned eval sets side by side.
If the effect survives decontamination it is real; if it collapses, the
project has zero tasks where task-conditioning beats universal repair.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS, _word_ngrams
from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.scores import score_awq, top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_data"
N_CALIB = 64
MAX_TOKENS = 128
BASE_BITS = 3
NGRAM_N = 8
K_FRACS = (0.02, 0.05)
N_BOOT = 2000
TASK = "code"


def dev_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def load_texts(p, n=10**9):
    o = []
    with open(p) as f:
        for line in f:
            o.append(json.loads(line)["text"])
            if len(o) >= n:
                break
    return o


def decontaminate(calib, evals, n=NGRAM_N):
    """Drop every eval text sharing an n-gram with any calibration text."""
    cal = set()
    for t in calib:
        cal |= _word_ngrams(t, n)
    kept, dropped = [], 0
    for t in evals:
        if _word_ngrams(t, n) & cal:
            dropped += 1
        else:
            kept.append(t)
    return kept, dropped


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
            k = ids.shape[1] - 1
            o.append((r.loss.item() * k, k))
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

    calib = load_texts(f"{DATA_DIR}/{TASK}_calib.jsonl", N_CALIB)
    ev_raw = load_texts(f"{DATA_DIR}/{TASK}_eval.jsonl", 100)
    ev_clean, dropped = decontaminate(calib, ev_raw)
    print(f"eval texts: {len(ev_raw)} raw -> {len(ev_clean)} clean ({dropped} dropped for {NGRAM_N}-gram overlap)")
    if len(ev_clean) < 10:
        print("WARNING: too few clean eval texts for a stable comparison")

    pooled_texts = []
    for t in TASKS:
        pooled_texts += load_texts(f"{DATA_DIR}/{t}_calib.jsonl", 16)
    print("pooled Hessian ...")
    H_pool = accum_H(model, tok, layers, pooled_texts, dev)
    print("base ...")
    base = {n: gptq_quantize(w_orig[n], H_pool[n], bits=BASE_BITS, group_size=GROUP_SIZE)[0] for n in order}

    print("task Hessian + act stats ...")
    H_code = accum_H(model, tok, layers, calib, dev)
    st = {n: ActivationAbsMean(m) for n, m in layers.items()}
    with torch.no_grad():
        for t in calib:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
            model(ids)
    act_code = {n: st[n].result().cpu() for n in order}
    for n in order:
        st[n].remove()

    def awq_sc(act):
        sc = {}
        for n in order:
            a = act[n]
            if a.shape[0] != w_orig[n].shape[1]:
                a = torch.nn.functional.pad(a, (0, w_orig[n].shape[1] - a.shape[0]))
            sc[n] = score_awq(w_orig[n], a, GROUP_SIZE)
        return sc

    def obs_sc(H):
        sc = {}
        for n in order:
            Hg = H[n].to(dev)
            sc[n] = score_obs_groups(w_orig[n].to(dev), base[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        return sc

    # pooled AWQ needs pooled activation stats
    st = {n: ActivationAbsMean(m) for n, m in layers.items()}
    with torch.no_grad():
        for t in pooled_texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(dev)
            model(ids)
    act_pool = {n: st[n].result().cpu() for n in order}
    for n in order:
        st[n].remove()

    srcs = {
        "code": (awq_sc(act_code), obs_sc(H_code), H_code),
        "pooled": (awq_sc(act_pool), obs_sc(H_pool), H_pool),
    }

    def build(src, kind, k):
        sc = srcs[src][0] if kind == "awq" else srcs[src][1]
        H = srcs[src][2]
        mk = gmask(sc, order, k)
        wd = {}
        for n in order:
            Hg = H[n].to(dev)
            wd[n] = base[n] + solve_residual_layer(w_orig[n].to(dev), base[n].to(dev), Hg, mk[n], GROUP_SIZE).cpu()
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        return wd

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    out = {"n_eval_raw": len(ev_raw), "n_eval_clean": len(ev_clean), "dropped": dropped, "res": {}}
    for kind in ("awq", "obs"):
        for k in K_FRACS:
            kk = int(k * 100)
            pts = {}
            for src in ("code", "pooled"):
                wd = build(src, kind, k)
                s = set_w(wd)
                pts[(src, "raw")] = per_text_nll(model, tok, ev_raw, dev)
                pts[(src, "clean")] = per_text_nll(model, tok, ev_clean, dev)
                unset(s)
                del wd
            for split in ("raw", "clean"):
                p = boot(pts[("code", split)], pts[("pooled", split)])
                out["res"][f"{kind}|k{kk}|{split}"] = {
                    "code_ppl": ppl(pts[("code", split)]),
                    "pooled_ppl": ppl(pts[("pooled", split)]),
                    "p_own_beats_pooled": p,
                }
                r = out["res"][f"{kind}|k{kk}|{split}"]
                print(f"  {kind} k{kk}% {split:5s}: own={r['code_ppl']:.4f} pooled={r['pooled_ppl']:.4f} p={p:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/leakage_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/leakage_check.json")


if __name__ == "__main__":
    main()
