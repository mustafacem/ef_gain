"""Isolate the bitmap encoding gain in perplexity, and measure seed variance.

Two standing gaps, both cheap to close:

1. BITMAP A/B. Idea F (dense bitmap mask instead of explicit (row, group)
   indices) was promoted on 12/12 activation-space probes at median 1.088x,
   and is now baked into every patch config -- but it was never isolated in a
   PPL A/B. Since it is pure byte accounting (an explicit index costs 8 B per
   selected group; a bitmap costs total_groups/8 flat), the prediction is that
   the freed budget buys ~14% more coverage and PPL improves accordingly. If
   it does not, the promotion was measured on the wrong metric.

2. SEED VARIANCE. Every result in this project is single-seed. The bootstrap
   resamples eval texts, so it captures eval noise but says nothing about
   variance from the CALIBRATION draw -- which feeds the Hessian, the base,
   the OBS scores and the mask. If that variance is comparable to the effects
   being reported, several conclusions are weaker than they look. This
   re-runs one headline config across disjoint calibration draws.

Both run on the full surface at the composite config (mixed attn@4 + mlp@3
base, 3-bit patch), which is the design the project actually recommends.
"""
import json
import math
import os
import random
import statistics as st

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/bitmap_ab.json")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "code,math,knowledge,chat").split(","))
GROUP_SIZE = 128
ATTN_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TYPES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN_TYPES + MLP_TYPES
PATCH_BITS = 3
BUDGET_FRAC = 0.05
N_CALIB_SEQ = 24
N_EVAL_SEQ = 16
N_SEEDS = 3
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


def is_attn(n):
    return "self_attn" in n


def accum_H(model, layers, seqs, dev):
    hs = {n: GPTQHessian(m.in_features, device=dev) for n, m in layers.items()}
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
    out = {n: h.H.cpu() for n, h in hs.items()}
    del hs
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


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
    n_w = {n: w_orig[n].numel() for n in order}
    tot_w = sum(n_w.values())
    grp = {n: n_w[n] // GROUP_SIZE for n in order}
    tot_g = sum(grp.values())
    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}

    def bpw(bit_of, groups, pb=PATCH_BITS, bitmap=True):
        v = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w
        v += patch_bytes(groups, GROUP_SIZE, pb,
                         total_groups=tot_g if bitmap else None) * 8 / tot_w
        return v

    def build(base, H_src, bitmap):
        obs = {}
        for n in order:
            Hg = H_src[n].to(dev)
            obs[n] = score_obs_groups(w_orig[n].to(dev), base[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        flat = torch.cat([obs[n].flatten() for n in order])
        budget = patch_bytes(int(BUDGET_FRAC * tot_g), GROUP_SIZE, 16)
        ng = groups_for_budget(tot_g, GROUP_SIZE, PATCH_BITS, budget, bitmap=bitmap)
        mflat = top_k_group_mask(flat, ng / flat.numel())
        wd, i, used = {}, 0, 0
        for n in order:
            c = grp[n]
            mk_ = mflat[i : i + c].reshape(obs[n].shape)
            i += c
            used += int(mk_.sum().item())
            Hg = H_src[n].to(dev)
            solved = solve_residual_layer(
                w_orig[n].to(dev), base[n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
            wd[n] = base[n] + quantize_delta(solved, mk_.cpu(), GROUP_SIZE, PATCH_BITS)
            if dev == "cuda":
                torch.cuda.empty_cache()
        del obs, flat
        return wd, used

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"model": MODEL_ID, "bitmap_ab": {}, "seeds": {}}

    # ---------- part 1: bitmap A/B at identical budget ----------
    print("=== PART 1: bitmap vs explicit indices, identical byte budget ===", flush=True)
    pooled = torch.cat([data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0)
    H_pool = accum_H(model, layers, pooled, dev)
    qc = {b: {n: gptq_quantize(w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE)[0]
              for n in order} for b in (3, 4)}
    mix43 = {n: qc[4][n] if is_attn(n) else qc[3][n] for n in order}

    for t in TASKS:
        H_t = accum_H(model, layers, data[t]["calib"][:N_CALIB_SEQ], dev)
        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache = {}, {}
        for tag, bm in (("explicit", False), ("bitmap", True)):
            wd, used = build(mix43, H_t, bm)
            s = set_w(wd)
            cache[tag] = chunked_nll(model, ev, dev)
            unset(s)
            row[tag] = {"ppl": ppl(cache[tag]), "groups": used,
                        "coverage": used / tot_g,
                        "bits": bpw(lambda n: 4 if is_attn(n) else 3, used, bitmap=bm)}
            print(f"  {t:10s} {tag:9s} cov={100*used/tot_g:5.1f}% "
                  f"ppl={row[tag]['ppl']:8.4f} bits/w={row[tag]['bits']:.3f}", flush=True)
            del wd
        row["bitmap_beats_explicit_p"] = boot(cache["bitmap"], cache["explicit"])
        print(f"  {t:10s} -> bitmap better p={row['bitmap_beats_explicit_p']:.3f}", flush=True)
        results["bitmap_ab"][t] = row
        del H_t
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---------- part 2: calibration-seed variance ----------
    print("\n=== PART 2: calibration-draw variance (composite config) ===", flush=True)
    for t in TASKS:
        ev = data[t]["eval"][:N_EVAL_SEQ]
        avail = data[t]["calib"].shape[0]
        per_seed = []
        for sd in range(N_SEEDS):
            lo = (sd * N_CALIB_SEQ) % max(1, avail - N_CALIB_SEQ)
            chunk = data[t]["calib"][lo : lo + N_CALIB_SEQ]
            H_s = accum_H(model, layers, chunk, dev)
            base_s = {n: gptq_quantize(w_orig[n], H_s[n],
                                       bits=4 if is_attn(n) else 3,
                                       group_size=GROUP_SIZE)[0] for n in order}
            wd, used = build(base_s, H_s, True)
            s = set_w(wd)
            p = ppl(chunked_nll(model, ev, dev))
            unset(s)
            per_seed.append(p)
            print(f"  {t:10s} seed{sd} (calib[{lo}:{lo+N_CALIB_SEQ}]) ppl={p:8.4f}", flush=True)
            del wd, H_s, base_s
            if dev == "cuda":
                torch.cuda.empty_cache()
        m, sd_ = st.mean(per_seed), (st.stdev(per_seed) if len(per_seed) > 1 else 0.0)
        results["seeds"][t] = {"ppl": per_seed, "mean": m, "stdev": sd_,
                               "rel_stdev": sd_ / m}
        print(f"  {t:10s} -> mean {m:.4f} sd {sd_:.4f} ({100*sd_/m:.2f}% relative)", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
