"""Does the sub-integer-bit frontier finding extend BELOW 3 bits?

The full-surface run showed patches own the Pareto frontier between 3.25 and
4.25 bits/weight -- not because they beat uniform 4-bit (they do not) but
because uniform quantization is integer-only and simply does not exist in that
band. If that argument is right, it should hold even more strongly one bit
lower: the band 2.25 -> 3.25 bits/weight, where compression actually matters
and where uniform's only options are "catastrophic" or "expensive".

There is more headroom here in principle -- 2-bit damage is far larger, so
there is more for a patch to recover -- but also a real risk the base is so
broken that nothing rescues it. That risk is why this is measured rather than
assumed: the script reports PPL as a multiple of fp16 and flags any config
beyond 10x as uninterpretable rather than quietly reporting a number.

Configs span the band:
    uniform 2-bit                      floor (2.25 b/w)
    mixed attn@3 + mlp@2               cheap asymmetric point
    u2 / mix32 base + pb2 or pb3 patch at several budgets
    uniform 3-bit                      the alternative to beat (3.25 b/w)

Patch precision is swept over {2, 3} rather than fixed at 3: the pb optimum
was measured against a 3-bit base, and a 2-bit base leaves a larger residual,
so the optimum may move. Bitmap encoding (promoted, 12/12 probes) is used
throughout.
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

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/lowbit_pareto.json")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "code,math,knowledge,chat").split(","))
GROUP_SIZE = 128
ATTN_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TYPES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN_TYPES + MLP_TYPES
PATCH_BITS = (2, 3)
BUDGET_FRACS = (0.02, 0.05, 0.10)
N_CALIB_SEQ = int(os.environ.get("SQ_CALIB_SEQ", "24"))
N_EVAL_SEQ = int(os.environ.get("SQ_EVAL_SEQ", "16"))
LOGIT_CHUNK = 256
N_BOOT = 4000
FLOOR_RATIO = 10.0  # PPL beyond this multiple of fp16 is not interpretable


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
    print(f"full surface: {len(order)} layers, {tot_w/1e6:.1f}M weights", flush=True)

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled = torch.cat([data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0)
    H_pool = accum_H(model, layers, pooled, dev)

    print("bases ...", flush=True)
    qc = {}
    for b in (2, 3):
        qc[b] = {n: gptq_quantize(w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE)[0]
                 for n in order}
        print(f"  {b}-bit", flush=True)

    base_defs = {
        "u2": lambda n: 2,
        "mix32": lambda n: 3 if is_attn(n) else 2,
    }
    base_w = {tag: {n: qc[f(n)][n] for n in order} for tag, f in base_defs.items()}

    def bpw(bit_of, patch_groups=0, pb=3):
        v = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w
        if patch_groups:
            v += patch_bytes(patch_groups, GROUP_SIZE, pb, total_groups=tot_g) * 8 / tot_w
        return v

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"model": MODEL_ID, "ppl": {}, "bits": {}, "sig": {}, "floored": {}}

    for t in TASKS:
        print(f"\n=== {t} ===", flush=True)
        H_t = accum_H(model, layers, data[t]["calib"][:N_CALIB_SEQ], dev)
        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache, bits = {}, {}, {}

        def score(key, wd, b):
            s = set_w(wd) if wd else {}
            cache[key] = chunked_nll(model, ev, dev)
            if s:
                unset(s)
            row[key] = ppl(cache[key])
            bits[key] = b
            ratio = row[key] / row["full_precision"] if "full_precision" in row else 1.0
            flag = "  <-- FLOORED, not interpretable" if ratio > FLOOR_RATIO else ""
            print(f"  {key:24s} ppl={row[key]:9.3f} ({ratio:5.1f}x fp16)  bits/w={b:.3f}{flag}",
                  flush=True)

        score("full_precision", None, 16.0)
        score("uniform2bit", qc[2], bpw(lambda n: 2))
        score("uniform3bit", qc[3], bpw(lambda n: 3))
        score("mixed_attn3_mlp2", base_w["mix32"], bpw(base_defs["mix32"]))

        for btag in ("u2", "mix32"):
            obs = {}
            for n in order:
                Hg = H_t[n].to(dev)
                obs[n] = score_obs_groups(w_orig[n].to(dev), base_w[btag][n].to(dev), Hg, GROUP_SIZE)
                del Hg
            if dev == "cuda":
                torch.cuda.empty_cache()
            flat = torch.cat([obs[n].flatten() for n in order])

            for pb in PATCH_BITS:
                for bf in BUDGET_FRACS:
                    budget = patch_bytes(int(bf * tot_g), GROUP_SIZE, 16)
                    ng = groups_for_budget(tot_g, GROUP_SIZE, pb, budget, bitmap=True)
                    if ng < 1:
                        continue
                    mflat = top_k_group_mask(flat, ng / flat.numel())
                    wd, i, used = {}, 0, 0
                    for n in order:
                        c = grp[n]
                        mk_ = mflat[i : i + c].reshape(obs[n].shape)
                        i += c
                        used += int(mk_.sum().item())
                        Hg = H_t[n].to(dev)
                        solved = solve_residual_layer(
                            w_orig[n].to(dev), base_w[btag][n].to(dev), Hg, mk_, GROUP_SIZE
                        ).cpu()
                        del Hg
                        wd[n] = base_w[btag][n] + quantize_delta(
                            solved, mk_.cpu(), GROUP_SIZE, pb
                        )
                        if dev == "cuda":
                            torch.cuda.empty_cache()
                    score(f"{btag}+pb{pb}_bf{int(bf*100)}", wd,
                          bpw(base_defs[btag], used, pb))
                    del wd
            del obs, flat
            if dev == "cuda":
                torch.cuda.empty_cache()

        fp = row["full_precision"]
        results["floored"][t] = {k: (v / fp > FLOOR_RATIO) for k, v in row.items()}
        # the band question: anything at <= uniform-3bit cost that beats it?
        sig = {}
        for k in row:
            if k in ("full_precision", "uniform3bit"):
                continue
            if bits[k] <= bits["uniform3bit"]:
                sig[f"{k}_beats_uniform3"] = boot(cache[k], cache["uniform3bit"])
        results["ppl"][t] = row
        results["bits"][t] = bits
        results["sig"][t] = sig
        top = sorted(sig.items(), key=lambda x: -x[1])[:4]
        for k, v in top:
            print(f"    {k:40s} p={v:.3f}")
        del H_t
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
