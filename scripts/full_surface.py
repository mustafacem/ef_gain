"""Full-surface re-baseline + composite: does the improved design beat uniform?

Two changes from every previous experiment, both of which make the numbers
mean something they did not before:

1. ALL SEVEN linear types are quantized (q/k/v/o/gate/up/down = 358M weights),
   not just o_proj+down_proj (124M, 35%). Every compression figure reported
   before this let 65% of the weights ride along in fp16, which inflated
   quality and undercut any claim about bits/weight. These numbers are NOT
   comparable to earlier ones -- that is the point of re-baselining.

2. The patch design carries the two changes that survived their gates:
     - patch_bits=3 (2.13x better than fp16 in PPL, p=1.000)
     - bitmap mask encoding (12/12 probes, median 1.088x, free)
   and drops the two that did not (solve/quantize iteration, residual-aware
   re-selection -- both rejected at matched bytes).

The composite hypothesis being tested: the two most byte-efficient
interventions measured so far are a mixed-precision base (one extra bit on
attention: 129% gap-closed per extra bit) and a small pb3 patch. They address
orthogonal structure -- across layer types vs within layers -- so stacking
them should land near ~3.7 bits/weight with quality that uniform 4-bit
(4.25 bits) cannot match at its own cost.

Every row prints effective bits/weight. Nothing is compared at unequal cost.
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
OUT_PATH = os.environ.get("SQ_OUT", "results/full_surface.json")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "code,math,knowledge,chat").split(","))
GROUP_SIZE = 128
ATTN_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TYPES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN_TYPES + MLP_TYPES
PATCH_BITS = 3
BUDGET_FRACS = (0.01, 0.02, 0.05)
N_CALIB_SEQ = int(os.environ.get("SQ_CALIB_SEQ", "24"))
N_EVAL_SEQ = int(os.environ.get("SQ_EVAL_SEQ", "16"))
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


def is_attn(name):
    return "self_attn" in name


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
    attn_w = sum(n_w[n] for n in order if is_attn(n))
    print(f"FULL SURFACE: {len(order)} layers, {tot_w/1e6:.1f}M weights "
          f"(attn {100*attn_w/tot_w:.1f}%, mlp {100*(1-attn_w/tot_w):.1f}%)", flush=True)

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled = torch.cat([data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0)
    smallest_in = min(layers[n].in_features for n in order)
    print(f"pooled calib {pooled.numel()} tokens "
          f"({pooled.numel()/max(layers[n].in_features for n in order):.1f} samples/dim worst case)",
          flush=True)
    H_pool = accum_H(model, layers, pooled, dev)

    print("quantizing bases ...", flush=True)
    qc = {}
    for b in (3, 4, 5):
        qc[b] = {}
        for n in order:
            qc[b][n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE)
        print(f"  {b}-bit", flush=True)

    def bpw(bit_of, patch_groups=0):
        base = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w
        if patch_groups:
            base += patch_bytes(patch_groups, GROUP_SIZE, PATCH_BITS,
                                total_groups=tot_g) * 8 / tot_w
        return base

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    # base variants for the composite
    base_defs = {
        "u3": lambda n: 3,                          # uniform 3-bit
        "mix43": lambda n: 4 if is_attn(n) else 3,  # +1 bit on attention
    }
    base_w = {
        tag: {n: qc[f(n)][n] for n in order} for tag, f in base_defs.items()
    }

    results = {"model": MODEL_ID, "n_weights": tot_w, "ppl": {}, "bits": {}, "sig": {}}

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
            print(f"  {key:26s} ppl={row[key]:8.4f}  bits/w={b:.3f}", flush=True)

        score("full_precision", None, 16.0)
        score("uniform3bit", qc[3], bpw(lambda n: 3))
        score("uniform4bit", qc[4], bpw(lambda n: 4))
        score("uniform5bit", qc[5], bpw(lambda n: 5))
        score("mixed_attn4_mlp3", base_w["mix43"], bpw(base_defs["mix43"]))

        # composite: base + pb3 patch (bitmap-encoded), swept over budget
        for btag in ("u3", "mix43"):
            obs = {}
            for n in order:
                Hg = H_t[n].to(dev)
                obs[n] = score_obs_groups(
                    w_orig[n].to(dev), base_w[btag][n].to(dev), Hg, GROUP_SIZE
                )
                del Hg
            if dev == "cuda":
                torch.cuda.empty_cache()
            flat = torch.cat([obs[n].flatten() for n in order])

            for bf in BUDGET_FRACS:
                budget = patch_bytes(int(bf * tot_g), GROUP_SIZE, 16)
                ng = groups_for_budget(tot_g, GROUP_SIZE, PATCH_BITS, budget, bitmap=True)
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
                        solved, mk_.cpu(), GROUP_SIZE, PATCH_BITS
                    )
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                score(f"{btag}+pb3_bf{int(bf*100)}", wd, bpw(base_defs[btag], used))
                del wd
            del obs, flat
            if dev == "cuda":
                torch.cuda.empty_cache()

        # the question: does anything beat uniform 4-bit, and at what cost?
        cands = [k for k in row if k not in ("full_precision", "uniform4bit")]
        sig = {}
        for k in cands:
            if bits[k] <= bits["uniform4bit"]:
                sig[f"{k}_beats_uniform4_at_lower_bits"] = boot(cache[k], cache["uniform4bit"])
        results["ppl"][t] = row
        results["bits"][t] = bits
        results["sig"][t] = sig
        for k, v in sorted(sig.items(), key=lambda x: -x[1])[:6]:
            print(f"    {k:44s} p={v:.3f}")
        del H_t
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
