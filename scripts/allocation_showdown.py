"""Do patches earn their complexity, or is per-layer-type mixed precision enough?

The PPL confirmation showed 3-bit patches close 46.9% of the quantization gap
(vs 22.0% for fp16 patches) but still lose to uniform 4-bit on 3 of 4 tasks.
The activation-space breakdown said why: `o_proj` reaches 0.97-1.12x of uniform
while `down_proj` sits near 1.44x, and down_proj holds 84% of the patchable
weights, so it dominates the average.

That points at two competing conclusions, and only one of them is good for the
patch idea:

  (a) the budget is being spread over the wrong layers -- fix the ALLOCATION
      and patches become competitive; or
  (b) what patches are really discovering is just "attention needs more bits
      than MLP", in which case plain per-layer-type MIXED PRECISION gets the
      same benefit with none of the machinery (no Hessian solve, no mask, no
      indices, no swappable artifact).

(b) is the baseline this project never ran, and it is the honest control. At
0.5B, o_proj is 15.6% of patchable weights, so `o_proj@8 + down_proj@3` costs
4.028 bits/weight -- the same as the 3-bit+3-bit-patch config at 4.075. If that
wins, the patch machinery is unnecessary at this budget.

Configs, all reported with exact bits/weight so nothing is compared unfairly:

    3-bit / uniform-4-bit                floor and the standing champion
    mixed o{4,6,8} + down_proj@3         no patches at all
    patch pb3, proportional allocation   current design (global top-k)
    patch pb3, attention-weighted        the (a) hypothesis, budget pushed
                                         toward o_proj
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
OUT_PATH = os.environ.get("SQ_OUT", "results/allocation_showdown.json")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "code,math,knowledge,chat").split(","))
GROUP_SIZE = 128
ATTN, MLP = "self_attn.o_proj", "mlp.down_proj"
MODULE_TYPES = (ATTN, MLP)
BASE_BITS = 3
PATCH_BITS = 3
BUDGET_FRAC = 0.05
MIXED = ((4, 3), (6, 3), (8, 3))          # (o_proj bits, down_proj bits)
ATTN_ALLOC = (None, 0.5, 1.0)             # None = proportional (global top-k)
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

    attn_names = [n for n in order if ATTN in n]
    mlp_names = [n for n in order if MLP in n]
    n_w = {n: w_orig[n].numel() for n in order}
    tot_w = sum(n_w.values())
    grp = {n: n_w[n] // GROUP_SIZE for n in order}
    tot_g = sum(grp.values())
    attn_g = sum(grp[n] for n in attn_names)
    mlp_g = sum(grp[n] for n in mlp_names)
    print(f"o_proj {sum(n_w[n] for n in attn_names)/1e6:.1f}M "
          f"({100*sum(n_w[n] for n in attn_names)/tot_w:.1f}%)  "
          f"down_proj {sum(n_w[n] for n in mlp_names)/1e6:.1f}M", flush=True)

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled = torch.cat([data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0)
    print("pooled Hessian ...", flush=True)
    H_pool = accum_H(model, layers, pooled, dev)

    print("bases ...", flush=True)
    qcache = {}
    for b in (3, 4, 6, 8):
        qcache[b] = {}
        for n in order:
            qcache[b][n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE)
        print(f"  {b}-bit", flush=True)

    def bits_per_weight(bit_of):
        return sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w

    budget = patch_bytes(int(BUDGET_FRAC * tot_g), GROUP_SIZE, 16)

    def build_patch(H_src, obs, attn_alloc):
        """attn_alloc: fraction of patch BYTES spent on o_proj. None =
        proportional, i.e. a single global top-k that lets the scores decide."""
        if attn_alloc is None:
            ng = groups_for_budget(tot_g, GROUP_SIZE, PATCH_BITS, budget)
            flat = torch.cat([obs[n].flatten() for n in order])
            mflat = top_k_group_mask(flat, ng / flat.numel())
            masks, i = {}, 0
            for n in order:
                c = grp[n]
                masks[n] = mflat[i : i + c].reshape(obs[n].shape)
                i += c
        else:
            masks = {}
            for names, total, alloc in (
                (attn_names, attn_g, attn_alloc),
                (mlp_names, mlp_g, 1.0 - attn_alloc),
            ):
                want = groups_for_budget(total, GROUP_SIZE, PATCH_BITS, int(alloc * budget))
                want = min(want, total)  # cannot patch more groups than exist
                sub = torch.cat([obs[n].flatten() for n in names])
                mf = top_k_group_mask(sub, want / sub.numel()) if want else torch.zeros_like(sub, dtype=torch.bool)
                i = 0
                for n in names:
                    c = grp[n]
                    masks[n] = mf[i : i + c].reshape(obs[n].shape)
                    i += c

        wd, used = {}, 0
        for n in order:
            mk_ = masks[n]
            used += int(mk_.sum().item())
            Hg = H_src[n].to(dev)
            solved = solve_residual_layer(
                w_orig[n].to(dev), qcache[BASE_BITS][n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
            wd[n] = qcache[BASE_BITS][n] + quantize_delta(solved, mk_.cpu(), GROUP_SIZE, PATCH_BITS)
            if dev == "cuda":
                torch.cuda.empty_cache()
        eb = BASE_BITS + 2 * 16 / GROUP_SIZE + (
            patch_bytes(used, GROUP_SIZE, PATCH_BITS) * 8 / tot_w
        )
        return wd, used, eb

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"model": MODEL_ID, "ppl": {}, "bits": {}, "sig": {}}

    for t in TASKS:
        print(f"\n=== {t} ===", flush=True)
        H_t = accum_H(model, layers, data[t]["calib"][:N_CALIB_SEQ], dev)
        obs = {}
        for n in order:
            Hg = H_t[n].to(dev)
            obs[n] = score_obs_groups(w_orig[n].to(dev), qcache[BASE_BITS][n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()

        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache, bits = {}, {}, {}

        def score(key, wd, eb):
            s = set_w(wd) if wd else {}
            cache[key] = chunked_nll(model, ev, dev)
            if s:
                unset(s)
            row[key] = ppl(cache[key])
            bits[key] = eb
            print(f"  {key:22s} ppl={row[key]:8.4f}  bits/w={eb:.3f}", flush=True)

        score("full_precision", None, 16.0)
        score("3bit", qcache[3], bits_per_weight(lambda n: 3))
        score("uniform4bit", qcache[4], bits_per_weight(lambda n: 4))

        for bo, bd in MIXED:
            wd = {n: qcache[bo][n] if ATTN in n else qcache[bd][n] for n in order}
            score(f"mixed_o{bo}_d{bd}", wd, bits_per_weight(lambda n, bo=bo, bd=bd: bo if ATTN in n else bd))

        for alloc in ATTN_ALLOC:
            wd, used, eb = build_patch(H_t, obs, alloc)
            tag = "prop" if alloc is None else f"attn{int(alloc*100)}"
            score(f"patch_pb3_{tag}", wd, eb)
            del wd

        results["ppl"][t] = row
        results["bits"][t] = bits
        best_mixed = min((k for k in row if k.startswith("mixed_")), key=lambda k: row[k])
        best_patch = min((k for k in row if k.startswith("patch_")), key=lambda k: row[k])
        results["sig"][t] = {
            f"{best_patch}_beats_{best_mixed}": boot(cache[best_patch], cache[best_mixed]),
            f"{best_patch}_beats_uniform4": boot(cache[best_patch], cache["uniform4bit"]),
            f"{best_mixed}_beats_uniform4": boot(cache[best_mixed], cache["uniform4bit"]),
        }
        for k, v in results["sig"][t].items():
            print(f"    {k:44s} p={v:.3f}")
        del H_t, obs
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
