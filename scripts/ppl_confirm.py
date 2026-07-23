"""PPL confirmation of the 3-bit-patch result.

The patch-precision optimum (3-4 bits, not fp16) and the near-parity-with-
uniform claim were both established in ACTIVATION-SPACE reconstruction error
on probe layers. That metric ranks designs well -- it independently picked the
same optimum the earlier PPL sweep found -- but a claim as load-bearing as
"patches match uniform quantization" should not rest on a proxy, especially
since activation-space error previously disagreed with PPL about how much
task-to-task variation exists.

This runs the decisive comparison end to end on real perplexity:

    3-bit base                         floor
    3-bit base + 3-bit patch           the optimised design
    3-bit base + fp16 patch            the original design, same bytes
    uniform 4-bit                      the alternative to beat

Two predictions are being tested: that pb3 beats pb16 in PPL (confirming the
diagnostic transfers), and how much of the gap to uniform that closes.

Performance note: Hessians are accumulated ON THE GPU here. The earlier sweep
put them on CPU, which meant every x^T x -- 2048 x 4864 by 4864 for one
down_proj sequence -- ran on CPU. That, not the solve, is what made a single
task take over an hour.
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
OUT_PATH = os.environ.get("SQ_OUT", "results/ppl_confirm.json")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "code,math,knowledge,chat").split(","))
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
BASE_BITS = 3
PATCH_BITS = (3, 16)
BUDGET_FRAC = 0.05
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


def accum_H(model, layers, seqs, dev, h_dev):
    hs = {n: GPTQHessian(m.in_features, device=h_dev) for n, m in layers.items()}
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
    # move to CPU for storage; the expensive accumulation is already done
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


def eff_bits(bits, k, pb):
    """Amortised bits/weight, counting the patch's own scales and indices."""
    base = bits + 2 * 16 / GROUP_SIZE
    return base + k * (pb + (0 if pb == 16 else 2 * 16 / GROUP_SIZE) + 64 / GROUP_SIZE)


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
    print(f"device={dev} layers={len(layers)} weights={n_weights/1e6:.1f}M", flush=True)

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled = torch.cat([data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0)

    h_dev = dev  # accumulate on GPU; see module docstring
    print("pooled Hessian (GPU accumulation) ...", flush=True)
    H_pool = accum_H(model, layers, pooled, dev, h_dev)

    print("bases ...", flush=True)
    base_w, uni_w = {}, {}
    for n in order:
        base_w[n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=BASE_BITS, group_size=GROUP_SIZE)
        uni_w[n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=BASE_BITS + 1, group_size=GROUP_SIZE)

    budget = patch_bytes(int(BUDGET_FRAC * total_groups), GROUP_SIZE, 16)
    print(f"budget {budget/1e6:.1f} MB", flush=True)

    def build(H_src, obs, pb):
        ng = groups_for_budget(total_groups, GROUP_SIZE, pb, budget)
        flat = torch.cat([obs[n].flatten() for n in order])
        mflat = top_k_group_mask(flat, ng / flat.numel())
        wd, i = {}, 0
        for n in order:
            sh = obs[n].shape
            c = sh[0] * sh[1]
            mk_ = mflat[i : i + c].reshape(sh)
            i += c
            Hg = H_src[n].to(dev)
            solved = solve_residual_layer(
                w_orig[n].to(dev), base_w[n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
            wd[n] = base_w[n] + quantize_delta(solved, mk_.cpu(), GROUP_SIZE, pb)
            if dev == "cuda":
                torch.cuda.empty_cache()
        return wd, ng

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"model": MODEL_ID, "ppl": {}, "sig": {}, "eff_bits": {}}

    for t in TASKS:
        print(f"\n=== {t} ===", flush=True)
        H_t = accum_H(model, layers, data[t]["calib"][:N_CALIB_SEQ], dev, h_dev)
        obs = {}
        for n in order:
            Hg = H_t[n].to(dev)
            obs[n] = score_obs_groups(w_orig[n].to(dev), base_w[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()

        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache = {}, {}

        def score(key, wd):
            s = set_w(wd) if wd else {}
            cache[key] = chunked_nll(model, ev, dev)
            if s:
                unset(s)
            row[key] = ppl(cache[key])
            print(f"  {key:16s} {row[key]:.4f}", flush=True)

        score("full_precision", None)
        score(f"{BASE_BITS}bit", base_w)
        score(f"uniform{BASE_BITS+1}bit", uni_w)
        for pb in PATCH_BITS:
            wd, ng = build(H_t, obs, pb)
            results["eff_bits"][f"patch_pb{pb}"] = eff_bits(BASE_BITS, ng / total_groups, pb)
            score(f"patch_pb{pb}", wd)
            del wd
        results["eff_bits"][f"uniform{BASE_BITS+1}bit"] = (BASE_BITS + 1) + 2 * 16 / GROUP_SIZE

        results["ppl"][t] = row
        results["sig"][t] = {
            "pb3_beats_pb16": boot(cache["patch_pb3"], cache["patch_pb16"]),
            "pb3_beats_base": boot(cache["patch_pb3"], cache[f"{BASE_BITS}bit"]),
            "pb3_beats_uniform": boot(cache["patch_pb3"], cache[f"uniform{BASE_BITS+1}bit"]),
        }
        for k, v in results["sig"][t].items():
            print(f"    {k:20s} p={v:.3f}")
        del H_t, obs
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
