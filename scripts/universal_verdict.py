"""The synthesis config: mix43 base + a small UNIVERSAL 3-bit correction patch.

This is the honest distillation of what survived the whole study:
  - mixed-precision base (attention@4, MLP@3) -- the robust efficiency win;
  - a UNIVERSAL (pooled-calibration) patch, not task-specific -- because task
    conditioning does not pay (a universal patch works as well);
  - a RESTORED/solved 3-bit patch, not learned -- because learned patches are
    a perplexity mirage that collapse on accuracy;
  - placed on the most quantization-damaged groups (OBS activation-space score).

Target: near-4-bit quality at slightly lower memory. The decisive test is
GSM8K accuracy, not perplexity -- the closest prior measurement used a
task-specific patch, and perplexity has repeatedly flattered configs in this
project. We report both metrics and the effective bits/weight for:

    base (mix43) | base + universal patch | uniform-4 | fp16

If base+universal-patch matches uniform-4 accuracy below 4.25 b/w, the config
is a genuine (if modest) win. If not, we say so.
"""
import json
import math
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT = os.environ.get("SQ_OUT", "results/universal_verdict.json")
DATA_DIR = "results/calibration_longform"
TASKS = ("code", "math", "knowledge", "chat")
GROUP_SIZE = 128
ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN + MLP
PATCH_BITS = 3
BUDGET_FRAC = float(os.environ.get("SQ_BUDGET_FRAC", "0.05"))  # of full surface
N_CALIB_SEQ = 32
N_EVAL_SEQ = 16
N_PROBLEMS = int(os.environ.get("SQ_N_PROBLEMS", "200"))
N_SHOT = 4
LOGIT_CHUNK = 256


def dev_of():
    return "cuda" if torch.cuda.is_available() else "cpu"


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


ANS_RE = re.compile(r"-?[\d,]*\.?\d+")


def extract(text):
    text = text.split("Question:")[0]
    nums = ANS_RE.findall(text.replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev)
    model.config.use_cache = True
    layers = target_layers(model)
    order = sorted(layers)
    w_orig = {n: layers[n].weight.data.float().cpu() for n in order}
    n_w = {n: w_orig[n].numel() for n in order}
    tot_w = sum(n_w.values())
    grp = {n: n_w[n] // GROUP_SIZE for n in order}
    tot_g = sum(grp.values())
    print(f"full surface {tot_w/1e6:.0f}M weights, {tot_g} groups", flush=True)

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    # UNIVERSAL calibration: pooled across all tasks
    pooled = torch.cat([data[t]["calib"][: N_CALIB_SEQ // len(TASKS)] for t in TASKS], 0)
    print(f"universal (pooled) calibration: {pooled.shape[0]} seqs", flush=True)
    H = accum_H(model, layers, pooled, dev)

    print("building mix43 base, uniform-4, universal patch ...", flush=True)
    mix43 = {n: gptq_quantize(w_orig[n], H[n], bits=(4 if is_attn(n) else 3), group_size=GROUP_SIZE)[0]
             for n in order}
    uni4 = {n: gptq_quantize(w_orig[n], H[n], bits=4, group_size=GROUP_SIZE)[0] for n in order}

    # universal patch: pooled OBS score -> most-damaged groups -> solve -> 3-bit
    obs = {}
    for n in order:
        Hg = H[n].to(dev)
        obs[n] = score_obs_groups(w_orig[n].to(dev), mix43[n].to(dev), Hg, GROUP_SIZE)
        del Hg
    if dev == "cuda":
        torch.cuda.empty_cache()
    flat = torch.cat([obs[n].flatten() for n in order])
    budget = patch_bytes(int(BUDGET_FRAC * tot_g), GROUP_SIZE, 16)
    ng = groups_for_budget(tot_g, GROUP_SIZE, PATCH_BITS, budget, bitmap=True)
    mflat = top_k_group_mask(flat, ng / flat.numel())
    patched, i, used = {}, 0, 0
    for n in order:
        c = grp[n]
        mk_ = mflat[i : i + c].reshape(obs[n].shape)
        i += c
        used += int(mk_.sum().item())
        Hg = H[n].to(dev)
        sol = solve_residual_layer(w_orig[n].to(dev), mix43[n].to(dev), Hg, mk_, GROUP_SIZE).cpu()
        del Hg
        patched[n] = mix43[n] + quantize_delta(sol.to(mix43[n].dtype), mk_.cpu(), GROUP_SIZE, PATCH_BITS)
        if dev == "cuda":
            torch.cuda.empty_cache()

    def bpw(bit_of, patch_groups=0):
        v = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w
        if patch_groups:
            v += patch_bytes(patch_groups, GROUP_SIZE, PATCH_BITS, total_groups=tot_g) * 8 / tot_w
        return v

    configs = {
        "base_mix43": (mix43, bpw(lambda n: 4 if is_attn(n) else 3)),
        "base+universal_patch": (patched, bpw(lambda n: 4 if is_attn(n) else 3, used)),
        "uniform4": (uni4, bpw(lambda n: 4)),
        "fp16": (None, 16.0),
    }
    print(f"patch covers {100*used/tot_g:.1f}% of groups", flush=True)

    def set_w(wd):
        s = {}
        for n in order:
            s[n] = layers[n].weight.data.clone()
            layers[n].weight.data = wd[n].to(layers[n].weight.dtype).to(dev)
        return s

    def unset(s):
        for n in order:
            layers[n].weight.data = s[n]

    # ---- perplexity on math eval (where the config is meant to shine) ----
    results = {"model": MODEL_ID, "budget_frac": BUDGET_FRAC, "patch_coverage": used / tot_g,
               "ppl": {}, "bits": {}, "gsm8k": {}}
    ev = data["math"]["eval"][:N_EVAL_SEQ, :256]
    for tag, (wd, b) in configs.items():
        s = set_w(wd) if wd else {}
        results["ppl"][tag] = ppl(chunked_nll(model, ev, dev))
        results["bits"][tag] = b
        if s:
            unset(s)
        print(f"  {tag:22s} ppl={results['ppl'][tag]:.4f}  bits/w={b:.3f}", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- GSM8K accuracy (the decisive metric) ----
    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    shots = [(train[i]["question"], train[i]["answer"]) for i in range(N_SHOT)]

    def gold(a):
        return float(a.split("####")[-1].strip().replace(",", ""))

    def prompt(q):
        return "\n\n".join([f"Question: {sq}\nAnswer: {sa}" for sq, sa in shots]
                           + [f"Question: {q}\nAnswer:"])

    probs = [(test[i]["question"], gold(test[i]["answer"])) for i in range(N_PROBLEMS)]
    per_item = {}
    for tag, (wd, b) in configs.items():
        s = set_w(wd) if wd else {}
        correct, items = 0, []
        for j, (q, g) in enumerate(probs):
            ids = tok(prompt(q), return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
            gen = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            p = extract(gen)
            ok = p is not None and abs(p - g) < 1e-4
            correct += ok
            items.append(bool(ok))
            del o, ids
        if s:
            unset(s)
        results["gsm8k"][tag] = correct / len(probs)
        per_item[tag] = items
        print(f"  GSM8K {tag:22s} {100*correct/len(probs):.1f}%  bits/w={b:.3f}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # significance: patch vs uniform4 (does patch match uniform-4 at lower mem?)
    import random
    a = per_item["base+universal_patch"]
    bb = per_item["uniform4"]
    rng = random.Random(0)
    idx = range(len(a))
    wins = ge = 0
    for _ in range(4000):
        sm = [rng.choice(idx) for _ in idx]
        la, lb = sum(a[i] for i in sm), sum(bb[i] for i in sm)
        wins += la > lb
        ge += la >= lb
    results["patch_beats_uniform4_p"] = wins / 4000
    results["patch_ge_uniform4_p"] = ge / 4000
    print(f"\npatch > uniform4: p={wins/4000:.3f}  patch >= uniform4: p={ge/4000:.3f}")
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
