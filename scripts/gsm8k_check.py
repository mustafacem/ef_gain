"""Downstream accuracy: does the math parity claim survive on real GSM8K?

Every quality number in this project is perplexity. The most interesting
single result -- that a 3-bit base plus a 3-bit patch is statistically tied
with uniform 4-bit on math (p=0.138) while using LESS memory -- therefore
rests entirely on a proxy. Perplexity and task accuracy come apart routinely,
so the claim is not load-bearing until it is checked against exact-match
accuracy on actual problems.

This runs greedy generation on GSM8K test with a few-shot prompt and scores
strict exact match on the final number. Configs are the ones the frontier
analysis cares about, all at known bits/weight:

    fp16                        ceiling
    uniform 3-bit               floor
    mixed attn@4 + mlp@3        the cheap efficient point (3.373 b/w)
    mix43 + pb3 patch @5%       the best composite (4.198 b/w)
    uniform 4-bit               the standing champion (4.250 b/w)

The honest question is narrow: at ~4.2 bits, does the patch config match
uniform 4-bit on accuracy the way it nearly does on perplexity, or does the
proxy flatter it?

Note on scale: Qwen2.5-0.5B-Instruct is weak at GSM8K (tens of percent), so
absolute numbers are low and the run is powered to detect large differences
only. A null result here means "no detectable difference at this sample
size", not "identical".
"""
import json
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
OUT_PATH = os.environ.get("SQ_OUT", "results/gsm8k_check.json")
DATA_DIR = "results/calibration_longform"
GROUP_SIZE = 128
ATTN_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TYPES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN_TYPES + MLP_TYPES
PATCH_BITS = 3
BUDGET_FRAC = 0.05
N_PROBLEMS = int(os.environ.get("SQ_N_PROBLEMS", "200"))
N_SHOT = 4
MAX_NEW = 256
N_CALIB_SEQ = 24


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


ANS_RE = re.compile(r"-?[\d,]*\.?\d+")


def extract_answer(text):
    """Last number in the completion -- the standard GSM8K strict-ish rule."""
    text = text.split("Question:")[0]  # stop at a hallucinated next problem
    nums = ANS_RE.findall(text.replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def gold_answer(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))


def build_prompt(shots, q):
    parts = []
    for sq, sa in shots:
        parts.append(f"Question: {sq}\nAnswer: {sa}")
    parts.append(f"Question: {q}\nAnswer:")
    return "\n\n".join(parts)


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

    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    shots = [(train[i]["question"], train[i]["answer"]) for i in range(N_SHOT)]
    probs = [(test[i]["question"], gold_answer(test[i]["answer"])) for i in range(N_PROBLEMS)]
    print(f"GSM8K: {len(probs)} problems, {N_SHOT}-shot, greedy", flush=True)

    calib = torch.load(f"{DATA_DIR}/math.pt")["calib"][:N_CALIB_SEQ]
    print("Hessians (math-task calibration) ...", flush=True)
    H = accum_H(model, layers, calib, dev)

    print("bases ...", flush=True)
    qc = {}
    for b in (3, 4):
        qc[b] = {n: gptq_quantize(w_orig[n], H[n], bits=b, group_size=GROUP_SIZE)[0] for n in order}
        print(f"  {b}-bit", flush=True)
    mix43 = {n: qc[4][n] if is_attn(n) else qc[3][n] for n in order}

    def bpw(bit_of, patch_groups=0):
        v = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w
        if patch_groups:
            v += patch_bytes(patch_groups, GROUP_SIZE, PATCH_BITS, total_groups=tot_g) * 8 / tot_w
        return v

    print("building composite patch ...", flush=True)
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
    composite, i, used = {}, 0, 0
    for n in order:
        c = grp[n]
        mk_ = mflat[i : i + c].reshape(obs[n].shape)
        i += c
        used += int(mk_.sum().item())
        Hg = H[n].to(dev)
        solved = solve_residual_layer(w_orig[n].to(dev), mix43[n].to(dev), Hg, mk_, GROUP_SIZE).cpu()
        del Hg
        composite[n] = mix43[n] + quantize_delta(solved, mk_.cpu(), GROUP_SIZE, PATCH_BITS)
        if dev == "cuda":
            torch.cuda.empty_cache()
    del obs, flat

    configs = [
        ("full_precision", None, 16.0),
        ("uniform3bit", qc[3], bpw(lambda n: 3)),
        ("mixed_attn4_mlp3", mix43, bpw(lambda n: 4 if is_attn(n) else 3)),
        ("mix43+pb3_bf5", composite, bpw(lambda n: 4 if is_attn(n) else 3, used)),
        ("uniform4bit", qc[4], bpw(lambda n: 4)),
    ]

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"model": MODEL_ID, "n_problems": len(probs), "n_shot": N_SHOT, "acc": {}, "bits": {}}
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for tag, wd, bits in configs:
        s = set_w(wd) if wd else {}
        correct, per_item = 0, []
        model.eval()
        for j, (q, gold) in enumerate(probs):
            prompt = build_prompt(shots, q)
            ids = tok(prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            ok = pred is not None and abs(pred - gold) < 1e-4
            correct += ok
            per_item.append(bool(ok))
            if (j + 1) % 50 == 0:
                print(f"    {tag}: {j+1}/{len(probs)} acc={correct/(j+1):.3f}", flush=True)
            del out
        if s:
            unset(s)
        acc = correct / len(probs)
        results["acc"][tag] = acc
        results["bits"][tag] = bits
        results.setdefault("per_item", {})[tag] = per_item
        print(f"  {tag:22s} acc={acc:.4f} ({correct}/{len(probs)})  bits/w={bits:.3f}", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # paired bootstrap on the item-level correctness vectors
    import random
    def paired(a, b, n=4000, seed=0):
        rng = random.Random(seed)
        idx = range(len(a))
        w = 0
        for _ in range(n):
            s = [rng.choice(idx) for _ in idx]
            if sum(a[i] for i in s) > sum(b[i] for i in s):
                w += 1
        return w / n

    pi = results["per_item"]
    results["sig"] = {
        "composite_beats_uniform4": paired(pi["mix43+pb3_bf5"], pi["uniform4bit"]),
        "composite_beats_mixed": paired(pi["mix43+pb3_bf5"], pi["mixed_attn4_mlp3"]),
        "composite_beats_u3": paired(pi["mix43+pb3_bf5"], pi["uniform3bit"]),
        "uniform4_beats_fp16": paired(pi["uniform4bit"], pi["full_precision"]),
    }
    print("\n=== significance (p that first > second) ===")
    for k, v in results["sig"].items():
        print(f"  {k:30s} p={v:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
