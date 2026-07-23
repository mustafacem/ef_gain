"""GSM8K accuracy across the whole Pareto frontier, not just one point.

The first GSM8K run established that perplexity badly understates quantization
damage: uniform 4-bit costs +2.8% PPL but -31.9% GSM8K accuracy, and uniform
3-bit costs +31.3% PPL but -75.4% accuracy. Every frontier conclusion in this
project is PPL-based and therefore describes a rosier world than the one a
user would actually experience.

So the frontier needs re-measuring in the metric that matters. This scores
accuracy at points spanning ~2.6 to ~5.1 bits/weight, which is the full range
the band experiments cover:

    uniform 2/3/4/5-bit                the integer rungs
    mixed attn@(B+1) + mlp@B           the cheap asymmetric points
    base + pb3 patch                   the composite, at two budgets

The question is no longer "does the composite tie uniform 4-bit" (answered:
yes, p=0.406) but "where on the frontier does accuracy actually survive, and
do patches still hold their frontier positions once measured honestly".

Accuracy is far more sensitive than PPL, so config ordering may differ from
the PPL frontier -- that is precisely what this is for.
"""
import json
import os
import random
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/gsm8k_frontier.json")
DATA_DIR = "results/calibration_longform"
GROUP_SIZE = 128
ATTN_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TYPES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN_TYPES + MLP_TYPES
N_PROBLEMS = int(os.environ.get("SQ_N_PROBLEMS", "200"))
N_SHOT = 4
MAX_NEW = 256
N_CALIB_SEQ = 24
PATCH_BITS = 3


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
    text = text.split("Question:")[0]
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
    return "\n\n".join([f"Question: {sq}\nAnswer: {sa}" for sq, sa in shots]
                       + [f"Question: {q}\nAnswer:"])


def paired(a, b, n=4000, seed=0):
    rng = random.Random(seed)
    idx = range(len(a))
    w = 0
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        if sum(a[i] for i in s) > sum(b[i] for i in s):
            w += 1
    return w / n


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
    print(f"GSM8K frontier: {len(probs)} problems, {N_SHOT}-shot", flush=True)

    calib = torch.load(f"{DATA_DIR}/math.pt")["calib"][:N_CALIB_SEQ]
    print("Hessians ...", flush=True)
    H = accum_H(model, layers, calib, dev)

    print("bases ...", flush=True)
    qc = {}
    for b in (2, 3, 4, 5):
        qc[b] = {n: gptq_quantize(w_orig[n], H[n], bits=b, group_size=GROUP_SIZE)[0]
                 for n in order}
        print(f"  {b}-bit", flush=True)

    def bpw(bit_of, patch_groups=0, pb=PATCH_BITS):
        v = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order) / tot_w
        if patch_groups:
            v += patch_bytes(patch_groups, GROUP_SIZE, pb, total_groups=tot_g) * 8 / tot_w
        return v

    def make_patch(base, bf, pb=PATCH_BITS):
        obs = {}
        for n in order:
            Hg = H[n].to(dev)
            obs[n] = score_obs_groups(w_orig[n].to(dev), base[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        flat = torch.cat([obs[n].flatten() for n in order])
        budget = patch_bytes(int(bf * tot_g), GROUP_SIZE, 16)
        ng = groups_for_budget(tot_g, GROUP_SIZE, pb, budget, bitmap=True)
        mflat = top_k_group_mask(flat, ng / flat.numel())
        wd, i, used = {}, 0, 0
        for n in order:
            c = grp[n]
            mk_ = mflat[i : i + c].reshape(obs[n].shape)
            i += c
            used += int(mk_.sum().item())
            Hg = H[n].to(dev)
            solved = solve_residual_layer(
                w_orig[n].to(dev), base[n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
            wd[n] = base[n] + quantize_delta(solved, mk_.cpu(), GROUP_SIZE, pb)
            if dev == "cuda":
                torch.cuda.empty_cache()
        del obs, flat
        return wd, used

    mix32 = {n: qc[3][n] if is_attn(n) else qc[2][n] for n in order}
    mix43 = {n: qc[4][n] if is_attn(n) else qc[3][n] for n in order}
    mix54 = {n: qc[5][n] if is_attn(n) else qc[4][n] for n in order}

    print("building patched configs ...", flush=True)
    p_mix32, u_mix32 = make_patch(mix32, 0.05)
    p_mix43, u_mix43 = make_patch(mix43, 0.05)
    p_mix54, u_mix54 = make_patch(mix54, 0.05)

    configs = [
        ("fp16", None, 16.0),
        ("uniform2bit", qc[2], bpw(lambda n: 2)),
        ("mix32", mix32, bpw(lambda n: 3 if is_attn(n) else 2)),
        ("mix32+pb3", p_mix32, bpw(lambda n: 3 if is_attn(n) else 2, u_mix32)),
        ("uniform3bit", qc[3], bpw(lambda n: 3)),
        ("mix43", mix43, bpw(lambda n: 4 if is_attn(n) else 3)),
        ("mix43+pb3", p_mix43, bpw(lambda n: 4 if is_attn(n) else 3, u_mix43)),
        ("uniform4bit", qc[4], bpw(lambda n: 4)),
        ("mix54", mix54, bpw(lambda n: 5 if is_attn(n) else 4)),
        ("mix54+pb3", p_mix54, bpw(lambda n: 5 if is_attn(n) else 4, u_mix54)),
        ("uniform5bit", qc[5], bpw(lambda n: 5)),
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

    results = {"model": MODEL_ID, "n_problems": len(probs), "acc": {}, "bits": {}, "per_item": {}}
    model.eval()
    for tag, wd, bits in configs:
        s = set_w(wd) if wd else {}
        correct, per_item = 0, []
        for j, (q, gold) in enumerate(probs):
            ids = tok(build_prompt(shots, q), return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            ok = pred is not None and abs(pred - gold) < 1e-4
            correct += ok
            per_item.append(bool(ok))
            del out
        if s:
            unset(s)
        results["acc"][tag] = correct / len(probs)
        results["bits"][tag] = bits
        results["per_item"][tag] = per_item
        print(f"  {tag:16s} bits/w={bits:6.3f}  acc={100*correct/len(probs):5.1f}%", flush=True)
        if dev == "cuda":
            torch.cuda.empty_cache()

    pi = results["per_item"]
    results["sig"] = {
        "mix43+pb3_vs_uniform4": paired(pi["mix43+pb3"], pi["uniform4bit"]),
        "mix54+pb3_vs_uniform5": paired(pi["mix54+pb3"], pi["uniform5bit"]),
        "mix32+pb3_vs_uniform3": paired(pi["mix32+pb3"], pi["uniform3bit"]),
    }
    print("\n=== significance (p that patch config > uniform rung) ===")
    for k, v in results["sig"].items():
        print(f"  {k:28s} p={v:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
