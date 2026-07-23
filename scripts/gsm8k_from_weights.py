"""GSM8K accuracy from saved weight dicts, one config at a time.

The inline GSM8K eval OOM'd because generation memory accumulated across
configs on a nearly-full card. The trained weights were saved, so this loads
them and evaluates each config in isolation, freeing the KV cache and model
state between configs. Decisive check: does the learned math patch's
perplexity parity with uniform-4 hold as *accuracy* parity?
"""
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
WEIGHTS = os.environ.get("SQ_WEIGHTS", "results/learned_gsm8k_math_math_weights.pt")
OUT = os.environ.get("SQ_OUT", "results/gsm8k_learned_verdict.json")
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
N_PROBLEMS = int(os.environ.get("SQ_N_PROBLEMS", "200"))
N_SHOT = 4
ONLY = os.environ.get("SQ_ONLY", "")  # comma list to restrict configs


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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev)
    model.config.use_cache = True
    model.eval()
    layers = target_layers(model)

    wd_all = torch.load(WEIGHTS, map_location="cpu")
    configs = list(wd_all.keys())
    if ONLY:
        configs = [c for c in ONLY.split(",") if c in wd_all]
    print(f"weights: {WEIGHTS}  configs: {configs}", flush=True)

    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    shots = [(train[i]["question"], train[i]["answer"]) for i in range(N_SHOT)]

    def gold(a):
        return float(a.split("####")[-1].strip().replace(",", ""))

    def prompt(q):
        return "\n\n".join([f"Question: {sq}\nAnswer: {sa}" for sq, sa in shots]
                           + [f"Question: {q}\nAnswer:"])

    probs = [(test[i]["question"], gold(test[i]["answer"])) for i in range(N_PROBLEMS)]

    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))

    for tag in configs:
        # install this config's weights, then drop the big CPU dict copy
        wd = wd_all[tag]
        for n, m in layers.items():
            m.weight.data = wd[n].to(m.weight.dtype).to(dev)
        correct, per_item = 0, []
        for j, (q, g) in enumerate(probs):
            ids = tok(prompt(q), return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                   pad_token_id=tok.pad_token_id, use_cache=True)
            gen = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            p = extract(gen)
            ok = p is not None and abs(p - g) < 1e-4
            correct += ok
            per_item.append(bool(ok))
            del o, ids
            if (j + 1) % 50 == 0:
                print(f"  {tag}: {j+1}/{len(probs)} acc={correct/(j+1):.3f}", flush=True)
        results[tag] = {"acc": correct / len(probs), "per_item": per_item}
        print(f"== {tag}: {100*results[tag]['acc']:.1f}% ({correct}/{len(probs)}) ==", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # paired bootstrap: learned vs uniform4
    import random
    if "learned_3bit" in results and "uniform4" in results:
        a = results["learned_3bit"]["per_item"]
        b = results["uniform4"]["per_item"]
        rng = random.Random(0)
        idx = range(len(a))
        wins = ties = 0
        for _ in range(4000):
            s = [rng.choice(idx) for _ in idx]
            la = sum(a[i] for i in s)
            lb = sum(b[i] for i in s)
            if la > lb:
                wins += 1
            elif la == lb:
                ties += 1
        results["learned_beats_uniform4_p"] = wins / 4000
        results["learned_ge_uniform4_p"] = (wins + ties) / 4000
        print(f"\nlearned > uniform4: p={wins/4000:.3f}  learned >= uniform4: p={(wins+ties)/4000:.3f}")
        json.dump(results, open(OUT, "w"), indent=2)

    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
