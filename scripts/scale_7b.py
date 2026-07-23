"""Scale test: does the patch approach behave differently on a large model?

Everything so far ran at 0.5B (and one structural probe at 1.5B), where the
verdict was that uniform precision out-competes patches -- but the 1.5B probe
showed the gap *halving*, so the open question is whether it closes at scale.

Nothing here can hold the model in VRAM: 7B in bf16 is ~15.2 GB against ~4.7 GB
free. Everything therefore runs through selfquant.quant.streaming, which keeps
the model on CPU and moves one transformer block (~543 MB) to the GPU at a
time. Hessians are built and discarded per block -- a single 7B down_proj
Hessian is 18944^2 x 4 B = 1.44 GB, so keeping one per layer would need ~40 GB.

The grid is deliberately small. At 0.5B a 24-config sweep was affordable; here
each eval is ~10x slower, so only the comparisons that decide something are
run:

    base            floor
    uniform b+1     the alternative patches must beat
    + own patch     task-conditional, at the best precision found at 0.5B
    + pooled patch  the control that separates task-conditioning from
                    generic repair (the comparison that retracted the
                    "code exception" at 0.5B)

SQ_MODEL selects the model so the same script covers 1.5B / 3B / 7B.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import gptq_quantize
from selfquant.quant.streaming import stream_quantize, streamed_nll
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/scale_7b.json")
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "code,chat").split(","))
BASE_BITS = int(os.environ.get("SQ_BASE_BITS", "3"))
PATCH_BITS = int(os.environ.get("SQ_PATCH_BITS", "4"))
BUDGET_FRAC = float(os.environ.get("SQ_BUDGET", "0.05"))
N_CALIB_SEQ = int(os.environ.get("SQ_CALIB_SEQ", "16"))
N_EVAL_SEQ = int(os.environ.get("SQ_EVAL_SEQ", "8"))
N_BOOT = 4000


def dev_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


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


def load_model(dev):
    # low_cpu_mem_usage keeps peak RAM near the checkpoint size rather than
    # double it; the model stays on CPU and streaming moves blocks to GPU.
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()


def build_bases(model, calib_ids, dev):
    """Pass 1: quantize each layer as its Hessian becomes available.

    A single 7B down_proj Hessian is 18944^2 x 4 B = 1.44 GB, so storing one
    per layer would need ~41 GB -- and three sources (pooled + one per task)
    would exceed RAM outright. Consuming each Hessian inside the streaming
    callback keeps exactly one live at a time, which is the whole reason this
    is tractable on a 6 GB card with 48 GB of RAM.
    """
    w_orig, base_w, uni_w = {}, {}, {}

    def qf(name, w, H):
        w_orig[name] = w
        base_w[name], _, _ = gptq_quantize(w, H, bits=BASE_BITS, group_size=GROUP_SIZE)
        uni_w[name], _, _ = gptq_quantize(w, H, bits=BASE_BITS + 1, group_size=GROUP_SIZE)
        return None  # leave the model at full precision for now

    print("  [pooled] streaming: Hessian -> base + uniform, per layer", flush=True)
    stream_quantize(model, calib_ids, MODULE_TYPES, dev, qf)
    return w_orig, base_w, uni_w


def build_patch_streaming(model, calib_ids, dev, w_orig, base_w, n_groups_frac, tag):
    """Pass 2: score and solve each layer's patch as its Hessian arrives.

    Selection is per-layer rather than global top-k across the model, because
    a global threshold would require all layers' scores -- and therefore all
    Hessians -- to exist simultaneously, which is exactly what does not fit.
    Each layer gets the same fraction of its own groups, so the byte budget
    is still matched.
    """
    patched = {}

    def qf(name, w, H):
        Hg = H.to(dev)
        wo, q = w_orig[name].to(dev), base_w[name].to(dev)
        obs = score_obs_groups(wo, q, Hg, GROUP_SIZE)
        mask = top_k_group_mask(obs, n_groups_frac)
        solved = solve_residual_layer(wo, q, Hg, mask, GROUP_SIZE).cpu()
        patched[name] = base_w[name] + quantize_delta(
            solved, mask.cpu(), GROUP_SIZE, PATCH_BITS
        )
        del Hg, wo, q, solved
        if dev == "cuda":
            torch.cuda.empty_cache()
        return None

    print(f"  [{tag}] streaming: Hessian -> score + solve + quantize, per layer", flush=True)
    stream_quantize(model, calib_ids, MODULE_TYPES, dev, qf)
    return patched


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"model={MODEL_ID} device={dev}")

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled = torch.cat(
        [data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0
    )
    print(f"pooled calibration: {pooled.shape[0]} seqs x {pooled.shape[1]} tok "
          f"({pooled.numel()} tokens)")

    model = load_model(dev)
    n_blocks = len(model.model.layers)
    print(f"blocks={n_blocks}")

    # --- pass 1: bases, built layer-by-layer as Hessians arrive -----------
    w_orig, base_w, uni_w = build_bases(model, pooled, dev)
    order = sorted(w_orig)
    n_weights = sum(w.numel() for w in w_orig.values())
    total_groups = n_weights // GROUP_SIZE
    print(f"targeted {len(order)} layers, {n_weights/1e6:.0f}M weights, {total_groups} groups")

    # Budget: express the fp16-equivalent budget as a fraction of groups, then
    # convert to how many groups PATCH_BITS can afford at the same bytes.
    fp16_groups = int(BUDGET_FRAC * total_groups)
    budget_bytes = patch_bytes(fp16_groups, GROUP_SIZE, 16)
    n_groups = groups_for_budget(total_groups, GROUP_SIZE, PATCH_BITS, budget_bytes)
    frac = n_groups / total_groups
    print(f"budget {budget_bytes/1e6:.1f} MB -> {100*frac:.1f}% of groups at {PATCH_BITS}-bit "
          f"(fp16 would afford {100*BUDGET_FRAC:.1f}%)")

    # --- pass 2: patches, each layer solved as its Hessian arrives ---------
    patch_pooled = build_patch_streaming(model, pooled, dev, w_orig, base_w, frac, "pooled")
    patch_own = {
        t: build_patch_streaming(
            model, data[t]["calib"][:N_CALIB_SEQ], dev, w_orig, base_w, frac, t
        )
        for t in TASKS
    }

    # --- evaluate ---------------------------------------------------------
    linears = {}
    for i, blk in enumerate(model.model.layers):
        for mt in MODULE_TYPES:
            m = blk
            for p in mt.split("."):
                m = getattr(m, p)
            linears[f"layer{i}.{mt}"] = m

    def set_w(wd):
        saved = {}
        for n, m in linears.items():
            if n in wd:
                saved[n] = m.weight.data
                m.weight.data = wd[n].to(m.weight.dtype)
        return saved

    def unset(saved):
        for n, w in saved.items():
            linears[n].weight.data = w

    results = {
        "model": MODEL_ID, "n_weights": n_weights, "base_bits": BASE_BITS,
        "patch_bits": PATCH_BITS, "budget_frac": BUDGET_FRAC,
        "n_patched_groups": n_groups, "ppl": {}, "sig": {},
    }

    for t in TASKS:
        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache = {}, {}

        def score(key, wd):
            saved = set_w(wd) if wd else {}
            cache[key] = streamed_nll(model, ev, dev)
            if saved:
                unset(saved)
            row[key] = ppl(cache[key])
            print(f"    {key:22s} {row[key]:.4f}", flush=True)

        print(f"\n=== {t} ===", flush=True)
        score("full_precision", None)
        score(f"{BASE_BITS}bit", base_w)
        score(f"uniform{BASE_BITS+1}bit", uni_w)
        score("own_patch", patch_own[t])
        score("pooled_patch", patch_pooled)

        results["ppl"][t] = row
        results["sig"][t] = {
            "own_beats_base": boot(cache["own_patch"], cache[f"{BASE_BITS}bit"]),
            "own_beats_uniform": boot(cache["own_patch"], cache[f"uniform{BASE_BITS+1}bit"]),
            "own_beats_pooled": boot(cache["own_patch"], cache["pooled_patch"]),
            "pooled_beats_uniform": boot(cache["pooled_patch"], cache[f"uniform{BASE_BITS+1}bit"]),
        }
        for k, v in results["sig"][t].items():
            print(f"      {k:22s} p={v:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
