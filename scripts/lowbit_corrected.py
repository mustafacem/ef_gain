"""4-bit (and 3-bit) base + patch vs. full precision, on properly conditioned data.

Every earlier pass ran on calibration that gave 0.8-1.6 activation samples per
Hessian dimension, so H was singular or nearly so for a 4864-dim layer. That
undermines anything that inverts H: the closed-form solve, OBS scores, and the
GPTQ base itself. This rerun uses results/calibration_longform (26-54
samples/dim, 2048-token sequences, ~98k eval tokens vs ~5k before).

Two questions:

  1. Does a patch on a 4-bit base beat spending the same bits uniformly,
     now that H is trustworthy? (The user's request; 4-bit is the regime
     where patching still has room to matter -- 8-bit is already lossless.)

  2. Does the CODE EXCEPTION survive? Code was the one task where an
     own-task patch beat a pooled/universal one -- and it also had the
     worst conditioning of any task (0.90 samples/dim, i.e. rank-deficient).
     That result is the most likely casualty of the fix, so it is re-tested
     directly rather than assumed.
"""
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_longform"
TASKS = ("code", "math", "knowledge", "chat")
BASE_BITS = (2, 3, 4)
UNIFORM_REF = {2: 3, 3: 4, 4: 5}   # bit-width a patched base is compared against
K_FRACS = (0.01, 0.02, 0.05)
N_CALIB_SEQ = 32               # 32 x 2048 = 65k tokens per task for H
N_EVAL_SEQ = 16                # 16 x 2048 = 33k eval tokens (was ~5k total)
N_BOOT = 4000
LOGIT_CHUNK = 256              # tokens per lm_head slice when scoring (see chunked_nll)


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
    hs = {n: GPTQHessian(m.in_features, device="cpu") for n, m in layers.items()}
    hd = []
    for n, m in layers.items():
        def mk(h, inf):
            def hook(mod, args):
                h.update(args[0].reshape(-1, inf).detach())
            return hook
        hd.append(m.register_forward_pre_hook(mk(hs[n], m.in_features)))
    with torch.no_grad():
        for i in range(seqs.shape[0]):
            # model.model(...) runs the transformer body only. The hooks live
            # on inner linears, so lm_head is never needed here -- and skipping
            # it avoids materialising a [1, 2048, 151936] logit tensor.
            model.model(seqs[i : i + 1].to(dev))
    for h in hd:
        h.remove()
    return {n: h.H for n, h in hs.items()}


def chunked_nll(model, seqs, dev):
    """Per-sequence NLL without ever holding full-sequence logits.

    Qwen's vocab is 151936, so logits for one 2048-token sequence are ~1.2 GB
    in fp32 / 0.6 GB in bf16 -- enough to OOM a 6 GB card on top of the model.
    Instead: run the body once for hidden states, then apply lm_head over
    LOGIT_CHUNK-token slices, accumulating loss and freeing each slice.
    Mathematically identical to model(ids, labels=ids); only the peak differs.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(seqs.shape[0]):
            ids = seqs[i : i + 1].to(dev)
            hidden = model.model(ids).last_hidden_state[0]     # [T, hidden]
            tgt = ids[0, 1:]                                    # next-token targets
            src = hidden[:-1]                                   # aligned states
            total, n = 0.0, src.shape[0]
            for s in range(0, n, LOGIT_CHUNK):
                e = min(s + LOGIT_CHUNK, n)
                logits = model.lm_head(src[s:e]).float()
                total += torch.nn.functional.cross_entropy(
                    logits, tgt[s:e], reduction="sum"
                ).item()
                del logits
            out.append((total, n))
            del hidden, src
    return out


# Kept as the public name used below; the chunked implementation is the
# only one that fits alongside a model on this card.
per_seq_nll = chunked_nll


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


def eff_bits(bits, k=0.0):
    base = bits + 2 * 16 / GROUP_SIZE
    return base + k * (16 - base)


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # bfloat16 here, not fp32: at 2-4 bit the quantization error dwarfs bf16
    # rounding error by orders of magnitude, so the dtype confound that
    # mattered for the 8-bit study is irrelevant, and bf16 halves both model
    # and logit memory.
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)
    layers = target_layers(model)
    order = sorted(layers)
    w_orig = {n: m.weight.data.float().cpu() for n, m in layers.items()}
    print(f"device={dev} layers={len(layers)} dtype={dt}")

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    for t in TASKS:
        print(f"  {t}: calib {tuple(data[t]['calib'].shape)}  eval {tuple(data[t]['eval'].shape)}")

    pooled_seqs = torch.cat([data[t]["calib"][: N_CALIB_SEQ // len(TASKS)] for t in TASKS], 0)
    print(f"\npooled Hessian from {pooled_seqs.shape[0]} seqs "
          f"({pooled_seqs.numel()} tokens = {pooled_seqs.numel()/4864:.1f} samples/dim) ...")
    H_pool = accum_H(model, layers, pooled_seqs, dev)

    # One generically-calibrated base per bit-width. Bits appearing in both
    # BASE_BITS and UNIFORM_REF.values() are quantized once and shared.
    all_bits = sorted(set(BASE_BITS) | set(UNIFORM_REF.values()))
    print(f"quantizing generic bases at {all_bits} ...")
    generic = {}
    for b in all_bits:
        generic[b] = {}
        for n in order:
            generic[b][n], _, _ = gptq_quantize(
                w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE
            )
        print(f"  {b}-bit done")

    print("\nper-task Hessians ...")
    H_task = {}
    for t in TASKS:
        H_task[t] = accum_H(model, layers, data[t]["calib"][:N_CALIB_SEQ], dev)
        print(f"  {t}: {N_CALIB_SEQ*2048/4864:.1f} samples/dim")

    # The baseline no earlier pass ran: a base calibrated entirely on the
    # task's own data. This needs no patch machinery and is standard
    # practice, so it is the alternative a task patch must actually beat --
    # not the generically-calibrated base it has been compared against.
    print("\nquantizing task-calibrated bases ...")
    task_base = {}
    for t in TASKS:
        task_base[t] = {}
        for b in BASE_BITS:
            task_base[t][b] = {}
            for n in order:
                task_base[t][b][n], _, _ = gptq_quantize(
                    w_orig[n], H_task[t][n], bits=b, group_size=GROUP_SIZE
                )
        print(f"  {t} done")

    def _apply_mask(base, H_src, mflat, shapes):
        """Solve the optimal correction on the masked groups and add it."""
        wd, i = {}, 0
        for n in order:
            sh = shapes[n]
            c = sh[0] * sh[1]
            mk_ = mflat[i : i + c].reshape(sh)
            i += c
            Hg = H_src[n].to(dev)
            wd[n] = base[n] + solve_residual_layer(
                w_orig[n].to(dev), base[n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        return wd

    def _group_shapes(base):
        return {
            n: (w_orig[n].shape[0], w_orig[n].shape[1] // GROUP_SIZE) for n in order
        }

    def build_random(base, H_src, k, seed=0):
        """Placebo: same budget and same solve, but groups chosen at random.
        Isolates 'does the ranking matter' from 'does correcting anything
        at this budget help'."""
        shapes = _group_shapes(base)
        g = torch.Generator().manual_seed(seed)
        flat = torch.rand(sum(s[0] * s[1] for s in shapes.values()), generator=g)
        return _apply_mask(base, H_src, top_k_group_mask(flat, k), shapes)

    def build(base, H_src, k):
        obs = {}
        for n in order:
            Hg = H_src[n].to(dev)
            obs[n] = score_obs_groups(w_orig[n].to(dev), base[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        flat = torch.cat([obs[n].flatten() for n in order])
        mflat = top_k_group_mask(flat, k)
        wd, i = {}, 0
        for n in order:
            sh = obs[n].shape
            c = sh[0] * sh[1]
            mk_ = mflat[i : i + c].reshape(sh)
            i += c
            Hg = H_src[n].to(dev)
            wd[n] = base[n] + solve_residual_layer(
                w_orig[n].to(dev), base[n].to(dev), Hg, mk_, GROUP_SIZE
            ).cpu()
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()
        return wd

    def set_w(wd):
        s = {}
        for n, m in layers.items():
            s[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return s

    def unset(s):
        for n, m in layers.items():
            m.weight.data = s[n]

    results = {"ppl": {}, "sig": {}, "eff_bits": {}, "conditioning": {}}
    for b in BASE_BITS:
        results["eff_bits"][f"{b}bit"] = eff_bits(b)
        # a task-calibrated base costs exactly what a generic one costs:
        # calibration data changes the values, not the storage
        results["eff_bits"][f"{b}bit_taskcalib"] = eff_bits(b)
        for k in K_FRACS:
            results["eff_bits"][f"{b}bit+k{int(k*100)}"] = eff_bits(b, k)
    for b in set(UNIFORM_REF.values()):
        results["eff_bits"][f"uniform{b}bit"] = eff_bits(b)
    results["conditioning"] = {"samples_per_dim_task_H": N_CALIB_SEQ * 2048 / 4864}

    for t in TASKS:
        ev = data[t]["eval"][:N_EVAL_SEQ]
        row, cache = {}, {}
        cache["full"] = per_seq_nll(model, ev, dev)
        row["full_precision"] = ppl(cache["full"])

        def score(key, wd):
            s = set_w(wd)
            cache[key] = per_seq_nll(model, ev, dev)
            unset(s)
            row[key] = ppl(cache[key])
            if dev == "cuda":
                torch.cuda.empty_cache()

        for b in BASE_BITS:
            score(f"{b}bit", generic[b])
            ub = UNIFORM_REF[b]
            if f"uniform{ub}bit" not in row:
                score(f"uniform{ub}bit", generic[ub])
            # task-calibrated base at the same bit-width: no patch at all
            score(f"{b}bit_taskcalib", task_base[t][b])

            for k in K_FRACS:
                kk = int(k * 100)
                score(f"{b}bit+own_k{kk}", build(generic[b], H_task[t], k))
                # the control that decides task-conditioning: same base,
                # same budget, universal (pooled) patch
                score(f"{b}bit+pooled_k{kk}", build(generic[b], H_pool, k))
                # placebo: same budget, groups chosen at random
                score(f"{b}bit+random_k{kk}", build_random(generic[b], H_pool, k, seed=kk))

        sig = {}
        for b in BASE_BITS:
            ub = UNIFORM_REF[b]
            for k in K_FRACS:
                kk = int(k * 100)
                own = cache[f"{b}bit+own_k{kk}"]
                sig[f"{b}bit+own_k{kk}_beats_base"] = boot(own, cache[f"{b}bit"])
                sig[f"{b}bit+own_k{kk}_beats_uniform{ub}"] = boot(own, cache[f"uniform{ub}bit"])
                sig[f"{b}bit+own_k{kk}_beats_pooled"] = boot(own, cache[f"{b}bit+pooled_k{kk}"])
                sig[f"{b}bit+own_k{kk}_beats_random"] = boot(own, cache[f"{b}bit+random_k{kk}"])
                # the baseline that decides whether patching is worth any
                # machinery at all, versus just calibrating on task data
                sig[f"{b}bit+own_k{kk}_beats_taskcalib"] = boot(own, cache[f"{b}bit_taskcalib"])
        results["ppl"][t] = row
        results["sig"][t] = sig

        print(f"\n=== {t} ===")
        for kk in sorted(row):
            print(f"  {kk:24s} {row[kk]:.5f}")
        for kk, v in sorted(sig.items()):
            print(f"    {kk:38s} p={v:.3f}")
        if dev == "cuda":
            torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/lowbit_corrected.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/lowbit_corrected.json")


if __name__ == "__main__":
    main()
