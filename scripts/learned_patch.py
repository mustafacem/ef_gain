"""Do LEARNED task patches break the restoration ceiling?

Every patch in this project so far only *restores* information already present
in the fp16 model: the closed-form solve finds the best correction that
reconstructs the original layer output. That is a hard ceiling -- a patch can
never make the quantized model behave better than fp16 on the calibration
distribution, only claw back toward it, group by group, locally.

A patch trained end-to-end against the fp16 teacher can do something
restoration provably cannot: compensate for quantization error that propagates
ACROSS layers. The local solve minimizes ||X(W - Q - d)||^2 for one layer in
isolation; distillation minimizes KL(teacher || student) through the whole
stack, so the correction on layer l can absorb error introduced at layers
!= l. This is the one proposed change (idea #4) that escapes the reconstruction
ceiling, and the only one with a real chance of beating uniform-4.

Design (kept honest and matched-memory):
  base    : mixed attn@4 / mlp@3, full surface, FROZEN
  surface : MLP groups (down/gate/up) -- where task structure concentrates
  select  : top-k by Fisher (task-loss gradient^2 x quant-error) -- idea #3
  learn   : a SPARSE trainable residual on the selected groups only, trained
            with KL(fp16 teacher || student) on task calibration text
  store   : quantize the learned residual to 3 bits (matched to the solved patch)
  compare : base | base+solved(3b) | base+learned(3b) | uniform-4,
            all at the SAME effective bits/weight

Success = learned beats solved at matched bytes AND closes materially more of
the gap to uniform-4 than the solved patch does. If learned ~ solved, the
ceiling is real and end-to-end training does not help at this scale.

Scoped to be tractable on 6 GB: 0.5B model, MLP surface, short sequences,
sparse trainable delta (only selected groups carry parameters), teacher logits
cached once.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.patch.residual import score_obs_groups, solve_residual_layer
from selfquant.patch.varbit import groups_for_budget, patch_bytes, quantize_delta
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.scores import top_k_group_mask

MODEL_ID = os.environ.get("SQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT_PATH = os.environ.get("SQ_OUT", "results/learned_patch.json")
DATA_DIR = "results/calibration_longform"
TASKS = tuple(os.environ.get("SQ_TASKS", "math,code").split(","))
GROUP_SIZE = 128
ATTN_TYPES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TYPES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MODULE_TYPES = ATTN_TYPES + MLP_TYPES
# Patches live on down_proj only: the atlas found MLP task-structure
# concentrates there, and training a sparse residual on all three MLP
# projections retains a dense weight clone per layer in the autograd graph
# (~1.2 GB), which OOMs a 6 GB card. down_proj alone keeps it tractable.
PATCH_SURFACE = ("mlp.down_proj",)
PATCH_BITS = 3
# Fraction of the down_proj (patch-surface) groups to patch. Kept modest so
# per-row solve systems stay small: the closed-form solve is O(b^3) in per-row
# patched width, and high coverage made torch's batched solver crawl.
PATCH_COVERAGE = float(os.environ.get("SQ_COVERAGE", "0.05"))
N_CALIB_SEQ = 24
N_TRAIN_SEQ = int(os.environ.get("SQ_TRAIN_SEQ", "24"))
N_EVAL_SEQ = 16
TRAIN_TOK = int(os.environ.get("SQ_TRAIN_TOK", "192"))
EVAL_TOK = 256
EPOCHS = int(os.environ.get("SQ_EPOCHS", "3"))
LR = float(os.environ.get("SQ_LR", "3e-4"))
LOGIT_CHUNK = 256


def dev_of():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def target_layers(model, types):
    ls = {}
    for d in range(len(model.model.layers)):
        blk = model.model.layers[d]
        for mt in types:
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
                total += F.cross_entropy(lg, tgt[s:e], reduction="sum").item()
                del lg
            out.append((total, n))
            del hidden, src
    return out


def ppl(pt):
    return math.exp(sum(a for a, _ in pt) / sum(b for _, b in pt))


# ---- trainable sparse residual: only selected groups carry parameters ----
class SparseResidualLinear(nn.Module):
    """Wraps a frozen quantized weight with a trainable residual supported on
    a fixed set of (row, group) cells. Only the selected groups carry
    parameters (delta_vals is [n_selected, group_size]); the effective weight
    is reconstructed by scatter-add each forward. This keeps the trainable
    parameter and its optimizer state proportional to the patch budget, not
    to the full layer -- a dense weight-shaped delta per MLP layer would need
    ~2 GB across the model and OOMs a 6 GB card."""

    def __init__(self, base_w, mask, bias):
        super().__init__()
        out_f, in_f = base_w.shape
        gs = GROUP_SIZE
        self.out_f, self.in_f, self.gs = out_f, in_f, gs
        self.n_groups = in_f // gs
        self.register_buffer("base_w", base_w)                     # frozen [out,in]
        rows, cols = torch.nonzero(mask, as_tuple=True)            # selected (row, group)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.delta_vals = nn.Parameter(torch.zeros(rows.numel(), gs, dtype=base_w.dtype))
        self.bias = bias

    def forward(self, x):
        w = self.base_w.view(self.out_f, self.n_groups, self.gs).clone()
        w[self.rows, self.cols] = w[self.rows, self.cols] + self.delta_vals
        return F.linear(x, w.view(self.out_f, self.in_f), self.bias)

    def learned_delta_dense(self):
        """Return the learned residual as a dense [out, in] tensor (zeros
        outside the selected groups), for quantization/storage."""
        with torch.no_grad():
            d = torch.zeros(self.out_f, self.n_groups, self.gs,
                            dtype=self.delta_vals.dtype, device=self.delta_vals.device)
            d[self.rows, self.cols] = self.delta_vals
            return d.view(self.out_f, self.in_f)


def install_wrappers(model, base_weights, masks, order):
    """Replace target linears with SparseResidualLinear. Returns handles to
    restore, and the list of trainable delta params."""
    originals = {}
    params = []
    for name in order:
        d, mt = name.split(".", 1)
        blk = model.model.layers[int(d.replace("layer", ""))]
        parent = blk
        parts = mt.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        orig = getattr(parent, leaf)
        wrap = SparseResidualLinear(
            base_weights[name].to(model.device, orig.weight.dtype),
            masks[name].to(model.device),
            orig.bias,
        ).to(model.device)
        originals[name] = (parent, leaf, orig)
        setattr(parent, leaf, wrap)
        params.append(wrap.delta_vals)
    return originals, params


def restore_wrappers(originals):
    for name, (parent, leaf, orig) in originals.items():
        setattr(parent, leaf, orig)


def main():
    dev = dev_of()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(dev)
    model.config.use_cache = False
    all_layers = target_layers(model, MODULE_TYPES)
    order_all = sorted(all_layers)
    w_orig = {n: all_layers[n].weight.data.float().cpu() for n in order_all}
    n_w = {n: w_orig[n].numel() for n in order_all}
    tot_w = sum(n_w.values())
    grp = {n: n_w[n] // GROUP_SIZE for n in order_all}
    tot_g = sum(grp.values())

    patch_order = sorted(target_layers(model, PATCH_SURFACE))
    patch_g = sum(grp[n] for n in patch_order)
    print(f"full surface {tot_w/1e6:.0f}M; patch surface (MLP) {sum(n_w[n] for n in patch_order)/1e6:.0f}M "
          f"({patch_g} groups)", flush=True)

    data = {t: torch.load(f"{DATA_DIR}/{t}.pt") for t in TASKS}
    pooled = torch.cat([data[t]["calib"][: max(1, N_CALIB_SEQ // len(TASKS))] for t in TASKS], 0)
    H_pool = accum_H(model, all_layers, pooled, dev)

    # mixed base: attn@4, mlp@3, frozen (dequantized values held in-place)
    print("building mixed base attn@4/mlp@3 ...", flush=True)
    base_w = {}
    for n in order_all:
        b = 4 if is_attn(n) else 3
        base_w[n], _, _ = gptq_quantize(w_orig[n], H_pool[n], bits=b, group_size=GROUP_SIZE)
    uni4 = {n: gptq_quantize(w_orig[n], H_pool[n], bits=4, group_size=GROUP_SIZE)[0] for n in order_all}
    del H_pool  # only needed for base construction; free before training

    def bpw(bit_of, patch_groups=0):
        v = sum(n_w[n] * (bit_of(n) + 2 * 16 / GROUP_SIZE) for n in order_all) / tot_w
        if patch_groups:
            v += patch_bytes(patch_groups, GROUP_SIZE, PATCH_BITS, total_groups=tot_g) * 8 / tot_w
        return v

    mixed_bits = bpw(lambda n: 4 if is_attn(n) else 3)

    def set_static(wd):
        saved = {}
        for n in order_all:
            m = all_layers[n]
            saved[n] = m.weight.data.clone()
            m.weight.data = wd[n].to(m.weight.dtype).to(m.weight.device)
        return saved

    def restore_static(saved):
        for n in order_all:
            all_layers[n].weight.data = saved[n]

    results = {"model": MODEL_ID, "mixed_bits": mixed_bits, "uni4_bits": bpw(lambda n: 4),
               "tasks": {}}

    for t in TASKS:
        print(f"\n=== TASK {t} ===", flush=True)
        H_t = accum_H(model, all_layers, data[t]["calib"][:N_CALIB_SEQ], dev)
        ev = data[t]["eval"][:N_EVAL_SEQ, :EVAL_TOK]

        # ---- select MLP groups by OBS (activation-space) score ----
        # A preliminary run confirmed Fisher/task-loss-gradient scoring is
        # *worse* than OBS reconstruction scoring (solved_fisher 5.226 vs
        # solved_obs 4.923 on math), consistent with the whole project's
        # finding that scoring is not the bottleneck. We therefore drop the
        # Fisher arm and give the learned patch the better (OBS) mask.
        obs_score = {}
        for n in patch_order:
            Hg = H_t[n].to(dev)
            obs_score[n] = score_obs_groups(w_orig[n].to(dev), base_w[n].to(dev), Hg, GROUP_SIZE)
            del Hg
        if dev == "cuda":
            torch.cuda.empty_cache()

        # Coverage defined directly as a fraction of the down_proj surface,
        # so the per-row solve systems stay small and fast regardless of how
        # small the patch surface is relative to the whole model.
        n_groups = int(PATCH_COVERAGE * patch_g)

        def masks_from(score):
            flat = torch.cat([score[n].flatten() for n in patch_order])
            mflat = top_k_group_mask(flat, min(1.0, n_groups / patch_g))
            out, i = {}, 0
            for n in patch_order:
                c = grp[n]
                out[n] = mflat[i : i + c].reshape(score[n].shape)
                i += c
            return out, int(mflat.sum().item())

        masks_obs, used_obs = masks_from(obs_score)

        # ---- solved patch (current method), on GPU (fast) ----
        # The GPU solve is ~100x faster than CPU here; it fits because
        # down_proj systems are modest and the solver caps its batch to free
        # memory. (An earlier CPU fallback ran ~1.5 h and was abandoned.)
        def solved_patch(masks):
            wd = dict(base_w)
            for n in patch_order:
                Hg = H_t[n].to(dev)
                sol = solve_residual_layer(
                    w_orig[n].to(dev), base_w[n].float().to(dev), Hg, masks[n].to(dev), GROUP_SIZE
                ).cpu()
                del Hg
                wd[n] = base_w[n] + quantize_delta(
                    sol.to(base_w[n].dtype), masks[n].cpu(), GROUP_SIZE, PATCH_BITS
                )
                if dev == "cuda":
                    torch.cuda.empty_cache()
            return wd

        # ---- LEARNED patch: distillation-trained residual on the Fisher mask ----
        def learned_patch(masks):
            # cache teacher (fp16) logits once
            teacher_lp = []
            model.eval()
            with torch.no_grad():
                for i in range(min(N_TRAIN_SEQ, data[t]['calib'].shape[0])):
                    ids = data[t]["calib"][i : i + 1, :TRAIN_TOK].to(dev)
                    h = model.model(ids).last_hidden_state[0]
                    lp = torch.empty(h.shape[0], model.config.vocab_size, dtype=torch.bfloat16)
                    for s in range(0, h.shape[0], LOGIT_CHUNK):
                        e = min(s + LOGIT_CHUNK, h.shape[0])
                        lp[s:e] = F.log_softmax(model.lm_head(h[s:e]).float(), dim=-1).to(torch.bfloat16).cpu()
                    teacher_lp.append(lp)
                    del h
            saved_b = set_static(base_w)
            originals, params = install_wrappers(model, base_w, masks, patch_order)
            opt = torch.optim.Adam(params, lr=LR)
            model.train()
            # Recompute activations in backward instead of retaining them for
            # all 24 blocks -- the standard memory-for-compute trade that makes
            # training a 0.5B model fit alongside its own dense weight clones.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            kl_chunk = 48  # tokens per logit slice in the backward
            first_kl = None
            for ep in range(EPOCHS):
                tot = 0.0
                for i in range(len(teacher_lp)):
                    ids = data[t]["calib"][i : i + 1, :TRAIN_TOK].to(dev)
                    opt.zero_grad(set_to_none=True)
                    # hidden states carry the grad path to delta; the [T, vocab]
                    # logit tensor is what OOMs, so materialise it one token
                    # slice at a time and accumulate the KL gradient. Blocks are
                    # gradient-checkpointed, so retaining the graph across slices
                    # is cheap (recomputed in backward).
                    h = model.model(ids).last_hidden_state[0]      # [T, hidden]
                    T = h.shape[0]
                    tlp_full = teacher_lp[i]
                    seq_kl = 0.0
                    starts = list(range(0, T, kl_chunk))
                    for j, s in enumerate(starts):
                        e = min(s + kl_chunk, T)
                        logits_c = model.lm_head(h[s:e]).float()
                        slp = F.log_softmax(logits_c, dim=-1)
                        tlp = tlp_full[s:e].to(dev).float()
                        kl_c = (tlp.exp() * (tlp - slp)).sum(-1).sum() / T
                        kl_c.backward(retain_graph=(j < len(starts) - 1))
                        seq_kl += kl_c.item()
                        del logits_c, slp, tlp, kl_c
                    # diagnostic: KL of the base (delta still ~0) on the very
                    # first step. If this is small (~base-to-fp16), the wrapper
                    # is correct and any blow-up is purely optimisation.
                    if first_kl is None:
                        first_kl = seq_kl
                        print(f"    [{t}] initial KL (delta~0): {first_kl:.4f}", flush=True)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt.step()
                    tot += seq_kl
                    del h
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                print(f"    [{t}] learned epoch {ep}: KL={tot/len(teacher_lp):.4f}", flush=True)
            # extract learned deltas; build BOTH a 3-bit-quantized version (the
            # deployable patch) and an fp16 version (to separate "learning
            # doesn't transfer" from "3-bit quantization of the learned delta
            # destroys it").
            wd_q, wd_fp16 = dict(base_w), dict(base_w)
            for name in patch_order:
                d, mt = name.split(".", 1)
                blk = model.model.layers[int(d.replace("layer", ""))]
                parent = blk
                for p in mt.split(".")[:-1]:
                    parent = getattr(parent, p)
                wrap = getattr(parent, mt.split(".")[-1])
                learned_delta = wrap.learned_delta_dense().float().cpu()
                wd_q[name] = base_w[name] + quantize_delta(learned_delta, masks[name], GROUP_SIZE, PATCH_BITS)
                wd_fp16[name] = base_w[name] + learned_delta.to(base_w[name].dtype)
            model.gradient_checkpointing_disable()
            restore_wrappers(originals)
            restore_static(saved_b)
            model.eval()
            if dev == "cuda":
                torch.cuda.empty_cache()
            return wd_q, wd_fp16

        row = {}
        def score_cfg(tag, wd, groups):
            saved2 = set_static(wd)
            p = ppl(chunked_nll(model, ev, dev))
            restore_static(saved2)
            row[tag] = {"ppl": p, "bits": bpw(lambda n: 4 if is_attn(n) else 3, groups)}
            print(f"  {tag:22s} ppl={p:.4f}  bits/w={row[tag]['bits']:.3f}", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()

        # references
        saved2 = set_static(base_w); row["base_mixed"] = {"ppl": ppl(chunked_nll(model, ev, dev)), "bits": mixed_bits}; restore_static(saved2)
        print(f"  base_mixed             ppl={row['base_mixed']['ppl']:.4f}  bits/w={mixed_bits:.3f}", flush=True)
        saved2 = set_static(uni4); row["uniform4"] = {"ppl": ppl(chunked_nll(model, ev, dev)), "bits": bpw(lambda n:4)}; restore_static(saved2)
        print(f"  uniform4               ppl={row['uniform4']['ppl']:.4f}  bits/w={row['uniform4']['bits']:.3f}", flush=True)
        saved2 = set_static(uni4)  # fp16 ref via teacher: just compute once from base fp16
        restore_static(saved2)
        row["fp16"] = {"ppl": ppl(chunked_nll(model, ev, dev)) if False else None}
        # true fp16
        p_fp16 = ppl(chunked_nll(model, ev, dev))  # weights currently original fp16
        row["fp16"] = {"ppl": p_fp16}
        print(f"  fp16                   ppl={p_fp16:.4f}", flush=True)

        score_cfg("solved_obs", solved_patch(masks_obs), used_obs)
        learned_q, learned_fp16 = learned_patch(masks_obs)
        score_cfg("learned_3bit", learned_q, used_obs)
        score_cfg("learned_fp16", learned_fp16, used_obs)  # diagnostic: unquantized

        results["tasks"][t] = row
        del H_t
        # save incrementally so a later task's failure never loses an earlier
        # task's result, and free GPU/CPU memory before the next task's
        # Hessian accumulation (which OOM'd when carried over).
        os.makedirs("results", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        import gc
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
