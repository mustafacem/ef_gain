"""E3-style main results grid (plan-v2 section 5), scoped to a same-day run:
build a real GPTQ 3-bit base (generic/task-agnostic calibration) over the
same 10 representative layers used in run_atlas.py, build a precision patch
per task (AWQ top-1%), plus the two required control patches (task-agnostic,
random), and evaluate all of it on held-out eval text via perplexity —
own-task patch vs. mismatched patches vs. task-agnostic patch vs. random
patch vs. a straight uniform 4-bit base, for every task.

Also reports the compute/storage economics: patch sizes, swap latency, and
effective average bits-per-weight with a patch applied vs. straight 4-bit
uniform quantization vs. fp16 — the actual "adapter" pitch, in numbers.
"""
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.data.calibration import TASKS
from selfquant.patch.apply import apply_patch_layer, hash_state_dict, save_patch, timed_apply_patch_layer
from selfquant.patch.build import build_patch_layer, patch_layer_bytes
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.scores import score_awq, top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
DEPTHS = (0, 6, 12, 18, 23)
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_data"
N_CALIB_SCORE = 64
N_GENERIC_PER_TASK = 8
N_PPL_EVAL = 20
MAX_TOKENS = 128
K_FRAC = 0.01
BASE_BITS = 3
UNIFORM_BITS = 4


def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def load_jsonl_texts(path: str, n: int) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


def target_layers(model):
    layers = {}
    for d in DEPTHS:
        block = model.model.layers[d]
        for mtype in MODULE_TYPES:
            mod = block
            for part in mtype.split("."):
                mod = getattr(mod, part)
            layers[f"layer{d}.{mtype}"] = mod
    return layers


def compute_ppl(model, tok, texts, device):
    model.eval()
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_nll += out.loss.item() * n
            total_tok += n
    return math.exp(total_nll / total_tok)


def set_layers(layers, weight_dict):
    saved = {}
    for name, mod in layers.items():
        saved[name] = mod.weight.data.clone()
        mod.weight.data = weight_dict[name].to(mod.weight.dtype).to(mod.weight.device)
    return saved


def restore_layers(layers, saved):
    for name, mod in layers.items():
        mod.weight.data = saved[name]


def main():
    device = get_device()
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=model_dtype).to(device)

    layers = target_layers(model)
    layer_order = sorted(layers.keys())
    w_orig = {name: mod.weight.data.float().cpu() for name, mod in layers.items()}
    print(f"target layers ({len(layers)}): {layer_order}")

    # --- 1. GPTQ bases (3-bit primary, 4-bit uniform-comparison), generic calibration ---
    print("\naccumulating Hessians (generic/task-agnostic calibration = chat corpus) ...")
    generic_calib_texts = load_jsonl_texts(f"{DATA_DIR}/chat_calib.jsonl", 64)
    hessians = {name: GPTQHessian(mod.in_features) for name, mod in layers.items()}

    def make_hess_hook(hess, in_features):
        def hook(module, args):
            x = args[0].reshape(-1, in_features).detach()
            hess.update(x)

        return hook

    handles = [mod.register_forward_pre_hook(make_hess_hook(hessians[name], mod.in_features)) for name, mod in layers.items()]
    with torch.no_grad():
        for t in generic_calib_texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
            model(ids)
    for h in handles:
        h.remove()

    print(f"quantizing base at {BASE_BITS}-bit (primary) and {UNIFORM_BITS}-bit (uniform comparison) ...")
    base3, base4 = {}, {}
    for name in layer_order:
        w = w_orig[name]
        q3, _, _ = gptq_quantize(w, hessians[name].H, bits=BASE_BITS, group_size=GROUP_SIZE)
        q4, _, _ = gptq_quantize(w, hessians[name].H, bits=UNIFORM_BITS, group_size=GROUP_SIZE)
        base3[name], base4[name] = q3, q4

    # --- 2. AWQ sensitivity scores per task + a genuinely task-independent generic set ---
    def awq_scores_for_texts(texts):
        act_stats = {name: ActivationAbsMean(mod) for name, mod in layers.items()}
        model.eval()
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
                model(ids)
        scores = {}
        for name in layer_order:
            w = w_orig[name]
            act = act_stats[name].result().cpu()
            if act.shape[0] != w.shape[1]:
                act = torch.nn.functional.pad(act, (0, w.shape[1] - act.shape[0]))
            scores[name] = score_awq(w, act, GROUP_SIZE)
            act_stats[name].remove()
        return scores

    print("\nscoring per-task sensitivity (AWQ) ...")
    task_scores = {}
    for task in TASKS:
        texts = load_jsonl_texts(f"{DATA_DIR}/{task}_calib.jsonl", N_CALIB_SCORE)
        task_scores[task] = awq_scores_for_texts(texts)
        print(f"  {task}: done")

    print("scoring generic/task-agnostic sensitivity (pooled sample across all 4 tasks) ...")
    generic_texts = []
    for task in TASKS:
        generic_texts += load_jsonl_texts(f"{DATA_DIR}/{task}_calib.jsonl", N_GENERIC_PER_TASK)
    generic_scores = awq_scores_for_texts(generic_texts)

    # --- 3. build patches: own (per task), task-agnostic, random ---
    layer_shapes = {name: task_scores[TASKS[0]][name].shape for name in layer_order}

    def flat(scores):
        return torch.cat([scores[n].flatten() for n in layer_order])

    def unflatten_mask(mask_flat):
        out, idx = {}, 0
        for name in layer_order:
            shp = layer_shapes[name]
            n = shp[0] * shp[1]
            out[name] = mask_flat[idx : idx + n].reshape(shp)
            idx += n
        return out

    def build_patch_set(scores):
        mask_flat = top_k_group_mask(flat(scores), K_FRAC)
        masks = unflatten_mask(mask_flat)
        return {name: build_patch_layer(w_orig[name], masks[name], GROUP_SIZE) for name in layer_order}

    print("\nbuilding patches (k=1%) ...")
    patches = {task: build_patch_set(task_scores[task]) for task in TASKS}
    patches["agnostic"] = build_patch_set(generic_scores)
    torch.manual_seed(0)
    random_scores = {name: torch.rand_like(task_scores[TASKS[0]][name]) for name in layer_order}
    patches["random"] = build_patch_set(random_scores)

    base_hash = hash_state_dict({f"{name}.weight": base3[name] for name in layer_order})
    os.makedirs("results/patches", exist_ok=True)
    for label, patch_set in patches.items():
        save_patch(
            f"results/patches/{label}.safetensors",
            patch_set,
            metadata={"task": label, "metric": "awq", "k_frac": str(K_FRAC), "base_hash": base_hash, "group_size": str(GROUP_SIZE)},
        )
    print(f"saved {len(patches)} patches to results/patches/")

    # --- 4. evaluate PPL per task under every config ---
    results = {"patch_sizes_mb": {}, "ppl": {}}
    for label, patch_set in patches.items():
        results["patch_sizes_mb"][label] = sum(patch_layer_bytes(p) for p in patch_set.values()) / 1e6

    rep_layer = "layer12.mlp.down_proj"
    _, lat = timed_apply_patch_layer(base3[rep_layer], patches["code"][rep_layer], n_trials=30)
    results["swap_latency_ms_repr_layer"] = lat * 1000

    print("\nevaluating PPL grid ...")
    for task in TASKS:
        eval_texts = load_jsonl_texts(f"{DATA_DIR}/{task}_eval.jsonl", N_PPL_EVAL)
        row = {}

        row["fp16"] = compute_ppl(model, tok, eval_texts, device)

        saved = set_layers(layers, base3)
        row["base_3bit"] = compute_ppl(model, tok, eval_texts, device)
        restore_layers(layers, saved)

        patched = {name: apply_patch_layer(base3[name], patches[task][name]) for name in layer_order}
        saved = set_layers(layers, patched)
        row["base+own_patch"] = compute_ppl(model, tok, eval_texts, device)
        restore_layers(layers, saved)

        for other in TASKS:
            if other == task:
                continue
            patched = {name: apply_patch_layer(base3[name], patches[other][name]) for name in layer_order}
            saved = set_layers(layers, patched)
            row[f"base+{other}_patch"] = compute_ppl(model, tok, eval_texts, device)
            restore_layers(layers, saved)

        patched = {name: apply_patch_layer(base3[name], patches["agnostic"][name]) for name in layer_order}
        saved = set_layers(layers, patched)
        row["base+agnostic_patch"] = compute_ppl(model, tok, eval_texts, device)
        restore_layers(layers, saved)

        patched = {name: apply_patch_layer(base3[name], patches["random"][name]) for name in layer_order}
        saved = set_layers(layers, patched)
        row["base+random_patch"] = compute_ppl(model, tok, eval_texts, device)
        restore_layers(layers, saved)

        saved = set_layers(layers, base4)
        row["uniform_4bit"] = compute_ppl(model, tok, eval_texts, device)
        restore_layers(layers, saved)

        results["ppl"][task] = row
        print(f"  {task}: {row}")
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- 5. memory / compute economics ---
    n_weights_total = sum(w.numel() for w in w_orig.values())
    overhead_per_weight = 16.0 * 2 / GROUP_SIZE  # fp16 scale+zero per group, amortized
    bits_3bit_base = BASE_BITS + overhead_per_weight
    bits_4bit_uniform = UNIFORM_BITS + overhead_per_weight
    eff_bits_patched = bits_3bit_base + K_FRAC * (16 - bits_3bit_base)

    results["memory_summary"] = {
        "n_weights_10_layers": n_weights_total,
        "fp16_bits_per_weight": 16.0,
        "base_3bit_bits_per_weight": bits_3bit_base,
        "uniform_4bit_bits_per_weight": bits_4bit_uniform,
        "effective_bits_with_1pct_patch": eff_bits_patched,
        "base_size_mb": n_weights_total * bits_3bit_base / 8 / 1e6,
        "uniform_4bit_size_mb": n_weights_total * bits_4bit_uniform / 8 / 1e6,
        "fp16_size_mb": n_weights_total * 16 / 8 / 1e6,
        "avg_patch_size_mb": sum(results["patch_sizes_mb"][t] for t in TASKS) / len(TASKS),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/patch_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/patch_eval_results.json")


if __name__ == "__main__":
    main()
