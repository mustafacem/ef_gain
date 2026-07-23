"""E1 + E2 on real data (plan-v2 sections 1, 4.4, 5): task-sensitivity atlas
across four real tasks (mbpp/gsm8k/ai2_arc/wikitext-2) on Qwen2.5-0.5B, using
a representative subset of layers (5 depths x {attention o_proj, mlp
down_proj}) to keep this a same-day run rather than the full 12-week study.

Produces: per-task/per-layer sensitivity scores, the cross-task overlap
matrix, a split-half noise ceiling, the random floor, the go/no-go signal
(plan-v2 section 1), a per-layer attention-vs-MLP breakdown (H4), and an E1
proxy-validation correlation on one layer. Results saved to
results/atlas_qwen0.5b.json.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.analysis.overlap import (
    go_no_go_signal,
    overlap_matrix,
    per_layer_breakdown,
    random_floor,
    split_half_noise_ceiling,
)
from selfquant.analysis.proxy_check import (
    sample_cells_across_score_range,
    validate_proxy_scores,
)
from selfquant.data.calibration import TASKS
from selfquant.quant.rtn import rtn_quantize
from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.fisher import FisherAccumulator
from selfquant.sensitivity.scores import score_awq, score_fisher

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128
N_CALIB_USED = 64  # subset of the 150 built in Phase 0
MAX_TOKENS = 128
DEPTHS = (0, 6, 12, 18, 23)
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")
DATA_DIR = "results/calibration_data"
K_FRAC = 0.01


def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def load_calib_texts(task: str, n: int) -> list[str]:
    path = os.path.join(DATA_DIR, f"{task}_calib.jsonl")
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


def score_task(model, tok, layers: dict, texts: list[str], device: str, rtn_cache: dict):
    act_stats = {name: ActivationAbsMean(mod) for name, mod in layers.items()}
    fisher_acc = {name: FisherAccumulator(mod, device="cpu") for name, mod in layers.items()}

    model.eval()
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
        for fa in fisher_acc.values():
            fa.accumulate()
        del out
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    scores_awq, scores_fisher = {}, {}
    for name, mod in layers.items():
        w = mod.weight.data.float().cpu()
        in_features = w.shape[1]
        if name not in rtn_cache:
            w_rtn, _, _ = rtn_quantize(w, bits=3, group_size=GROUP_SIZE)
            rtn_cache[name] = w_rtn
        w_rtn = rtn_cache[name]

        act = act_stats[name].result().cpu()
        if act.shape[0] != in_features:
            act = torch.nn.functional.pad(act, (0, in_features - act.shape[0]))
        scores_awq[name] = score_awq(w, act, GROUP_SIZE)

        fisher = fisher_acc[name].result().float().cpu()
        scores_fisher[name] = score_fisher(w, w_rtn, fisher, GROUP_SIZE)

        act_stats[name].remove()

    return scores_awq, scores_fisher


def flatten_scores(scores_by_layer: dict, layer_order: list[str]) -> torch.Tensor:
    return torch.cat([scores_by_layer[name].flatten() for name in layer_order])


def main():
    device = get_device()
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=model_dtype).to(device)

    layers = target_layers(model)
    layer_order = sorted(layers.keys())
    print(f"target layers ({len(layers)}): {layer_order}")

    rtn_cache: dict = {}
    task_awq: dict[str, torch.Tensor] = {}
    task_fisher: dict[str, torch.Tensor] = {}
    task_awq_by_layer: dict[str, dict] = {}

    for task in TASKS:
        print(f"\n=== scoring task: {task} ===")
        texts = load_calib_texts(task, N_CALIB_USED)
        print(f"  {len(texts)} calibration texts (subset of Phase 0's 64)")
        s_awq, s_fisher = score_task(model, tok, layers, texts, device, rtn_cache)
        task_awq[task] = flatten_scores(s_awq, layer_order)
        task_fisher[task] = flatten_scores(s_fisher, layer_order)
        task_awq_by_layer[task] = s_awq

    print("\n=== overlap matrix (AWQ, top-1%) ===")
    matrix_awq = overlap_matrix(task_awq, k_frac=K_FRAC)
    for a in TASKS:
        print("  " + " ".join(f"{matrix_awq[a][b]:.3f}" for b in TASKS))

    print("\n=== overlap matrix (Fisher, top-1%) ===")
    matrix_fisher = overlap_matrix(task_fisher, k_frac=K_FRAC)
    for a in TASKS:
        print("  " + " ".join(f"{matrix_fisher[a][b]:.3f}" for b in TASKS))

    # --- split-half noise ceiling on "code" (independent calibration halves) ---
    print("\n=== split-half noise ceiling (code task, AWQ) ===")
    code_texts_full = load_calib_texts("code", N_CALIB_USED * 2)
    half_a, half_b = code_texts_full[:N_CALIB_USED], code_texts_full[N_CALIB_USED : N_CALIB_USED * 2]
    s_awq_a, _ = score_task(model, tok, layers, half_a, device, rtn_cache)
    s_awq_b, _ = score_task(model, tok, layers, half_b, device, rtn_cache)
    flat_a = flatten_scores(s_awq_a, layer_order)
    flat_b = flatten_scores(s_awq_b, layer_order)
    ceiling = split_half_noise_ceiling(flat_a, flat_b, k_frac=K_FRAC)
    floor = random_floor(K_FRAC)
    print(f"  noise ceiling: {ceiling:.3f}   random floor: {floor:.3f}")

    cross_task_overlaps = [
        matrix_awq[a][b] for i, a in enumerate(TASKS) for b in TASKS[i + 1 :]
    ]
    signal = go_no_go_signal([ceiling], cross_task_overlaps)
    print(f"\n=== GO/NO-GO SIGNAL: {signal:.3f} ===")
    if signal >= 0.15:
        verdict = "GO — full speed ahead"
    elif signal >= 0.05:
        verdict = "PROCEED WITH REDUCED SCOPE — emphasize composition/systems story"
    else:
        verdict = "PIVOT — task-universal sensitivity; reframe as negative result"
    print(f"verdict (n={N_CALIB_USED} calib texts, {len(layers)} layers — NOT the full study): {verdict}")

    # --- per-layer breakdown: code vs chat, attention vs MLP (H4) ---
    print("\n=== per-layer overlap: code vs chat (AWQ, top-1%) ===")
    breakdown = per_layer_breakdown(task_awq_by_layer["code"], task_awq_by_layer["chat"], k_frac=K_FRAC)
    attn_vals, mlp_vals = [], []
    for name in layer_order:
        v = breakdown[name]
        kind = "attn" if "self_attn" in name else "mlp"
        (attn_vals if kind == "attn" else mlp_vals).append(v)
        print(f"  {name:30s} {v:.3f}")
    if attn_vals and mlp_vals:
        print(f"  mean attn overlap: {sum(attn_vals)/len(attn_vals):.3f}  mean mlp overlap: {sum(mlp_vals)/len(mlp_vals):.3f}")

    # --- E1: proxy validation, both metrics, multiple (task, layer) combos ---
    print("\n=== E1 proxy validation (both metrics, 3 task/layer combos) ===")
    E1_N_SAMPLES = 24
    E1_N_EVAL_TEXTS = 4
    E1_COMBOS = [
        ("code", "layer12.mlp.down_proj"),
        ("code", "layer12.self_attn.o_proj"),
        ("math", "layer12.mlp.down_proj"),
    ]

    def make_kl_loss_fn(mod, eval_ids, w_teacher: torch.Tensor, w_baseline_ref: torch.Tensor):
        """KL(teacher || candidate) against a cached fp32-weight teacher pass,
        not CE loss. A single 128-weight group is ~0.003% of a linear
        layer's parameters — on 4 short texts, CE loss (averaged over
        discrete next-token targets) frequently doesn't move at all in
        bf16, which is why the CE-based check above produced exact-zero
        deltas and NaN correlations for 2 of 3 combos. KL against the
        model's own (near-)original output distribution is continuous and
        far more sensitive to small weight nudges, and is the same
        "KL-teacher variant" the sensitivity metric itself supports
        (sensitivity/fisher.py docstring, plan-v2 4.2 metric 3).

        w_baseline_ref: the exact object measure_actual_group_delta_loss
        calls with the unquantized-candidate weight (w_quantized) each
        cell — only THIS call is safe to cache, by Python object identity
        (`is`), not by data_ptr(): per-cell w_restored clones are freed
        immediately after use, and PyTorch's CPU allocator reliably hands
        the next same-size clone() the just-freed address, so a
        data_ptr()-keyed cache silently returns cell 0's cached result for
        every later cell instead of recomputing — that bug, not precision,
        was the original source of the constant-array/NaN correlations."""
        orig = mod.weight.data.clone()
        mod.weight.data = w_teacher.to(mod.weight.dtype).to(mod.weight.device)
        teacher_logprobs = []
        with torch.no_grad():
            for ids in eval_ids:
                out = model(ids.to(device))
                teacher_logprobs.append(torch.log_softmax(out.logits.float(), dim=-1).cpu())
        mod.weight.data = orig

        baseline_result = {}

        def loss_fn(w_candidate: torch.Tensor) -> float:
            is_baseline = w_candidate is w_baseline_ref
            if is_baseline and "value" in baseline_result:
                return baseline_result["value"]
            orig2 = mod.weight.data.clone()
            mod.weight.data = w_candidate.to(mod.weight.dtype).to(mod.weight.device)
            total_kl, n_tok = 0.0, 0
            with torch.no_grad():
                for ids, t_logprob in zip(eval_ids, teacher_logprobs):
                    out = model(ids.to(device))
                    s_logprob = torch.log_softmax(out.logits.float(), dim=-1).cpu()
                    kl = torch.sum(torch.exp(t_logprob) * (t_logprob - s_logprob), dim=-1)
                    total_kl += kl.sum().item()
                    n_tok += kl.numel()
            mod.weight.data = orig2
            result = total_kl / n_tok
            if is_baseline:
                baseline_result["value"] = result
            return result

        return loss_fn

    e1_results = {}
    for task, layer_name in E1_COMBOS:
        print(f"\n  --- {task} / {layer_name} ---")
        mod = layers[layer_name]
        w_orig = mod.weight.data.float().cpu()
        w_quant = rtn_cache[layer_name]

        eval_texts = load_calib_texts(task, E1_N_EVAL_TEXTS)
        eval_ids = [
            tok(t, return_tensors="pt", truncation=True, max_length=96)["input_ids"]
            for t in eval_texts
        ]
        loss_fn = make_kl_loss_fn(mod, eval_ids, w_orig, w_quant)

        # single-layer rescoring: cheap (hooks scoped to just this layer),
        # and keeps AWQ/Fisher scores consistent with each other here.
        s_awq_full, s_fisher_full = score_task(
            model, tok, {layer_name: mod}, load_calib_texts(task, N_CALIB_USED), device, rtn_cache
        )
        s_awq_layer, s_fisher_layer = s_awq_full[layer_name], s_fisher_full[layer_name]

        for metric_name, scores in (("awq", s_awq_layer), ("fisher", s_fisher_layer)):
            cells = sample_cells_across_score_range(scores, n_samples=E1_N_SAMPLES, seed=0)
            rho = validate_proxy_scores(scores, w_quant, w_orig, GROUP_SIZE, loss_fn, cells)
            print(f"    {metric_name:7s} Spearman(proxy, measured KL-divergence-reduction) over {len(cells)} cells: {rho:.3f}")
            e1_results[f"{task}__{layer_name}__{metric_name}"] = rho

    # --- save everything ---
    results = {
        "model": MODEL_ID,
        "n_calib_used": N_CALIB_USED,
        "layers": layer_order,
        "overlap_matrix_awq": matrix_awq,
        "overlap_matrix_fisher": matrix_fisher,
        "split_half_noise_ceiling_awq": ceiling,
        "random_floor": floor,
        "go_no_go_signal": signal,
        "verdict": verdict,
        "per_layer_code_vs_chat_awq": breakdown,
        "mean_attn_overlap": sum(attn_vals) / len(attn_vals) if attn_vals else None,
        "mean_mlp_overlap": sum(mlp_vals) / len(mlp_vals) if mlp_vals else None,
        "e1_proxy_spearman": e1_results,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/atlas_qwen0.5b.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/atlas_qwen0.5b.json")


if __name__ == "__main__":
    main()
