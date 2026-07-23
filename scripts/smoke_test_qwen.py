"""Week-1 smoke test (plan-v2 section 11, day 6-7): run the real pipeline
end-to-end on one linear layer of Qwen2.5-0.5B-Instruct with real
calibration text, to sanity-check shapes/dtypes/memory before scaling up.
Not a unit test — a manual script, run once to validate the plumbing.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfquant.analysis.overlap import jaccard_topk, random_floor, split_half_noise_ceiling
from selfquant.patch.apply import apply_patch_layer, timed_apply_patch_layer
from selfquant.patch.build import build_patch_layer, patch_layer_bytes
from selfquant.quant.gptq import GPTQHessian, gptq_quantize
from selfquant.quant.rtn import rtn_quantize
from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.fisher import FisherAccumulator
from selfquant.sensitivity.scores import score_awq, score_fisher, top_k_group_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 128

CODE_TEXTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
    "import numpy as np\narr = np.array([1,2,3])\nresult = np.sum(arr) / len(arr)",
    "for i in range(10):\n    if i % 2 == 0:\n        print(f'{i} is even')\n    else:\n        print(f'{i} is odd')",
]
CHAT_TEXTS = [
    "The weather today is quite pleasant with a gentle breeze and clear skies.",
    "My favorite way to relax on the weekend is reading a good book by the fireplace.",
    "She smiled warmly and thanked everyone for coming to the small gathering.",
    "The recipe calls for two cups of flour, a pinch of salt, and fresh herbs.",
]


def main():
    print(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    layer = model.model.layers[6].mlp.down_proj  # [hidden, intermediate] -> in=intermediate
    w = layer.weight.data.clone()
    print(f"target layer: down_proj, shape {tuple(w.shape)}")
    in_features = w.shape[1]
    if in_features % GROUP_SIZE != 0:
        pad = GROUP_SIZE - (in_features % GROUP_SIZE)
        print(f"in_features={in_features} not divisible by {GROUP_SIZE}, padding by {pad}")
        w = torch.nn.functional.pad(w, (0, pad))
        in_features += pad

    # --- RTN baseline ---
    w_rtn, _, _ = rtn_quantize(w, bits=3, group_size=GROUP_SIZE)
    rtn_err = (w - w_rtn).pow(2).mean().item()
    print(f"RTN 3-bit MSE: {rtn_err:.6f}")

    # --- sensitivity scores per task (real calibration text) ---
    def compute_scores_for_texts(texts, use_fisher=True):
        act_stats = ActivationAbsMean(layer)
        fisher_acc = FisherAccumulator(layer) if use_fisher else None
        for text in texts:
            ids = tok(text, return_tensors="pt")["input_ids"]
            if use_fisher:
                model.zero_grad()
                out = model(ids, labels=ids)
                out.loss.backward()
                fisher_acc.accumulate()
            else:
                with torch.no_grad():
                    model(ids)
        act = act_stats.result()
        act_stats.remove()
        if in_features != act.shape[0]:
            act = torch.nn.functional.pad(act, (0, in_features - act.shape[0]))
        s_awq = score_awq(w, act, GROUP_SIZE)
        s_fisher = None
        if use_fisher:
            fisher = fisher_acc.result()
            if fisher.shape[1] != in_features:
                fisher = torch.nn.functional.pad(fisher, (0, in_features - fisher.shape[1]))
            s_fisher = score_fisher(w, w_rtn, fisher, GROUP_SIZE)
        return s_awq, s_fisher

    print("scoring on code calibration text ...")
    code_awq, code_fisher = compute_scores_for_texts(CODE_TEXTS)
    print("scoring on chat calibration text ...")
    chat_awq, chat_fisher = compute_scores_for_texts(CHAT_TEXTS)

    print(f"AWQ score shape: {code_awq.shape}")

    # --- overlap: code vs chat, plus a random-floor sanity check ---
    j_awq = jaccard_topk(code_awq, chat_awq, k_frac=0.01)
    floor = random_floor(0.01)
    print(f"code-vs-chat Jaccard (AWQ, top-1%): {j_awq:.3f}  (random floor: {floor:.3f})")

    if code_fisher is not None:
        j_fisher = jaccard_topk(code_fisher, chat_fisher, k_frac=0.01)
        print(f"code-vs-chat Jaccard (Fisher, top-1%): {j_fisher:.3f}")

    # split-half noise ceiling for the code task (halves of the 4 texts)
    code_half_a, _ = compute_scores_for_texts(CODE_TEXTS[:2], use_fisher=False)
    code_half_b, _ = compute_scores_for_texts(CODE_TEXTS[2:], use_fisher=False)
    ceiling = split_half_noise_ceiling(code_half_a, code_half_b, k_frac=0.01)
    print(f"code split-half noise ceiling (AWQ, top-1%): {ceiling:.3f}")

    # --- GPTQ with real Hessian from code calibration text ---
    print("accumulating Hessian for GPTQ ...")
    hess = GPTQHessian(in_features)

    def hess_hook(module, args):
        x = args[0].reshape(-1, in_features).detach()
        hess.update(x)

    handle = layer.register_forward_pre_hook(hess_hook)
    with torch.no_grad():
        for text in CODE_TEXTS:
            ids = tok(text, return_tensors="pt")["input_ids"]
            model(ids)
    handle.remove()

    q_gptq, scale, zero = gptq_quantize(w, hess.H, bits=3, group_size=GROUP_SIZE)
    gptq_err = (w - q_gptq).pow(2).mean().item()
    print(f"GPTQ 3-bit MSE (code-calibrated): {gptq_err:.6f} (RTN was {rtn_err:.6f})")

    # --- patch build/apply on the real layer ---
    mask = top_k_group_mask(code_awq, k_frac=0.01)
    patch = build_patch_layer(w, mask, GROUP_SIZE)
    print(f"patch: {mask.sum().item()} groups, {patch_layer_bytes(patch) / 1e6:.3f} MB")

    patched = apply_patch_layer(q_gptq, patch)
    patched_err = (w - patched).pow(2).mean().item()
    print(f"base+patch MSE: {patched_err:.6f} (base alone: {gptq_err:.6f})")

    _, latency = timed_apply_patch_layer(q_gptq, patch, n_trials=20)
    print(f"patch swap latency (median over 20): {latency * 1000:.3f} ms")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
