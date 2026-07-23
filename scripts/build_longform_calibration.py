"""Rebuild calibration/eval at a token scale that makes the Hessian usable.

The original data (128-token truncations of short examples) gave 0.8-1.6
activation samples per Hessian dimension for down_proj, so H was singular or
near-singular for every task and the damping term was carrying structural
load. Results that invert H -- the closed-form solve, OBS scores, the GPTQ
base -- were all built on that.

Two changes:

  1. Concatenate-and-chunk into 2048-token sequences (standard GPTQ/AWQ
     practice) instead of truncating each short example at 128 tokens. This
     extracts several times more usable tokens from the same corpus.

  2. Swap mbpp for codeparrot-clean-valid as the code corpus. mbpp holds
     ~30k tokens in total, so no amount of re-chunking can reach adequate
     conditioning for a 4864-dim Hessian; it is simply too small. mbpp is
     retained for EVAL, keeping the eval distribution comparable with the
     earlier passes while fixing the calibration side.

Calibration and eval stay disjoint by construction (different corpora for
code; different splits elsewhere), and the n-gram check is re-run regardless.
"""
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from selfquant.data.calibration import ngram_overlap_ratio, text_list_hash
from selfquant.data.longform import build_sequences, report_conditioning

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LEN = 2048
N_CALIB_SEQ = 128          # 128 x 2048 = 262k tokens, the standard budget
N_EVAL_SEQ = 48            # ~98k tokens of eval, vs ~5k before
OUT_DIR = "results/calibration_longform"
DOWN_PROJ_IN = 4864        # the widest Hessian in the 0.5B model


def code_calib():
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train[:4000]")
    return [c for c in ds["content"] if c and len(c) > 200]


def code_eval():
    ds = load_dataset("mbpp", split="test")
    return [f"{e['text']}\n{e['code']}" for e in ds]


def math_split(split):
    ds = load_dataset("gsm8k", "main", split=split)
    return [f"Question: {e['question']}\nAnswer: {e['answer']}" for e in ds]


def know_split(split):
    ds = load_dataset("ai2_arc", "ARC-Easy", split=split)
    out = []
    for e in ds:
        ch = "\n".join(f"{l}. {t}" for l, t in zip(e["choices"]["label"], e["choices"]["text"]))
        out.append(f"Question: {e['question']}\n{ch}\nAnswer: {e['answerKey']}")
    return out


def chat_split(split):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return [t for t in ds["text"] if len(t.strip()) > 200]


SOURCES = {
    "code": (code_calib, code_eval),
    "math": (lambda: math_split("train"), lambda: math_split("test")),
    "knowledge": (lambda: know_split("train"), lambda: know_split("test")),
    "chat": (lambda: chat_split("train"), lambda: chat_split("validation")),
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}

    for task, (calib_fn, eval_fn) in SOURCES.items():
        print(f"\n=== {task} ===")
        calib_txt, eval_txt = calib_fn(), eval_fn()
        print(f"  source texts: {len(calib_txt)} calib, {len(eval_txt)} eval")

        cs = build_sequences(calib_txt, tok, SEQ_LEN, N_CALIB_SEQ, seed=0)
        es = build_sequences(eval_txt, tok, SEQ_LEN, N_EVAL_SEQ, seed=1)
        print(f"  calib: {cs.summary(DOWN_PROJ_IN)}")
        print(f"  eval : {es.summary(DOWN_PROJ_IN)}")

        cond = report_conditioning(f"{task}_calib", cs, DOWN_PROJ_IN)
        if cond["status"] != "ok":
            print(f"  !! conditioning {cond['status']}: {cond['samples_per_dim']:.1f} samples/dim")

        # leakage check on the raw text, before chunking
        overlap = ngram_overlap_ratio(calib_txt[:400], eval_txt[:400], n=8)
        print(f"  8-gram calib/eval overlap: {overlap:.4f}")

        import torch

        torch.save({"calib": cs.ids, "eval": es.ids}, f"{OUT_DIR}/{task}.pt")
        manifest[task] = {
            "calib_tokens": cs.n_tokens,
            "eval_tokens": es.n_tokens,
            "calib_seqs": int(cs.ids.shape[0]),
            "eval_seqs": int(es.ids.shape[0]),
            "seq_len": SEQ_LEN,
            "samples_per_dim_down_proj": cond["samples_per_dim"],
            "conditioning": cond["status"],
            "ngram_overlap": overlap,
            "calib_hash": text_list_hash(calib_txt[:200]),
            "eval_hash": text_list_hash(eval_txt[:200]),
        }

    with open(f"{OUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== conditioning summary (was 0.8-1.6 samples/dim) ===")
    for t, m in manifest.items():
        print(f"  {t:10s} {m['calib_tokens']:7d} calib tok  {m['samples_per_dim_down_proj']:6.1f} samples/dim  [{m['conditioning']}]")
    print(f"\nsaved {OUT_DIR}/")


if __name__ == "__main__":
    main()
