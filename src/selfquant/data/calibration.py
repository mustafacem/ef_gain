"""Calibration/eval dataset pairs for the four tasks (plan-v2 section 6).

Each task loader pulls calibration text from a dataset's *train* split and
eval text from its *test* (or *validation*) split — different splits by
construction, but `ngram_overlap_ratio` below is still run as a hygiene
check, since some HF datasets have near-duplicate rows across splits.

Datasets chosen for being small and fast to pull (plan-v2's own picks like
The Stack / GSM8K-train / NQ / UltraChat are multi-GB; these are the same
task categories at a scale that fits a laptop iteration loop):
  - code:      mbpp        (train -> calib, test -> eval)
  - math:      gsm8k/main  (train -> calib, test -> eval)
  - knowledge: ai2_arc/ARC-Easy (train -> calib, test -> eval)
  - chat/general: wikitext-2-raw-v1 (train -> calib, validation -> eval)
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from datasets import load_dataset

TASKS = ("code", "math", "knowledge", "chat")


def _mbpp_texts(split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset("mbpp", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        out.append(f"{ex['text']}\n{ex['code']}")
        if len(out) >= n:
            break
    return out


def _gsm8k_texts(split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset("gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        out.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
        if len(out) >= n:
            break
    return out


def _arc_texts(split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset("ai2_arc", "ARC-Easy", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        ex = ds[i]
        choices = "\n".join(
            f"{lab}. {txt}"
            for lab, txt in zip(ex["choices"]["label"], ex["choices"]["text"])
        )
        out.append(f"Question: {ex['question']}\n{choices}\nAnswer: {ex['answerKey']}")
        if len(out) >= n:
            break
    return out


def _wikitext_texts(split: str, n: int, seed: int) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    lines = [t for t in ds["text"] if len(t.strip()) > 200]  # skip headers/blank lines
    random.Random(seed).shuffle(lines)
    return lines[:n]


_LOADERS = {
    "code": (_mbpp_texts, "train", "test"),
    "math": (_gsm8k_texts, "train", "test"),
    "knowledge": (_arc_texts, "train", "test"),
    "chat": (_wikitext_texts, "train", "validation"),
}


def load_task_texts(
    task: str, n_calib: int, n_eval: int, seed: int = 0
) -> tuple[list[str], list[str]]:
    if task not in _LOADERS:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    loader, calib_split, eval_split = _LOADERS[task]
    calib = loader(calib_split, n_calib, seed)
    eval_texts = loader(eval_split, n_eval, seed + 1)
    return calib, eval_texts


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap_ratio(calib_texts: list[str], eval_texts: list[str], n: int = 8) -> float:
    """Fraction of eval n-grams that also appear somewhere in calibration
    text — a leakage check, not just a split-name check (plan-v2 section 6
    hygiene note)."""
    calib_grams: set[tuple[str, ...]] = set()
    for t in calib_texts:
        calib_grams |= _word_ngrams(t, n)
    eval_grams: set[tuple[str, ...]] = set()
    for t in eval_texts:
        eval_grams |= _word_ngrams(t, n)
    if not eval_grams:
        return 0.0
    overlap = len(calib_grams & eval_grams)
    return overlap / len(eval_grams)


def text_list_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


@dataclass
class TaskManifest:
    task: str
    n_calib: int
    n_eval: int
    seed: int
    calib_hash: str
    eval_hash: str
    ngram_overlap_ratio: float

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "n_calib": self.n_calib,
            "n_eval": self.n_eval,
            "seed": self.seed,
            "calib_hash": self.calib_hash,
            "eval_hash": self.eval_hash,
            "ngram_overlap_ratio": self.ngram_overlap_ratio,
        }


def build_manifest(task: str, n_calib: int, n_eval: int, seed: int = 0) -> tuple[TaskManifest, list[str], list[str]]:
    calib, eval_texts = load_task_texts(task, n_calib, n_eval, seed)
    overlap = ngram_overlap_ratio(calib, eval_texts, n=8)
    manifest = TaskManifest(
        task=task,
        n_calib=len(calib),
        n_eval=len(eval_texts),
        seed=seed,
        calib_hash=text_list_hash(calib),
        eval_hash=text_list_hash(eval_texts),
        ngram_overlap_ratio=overlap,
    )
    return manifest, calib, eval_texts
