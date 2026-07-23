"""Calibration/eval built as long token sequences, not short per-example texts.

The original loader took N short texts and truncated each to 128 tokens. That
produced far too few activation samples: a down_proj Hessian is 4864x4864 but
was being estimated from ~4400 samples for the code task, i.e. LESS THAN ONE
sample per dimension. H was singular and the damping term was doing
structural work rather than numerical stabilisation, which puts every result
that inverts H (the closed-form solve, OBS scores, the GPTQ base itself)
on shaky ground.

Standard practice for GPTQ/AWQ calibration is 128 sequences of 2048 tokens
= 262k tokens, or ~54 samples per dimension for this layer shape. This module
reaches that by concatenating a task's corpus and chunking it into full-length
sequences, which extracts far more usable tokens from the same source data.

Where a corpus is simply too small to reach the target (mbpp is ~30k tokens
total), `build_sequences` returns what it can and reports the shortfall rather
than silently under-sampling -- the caller is expected to log samples-per-dim
alongside any result that depends on H.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class SequenceSet:
    ids: torch.Tensor          # [n_seq, seq_len] token ids
    n_tokens: int
    n_source_texts: int
    seq_len: int

    def samples_per_dim(self, in_features: int) -> float:
        return self.n_tokens / in_features

    def summary(self, in_features: int = 4864) -> str:
        return (
            f"{self.ids.shape[0]} seqs x {self.seq_len} tok = {self.n_tokens} tokens "
            f"({self.samples_per_dim(in_features):.1f} samples/dim at in_features={in_features})"
        )


def build_sequences(
    texts: list[str],
    tok,
    seq_len: int = 2048,
    max_seqs: int = 128,
    seed: int = 0,
    separator: str = "\n\n",
) -> SequenceSet:
    """Concatenate texts and chunk into fixed-length sequences.

    Shuffling before concatenation keeps any single long document from
    dominating a contiguous run of sequences.
    """
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    buf: list[int] = []
    seqs: list[list[int]] = []
    sep_ids = tok(separator, add_special_tokens=False)["input_ids"]

    for i in order:
        buf.extend(tok(texts[i], add_special_tokens=False)["input_ids"])
        buf.extend(sep_ids)
        while len(buf) >= seq_len and len(seqs) < max_seqs:
            seqs.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(seqs) >= max_seqs:
            break

    # Keep a final partial chunk only if we would otherwise have nothing.
    if not seqs and buf:
        seqs.append(buf)

    ids = torch.tensor(seqs, dtype=torch.long) if len(set(map(len, seqs))) == 1 else None
    if ids is None:  # ragged single partial sequence
        ids = torch.tensor([seqs[0]], dtype=torch.long)
    return SequenceSet(
        ids=ids,
        n_tokens=int(ids.numel()),
        n_source_texts=len(texts),
        seq_len=int(ids.shape[1]),
    )


def report_conditioning(name: str, seqs: SequenceSet, in_features: int) -> dict:
    """Emit the diagnostic that should accompany any H-dependent result."""
    spd = seqs.samples_per_dim(in_features)
    status = "RANK-DEFICIENT" if spd < 1.0 else ("thin" if spd < 10 else "ok")
    return {
        "name": name,
        "n_tokens": seqs.n_tokens,
        "in_features": in_features,
        "samples_per_dim": spd,
        "status": status,
    }
