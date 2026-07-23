"""E1 — proxy validation (plan-v2 section 5): the Taylor-approximation
sensitivity scores are proxies for the actual loss increase a group's
quantization causes. Before trusting any overlap atlas built from them,
check the correlation between predicted score and measured Delta-loss on a
sample of groups. Whichever metric correlates best becomes primary; the
others become ablations. If nothing correlates, the atlas is not trustworthy
yet — debug before drawing conclusions from it (plan-v2 section 5, E1).

Caller pitfall: if your `loss_fn` memoizes by the candidate tensor's
`data_ptr()`, don't — per-cell restored weights in
`measure_actual_group_delta_loss` are short-lived clones freed right after
use, and a CPU allocator will happily hand the next same-size clone() the
just-freed address, so a data_ptr-keyed cache silently returns a stale
result instead of recomputing (this produced constant-array/NaN
correlations before it was caught). Only the literal, still-referenced
`w_quantized` object passed in is safe to memoize, and only by identity
(`is`), not by address.
"""
from __future__ import annotations

import torch
from scipy.stats import spearmanr


def measure_actual_group_delta_loss(
    w_quantized: torch.Tensor,
    w_original: torch.Tensor,
    group_size: int,
    row: int,
    group: int,
    loss_fn,
) -> float:
    """Restore a single (row, group) cell to its original value in an
    otherwise-quantized weight matrix, measure the change in `loss_fn(w)`.

    loss_fn: callable(w: Tensor[out,in]) -> float, e.g. task loss or KL to
    the fp16 teacher, evaluated with this weight matrix substituted in.
    """
    baseline_loss = loss_fn(w_quantized)
    w_restored = w_quantized.clone()
    out_features, in_features = w_quantized.shape
    num_groups = in_features // group_size
    w_restored_grouped = w_restored.reshape(out_features, num_groups, group_size)
    orig_grouped = w_original.reshape(out_features, num_groups, group_size)
    w_restored_grouped[row, group] = orig_grouped[row, group]
    restored_loss = loss_fn(w_restored.reshape(out_features, in_features))
    return baseline_loss - restored_loss  # positive = restoring this group helped


def validate_proxy_scores(
    scores: torch.Tensor,
    w_quantized: torch.Tensor,
    w_original: torch.Tensor,
    group_size: int,
    loss_fn,
    sample_cells: list[tuple[int, int]],
) -> float:
    """Spearman correlation between proxy `scores` and measured actual
    Delta-loss over a sample of (row, group) cells. Returns the correlation;
    the caller (E1 driver) compares this across AWQ/Fisher/KL metrics."""
    predicted = []
    actual = []
    for row, group in sample_cells:
        predicted.append(scores[row, group].item())
        actual.append(
            measure_actual_group_delta_loss(
                w_quantized, w_original, group_size, row, group, loss_fn
            )
        )
    rho, _ = spearmanr(predicted, actual)
    return float(rho)


def sample_cells_across_score_range(
    scores: torch.Tensor, n_samples: int, seed: int = 0
) -> list[tuple[int, int]]:
    """Stratified sample spanning the full score range (not just the top),
    so the correlation check isn't biased toward already-obvious outliers."""
    flat = scores.flatten()
    order = torch.argsort(flat)
    n_total = flat.numel()
    g = torch.Generator().manual_seed(seed)
    stratum_edges = torch.linspace(0, n_total, steps=n_samples + 1).long()
    picks = []
    for i in range(n_samples):
        lo, hi = stratum_edges[i].item(), stratum_edges[i + 1].item()
        hi = max(hi, lo + 1)
        idx_in_stratum = order[lo:hi]
        pick = idx_in_stratum[torch.randint(0, idx_in_stratum.numel(), (1,), generator=g)]
        picks.append(pick.item())
    num_groups = scores.shape[1]
    return [(p // num_groups, p % num_groups) for p in picks]
