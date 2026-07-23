"""Per-group sensitivity scores (plan-v2 section 4.2), all reduced to shape
[out_features, num_groups] — the same shape as the (scale, zero) qparams in
quant/rtn.py and quant/gptq.py, so a score and a patch group always line up.
"""
from __future__ import annotations

import torch


def _sum_over_groups(elementwise: torch.Tensor, group_size: int) -> torch.Tensor:
    out_features, in_features = elementwise.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )
    num_groups = in_features // group_size
    return elementwise.reshape(out_features, num_groups, group_size).sum(dim=-1)


def score_awq(
    w: torch.Tensor, act_abs_mean: torch.Tensor, group_size: int
) -> torch.Tensor:
    """S_awq(g) = sum_{ij in g} |W_ij| * E[|X_j|]. No gradients needed."""
    elementwise = w.abs() * act_abs_mean.unsqueeze(0)
    return _sum_over_groups(elementwise, group_size)


def score_fisher(
    w: torch.Tensor,
    w_quant: torch.Tensor,
    fisher: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """S_fisher(g) = sum_{ij in g} F_ij * (W_ij - Wq_ij)^2 — the expected
    task-loss increase from quantizing exactly this group (diagonal
    second-order Taylor). Feed `fisher` computed from task loss or from
    KL(p_fp16 || p_model) to get either metric variant in plan-v2 4.2."""
    elementwise = fisher * (w - w_quant).pow(2)
    return _sum_over_groups(elementwise, group_size)


def top_k_group_mask(scores: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Boolean mask, same shape as scores, marking the top k_frac fraction of
    groups by score (global top-k across the whole tensor, not per-row)."""
    flat = scores.flatten()
    k = max(1, int(round(k_frac * flat.numel())))
    threshold = torch.topk(flat, k, largest=True).values.min()
    mask = scores >= threshold
    # Break ties deterministically so the mask has exactly k True entries.
    n_true = mask.sum().item()
    if n_true > k:
        idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        keep = idx[torch.argsort(flat[idx], descending=True)[:k]]
        mask = torch.zeros_like(mask).flatten()
        mask[keep] = True
        mask = mask.reshape(scores.shape)
    return mask
