"""Patch composition (plan-v2 section 5, experiment E5): can two tasks'
patches be merged into one and serve both tasks at once?

Composition operates on score tensors (not on already-built patches), since
merging needs to re-rank groups under a shared budget: where two tasks both
want a group, one selection covers both; where budgets collide, keep the
group with the higher normalized score.
"""
from __future__ import annotations

import torch

from selfquant.sensitivity.scores import top_k_group_mask


def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp(min=1e-8)


def compose_masks(scores_list: list[torch.Tensor], k_frac_total: float) -> torch.Tensor:
    """Merge N tasks' per-group score tensors into one mask under a shared
    total budget k_frac_total, keeping each group by its max normalized
    score across tasks (so a group critical to any one task survives)."""
    if len(scores_list) < 2:
        raise ValueError("compose_masks needs at least 2 score tensors")
    shape = scores_list[0].shape
    for s in scores_list:
        if s.shape != shape:
            raise ValueError("all score tensors must have the same shape")
    normalized = torch.stack([_zscore(s) for s in scores_list], dim=0)
    combined = normalized.max(dim=0).values
    return top_k_group_mask(combined, k_frac_total)
