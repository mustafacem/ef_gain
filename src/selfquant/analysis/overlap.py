"""Overlap methodology for the task-sensitivity atlas (plan-v2 section 4.4,
experiment E2 — the go/no-go gate).

This module is deliberately generic: it takes already-computed per-group
score tensors (any shape) and produces Jaccard/Spearman overlap numbers,
plus the noise-ceiling and random-floor controls that make those numbers
interpretable. Driving the actual calibration-split / multi-task scoring
runs is a separate concern (a script that calls into sensitivity/scores.py
twice per task and hands the results here).
"""
from __future__ import annotations

import torch
from scipy.stats import spearmanr

from selfquant.sensitivity.scores import top_k_group_mask


def jaccard_topk(scores_a: torch.Tensor, scores_b: torch.Tensor, k_frac: float) -> float:
    if scores_a.shape != scores_b.shape:
        raise ValueError("score tensors must have the same shape")
    mask_a = top_k_group_mask(scores_a, k_frac)
    mask_b = top_k_group_mask(scores_b, k_frac)
    intersection = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return intersection / union if union > 0 else 0.0


def jaccard_sweep(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    k_fracs: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
) -> dict[float, float]:
    return {k: jaccard_topk(scores_a, scores_b, k) for k in k_fracs}


def spearman_correlation(scores_a: torch.Tensor, scores_b: torch.Tensor) -> float:
    """Threshold-free overlap: rank correlation over the full score vectors."""
    if scores_a.shape != scores_b.shape:
        raise ValueError("score tensors must have the same shape")
    a = scores_a.flatten().cpu().numpy()
    b = scores_b.flatten().cpu().numpy()
    rho, _ = spearmanr(a, b)
    return float(rho)


def random_floor(k_frac: float) -> float:
    """Closed-form expected Jaccard of two independent random top-k_frac
    subsets of the same population: E[|A∩B|]=k^2/N, E[|A∪B|]=2k-k^2/N, so
    Jaccard = k_frac / (2 - k_frac)."""
    return k_frac / (2 - k_frac)


def split_half_noise_ceiling(
    scores_half_a: torch.Tensor, scores_half_b: torch.Tensor, k_frac: float
) -> float:
    """Within-task overlap: same task, two calibration halves. This is the
    ceiling that cross-task overlap must be compared against — the go/no-go
    signal in plan-v2 section 1 is `mean(within) - mean(cross)`, not the raw
    cross-task number."""
    return jaccard_topk(scores_half_a, scores_half_b, k_frac)


def overlap_matrix(
    task_scores: dict[str, torch.Tensor], k_frac: float
) -> dict[str, dict[str, float]]:
    """All-pairs Jaccard overlap at a fixed k_frac, including the diagonal
    (trivially 1.0) for a complete heatmap."""
    names = list(task_scores.keys())
    matrix: dict[str, dict[str, float]] = {a: {} for a in names}
    for i, a in enumerate(names):
        for b in names[i:]:
            j = jaccard_topk(task_scores[a], task_scores[b], k_frac)
            matrix[a][b] = j
            matrix[b][a] = j
    return matrix


def go_no_go_signal(
    within_task_overlaps: list[float], cross_task_overlaps: list[float]
) -> float:
    """mean(within-task) - mean(cross-task), the decision statistic from
    plan-v2 section 1. >=0.15 -> full speed ahead; <0.05 -> pivot to the
    negative-result framing; in between -> proceed with reduced scope."""
    if not within_task_overlaps or not cross_task_overlaps:
        raise ValueError("need at least one within- and cross-task overlap value")
    mean_within = sum(within_task_overlaps) / len(within_task_overlaps)
    mean_cross = sum(cross_task_overlaps) / len(cross_task_overlaps)
    return mean_within - mean_cross


def per_layer_breakdown(
    layer_scores_a: dict[str, torch.Tensor],
    layer_scores_b: dict[str, torch.Tensor],
    k_frac: float,
) -> dict[str, float]:
    """Per-layer-name Jaccard overlap, so attention-vs-MLP or depth trends
    (hypothesis H4 in plan-v2) can be read off directly."""
    common = sorted(set(layer_scores_a) & set(layer_scores_b))
    return {name: jaccard_topk(layer_scores_a[name], layer_scores_b[name], k_frac) for name in common}
