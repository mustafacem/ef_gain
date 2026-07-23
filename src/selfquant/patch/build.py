"""Patch construction (plan-v2 section 4.3).

A patch's atomic unit is one (output_row, group) cell — group_size
consecutive input-dim weights for a single output row — matching the
group-wise scale/zero granularity in quant/rtn.py and quant/gptq.py, and the
per-group scores in sensitivity/scores.py. A patch is just the *original*
fp16 values for a selected subset of these cells, plus enough bookkeeping to
scatter them back.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatchLayer:
    """One layer's worth of patch data.

    indices: LongTensor [n_groups, 2] of (row, group) coordinates.
    values:  Tensor [n_groups, group_size], the original fp16 weights.
    """

    indices: torch.Tensor
    values: torch.Tensor
    group_size: int
    in_features: int


def build_patch_layer(
    w_original: torch.Tensor, mask: torch.Tensor, group_size: int
) -> PatchLayer:
    """Extract the fp16 values for the selected (row, group) cells.

    w_original: [out_features, in_features] — the *un-quantized* weights.
    mask: [out_features, num_groups] bool, e.g. from top_k_group_mask.
    """
    out_features, in_features = w_original.shape
    num_groups = in_features // group_size
    if mask.shape != (out_features, num_groups):
        raise ValueError(f"mask shape {mask.shape} != {(out_features, num_groups)}")

    rows, groups = torch.nonzero(mask, as_tuple=True)
    w_grouped = w_original.reshape(out_features, num_groups, group_size)
    values = w_grouped[rows, groups].clone().half()
    indices = torch.stack([rows, groups], dim=1).to(torch.int64)
    return PatchLayer(
        indices=indices, values=values, group_size=group_size, in_features=in_features
    )


def patch_layer_n_weights(patch: PatchLayer) -> int:
    return patch.indices.shape[0] * patch.group_size


def patch_layer_bytes(patch: PatchLayer) -> int:
    """Storage cost: fp16 values + int32-packable indices (row, group each
    fit in int32 for any realistic model, so we cost them at 4B each)."""
    value_bytes = patch.values.numel() * 2
    index_bytes = patch.indices.shape[0] * 2 * 4
    return value_bytes + index_bytes
