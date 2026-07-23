"""Group-wise round-to-nearest (RTN) fake quantization.

Weights are shaped [out_features, in_features]. Groups are contiguous runs of
`group_size` columns (input-dim positions) sharing one asymmetric scale/zero
per output row. This grouping matches the patch-group definition in plan-v2
section 2, so a "group" here is always exactly one restorable patch unit.
"""
from __future__ import annotations

import torch


def compute_group_qparams(
    w: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(row, group) asymmetric scale/zero for weight matrix w [out, in].

    Returns scale, zero of shape [out, num_groups], float32.
    Assumes in_features is divisible by group_size (caller pads otherwise).
    """
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )
    num_groups = in_features // group_size
    wg = w.reshape(out_features, num_groups, group_size).float()

    qmax = 2**bits - 1
    w_min = wg.min(dim=-1).values
    w_max = wg.max(dim=-1).values
    # Ensure zero is representable and range is non-degenerate.
    w_min = torch.minimum(w_min, torch.zeros_like(w_min))
    w_max = torch.maximum(w_max, torch.zeros_like(w_max))
    scale = (w_max - w_min).clamp(min=1e-8) / qmax
    zero = torch.round(-w_min / scale)
    return scale, zero


def quantize_dequantize_group(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Fake-quantize w using precomputed per-group (scale, zero).

    Returns a tensor of the same shape/dtype as w holding dequantized
    (i.e. quantized-then-reconstructed) values.
    """
    out_features, in_features = w.shape
    num_groups = in_features // group_size
    qmax = 2**bits - 1
    wg = w.reshape(out_features, num_groups, group_size).float()
    s = scale.unsqueeze(-1)
    z = zero.unsqueeze(-1)
    q = torch.clamp(torch.round(wg / s + z), 0, qmax)
    dq = (q - z) * s
    return dq.reshape(out_features, in_features).to(w.dtype)


def rtn_quantize(
    w: torch.Tensor, bits: int, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full RTN fake-quant pipeline: compute qparams then dequantize.

    Returns (w_fake_quant, scale, zero).
    """
    scale, zero = compute_group_qparams(w, bits, group_size)
    w_dq = quantize_dequantize_group(w, scale, zero, bits, group_size)
    return w_dq, scale, zero
