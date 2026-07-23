"""Own GPTQ implementation (Cholesky-based, Frantar et al. 2022 algorithm).

Kept minimal and hookable: the `hold_out_mask` argument implements the
"candidate-aware base" (design B, plan-v2 section 3) — columns in the mask
are excluded from the sequential error-compensation loop entirely (no error
propagates into or out of them via GPTQ), then quantized independently with
plain RTN at the end. Any patch drawn from the held-out candidate set can
then be restored to its exact original fp16 value with no residual
compensation error left behind in the rest of the matrix.
"""
from __future__ import annotations

import torch

from selfquant.quant.rtn import compute_group_qparams, quantize_dequantize_group


class GPTQHessian:
    """Accumulates H = 2 * X^T X / n from a linear layer's input activations."""

    def __init__(self, in_features: int, device: str | torch.device = "cpu"):
        self.in_features = in_features
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.n_samples = 0

    def update(self, x: torch.Tensor) -> None:
        """x: activations of shape [..., in_features] (a calibration batch),
        on any device — moved to the accumulator's device here, so a CPU
        accumulator can stream activations from a GPU-resident model."""
        x = x.reshape(-1, self.in_features).float().to(self.H.device)
        n = x.shape[0]
        if n == 0:
            return
        # Running average: rescale existing H, then add new contribution.
        total = self.n_samples + n
        self.H.mul_(self.n_samples / total)
        self.H.add_(x.t() @ x, alpha=2.0 / total)
        self.n_samples = total


def _inverse_cholesky(H: torch.Tensor, percdamp: float = 0.01) -> torch.Tensor:
    """Return upper-triangular Cholesky factor of H^{-1}, with dampening."""
    n = H.shape[0]
    dead = torch.diag(H) == 0
    H = H.clone()
    H[dead, dead] = 1.0
    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(n, device=H.device)
    H[idx, idx] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv_upper = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv_upper


def gptq_quantize(
    w: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    group_size: int = 128,
    percdamp: float = 0.01,
    hold_out_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weight matrix w [out_features, in_features] with GPTQ.

    H: Hessian [in_features, in_features] from calibration activations.
    hold_out_mask: optional bool tensor [in_features]; True = column is
        excluded from the GPTQ loop (no compensation in or out) and instead
        RTN-quantized independently afterward (design B).

    Returns (w_fake_quant, scale, zero) with scale/zero of shape
    [out_features, num_groups] as in rtn.py.
    """
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )
    device = w.device
    W = w.clone().float()
    Hinv = _inverse_cholesky(H.to(device), percdamp=percdamp)

    if hold_out_mask is None:
        hold_out_mask = torch.zeros(in_features, dtype=torch.bool, device=device)
    else:
        hold_out_mask = hold_out_mask.to(device=device, dtype=torch.bool)

    num_groups = in_features // group_size
    qmax = 2**bits - 1
    scale = torch.zeros(out_features, num_groups, device=device)
    zero = torch.zeros(out_features, num_groups, device=device)
    Q = torch.zeros_like(W)

    for g in range(num_groups):
        cols = range(g * group_size, (g + 1) * group_size)
        # Group qparams computed once, from the (still largely original)
        # weights at the start of this group's processing.
        w_group = W[:, g * group_size : (g + 1) * group_size]
        s, z = compute_group_qparams(w_group, bits, group_size)
        scale[:, g] = s.squeeze(-1) if s.dim() > 1 else s[:, 0]
        # compute_group_qparams returns [out, num_groups_in_input]; here
        # w_group has exactly one group, so num_groups_in_input == 1.
        scale[:, g] = s[:, 0]
        zero[:, g] = z[:, 0]

        for i in cols:
            w_col = W[:, i]
            if hold_out_mask[i]:
                # Skip entirely: no quant, no error, no propagation.
                Q[:, i] = w_col
                continue
            d = Hinv[i, i]
            qmax_t = qmax
            q_col = torch.clamp(
                torch.round(w_col / scale[:, g] + zero[:, g]), 0, qmax_t
            )
            dq_col = (q_col - zero[:, g]) * scale[:, g]
            Q[:, i] = dq_col
            err = (w_col - dq_col) / d

            if i + 1 < in_features:
                update_row = Hinv[i, i + 1 :].clone()
                # Zero propagation into held-out columns (design B guarantee).
                update_row[hold_out_mask[i + 1 :]] = 0.0
                W[:, i + 1 :] -= err.unsqueeze(1) @ update_row.unsqueeze(0)

    # Independently RTN-quantize held-out columns using their *original*
    # (untouched, since propagation into them was zeroed) values.
    if hold_out_mask.any():
        for g in range(num_groups):
            group_cols = torch.arange(g * group_size, (g + 1) * group_size, device=device)
            mask_in_group = hold_out_mask[group_cols]
            if not mask_in_group.any():
                continue
            w_group = W[:, group_cols]
            s, z = compute_group_qparams(w_group, bits, group_size)
            dq_group = quantize_dequantize_group(w_group, s, z, bits, group_size)
            held_cols = group_cols[mask_in_group]
            Q[:, held_cols] = dq_group[:, mask_in_group]
            # Held-out columns keep the group's own qparams for their slice;
            # since they're a strict subset of the group, store the RTN
            # qparams only informationally (main scale/zero above already
            # cover the non-held-out portion of this group).

    return Q.to(w.dtype), scale, zero
