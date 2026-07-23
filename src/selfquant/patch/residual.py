"""Closed-form activation-space residual correction.

The original patch design restored selected groups to their *original* fp16
values. That is only optimal if the base is an unbiased per-weight
approximation — which GPTQ's base is not, since its error compensation has
already redistributed each group's quantization error into surrounding
weights. Restoring a group then removes an error the rest of the matrix is
still compensating for (measured: patching a GPTQ base made things worse in
13 of 16 configurations).

This module learns the correction instead of assuming it. For a fixed mask
S, minimising activation-space reconstruction error

    argmin_delta  || X (W - Q(W) - delta) ||^2     s.t. supp(delta) subset S

is a least-squares problem with a closed form, per output row:

    delta[S] = (H_SS)^-1 (H r)[S]        where H = X^T X, r = w - q

H is exactly the Hessian already accumulated for GPTQ, so no new
calibration machinery is needed and no gradient descent is involved.
Restoration is the special case delta[S] = r[S], which this strictly
dominates: it is one point in the space being optimised over.

Because the objective is stated against the actual quantized base, the
solution automatically accounts for whatever compensation the base carries
— which is why this is expected to rescue the coupled (GPTQ) base rather
than requiring an uncoupled one.
"""
from __future__ import annotations

import torch


# Ceiling on the [B, b, b] batched-system stack built during a solve. The
# per-row system size b scales with the patch budget, so without this a
# large-k solve silently tries to allocate many GB at once.
_BATCH_BYTES_MAX = 256 * 1024 * 1024


def _batch_bytes(device: torch.device) -> int:
    """Cap the batched-solve working set at a fraction of memory that is
    actually free, not a fixed constant — the same budget that is roomy
    beside a 0.5B model will OOM beside a larger one."""
    if device.type != "cuda":
        return _BATCH_BYTES_MAX
    free_b, _ = torch.cuda.mem_get_info(device)
    return max(16 * 1024 * 1024, min(_BATCH_BYTES_MAX, int(0.25 * free_b)))


def _cols_for_groups(group_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand group indices to the column indices they cover."""
    offsets = torch.arange(group_size, device=group_idx.device)
    return (group_idx.unsqueeze(1) * group_size + offsets.unsqueeze(0)).reshape(-1)


def solve_residual_layer(
    w: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    damp: float = 0.01,
    max_batch: int = 64,
) -> torch.Tensor:
    """Optimal activation-space correction for one layer under a fixed mask.

    w, q: [out_features, in_features] original and quantized weights.
    H:    [in_features, in_features] Hessian from calibration activations
          (any positive scaling; it cancels in the solve).
    mask: [out_features, num_groups] bool, which groups may be corrected.

    Returns delta [out_features, in_features], zero outside the mask, to be
    added to q. Rows are solved in batches grouped by how many groups they
    select, so identically-shaped systems share one batched solve.
    """
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by {group_size}")
    num_groups = in_features // group_size
    if mask.shape != (out_features, num_groups):
        raise ValueError(f"mask shape {mask.shape} != {(out_features, num_groups)}")

    device = H.device
    R = (w - q).to(device=device, dtype=torch.float32)
    H = H.to(torch.float32)
    delta = torch.zeros_like(R)

    lam = damp * torch.diagonal(H).mean().clamp(min=1e-8)
    HR = R @ H  # row j == (H r_j)^T, H symmetric

    mask = mask.to(device)
    counts = mask.sum(dim=1)
    for n_sel in counts.unique().tolist():
        if n_sel == 0:
            continue
        rows = torch.nonzero(counts == n_sel, as_tuple=False).flatten()
        b = int(n_sel) * group_size
        # Cap the batch so the [B, b, b] system stack stays well under a
        # few hundred MB; b grows with the budget, so this matters.
        budget_rows = max(1, int(_batch_bytes(device) / (4 * b * b)))
        step = max(1, min(max_batch, budget_rows))
        # All rows here select the same NUMBER of groups (not the same ones),
        # so their linear systems share a shape and can be solved batched.
        for start in range(0, rows.numel(), step):
            chunk = rows[start : start + step]
            gsel = torch.stack([torch.nonzero(mask[j], as_tuple=False).flatten() for j in chunk])
            cols = torch.stack([_cols_for_groups(g, group_size) for g in gsel])  # [B, b]
            # Build [B, b, b] directly via broadcast advanced indexing; going
            # through H[cols.reshape(-1)] would materialise [B*b, in_features]
            # first, which is orders of magnitude larger.
            Hss = H[cols.unsqueeze(2), cols.unsqueeze(1)]  # [B, b, b]
            Hss = Hss + lam * torch.eye(b, device=device).unsqueeze(0)
            rhs = torch.gather(HR[chunk], 1, cols).unsqueeze(-1)  # [B, b, 1]
            sol = torch.linalg.solve(Hss, rhs).squeeze(-1)  # [B, b]
            delta[chunk.unsqueeze(1).expand(-1, b), cols] = sol

    return delta.to(w.dtype)


def score_obs_groups(
    w: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
    damp: float = 0.01,
) -> torch.Tensor:
    """Per-group activation-space error reduction from correcting that group
    optimally, in isolation:

        score(j, g) = (H r_j)[g]^T (H_gg)^-1 (H r_j)[g]

    This is the exact drop in || X (w_j - q_j - delta) ||^2 achievable by
    correcting only group g of row j — an OBS/OBC-style saliency, stated in
    the same units as the objective the patch actually optimises.

    Compare with score_awq in sensitivity/scores.py, which uses |W| * E[|X|]:
    that ignores both the quantization error actually present and the
    correlation structure in H. The measured proxy correlation of AWQ
    against real restoration benefit was only 0.07-0.43.

    One Cholesky per group, batched over output rows.
    Returns [out_features, num_groups].
    """
    out_features, in_features = w.shape
    num_groups = in_features // group_size
    device = H.device
    R = (w - q).to(device=device, dtype=torch.float32)
    H = H.to(torch.float32)
    lam = damp * torch.diagonal(H).mean().clamp(min=1e-8)
    HR = R @ H

    scores = torch.zeros(out_features, num_groups, device=device)
    eye = torch.eye(group_size, device=device)
    for g in range(num_groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        Hgg = H[sl, sl] + lam * eye
        L = torch.linalg.cholesky(Hgg)
        B = HR[:, sl]  # [out, gs]
        sol = torch.cholesky_solve(B.t(), L).t()  # [out, gs]
        scores[:, g] = (B * sol).sum(dim=1)
    return scores.cpu()


def restoration_delta(
    w: torch.Tensor, q: torch.Tensor, mask: torch.Tensor, group_size: int
) -> torch.Tensor:
    """The old method expressed as a delta, for apples-to-apples comparison:
    delta[S] = (w - q)[S], i.e. restore the original fp16 values."""
    R = (w - q).float()
    keep = mask.to(R.device).repeat_interleave(group_size, dim=1)
    return (R * keep).to(w.dtype)
