"""Variable-bit patches: store the correction at N bits, not fp16.

Every patch so far upgraded its selected groups from the base precision
straight to fp16 -- spending ~12.75 extra bits per weight on a 3-bit base.
That is hard to justify against the measured bit-scaling law: each added bit
removes ~79% of the remaining quantization error (matching the Delta^2/12
theory, where halving the step size is a 4x squared-error reduction). So

    3 -> 6 bit   captures ~98% of the benefit for ~0.15 bits/weight at k=5%
    3 -> fp16    captures    100%             for ~0.64 bits/weight

i.e. roughly a quarter of the storage for nearly all of the gain. Since
uniform quantization was out-competing fp16 patches by about that same
factor, the choice of patch precision -- not the choice of mask or of
correction values -- may be what actually decided the earlier results.

The correction is quantized with the same group-wise machinery as the base
(compute_group_qparams / quantize_dequantize_group from quant/rtn.py), so a
patch group carries its own scale/zero and nothing about the base's
quantization is disturbed.

fp16 storage is the patch_bits=16 special case, which makes the old design a
point on the new design's sweep rather than a separate thing to compare
against.
"""
from __future__ import annotations

import torch

from selfquant.quant.rtn import compute_group_qparams, quantize_dequantize_group


def quantize_delta(
    delta: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    patch_bits: int,
) -> torch.Tensor:
    """Quantize a dense correction to `patch_bits`, keeping only masked groups.

    delta: [out_features, in_features] correction to add to the base. Supply
        either the raw residual (w - q) or, better, the closed-form optimum
        from solve_residual_layer -- quantizing the *solved* correction keeps
        the activation-space optimality that the solve bought.
    mask:  [out_features, num_groups] bool, which groups the patch stores.

    Returns a delta of the same shape, zero outside the mask, holding
    dequantized values at `patch_bits` precision inside it.
    """
    out_features, in_features = delta.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by {group_size}")
    num_groups = in_features // group_size
    if mask.shape != (out_features, num_groups):
        raise ValueError(f"mask shape {mask.shape} != {(out_features, num_groups)}")
    if not 1 <= patch_bits <= 16:
        raise ValueError(f"patch_bits must be in [1, 16], got {patch_bits}")

    keep = mask.to(delta.device).repeat_interleave(group_size, dim=1)
    masked = delta * keep

    if patch_bits == 16:
        # fp16 storage: no integer quantization, just the storage dtype.
        return masked.half().to(delta.dtype)

    scale, zero = compute_group_qparams(masked, patch_bits, group_size)
    dq = quantize_dequantize_group(masked, scale, zero, patch_bits, group_size)
    # Re-apply the mask: unselected groups must stay exactly zero, and their
    # qparams are meaningless (an all-zero group quantizes to zero anyway,
    # but this makes the guarantee explicit rather than incidental).
    return dq * keep


def iterate_solve_quantize(
    w: torch.Tensor,
    q: torch.Tensor,
    H: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    patch_bits: int,
    rounds: int = 2,
    damp: float = 0.01,
) -> torch.Tensor:
    """Alternate solving and quantizing so the solve sees its own rounding.

    The one-shot pipeline is: solve for the optimal fp32 correction, then
    quantize it to `patch_bits`. Those two steps do not talk to each other --
    the solve optimises a correction it will never get to store, and the
    quantizer discards part of the optimality the solve just bought. At
    patch_bits=3 that discarded part is the dominant remaining error.

    This closes the loop. After quantizing round k's correction, the residual
    w - (q + accumulated) is recomputed and re-solved on the SAME mask, and
    the new increment is quantized and added. Each round therefore attacks
    the error the previous round's rounding left behind.

    Storage is unchanged in group count but not in bytes: each round stores
    its own quantized increment, so `rounds` rounds cost `rounds` x the
    single-round patch. Callers must price it that way -- see
    `patch_bytes(..., rounds=)`. Round 1 is exactly the existing one-shot
    path, so this strictly generalises it.

    Returns the accumulated correction to add to `q`.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    # Imported here to avoid a circular import at module load: residual.py
    # imports nothing from varbit, but varbit is imported by scripts that
    # also pull residual, and keeping this local makes the dependency
    # one-directional and obvious.
    from selfquant.patch.residual import solve_residual_layer

    total = torch.zeros_like(w)
    for _ in range(rounds):
        target = q + total
        solved = solve_residual_layer(w, target, H, mask, group_size, damp=damp)
        step = quantize_delta(solved, mask, group_size, patch_bits)
        if step.abs().max() == 0:
            break  # nothing left this round can represent
        total = total + step
    return total


def patch_bytes(
    n_selected_groups: int,
    group_size: int,
    patch_bits: int,
    index_bytes_per_group: int = 8,
    qparam_bits: int = 16,
    rounds: int = 1,
    total_groups: int | None = None,
) -> int:
    """Storage for a variable-bit patch, counting everything.

    Three components, all of which matter at small patch_bits:
      values   n_selected_groups * group_size * patch_bits
      qparams  scale + zero per stored group (fp16 each), unless patch_bits
               is 16, where values are stored directly and no qparams exist
      indices  (row, group) coordinate per stored group

    At patch_bits=4 the qparams are 32 bits per 128-weight group = 0.25
    bits/weight, which is a real fraction of the 4 bits/weight of payload --
    ignoring it would flatter low-bit patches.

    `rounds` > 1 accounts for iterate_solve_quantize: every round stores its
    own quantized increment over the same groups, so values and qparams scale
    with rounds while the (row, group) indices are shared and do not.

    `total_groups` switches the mask encoding from an explicit (row, group)
    list to a dense BITMAP over all groups -- one bit each, selected or not.
    At pb3 an explicit index costs 8 B against 52 B of payload, i.e. 13% of
    every patched group, and that fraction is paid again for each additional
    group. A bitmap costs total_groups/8 bytes regardless of how many are
    selected, so it wins as soon as coverage exceeds ~1.5%, and at the 22%
    coverage these patches actually run at it frees ~12% of the budget --
    which buys ~14% more coverage, the quantity that dominates patch quality.
    """
    value_bits = n_selected_groups * group_size * patch_bits * rounds
    per_round_qparams = 0 if patch_bits == 16 else n_selected_groups * 2 * qparam_bits
    qparam_total = per_round_qparams * rounds
    payload = (value_bits + qparam_total) // 8
    if total_groups is not None:
        return payload + (total_groups + 7) // 8
    return payload + n_selected_groups * index_bytes_per_group


def effective_bits_per_weight(
    total_weights: int,
    n_selected_groups: int,
    group_size: int,
    base_bits: int,
    patch_bits: int,
    group_qparam_bits: int = 32,
) -> float:
    """Amortised bits/weight for base + patch, for matched-budget comparisons.

    Includes the base's own per-group scale/zero, so figures here are
    comparable with the eff_bits() helpers used in the experiment scripts.
    """
    base = base_bits + group_qparam_bits / group_size
    patched_weights = n_selected_groups * group_size
    extra_bits = patch_bytes(n_selected_groups, group_size, patch_bits) * 8
    return base + extra_bits / total_weights if total_weights else base


def groups_for_budget(
    total_groups: int,
    group_size: int,
    patch_bits: int,
    target_bytes: int,
    index_bytes_per_group: int = 8,
    qparam_bits: int = 16,
    rounds: int = 1,
    bitmap: bool = False,
) -> int:
    """How many groups fit in `target_bytes` at `patch_bits`.

    This is what makes the sweep fair: comparing 6-bit and fp16 patches at
    equal k would just be comparing different storage budgets. The real
    question is whether, at a FIXED byte budget, it is better to patch few
    groups precisely or many groups coarsely.

    `rounds` mirrors patch_bytes: an iterated patch stores one quantized
    increment per round over the same groups, so it buys proportionally
    fewer groups. Without this, comparing rounds=2 against rounds=1 at the
    same group count would silently be comparing 2x the storage.
    """
    per_group_bits = (
        group_size * patch_bits + (0 if patch_bits == 16 else 2 * qparam_bits)
    ) * rounds
    if bitmap:
        # the mask costs a flat total_groups/8 bytes up front, then every
        # group is pure payload
        avail = target_bytes - (total_groups + 7) // 8
        if avail <= 0:
            return 0
        return max(0, min(total_groups, int(avail // (per_group_bits / 8))))
    per_group_bytes = per_group_bits / 8 + index_bytes_per_group
    return max(0, min(total_groups, int(target_bytes // per_group_bytes)))
