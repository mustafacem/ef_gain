import torch

from selfquant.patch.residual import restoration_delta
from selfquant.patch.varbit import (
    effective_bits_per_weight,
    groups_for_budget,
    patch_bytes,
    quantize_delta,
)
from selfquant.quant.rtn import rtn_quantize
from selfquant.sensitivity.scores import top_k_group_mask


def _setup(out_f=16, in_f=256, bits=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g)
    q, _, _ = rtn_quantize(w, bits=bits, group_size=128)
    return w, q


def test_masked_groups_stay_exactly_zero():
    w, q = _setup()
    mask = torch.zeros(16, 2, dtype=torch.bool)
    mask[:, 0] = True
    d = quantize_delta(w - q, mask, 128, patch_bits=4)
    assert d[:, 128:].abs().max().item() == 0.0
    assert d[:, :128].abs().max().item() > 0.0


def test_error_decreases_monotonically_with_patch_bits():
    """More bits in the patch must mean a closer correction."""
    w, q = _setup()
    mask = torch.ones(16, 2, dtype=torch.bool)
    target = w - q
    errs = {}
    for pb in (2, 4, 6, 8, 16):
        d = quantize_delta(target, mask, 128, patch_bits=pb)
        errs[pb] = (target - d).pow(2).mean().item()
    bits = sorted(errs)
    for a, b in zip(bits, bits[1:]):
        assert errs[a] > errs[b], f"{a}-bit ({errs[a]:.3e}) should be worse than {b}-bit ({errs[b]:.3e})"


def test_patch_bits_16_matches_the_fp16_restoration_path():
    """The old design must be recoverable as the patch_bits=16 special case."""
    w, q = _setup()
    torch.manual_seed(0)
    mask = top_k_group_mask(torch.rand(16, 2), k_frac=0.5)
    old = restoration_delta(w, q, mask, 128)
    new = quantize_delta(w - q, mask, 128, patch_bits=16)
    assert torch.allclose(old.float(), new.float(), atol=1e-3)


def test_six_bit_captures_most_of_fp16_benefit():
    """The premise of the redesign: a 6-bit patch should recover the large
    majority of what an fp16 patch recovers, at a fraction of the storage."""
    w, q = _setup(out_f=32, in_f=512, bits=3)
    mask = torch.ones(32, 4, dtype=torch.bool)
    target = w - q
    base_err = target.pow(2).mean().item()

    e16 = (target - quantize_delta(target, mask, 128, 16)).pow(2).mean().item()
    e6 = (target - quantize_delta(target, mask, 128, 6)).pow(2).mean().item()

    recovered_16 = 1 - e16 / base_err
    recovered_6 = 1 - e6 / base_err
    assert recovered_6 > 0.9 * recovered_16

    b16 = patch_bytes(32 * 4, 128, 16)
    b6 = patch_bytes(32 * 4, 128, 6)
    assert b6 < b16 / 2  # and it costs less than half as much


def test_patch_bytes_matches_hand_computation():
    # 10 groups of 128 weights at 4 bits: 10*128*4 = 5120 value bits,
    # plus 10 groups * 2 qparams * 16 bits = 320 bits -> 5440 bits = 680 B,
    # plus 10 * 8 B of indices = 80 B.
    assert patch_bytes(10, 128, 4) == 680 + 80
    # fp16 stores values directly, so no qparams: 10*128*16 = 20480 bits
    # = 2560 B, plus 80 B indices.
    assert patch_bytes(10, 128, 16) == 2560 + 80


def test_groups_for_budget_is_inverse_of_patch_bytes():
    for pb in (4, 6, 8, 16):
        n = groups_for_budget(10_000, 128, pb, target_bytes=100_000)
        assert patch_bytes(n, 128, pb) <= 100_000
        assert patch_bytes(n + 1, 128, pb) > 100_000


def test_lower_bits_buy_more_groups_at_fixed_budget():
    """The whole point of the sweep: a fixed byte budget patches far more
    groups at 4 bits than at fp16."""
    budget = 200_000
    n4 = groups_for_budget(100_000, 128, 4, budget)
    n16 = groups_for_budget(100_000, 128, 16, budget)
    assert n4 > 2 * n16


def test_effective_bits_increases_with_patch_bits():
    tot = 128 * 1000
    lo = effective_bits_per_weight(tot, 50, 128, base_bits=3, patch_bits=4)
    hi = effective_bits_per_weight(tot, 50, 128, base_bits=3, patch_bits=16)
    assert 3.0 < lo < hi


def test_rejects_bad_patch_bits():
    w, q = _setup()
    mask = torch.ones(16, 2, dtype=torch.bool)
    for bad in (0, 17, -1):
        try:
            quantize_delta(w - q, mask, 128, patch_bits=bad)
            assert False, f"expected ValueError for patch_bits={bad}"
        except ValueError:
            pass


# --- idea C: solve/quantize iteration -------------------------------------

def _iter_setup(out_f=16, in_f=256, n=800, bits=3, seed=0):
    from selfquant.quant.rtn import rtn_quantize
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g)
    x = torch.randn(n, in_f, generator=g)
    q, _, _ = rtn_quantize(w, bits=bits, group_size=128)
    H = (x.t() @ x).float()
    return w, q, x, H


def _act_err(x, w, w_eff):
    return (x @ (w - w_eff).t()).pow(2).sum().item()


def test_round1_equals_the_one_shot_path():
    """rounds=1 must reproduce solve-then-quantize exactly, so the iteration
    strictly generalises the current implementation."""
    from selfquant.patch.residual import solve_residual_layer
    from selfquant.patch.varbit import iterate_solve_quantize

    w, q, x, H = _iter_setup()
    torch.manual_seed(0)
    mask = top_k_group_mask(torch.rand(16, 2), k_frac=0.5)

    one_shot = quantize_delta(solve_residual_layer(w, q, H, mask, 128), mask, 128, 3)
    iterated = iterate_solve_quantize(w, q, H, mask, 128, patch_bits=3, rounds=1)
    assert torch.allclose(one_shot, iterated, atol=1e-6)


def test_more_rounds_never_increase_activation_error():
    from selfquant.patch.varbit import iterate_solve_quantize

    w, q, x, H = _iter_setup()
    torch.manual_seed(1)
    mask = top_k_group_mask(torch.rand(16, 2), k_frac=0.5)

    errs = []
    for r in (1, 2, 3):
        d = iterate_solve_quantize(w, q, H, mask, 128, patch_bits=3, rounds=r)
        errs.append(_act_err(x, w, q + d))
    assert errs[1] <= errs[0] + 1e-6
    assert errs[2] <= errs[1] + 1e-6
    assert errs[1] < errs[0]  # round 2 should actually help at 3 bits


def test_iteration_support_stays_inside_mask():
    from selfquant.patch.varbit import iterate_solve_quantize

    w, q, x, H = _iter_setup()
    mask = torch.zeros(16, 2, dtype=torch.bool)
    mask[:, 0] = True
    d = iterate_solve_quantize(w, q, H, mask, 128, patch_bits=3, rounds=3)
    assert d[:, 128:].abs().max().item() == 0.0


def test_iteration_rejects_zero_rounds():
    from selfquant.patch.varbit import iterate_solve_quantize

    w, q, x, H = _iter_setup()
    mask = torch.ones(16, 2, dtype=torch.bool)
    try:
        iterate_solve_quantize(w, q, H, mask, 128, patch_bits=3, rounds=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_patch_bytes_scales_values_with_rounds_but_not_indices():
    from selfquant.patch.varbit import patch_bytes as pb
    one = pb(10, 128, 4, rounds=1)
    two = pb(10, 128, 4, rounds=2)
    idx = 10 * 8
    assert (two - idx) == 2 * (one - idx)


def test_groups_for_budget_respects_rounds():
    """rounds=2 must buy roughly half as many groups as rounds=1, and the
    result must still round-trip through patch_bytes at the same rounds."""
    from selfquant.patch.varbit import groups_for_budget as gfb, patch_bytes as pb
    budget = 500_000
    n1 = gfb(100_000, 128, 3, budget, rounds=1)
    n2 = gfb(100_000, 128, 3, budget, rounds=2)
    assert 0 < n2 < n1
    assert pb(n2, 128, 3, rounds=2) <= budget
    assert pb(n2 + 1, 128, 3, rounds=2) > budget


def test_bitmap_beats_explicit_indices_above_low_coverage():
    """A dense bitmap costs total_groups/8 bytes flat; explicit indices cost
    8 B per selected group. At the coverages these patches run at, bitmap
    must buy strictly more groups for the same budget."""
    from selfquant.patch.varbit import groups_for_budget as gfb, patch_bytes as pb
    tg = 968_000
    budget = pb(int(0.05 * tg), 128, 16)
    ne = gfb(tg, 128, 3, budget)
    nb = gfb(tg, 128, 3, budget, bitmap=True)
    assert nb > ne
    assert pb(nb, 128, 3, total_groups=tg) <= budget      # honest accounting
    assert pb(nb + 1, 128, 3, total_groups=tg) > budget   # and tight
