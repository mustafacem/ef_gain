import torch

from selfquant.patch.residual import (
    restoration_delta,
    score_obs_groups,
    solve_residual_layer,
)
from selfquant.quant.rtn import rtn_quantize
from selfquant.sensitivity.scores import top_k_group_mask


def _setup(out_f=16, in_f=256, n=800, bits=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g)
    x = torch.randn(n, in_f, generator=g)
    q, _, _ = rtn_quantize(w, bits=bits, group_size=128)
    H = (x.t() @ x).float()
    return w, q, x, H


def _act_err(x, w, w_eff):
    return (x @ (w - w_eff).t()).pow(2).sum().item()


def test_solve_respects_mask_support():
    w, q, x, H = _setup()
    mask = torch.zeros(16, 2, dtype=torch.bool)
    mask[:, 0] = True  # only first group correctable
    delta = solve_residual_layer(w, q, H, mask, group_size=128)
    assert delta[:, 128:].abs().max().item() == 0.0
    assert delta[:, :128].abs().max().item() > 0.0


def test_solve_beats_restoration_on_its_objective():
    """The central claim: for a fixed mask, the closed-form correction
    achieves lower activation-space error than restoring fp16 values,
    because restoration is just one point in the space being optimised."""
    w, q, x, H = _setup()
    torch.manual_seed(0)
    mask = top_k_group_mask(torch.rand(16, 2), k_frac=0.5)

    d_solve = solve_residual_layer(w, q, H, mask, group_size=128)
    d_restore = restoration_delta(w, q, mask, group_size=128)

    err_solve = _act_err(x, w, q + d_solve)
    err_restore = _act_err(x, w, q + d_restore)
    err_base = _act_err(x, w, q)

    assert err_solve < err_restore
    assert err_solve < err_base


def test_solve_with_full_mask_approaches_zero_error():
    """With every group correctable the solution should nearly recover w."""
    w, q, x, H = _setup()
    mask = torch.ones(16, 2, dtype=torch.bool)
    delta = solve_residual_layer(w, q, H, mask, group_size=128, damp=1e-6)
    err_full = _act_err(x, w, q + delta)
    err_base = _act_err(x, w, q)
    assert err_full < err_base * 1e-3


def test_solve_empty_mask_is_noop():
    w, q, x, H = _setup()
    mask = torch.zeros(16, 2, dtype=torch.bool)
    delta = solve_residual_layer(w, q, H, mask, group_size=128)
    assert delta.abs().max().item() == 0.0


def test_obs_scores_shape_and_nonnegativity():
    """Each score is a quadratic form with a PD matrix, so it must be >= 0."""
    w, q, x, H = _setup()
    s = score_obs_groups(w, q, H, group_size=128)
    assert s.shape == (16, 2)
    assert (s >= -1e-6).all()


def test_obs_score_predicts_actual_error_reduction():
    """An OBS score should equal the measured drop in activation-space error
    from correcting exactly that group — that is what it is defined to be."""
    w, q, x, H = _setup(out_f=4, in_f=256)
    scores = score_obs_groups(w, q, H, group_size=128, damp=1e-6)

    base_err_rows = (x @ (w - q).t()).pow(2).sum(dim=0)  # per output row
    for j in range(4):
        for g in range(2):
            mask = torch.zeros(4, 2, dtype=torch.bool)
            mask[j, g] = True
            d = solve_residual_layer(w, q, H, mask, group_size=128, damp=1e-6)
            new_err = (x @ (w - q - d).t()).pow(2).sum(dim=0)[j]
            measured_drop = base_err_rows[j].item() - new_err.item()
            predicted = scores[j, g].item()
            rel = abs(measured_drop - predicted) / max(abs(measured_drop), 1e-8)
            assert rel < 0.05, f"row {j} group {g}: predicted {predicted}, measured {measured_drop}"


def test_obs_mask_beats_random_mask_at_equal_budget():
    w, q, x, H = _setup(out_f=32, in_f=512)
    obs = score_obs_groups(w, q, H, group_size=128)
    torch.manual_seed(1)
    rnd = torch.rand_like(obs)

    m_obs = top_k_group_mask(obs, k_frac=0.25)
    m_rnd = top_k_group_mask(rnd, k_frac=0.25)
    assert m_obs.sum() == m_rnd.sum()

    e_obs = _act_err(x, w, q + solve_residual_layer(w, q, H, m_obs, 128))
    e_rnd = _act_err(x, w, q + solve_residual_layer(w, q, H, m_rnd, 128))
    assert e_obs < e_rnd


def test_batched_solve_matches_unbatched():
    """Row-batching is an optimisation, not a semantic change."""
    w, q, x, H = _setup(out_f=24, in_f=384)
    torch.manual_seed(2)
    mask = top_k_group_mask(torch.rand(24, 3), k_frac=0.4)
    big = solve_residual_layer(w, q, H, mask, 128, max_batch=64)
    small = solve_residual_layer(w, q, H, mask, 128, max_batch=1)
    assert torch.allclose(big, small, atol=1e-4)
