import torch

from selfquant.analysis.proxy_check import (
    measure_actual_group_delta_loss,
    sample_cells_across_score_range,
    validate_proxy_scores,
)


def _make_loss_fn(w_target: torch.Tensor):
    """A toy loss: squared distance to a fixed target weight matrix. A
    group that's further from the target contributes more loss, so
    restoring it toward w_original (if w_original is closer to target)
    should reduce loss measurably — giving us a ground truth to correlate
    proxy scores against."""

    def loss_fn(w: torch.Tensor) -> float:
        return (w - w_target).pow(2).mean().item()

    return loss_fn


def test_measure_actual_group_delta_loss_positive_when_restoration_helps():
    torch.manual_seed(0)
    w_original = torch.randn(8, 128)
    w_target = w_original.clone()  # original == target: restoring always helps
    w_quantized = w_original + 5.0  # badly off everywhere

    loss_fn = _make_loss_fn(w_target)
    delta = measure_actual_group_delta_loss(
        w_quantized, w_original, group_size=128, row=0, group=0, loss_fn=loss_fn
    )
    assert delta > 0  # restoring toward the target reduced loss


def test_validate_proxy_scores_recovers_perfect_correlation():
    torch.manual_seed(1)
    w_original = torch.randn(8, 256)
    w_target = w_original.clone()
    # quantization error proportional to a synthetic "damage" tensor so we
    # can build proxy scores that should correlate perfectly with the
    # measured delta-loss.
    damage = torch.rand(8, 256)
    w_quantized = w_original + damage

    group_size = 128
    num_groups = 256 // group_size
    # proxy score = sum of squared damage per group (should predict delta loss)
    scores = damage.pow(2).reshape(8, num_groups, group_size).sum(-1)

    loss_fn = _make_loss_fn(w_target)
    cells = [(r, g) for r in range(8) for g in range(num_groups)]
    rho = validate_proxy_scores(scores, w_quantized, w_original, group_size, loss_fn, cells)
    assert rho > 0.9


def test_sample_cells_across_score_range_spans_full_range():
    scores = torch.arange(100).float().reshape(10, 10)
    cells = sample_cells_across_score_range(scores, n_samples=10, seed=0)
    assert len(cells) == 10
    values = [scores[r, g].item() for r, g in cells]
    assert min(values) < 20  # some low-score cells included
    assert max(values) > 80  # some high-score cells included
