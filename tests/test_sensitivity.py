import torch
import torch.nn as nn
import torch.nn.functional as F

from selfquant.sensitivity.activations import ActivationAbsMean
from selfquant.sensitivity.fisher import FisherAccumulator
from selfquant.sensitivity.scores import score_awq, score_fisher, top_k_group_mask


def test_activation_abs_mean_matches_manual_computation():
    torch.manual_seed(0)
    layer = nn.Linear(64, 16, bias=False)
    stats = ActivationAbsMean(layer)
    batches = [torch.randn(8, 64) for _ in range(5)]
    for b in batches:
        layer(b)
    got = stats.result()
    all_x = torch.cat(batches, dim=0)
    expected = all_x.abs().mean(dim=0)
    assert torch.allclose(got, expected, atol=1e-5)
    stats.remove()


def test_fisher_accumulator_matches_manual_grad_sq():
    torch.manual_seed(0)
    layer = nn.Linear(32, 8, bias=False)
    fisher = FisherAccumulator(layer)
    manual_accum = torch.zeros_like(layer.weight)
    for _ in range(4):
        layer.zero_grad()
        x = torch.randn(16, 32)
        target = torch.randn(16, 8)
        out = layer(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        fisher.accumulate()
        manual_accum += layer.weight.grad.detach() ** 2
    expected = manual_accum / 4
    assert torch.allclose(fisher.result(), expected, atol=1e-5)


def test_score_awq_shape_and_monotonicity():
    w = torch.randn(16, 256)
    act = torch.rand(256) + 0.1
    s = score_awq(w, act, group_size=128)
    assert s.shape == (16, 2)
    # scaling one group's activations up should increase its score
    act2 = act.clone()
    act2[:128] *= 10
    s2 = score_awq(w, act2, group_size=128)
    assert s2[:, 0].mean() > s[:, 0].mean()
    assert torch.allclose(s2[:, 1], s[:, 1])


def test_score_fisher_zero_when_no_quant_error():
    w = torch.randn(8, 128)
    fisher = torch.rand(8, 128)
    s = score_fisher(w, w.clone(), fisher, group_size=128)
    assert torch.allclose(s, torch.zeros_like(s))


def test_top_k_group_mask_selects_correct_fraction():
    torch.manual_seed(1)
    scores = torch.rand(10, 20)  # 200 groups total
    mask = top_k_group_mask(scores, k_frac=0.05)
    assert mask.sum().item() == 10  # 5% of 200
    # the selected groups should indeed be the highest-scoring ones
    selected_scores = scores[mask]
    unselected_scores = scores[~mask]
    assert selected_scores.min() >= unselected_scores.max()
