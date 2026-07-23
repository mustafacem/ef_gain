import torch

from selfquant.analysis.overlap import (
    go_no_go_signal,
    jaccard_sweep,
    jaccard_topk,
    overlap_matrix,
    per_layer_breakdown,
    random_floor,
    spearman_correlation,
    split_half_noise_ceiling,
)


def test_jaccard_identical_scores_is_one():
    s = torch.rand(20, 20)
    assert jaccard_topk(s, s.clone(), k_frac=0.05) == 1.0


def test_jaccard_disjoint_top_regions_is_zero():
    scores_a = torch.zeros(10, 10)
    scores_b = torch.zeros(10, 10)
    scores_a[0, 0] = 1.0
    scores_a[0, 1] = 1.0
    scores_b[5, 5] = 1.0
    scores_b[5, 6] = 1.0
    j = jaccard_topk(scores_a, scores_b, k_frac=0.02)
    assert j == 0.0


def test_jaccard_sweep_returns_all_ks():
    s1, s2 = torch.rand(10, 10), torch.rand(10, 10)
    result = jaccard_sweep(s1, s2, k_fracs=(0.01, 0.05, 0.1))
    assert set(result.keys()) == {0.01, 0.05, 0.1}
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_spearman_correlation_perfect_for_identical():
    s = torch.rand(15, 15)
    assert spearman_correlation(s, s.clone()) > 0.999


def test_spearman_correlation_near_zero_for_independent_random():
    torch.manual_seed(0)
    a = torch.rand(200, 200)
    b = torch.rand(200, 200)
    rho = spearman_correlation(a, b)
    assert abs(rho) < 0.05


def test_random_floor_matches_closed_form():
    # k_frac=0.5 -> floor should equal 0.5/(2-0.5) = 1/3
    assert abs(random_floor(0.5) - (1 / 3)) < 1e-9
    assert random_floor(0.0) == 0.0


def test_split_half_noise_ceiling_high_for_correlated_halves():
    torch.manual_seed(1)
    base = torch.rand(50, 50)
    half_a = base + 0.01 * torch.randn(50, 50)
    half_b = base + 0.01 * torch.randn(50, 50)
    ceiling = split_half_noise_ceiling(half_a, half_b, k_frac=0.1)
    assert ceiling > 0.7


def test_overlap_matrix_symmetric_and_diagonal_one():
    torch.manual_seed(2)
    task_scores = {"code": torch.rand(10, 10), "math": torch.rand(10, 10)}
    m = overlap_matrix(task_scores, k_frac=0.1)
    assert m["code"]["code"] == 1.0
    assert m["math"]["math"] == 1.0
    assert m["code"]["math"] == m["math"]["code"]


def test_go_no_go_signal_thresholds():
    signal_high = go_no_go_signal(within_task_overlaps=[0.8, 0.75], cross_task_overlaps=[0.5, 0.55])
    assert signal_high >= 0.15
    signal_low = go_no_go_signal(within_task_overlaps=[0.6], cross_task_overlaps=[0.58])
    assert signal_low < 0.05


def test_per_layer_breakdown_only_common_layers():
    a = {"layer0": torch.rand(4, 4), "layer1": torch.rand(4, 4)}
    b = {"layer0": torch.rand(4, 4), "layer2": torch.rand(4, 4)}
    result = per_layer_breakdown(a, b, k_frac=0.25)
    assert set(result.keys()) == {"layer0"}
