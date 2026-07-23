import os

import torch

from selfquant.patch.apply import (
    apply_patch_layer,
    hash_state_dict,
    load_patch,
    save_patch,
    timed_apply_patch_layer,
)
from selfquant.patch.build import build_patch_layer, patch_layer_bytes, patch_layer_n_weights
from selfquant.patch.compose import compose_masks
from selfquant.sensitivity.scores import top_k_group_mask


def test_build_and_apply_patch_restores_exact_values():
    torch.manual_seed(0)
    w_original = torch.randn(16, 256)
    w_base = torch.zeros(16, 256)  # stand-in for a badly quantized base

    scores = torch.rand(16, 2)
    mask = top_k_group_mask(scores, k_frac=0.25)  # 8 of 32 groups
    patch = build_patch_layer(w_original, mask, group_size=128)

    assert patch_layer_n_weights(patch) == mask.sum().item() * 128
    assert patch_layer_bytes(patch) > 0

    patched = apply_patch_layer(w_base, patch)
    w_grouped_orig = w_original.reshape(16, 2, 128)
    w_grouped_patched = patched.reshape(16, 2, 128)
    rows, groups = patch.indices[:, 0], patch.indices[:, 1]
    assert torch.allclose(
        w_grouped_patched[rows, groups].float(), w_grouped_orig[rows, groups], atol=1e-3
    )
    # unpatched cells should remain the base's (zero) values
    mask_inv = ~mask
    r2, g2 = torch.nonzero(mask_inv, as_tuple=True)
    assert torch.allclose(w_grouped_patched[r2, g2], torch.zeros_like(w_grouped_patched[r2, g2]))


def test_save_and_load_patch_roundtrip(tmp_path):
    torch.manual_seed(1)
    w_original = torch.randn(8, 128)
    mask = top_k_group_mask(torch.rand(8, 1), k_frac=0.5)
    patch = build_patch_layer(w_original, mask, group_size=128)

    path = os.path.join(tmp_path, "patch.safetensors")
    save_patch(
        path,
        {"layer0": patch},
        metadata={"task": "code", "metric": "fisher", "k_frac": "0.5"},
    )
    layers, meta = load_patch(path)

    assert meta["task"] == "code"
    assert "layer0" in layers
    loaded = layers["layer0"]
    assert torch.equal(loaded.indices, patch.indices)
    assert torch.allclose(loaded.values.float(), patch.values.float())


def test_base_hash_detects_mismatch():
    torch.manual_seed(2)
    sd1 = {"w": torch.randn(4, 4)}
    sd2 = {"w": torch.randn(4, 4)}
    assert hash_state_dict(sd1) != hash_state_dict(sd2)
    assert hash_state_dict(sd1) == hash_state_dict(sd1)


def test_timed_apply_returns_positive_latency():
    torch.manual_seed(3)
    w_original = torch.randn(32, 256)
    mask = top_k_group_mask(torch.rand(32, 2), k_frac=0.1)
    patch = build_patch_layer(w_original, mask, group_size=128)
    w_base = torch.zeros(32, 256)
    result, latency = timed_apply_patch_layer(w_base, patch, n_trials=5)
    assert latency >= 0
    assert result.shape == w_base.shape


def test_compose_masks_covers_union_when_budget_allows():
    torch.manual_seed(4)
    scores_a = torch.zeros(10, 10)
    scores_b = torch.zeros(10, 10)
    scores_a[0, 0] = 100.0  # task A cares a lot about this group
    scores_b[5, 5] = 100.0  # task B cares a lot about this other group
    merged = compose_masks([scores_a, scores_b], k_frac_total=0.02)  # 2 of 100
    assert merged[0, 0]
    assert merged[5, 5]


def test_compose_masks_requires_matching_shapes():
    a = torch.rand(4, 4)
    b = torch.rand(4, 5)
    try:
        compose_masks([a, b], k_frac_total=0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass
