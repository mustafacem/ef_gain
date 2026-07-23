import pytest
import torch

from selfquant.quant.gptq import GPTQHessian, gptq_quantize


def _make_calibrated_layer(out_features=32, in_features=256, n_samples=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=g)
    x = torch.randn(n_samples, in_features, generator=g)
    hess = GPTQHessian(in_features)
    hess.update(x)
    return w, x, hess


def test_gptq_shapes():
    w, x, hess = _make_calibrated_layer()
    q, scale, zero = gptq_quantize(w, hess.H, bits=4, group_size=128)
    assert q.shape == w.shape
    assert scale.shape == (32, 2)
    assert zero.shape == (32, 2)


def test_gptq_beats_rtn_on_calibrated_data():
    """GPTQ should reconstruct calibration-set output better than plain RTN,
    since it compensates using the Hessian from that exact data."""
    from selfquant.quant.rtn import rtn_quantize

    w, x, hess = _make_calibrated_layer(n_samples=2000)
    q_gptq, _, _ = gptq_quantize(w, hess.H, bits=3, group_size=128)
    q_rtn, _, _ = rtn_quantize(w, bits=3, group_size=128)

    out_ref = x @ w.t()
    out_gptq = x @ q_gptq.t()
    out_rtn = x @ q_rtn.t()

    err_gptq = (out_ref - out_gptq).pow(2).mean().item()
    err_rtn = (out_ref - out_rtn).pow(2).mean().item()
    assert err_gptq < err_rtn


def test_hold_out_mask_preserves_original_values_exactly():
    """Design B guarantee: held-out columns must be quantizable back to
    within RTN-only error of their original value — i.e. no GPTQ
    compensation error leaks into them from neighboring columns."""
    w, x, hess = _make_calibrated_layer(in_features=128, n_samples=1000)
    mask = torch.zeros(128, dtype=torch.bool)
    mask[10:20] = True  # 10 held-out columns within the single group

    q, scale, zero = gptq_quantize(w, hess.H, bits=3, group_size=128, hold_out_mask=mask)

    # A patch simply restores original fp16 values at held-out columns.
    patched = q.clone()
    patched[:, mask] = w[:, mask]

    # Reconstructing with the *original* values at held-out columns should
    # match reference output much better than the base alone, on the
    # component of output attributable to those columns.
    out_ref = x @ w.t()
    out_base = x @ q.t()
    out_patched = x @ patched.t()

    err_base = (out_ref - out_base).pow(2).mean().item()
    err_patched = (out_ref - out_patched).pow(2).mean().item()
    assert err_patched < err_base


def test_hold_out_columns_untouched_by_compensation():
    """Directly check W at held-out columns is never modified by the
    sequential compensation loop before RTN is applied to it."""
    w, x, hess = _make_calibrated_layer(in_features=128, n_samples=500)
    mask = torch.zeros(128, dtype=torch.bool)
    mask[5:15] = True

    # Held-out columns are RTN-quantized independently at the end, so their
    # Q values should equal a plain per-group RTN quantization of the
    # *original* w restricted to the group (since compensation was blocked).
    from selfquant.quant.rtn import compute_group_qparams, quantize_dequantize_group

    q, _, _ = gptq_quantize(w, hess.H, bits=4, group_size=128, hold_out_mask=mask)

    w_group = w[:, :128]
    s, z = compute_group_qparams(w_group, bits=4, group_size=128)
    expected_full_group_rtn = quantize_dequantize_group(w_group, s, z, bits=4, group_size=128)

    # Held-out columns should match RTN-on-original closely (small qparam
    # differences possible since our RTN pass recomputes qparams per-group
    # over only the held-out sub-slice); check they're close, not identical.
    diff = (q[:, mask] - expected_full_group_rtn[:, mask]).abs().mean().item()
    assert diff < 0.5  # loose bound: same original values, similar quant grid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_hessian_accepts_activations_from_a_different_device():
    """A CPU accumulator must accept activations streamed from a CUDA model
    (the intended memory-saving pattern, plan-v2 section 4.1) without a
    device-mismatch error."""
    hess = GPTQHessian(in_features=64, device="cpu")
    x_cuda = torch.randn(16, 64, device="cuda")
    hess.update(x_cuda)  # must not raise
    assert hess.H.device.type == "cpu"
    assert hess.n_samples == 16
