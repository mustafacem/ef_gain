import torch

from selfquant.quant.rtn import rtn_quantize


def test_rtn_shapes():
    w = torch.randn(64, 256)
    w_dq, scale, zero = rtn_quantize(w, bits=4, group_size=128)
    assert w_dq.shape == w.shape
    assert scale.shape == (64, 2)
    assert zero.shape == (64, 2)


def test_rtn_error_decreases_with_bits():
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    errs = {}
    for bits in (2, 3, 4, 8):
        w_dq, _, _ = rtn_quantize(w, bits=bits, group_size=128)
        errs[bits] = (w - w_dq).pow(2).mean().item()
    assert errs[2] > errs[3] > errs[4] > errs[8]
    assert errs[8] < 1e-4


def test_rtn_rejects_bad_group_size():
    w = torch.randn(8, 100)
    try:
        rtn_quantize(w, bits=4, group_size=128)
        assert False, "expected ValueError"
    except ValueError:
        pass
