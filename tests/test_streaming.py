import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from selfquant.quant.streaming import (  # noqa: E402
    block_linears,
    capture_block_inputs,
    iter_blocks,
    stream_quantize,
    streamed_nll,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODULE_TYPES = ("self_attn.o_proj", "mlp.down_proj")


@pytest.fixture
def tiny_model():
    """A 2-block randomly-initialised Qwen2. Fast, offline, and enough to
    prove the streaming path is equivalent to the ordinary forward path.

    Function-scoped deliberately: several tests here mutate weights (that is
    what they are testing), and a shared instance let one test's zeroed
    down_proj silently invalidate another's premise.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m


def test_capture_block_inputs_shape(tiny_model):
    ids = torch.randint(0, 512, (3, 32))
    hidden, kwargs = capture_block_inputs(tiny_model, ids, "cpu")
    assert len(hidden) == 3
    assert hidden[0].shape == (1, 32, 64)
    # block kwargs must carry the rotary embeddings, or manually-driven
    # blocks would silently compute different attention than the real model
    assert "position_embeddings" in kwargs


def test_streamed_nll_matches_standard_forward(tiny_model):
    """The streaming scorer must agree with model(ids, labels=ids)."""
    ids = torch.randint(0, 512, (2, 32))
    streamed = streamed_nll(tiny_model, ids, "cpu")

    ref = []
    with torch.no_grad():
        for i in range(ids.shape[0]):
            x = ids[i : i + 1]
            r = tiny_model(x, labels=x)
            ref.append((r.loss.item() * (x.shape[1] - 1), x.shape[1] - 1))

    for (a, na), (b, nb) in zip(streamed, ref):
        assert na == nb
        assert abs(a - b) / max(abs(b), 1e-6) < 1e-4

    p_stream = math.exp(sum(a for a, _ in streamed) / sum(n for _, n in streamed))
    p_ref = math.exp(sum(a for a, _ in ref) / sum(n for _, n in ref))
    assert abs(p_stream - p_ref) / p_ref < 1e-4


def test_stream_quantize_visits_every_targeted_linear(tiny_model):
    ids = torch.randint(0, 512, (2, 32))
    seen = {}

    def qf(name, w, H):
        seen[name] = (tuple(w.shape), tuple(H.shape))
        return None  # collection-only pass

    stream_quantize(tiny_model, ids, MODULE_TYPES, "cpu", qf, progress=False)
    assert set(seen) == {
        "layer0.self_attn.o_proj",
        "layer0.mlp.down_proj",
        "layer1.self_attn.o_proj",
        "layer1.mlp.down_proj",
    }
    # Hessian must be [in_features, in_features] for each layer
    for name, (wshape, hshape) in seen.items():
        assert hshape == (wshape[1], wshape[1])


def test_stream_quantize_applies_returned_weights(tiny_model):
    """A returned weight must actually land on the module."""
    ids = torch.randint(0, 512, (2, 32))
    before = tiny_model.model.layers[0].mlp.down_proj.weight.data.clone()

    def qf(name, w, H):
        return torch.zeros_like(w) if name == "layer0.mlp.down_proj" else None

    out = stream_quantize(tiny_model, ids, MODULE_TYPES, "cpu", qf, progress=False)
    after = tiny_model.model.layers[0].mlp.down_proj.weight.data
    assert "layer0.mlp.down_proj" in out
    assert after.abs().max().item() == 0.0
    assert not torch.equal(before, after)


def test_hessian_reflects_sequential_propagation(tiny_model):
    """Later blocks must be calibrated on the *quantized* outputs of earlier
    ones -- that is the point of the sequential formulation. Zeroing block 0's
    down_proj must therefore change block 1's Hessian."""
    ids = torch.randint(0, 512, (2, 32))

    def collect(store, zero_first):
        def qf(name, w, H):
            store[name] = H.clone()
            if zero_first and name == "layer0.mlp.down_proj":
                return torch.zeros_like(w)
            return None
        return qf

    from transformers import AutoModelForCausalLM

    # Clone the reference weights: load_state_dict would otherwise hand both
    # models tensors that alias the fixture's, so mutating one contaminates
    # the comparison this test exists to make.
    sd = {k: v.clone() for k, v in tiny_model.state_dict().items()}

    baseline, perturbed = {}, {}
    m1 = AutoModelForCausalLM.from_config(tiny_model.config)
    m1.load_state_dict(sd)
    m1.eval()
    stream_quantize(m1, ids, MODULE_TYPES, "cpu", collect(baseline, False), progress=False)

    m2 = AutoModelForCausalLM.from_config(tiny_model.config)
    m2.load_state_dict(sd)
    m2.eval()
    stream_quantize(m2, ids, MODULE_TYPES, "cpu", collect(perturbed, True), progress=False)

    h1 = baseline["layer1.mlp.down_proj"]
    h2 = perturbed["layer1.mlp.down_proj"]
    assert not torch.allclose(h1, h2), "block 1 Hessian ignored block 0's quantization"
