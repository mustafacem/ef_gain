"""Block-by-block quantization for models that do not fit in VRAM.

A 7B model is ~15.2 GB in bf16 against 4.7 GB of free VRAM, so nothing that
holds the whole model on the GPU can work. One transformer block is ~543 MB,
which fits comfortably -- so the model lives on CPU and blocks are moved to
the GPU one at a time.

The second constraint is subtler and bites harder: a single 7B `down_proj`
Hessian is 18944^2 x 4 B = 1.44 GB, so keeping one per layer would need ~40 GB
of RAM. Hessians must therefore be built, used, and discarded within a block,
never accumulated across the model.

This is the standard GPTQ sequential formulation: propagate calibration
activations block by block, where each block's inputs are the *previous*
block's outputs. Because earlier blocks are already quantized when later ones
are calibrated, each block sees the activations it will actually receive at
inference -- which is more faithful than calibrating every layer against
full-precision activations, not merely a memory workaround.
"""
from __future__ import annotations

from typing import Callable, Iterator

import torch
import torch.nn as nn

from selfquant.quant.gptq import GPTQHessian


def iter_blocks(model) -> Iterator[tuple[int, nn.Module]]:
    """Yield (index, transformer block) for a Qwen/Llama-style model."""
    for i, blk in enumerate(model.model.layers):
        yield i, blk


def block_linears(block: nn.Module, module_types: tuple[str, ...]) -> dict[str, nn.Linear]:
    out = {}
    for mt in module_types:
        m = block
        for part in mt.split("."):
            m = getattr(m, part)
        out[mt] = m
    return out


@torch.no_grad()
def capture_block_inputs(
    model,
    input_ids: torch.Tensor,
    device: str | torch.device,
    max_seqs: int | None = None,
) -> tuple[list[torch.Tensor], dict]:
    """Run the embedding layer and capture what block 0 receives.

    Raises a sentinel exception to stop the forward pass at block 0 rather
    than running the whole model -- the point is to avoid touching layers we
    cannot afford to have resident.

    Returns (list of per-sequence hidden states on CPU, kwargs for the block).
    """
    inputs: list[torch.Tensor] = []
    captured_kwargs: dict = {}

    class _Stop(Exception):
        pass

    first = model.model.layers[0]

    def hook(module, args, kwargs):
        inputs.append(args[0].detach().cpu() if args else kwargs["hidden_states"].detach().cpu())
        # position_embeddings / attention_mask travel with the block and must
        # be replayed exactly when we drive blocks manually below.
        #
        # The KV cache must NOT be: it is a live, mutating object. Replaying
        # blocks with it would append every call's keys and values, so a
        # second sequence would attend to the first one's, and the Hessian
        # pass would contaminate the propagation pass. Drop it and disable
        # caching -- neither calibration nor scoring wants it.
        drop = {"hidden_states", "past_key_values", "past_key_value", "use_cache"}
        captured_kwargs.update({k: v for k, v in kwargs.items() if k not in drop})
        captured_kwargs["use_cache"] = False
        raise _Stop

    h = first.register_forward_pre_hook(hook, with_kwargs=True)
    n = input_ids.shape[0] if max_seqs is None else min(max_seqs, input_ids.shape[0])
    emb = model.model.embed_tokens
    was = next(emb.parameters()).device
    emb.to(device)
    try:
        for i in range(n):
            try:
                model(input_ids[i : i + 1].to(device))
            except _Stop:
                pass
    finally:
        h.remove()
        emb.to(was)
    return inputs, captured_kwargs


@torch.no_grad()
def run_block(
    block: nn.Module,
    hidden_states: list[torch.Tensor],
    block_kwargs: dict,
    device: str | torch.device,
) -> list[torch.Tensor]:
    """Push activations through one block, returning its outputs on CPU."""
    outs = []
    for hs in hidden_states:
        kw = {
            k: (tuple(x.to(device) for x in v) if isinstance(v, tuple)
                else v.to(device) if torch.is_tensor(v) else v)
            for k, v in block_kwargs.items()
        }
        out = block(hs.to(device), **kw)
        out = out[0] if isinstance(out, tuple) else out
        outs.append(out.detach().cpu())
        del out
    return outs


def stream_quantize(
    model,
    calib_ids: torch.Tensor,
    module_types: tuple[str, ...],
    device: str | torch.device,
    quantize_fn: Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor],
    hessian_device: str = "cpu",
    progress: bool = True,
) -> dict[str, torch.Tensor]:
    """Sequentially quantize every targeted linear, one block at a time.

    quantize_fn(name, weight_fp32_cpu, hessian) -> new weight, called once per
    targeted linear with that layer's Hessian. Returning the weight unchanged
    makes this a pure Hessian-collection pass.

    Only one block and one block's Hessians are resident at a time, so peak
    memory is set by the largest block rather than by the model.
    """
    model.eval()
    hidden, block_kwargs = capture_block_inputs(model, calib_ids, device)
    if progress:
        mb = sum(h.numel() * h.element_size() for h in hidden) / 1e6
        print(f"  captured {len(hidden)} sequences of block input ({mb:.0f} MB on CPU)")

    new_weights: dict[str, torch.Tensor] = {}

    for idx, block in iter_blocks(model):
        block.to(device)
        lins = block_linears(block, module_types)
        hess = {
            mt: GPTQHessian(lin.in_features, device=hessian_device)
            for mt, lin in lins.items()
        }

        handles = []
        for mt, lin in lins.items():
            def mk(h, inf):
                def hook(mod, args):
                    h.update(args[0].reshape(-1, inf).detach())
                return hook
            handles.append(lin.register_forward_pre_hook(mk(hess[mt], lin.in_features)))

        # One pass to build this block's Hessians ...
        run_block(block, hidden, block_kwargs, device)
        for h in handles:
            h.remove()

        # ... then quantize, so the outputs we propagate are post-quantization.
        for mt, lin in lins.items():
            name = f"layer{idx}.{mt}"
            w = lin.weight.data.float().cpu()
            new_w = quantize_fn(name, w, hess[mt].H)
            if new_w is not None:
                lin.weight.data = new_w.to(lin.weight.dtype).to(lin.weight.device)
                new_weights[name] = new_w
            del w
        del hess

        hidden = run_block(block, hidden, block_kwargs, device)
        block.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        if progress:
            print(f"  block {idx} done", flush=True)

    return new_weights


@torch.no_grad()
def streamed_nll(
    model,
    seqs: torch.Tensor,
    device: str | torch.device,
    logit_chunk: int = 256,
) -> list[tuple[float, int]]:
    """Per-sequence NLL with only one block resident at a time.

    Mirrors the chunked-logit trick used elsewhere (a 7B vocab x 2048 logit
    tensor is ~1.2 GB), but also streams the blocks, so a model far larger
    than VRAM can still be scored.
    """
    hidden, block_kwargs = capture_block_inputs(model, seqs, device)

    for _, block in iter_blocks(model):
        block.to(device)
        hidden = run_block(block, hidden, block_kwargs, device)
        block.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    norm, head = model.model.norm, model.lm_head
    norm.to(device)
    head.to(device)
    out = []
    try:
        for i, hs in enumerate(hidden):
            h = norm(hs.to(device))[0]
            tgt = seqs[i, 1:].to(device)
            src = h[:-1]
            total, n = 0.0, src.shape[0]
            for s in range(0, n, logit_chunk):
                e = min(s + logit_chunk, n)
                lg = head(src[s:e]).float()
                total += torch.nn.functional.cross_entropy(
                    lg, tgt[s:e], reduction="sum"
                ).item()
                del lg
            out.append((total, n))
            del h, src
    finally:
        norm.to("cpu")
        head.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return out
