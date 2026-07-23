"""Apply / swap patches onto a quantized base, and (de)serialize to disk.

Format: one safetensors file per patch. For each patched layer `name` we
store `{name}.indices` (int64 [n,2]) and `{name}.values` (fp16 [n, group]),
plus a metadata dict (base hash, metric, task, k, calib seed, group_size)
that safetensors carries natively as str->str header fields. The base hash
check prevents silently applying a patch built for a different base.
"""
from __future__ import annotations

import hashlib
import time

import torch
from safetensors.torch import load_file, save_file

from selfquant.patch.build import PatchLayer


def hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Cheap fingerprint of a base model's weights, used to bind a patch to
    the exact base it was built against."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        h.update(name.encode())
        h.update(t.detach().cpu().numpy().tobytes()[:4096])  # sample, not full tensor
        h.update(str(tuple(t.shape)).encode())
    return h.hexdigest()[:16]


def save_patch(
    path: str, layers: dict[str, PatchLayer], metadata: dict[str, str]
) -> None:
    tensors = {}
    meta = dict(metadata)
    for name, p in layers.items():
        tensors[f"{name}.indices"] = p.indices
        tensors[f"{name}.values"] = p.values
        meta[f"{name}.group_size"] = str(p.group_size)
        meta[f"{name}.in_features"] = str(p.in_features)
    meta["layer_names"] = ",".join(layers.keys())
    save_file(tensors, path, metadata=meta)


def load_patch(path: str) -> tuple[dict[str, PatchLayer], dict[str, str]]:
    tensors = load_file(path)
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}

    layer_names = meta.get("layer_names", "").split(",") if meta.get("layer_names") else []
    layers = {}
    for name in layer_names:
        if not name:
            continue
        layers[name] = PatchLayer(
            indices=tensors[f"{name}.indices"],
            values=tensors[f"{name}.values"],
            group_size=int(meta[f"{name}.group_size"]),
            in_features=int(meta[f"{name}.in_features"]),
        )
    return layers, meta


def apply_patch_layer(
    w_base: torch.Tensor, patch: PatchLayer, inplace: bool = False
) -> torch.Tensor:
    """Scatter patch values onto w_base, returning the patched tensor."""
    out_features, in_features = w_base.shape
    if in_features != patch.in_features:
        raise ValueError("patch in_features mismatch with base layer")
    num_groups = in_features // patch.group_size
    w = w_base if inplace else w_base.clone()
    w_grouped = w.reshape(out_features, num_groups, patch.group_size)
    rows, groups = patch.indices[:, 0], patch.indices[:, 1]
    w_grouped[rows, groups] = patch.values.to(w.dtype)
    return w_grouped.reshape(out_features, in_features)


def timed_apply_patch_layer(
    w_base: torch.Tensor, patch: PatchLayer, n_trials: int = 20
) -> tuple[torch.Tensor, float]:
    """Applies the patch and returns (result, median_latency_seconds) over
    n_trials in-place scatters on a scratch copy — for reporting swap
    latency honestly (plan-v2 section 3/4.3)."""
    scratch = w_base.clone()
    times = []
    for _ in range(n_trials):
        if scratch.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        apply_patch_layer(scratch, patch, inplace=True)
        if scratch.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return scratch, median
