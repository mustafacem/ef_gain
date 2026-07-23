"""Diagonal-Fisher accumulation for the OBD-style quantization sensitivity
score. The loss used to produce `weight.grad` determines which variant this
becomes: task cross-entropy loss -> S_fisher; KL(p_fp16 || p_model) on
unlabeled calibration text -> the KL-teacher variant (plan-v2 section 4.2,
metric 3). Same accumulator either way — only the caller's loss differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FisherAccumulator:
    """Accumulates E[(dL/dW)^2] for one nn.Linear's weight across calibration
    batches. Call `.accumulate()` right after each `loss.backward()`."""

    def __init__(self, module: nn.Linear, device: str | torch.device = "cpu"):
        self.module = module
        self.accum = torch.zeros(
            module.weight.shape, dtype=torch.float64, device=device
        )
        self.n_batches = 0
        self._device = device

    def accumulate(self) -> None:
        grad = self.module.weight.grad
        if grad is None:
            raise RuntimeError(
                "weight.grad is None — call loss.backward() before accumulate()"
            )
        self.accum += grad.detach().double().to(self._device).pow(2)
        self.n_batches += 1

    def result(self) -> torch.Tensor:
        if self.n_batches == 0:
            raise RuntimeError("no batches accumulated")
        return (self.accum / self.n_batches).float()
