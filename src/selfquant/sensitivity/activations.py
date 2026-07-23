"""Activation statistics for the AWQ-style saliency score."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivationAbsMean:
    """Accumulates E[|X|] per input channel for one nn.Linear via a forward
    pre-hook, streamed across calibration batches (no need to hold all
    activations in memory at once)."""

    def __init__(self, module: nn.Linear):
        self.module = module
        in_features = module.in_features
        self.sum_abs = torch.zeros(in_features, dtype=torch.float64)
        self.n = 0
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = args[0]
        x = x.reshape(-1, self.sum_abs.shape[0]).detach()
        self.sum_abs += x.abs().sum(dim=0).double().to(self.sum_abs.device)
        self.n += x.shape[0]

    def result(self) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("no calibration batches observed")
        return (self.sum_abs / self.n).float()

    def remove(self) -> None:
        self._handle.remove()
