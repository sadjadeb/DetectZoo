"""P2SGrad activation layer for Raw-PC-DARTS (vendored from upstream func/p2sgrad.py)."""

from __future__ import annotations

import torch
import torch.nn as torch_nn
from torch.nn import Parameter


class P2SActivationLayer(torch_nn.Module):
    """Cosine activation layer used as the Raw-PC-DARTS classifier head."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        return cos_theta.clamp(-1, 1)
