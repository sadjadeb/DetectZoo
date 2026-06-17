"""Raw-PC-DARTS network (vendored from eurecom-asp/raw-pc-darts-anti-spoofing/models/model.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from detectzoo.detectors.audio.raw_pc_darts._genotype import DEFAULT_GENOTYPE, Genotype
from detectzoo.detectors.audio.raw_pc_darts._operations1d import (
    FactorizedReduce,
    Identity,
    OPS,
    ReLUConvBN_half,
    ReLUConvBN_same,
)
from detectzoo.detectors.audio.raw_pc_darts._p2sgrad import P2SActivationLayer
from detectzoo.detectors.audio.raw_pc_darts._sinc import Conv_0, SincConv_fast

DEFAULT_LAYERS = 8
DEFAULT_INIT_CHANNELS = 64
DEFAULT_GRU_HSIZE = 1024
DEFAULT_GRU_LAYERS = 3
DEFAULT_SINC_SCALE = "mel"
DEFAULT_SINC_KERNEL = 128


def _drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.empty(x.size(0), 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        x = x.div(keep_prob)
        x = x.mul(mask)
    return x


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN_half(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN_same(C_prev, C, 1, 1, 0, affine=False)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)
        self.pooling_layer = nn.MaxPool1d(2)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, _index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = _drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = _drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        out = torch.cat([states[i] for i in self._concat], dim=1)
        out = self.pooling_layer(out)
        return out


class RawPCDartsNetwork(nn.Module):
    """Raw-PC-DARTS anti-spoofing model."""

    def __init__(
        self,
        C: int,
        layers: int,
        args: Any,
        num_classes: int,
        genotype: Genotype,
    ):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self.is_mask = args.is_mask
        self.drop_path_prob = 0.0

        if args.sinc_scale == "conv":
            self.sinc = Conv_0(C, kernel_size=args.sinc_kernel, is_mask=args.is_mask)
        else:
            self.sinc = SincConv_fast(
                C,
                kernel_size=args.sinc_kernel,
                freq_scale=args.sinc_scale,
                is_mask=args.is_mask,
                is_trainable=args.is_trainable,
            )

        self.mp = nn.MaxPool1d(3)
        self.bn = nn.BatchNorm1d(C)
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.stem = nn.Sequential(
            nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C),
            nn.LeakyReLU(negative_slope=0.3),
        )

        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.gru = nn.GRU(
            input_size=C_prev,
            hidden_size=args.gru_hsize,
            num_layers=args.gru_layers,
            batch_first=True,
        )
        self.fc_gru = nn.Linear(args.gru_hsize, args.gru_hsize)
        self.l_layer = P2SActivationLayer(args.gru_hsize, out_dim=num_classes)

    def forward(self, input: torch.Tensor):
        input = input.unsqueeze(1)
        s0 = self.sinc(input, self.training)
        s0 = self.mp(s0)
        s0 = self.bn(s0)
        s0 = self.lrelu(s0)
        s1 = self.stem(s0)

        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        v = s1
        v = v.permute(0, 2, 1)
        self.gru.flatten_parameters()
        v, _ = self.gru(v)
        v = v[:, -1, :]
        embeddings = self.fc_gru(v)

        if not self.training:
            return embeddings
        logits = self.l_layer(embeddings)
        return logits, embeddings

    def forward_classifier(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.l_layer(embeddings)


def build_default_args(
    *,
    sinc_scale: str = DEFAULT_SINC_SCALE,
    sinc_kernel: int = DEFAULT_SINC_KERNEL,
    gru_hsize: int = DEFAULT_GRU_HSIZE,
    gru_layers: int = DEFAULT_GRU_LAYERS,
    is_mask: bool = False,
    is_trainable: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        sinc_scale=sinc_scale,
        sinc_kernel=sinc_kernel,
        gru_hsize=gru_hsize,
        gru_layers=gru_layers,
        is_mask=is_mask,
        is_trainable=is_trainable,
    )


def build_raw_pc_darts_network(
    *,
    init_channels: int = DEFAULT_INIT_CHANNELS,
    layers: int = DEFAULT_LAYERS,
    args: Any | None = None,
    genotype: Genotype = DEFAULT_GENOTYPE,
    num_classes: int = 2,
) -> RawPCDartsNetwork:
    model_args = args or build_default_args()
    return RawPCDartsNetwork(init_channels, layers, model_args, num_classes, genotype)
