"""1-D DARTS operations for Raw-PC-DARTS (vendored from eurecom-asp/raw-pc-darts-anti-spoofing)."""

from __future__ import annotations

import torch
import torch.nn as nn

OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "max_pool_3": lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    "avg_pool_3": lambda C, stride, affine: nn.AvgPool1d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "std_conv_3": lambda C, stride, affine: StdConv1d(C, C, 3, stride, 1, affine=affine),
    "std_conv_5": lambda C, stride, affine: StdConv1d(C, C, 5, stride, 2, affine=affine),
    "std_conv_7": lambda C, stride, affine: StdConv1d(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3": lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_5": lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "dil_conv_7": lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
}


class ReLUConvBN_half(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ReLUConvBN_same(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class StdConv1d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_in, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
        self.pl = nn.MaxPool1d(2)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.pl(out)
        out = self.bn(out)
        return out
