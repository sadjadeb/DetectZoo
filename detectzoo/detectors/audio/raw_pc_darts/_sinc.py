"""SincNet front-end for Raw-PC-DARTS (vendored from upstream func/sinc.py)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_0(nn.Module):
    """Convolutional front-end used when ``sinc_scale='conv'``."""

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        padding=2,
        dilation=1,
        bias=False,
        groups=1,
        is_mask=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride, padding, dilation, groups)
        self.channel_number = out_channels
        self.is_mask = is_mask

    def forward(self, x, is_training):
        x = self.conv(x)
        if is_training and self.is_mask:
            v = self.channel_number
            f = int(np.random.uniform(low=0.0, high=16))
            f0 = np.random.randint(0, v - f)
            x[:, f0 : f0 + f, :] = 0
        return x


class SincConv_fast(nn.Module):
    """Sinc-based convolution (mel / linear / inverse-mel scales)."""

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=2,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
        freq_scale="mel",
        is_trainable=False,
        is_mask=False,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                f"SincConv only support one input channel (here, in_channels = {in_channels})"
            )

        self.out_channels = out_channels + 4
        self.kernel_size = kernel_size
        self.is_mask = is_mask

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        if freq_scale == "mel":
            mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
            hz = self.to_hz(mel)
        elif freq_scale == "lem":
            mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
            hz = self.to_hz(mel)
            hz = np.abs(np.flip(hz) - 1)
        elif freq_scale == "linear":
            hz = np.linspace(low_hz, high_hz, self.out_channels + 1)
        else:
            raise ValueError(f"Unknown freq_scale: {freq_scale!r}")

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1), requires_grad=is_trainable)
        self.band_hz_ = nn.Parameter(
            torch.Tensor(np.diff(hz)).view(-1, 1), requires_grad=is_trainable
        )

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms, is_training=False):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        self.filters = self.filters[: self.out_channels - 4, :, :]

        if is_training and self.is_mask:
            v = self.filters.shape[0]
            f = int(np.random.uniform(low=0.0, high=16))
            f0 = np.random.randint(0, v - f)
            self.filters[f0 : f0 + f, :, :] = 0

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
