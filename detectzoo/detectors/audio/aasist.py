"""AASIST — Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.

Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks", ICASSP 2022.
    https://arxiv.org/abs/2110.01200

GitHub:  https://github.com/clovaai/aasist
Weights: https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth

The model classes below (``GraphAttentionLayer``, ``HtrgGraphAttentionLayer``,
``GraphPool``, ``CONV`` [SincConv], ``_ResidualBlock``, ``_AASISTModel``) are
vendored from the official reference implementation at
``clovaai/aasist/models/AASIST.py`` (MIT license, Copyright (c) 2021-present
NAVER Corp.). Input is a **raw 4-second 16 kHz waveform** ``[B, T=64600]`` —
the model's fixed (non-learnable) SincConv front-end ``conv_time`` performs the
filterbank step internally. Do NOT pre-compute an STFT.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CKPT_URL = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"
_CKPT_URL_L = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST-L.pth"
_CKPT_NAME = "AASIST.pth"
_CKPT_NAME_L = "AASIST-L.pth"

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_600  # ~4 s at 16 kHz (standard ASVspoof eval length)

# Model configs (from clovaai/aasist/config/AASIST{,-L}.conf)
_D_ARGS_BASE: dict = {
    "nb_samp": 64_600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
_D_ARGS_LIGHT: dict = {
    "nb_samp": 64_600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


# ---------------------------------------------------------------------------
# Graph Attention Layer  (upstream: GraphAttentionLayer)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any) -> None:
        super().__init__()

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)

        self.temp = float(kwargs.get("temperature", 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, N, in_dim) -> (B, N, out_dim)"""
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return self.act(x)

    @staticmethod
    def _pairwise_mul_nodes(x: torch.Tensor) -> torch.Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map(self, x: torch.Tensor) -> torch.Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: torch.Tensor) -> torch.Tensor:
        org_size = x.size()
        x = self.bn(x.view(-1, org_size[-1]))
        return x.view(org_size)

    @staticmethod
    def _init_new_params(*size: int) -> nn.Parameter:
        p = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(p)
        return p


# ---------------------------------------------------------------------------
# Heterogeneous Graph Attention Layer  (upstream: HtrgGraphAttentionLayer)
# ---------------------------------------------------------------------------
class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any) -> None:
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)

        self.temp = float(kwargs.get("temperature", 1.0))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        master: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x1,x2 : (B, N, in_dim). Returns (x1', x2', master)."""
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    @staticmethod
    def _pairwise_mul_nodes(x: torch.Tensor) -> torch.Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map_master(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x: torch.Tensor, num_type1: int, num_type2: int) -> torch.Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )

        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(
        self, x: torch.Tensor, master: torch.Tensor, att_map: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: torch.Tensor) -> torch.Tensor:
        org_size = x.size()
        x = self.bn(x.view(-1, org_size[-1]))
        return x.view(org_size)

    @staticmethod
    def _init_new_params(*size: int) -> nn.Parameter:
        p = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(p)
        return p


# ---------------------------------------------------------------------------
# Graph Pool  (upstream: GraphPool — sigmoid-weighted top-k node selection)
# ---------------------------------------------------------------------------
class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]) -> None:
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        return self._top_k_graph(scores, h, self.k)

    @staticmethod
    def _top_k_graph(scores: torch.Tensor, h: torch.Tensor, k: float) -> torch.Tensor:
        _, n_nodes, n_feat = h.size()
        n_keep = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


# ---------------------------------------------------------------------------
# SincConv  (upstream: CONV — fixed mel-spaced bandpass filterbank; 0 params)
# ---------------------------------------------------------------------------
class CONV(nn.Module):
    @staticmethod
    def _to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16_000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        mask: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"SincConv only supports in_channels=1 (got {in_channels})")
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.out_channels = out_channels
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self._to_mel(f)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        self.mel = self._to_hz(filbandwidthsmel)
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp.numpy() / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp.numpy() / self.sample_rate
            )
            hideal = hHigh - hLow
            band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)
        # Register as a buffer so it follows the module across .to(device)
        # and is excluded from parameters (upstream stores it as a plain
        # tensor attribute, which works on CPU but is fragile on CUDA).
        self.register_buffer("band_pass", band_pass)

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        band_pass_filter = self.band_pass.clone()
        if mask:
            A = int(np.random.uniform(0, 20))
            A0 = random.randint(0, band_pass_filter.shape[0] - A) if A > 0 else 0
            band_pass_filter[A0 : A0 + A, :] = 0
        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


# ---------------------------------------------------------------------------
# Residual block  (upstream: Residual_block)
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, nb_filts: list[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


# ---------------------------------------------------------------------------
# AASIST model  (upstream: Model). Expects RAW waveform [B, T].
# ---------------------------------------------------------------------------
class _AASISTModel(nn.Module):
    def __init__(self, d_args: dict) -> None:
        super().__init__()
        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.conv_time = CONV(
            out_channels=filts[0],
            kernel_size=d_args["first_conv"],
            in_channels=1,
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(_ResidualBlock(nb_filts=filts[1], first=True)),
            nn.Sequential(_ResidualBlock(nb_filts=filts[2])),
            nn.Sequential(_ResidualBlock(nb_filts=filts[3])),
            nn.Sequential(_ResidualBlock(nb_filts=filts[4])),
            nn.Sequential(_ResidualBlock(nb_filts=filts[4])),
            nn.Sequential(_ResidualBlock(nb_filts=filts[4])),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: torch.Tensor, Freq_aug: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """x : (B, T) raw waveform at 16 kHz. Returns (hidden[B,160], logits[B,2])."""
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv_time(x, mask=Freq_aug)  # SincConv -> [B, filts0, T']
        x = x.unsqueeze(dim=1)  # [B, 1, filts0, T']
        x = F.max_pool2d(torch.abs(x), (3, 3))  # [B, 1, 23, T'']
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.encoder(x)  # [B, C, F', T''']

        # ---- spectral / temporal branches -----------------------------------
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max over time
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        e_T, _ = torch.max(torch.abs(e), dim=2)  # max over freq
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # ---- HS-GAL inference 1 --------------------------------------------
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # ---- HS-GAL inference 2 --------------------------------------------
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return last_hidden, output


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _load_audio(path: Union[str, Path], target_sr: int = _SAMPLE_RATE) -> torch.Tensor:
    """Load audio file -> mono float32 [1, T] at target_sr."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(data.T.astype(np.float32))
        if sr != target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav  # [1, T]


def _pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    """Repeat-pad (not zero-pad — matches upstream eval) or trim to length."""
    T = wav.shape[-1]
    if T < length:
        repeats = (length + T - 1) // T
        wav = wav.repeat(1, repeats)
    return wav[:, :length]


# ---------------------------------------------------------------------------
# DetectZoo detector wrapper
# ---------------------------------------------------------------------------


@register_detector("aasist", aliases=["aasist_audio"])
class AASISTDetector(BaseDetector):
    """AASIST audio deepfake detector (Jung et al., ICASSP 2022).

    Parameters
    ----------
    variant : str
        ``"base"`` (AASIST, ~297k params) or ``"light"`` (AASIST-L, ~85k params).
    checkpoint_path : str or Path, optional
        Local ``.pth`` file. Defaults to auto-download from official GitHub.
    threshold : float
        Score threshold for ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Notes
    -----
    The model takes a **raw 4-second 16 kHz waveform** (``[B, 64600]``). The
    SincConv front-end (``conv_time``) is applied internally — do NOT feed a
    spectrogram.
    """

    modality = "audio"

    def __init__(
        self,
        variant: str = "base",
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if variant not in ("base", "light"):
            raise ValueError(f"variant must be 'base' or 'light', got {variant!r}")
        self.variant = variant

        # resolve checkpoint
        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            cache = get_cache_dir("aasist", cache_dir)
            name = _CKPT_NAME if variant == "base" else _CKPT_NAME_L
            url = _CKPT_URL if variant == "base" else _CKPT_URL_L
            self._weight_path = cache / name
            download_file(url, self._weight_path)

        # build model and load weights
        cfg = _D_ARGS_BASE if variant == "base" else _D_ARGS_LIGHT
        self._model = _AASISTModel(cfg)
        self._load_weights()
        self._model.to(self._device).eval()

    def _load_weights(self) -> None:
        state = torch.load(self._weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        # strip DataParallel prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        # SincConv `band_pass` is a buffer we compute deterministically; it's
        # OK if it's missing from the checkpoint (upstream stored it as a
        # plain attribute, not a buffer).
        missing = [m for m in missing if m != "conv_time.band_pass"]
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {missing[:5]} ...")
        if unexpected:
            raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected[:5]} ...")

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        """Accept path / numpy / tensor -> raw waveform tensor [1, T=64600]."""
        if isinstance(input_data, torch.Tensor):
            wav = input_data.float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        elif isinstance(input_data, np.ndarray):
            wav = torch.from_numpy(input_data.astype(np.float32))
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        else:
            wav = _load_audio(input_data, _SAMPLE_RATE)
        return _pad_or_trim(wav, _MAX_SAMPLES)  # [1, T]

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Predict whether audio is AI-generated (spoofed).

        Parameters
        ----------
        input_data
            Audio file path, numpy array, or torch tensor (16 kHz mono).

        Returns
        -------
        DetectionResult
            score=P(ai), label='ai'/'human', confidence in [0,1].
        """
        wav = self._normalize_input(input_data).to(self._device)  # [1, T]
        _, logits = self._model(wav)  # ([1,160], [1,2])
        probs = torch.softmax(logits, dim=-1)  # [1, 2]

        # AASIST training convention: index 0 = spoof/ai, index 1 = bonafide/human
        # (see clovaai/aasist data_utils.py: `d_meta[key] = 1 if label == "bonafide" else 0`,
        #  and main.py inference: `batch_score = batch_out[:, 1]` is the bonafide CM score)
        score_ai = float(probs[0, 0])

        return self._make_result(
            score_ai,
            score_bonafide=float(probs[0, 1]),
            score_spoof=float(probs[0, 0]),
            logit_bonafide=float(logits[0, 1]),
            logit_spoof=float(logits[0, 0]),
            variant=self.variant,
        )
