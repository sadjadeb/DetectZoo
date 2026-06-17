"""SAMO — Speaker Attractor Multi-center One-class learning for voice anti-spoofing.

Reference:
    Ding, Zhang, Duan, "SAMO: Speaker Attractor Multi-Center One-Class Learning
    for Voice Anti-Spoofing", ICASSP 2023.
    https://arxiv.org/abs/2211.02718

GitHub:  https://github.com/sivannavis/samo
Weights: https://github.com/sivannavis/samo/raw/master/models/samo.pt
         (1.37 MB, ASVspoof 2019 LA EER ~0.88%)

Notes
-----
The authors release ``samo.pt`` as a **pickled full** ``nn.Module`` (not a plain
state-dict) trained on the raw-waveform AASIST backbone from
``sivannavis/samo/samo/aasist/AASIST.py``. To make ``torch.load`` succeed we
register a tiny ``sys.modules['aasist.AASIST']`` shim that exposes the expected
class names (``Model``, ``Residual_block``, ``GraphAttentionLayer``,
``HtrgGraphAttentionLayer``, ``GraphPool``, ``CONV``).

Scoring
-------
The SAMO paper reports **EER ~0.88 %** on ASVspoof 2019 LA using
**speaker-aware similarity scoring** — at test time the model is conditioned
on per-trial speaker attractors built from the LA *ASV enrollment* protocols
(``ASVspoof2019.LA.asv.eval.{female,male}.trn.txt``) and bonafide enrollment
audio. The released ``samo.pt`` ships only the AASIST-style encoder; the
SAMO loss head's ``center`` parameter is **not** in the checkpoint, so this
speaker-aware scoring **cannot be reproduced without enrolment data**.

This wrapper exposes two scoring modes, each a faithful port of upstream
``samo/main.py::test``:

* ``"fc"`` (default) — classification-head scoring. Mirrors upstream
  ``--scoring fc`` on the SAMO-pretrained checkpoint
  (``score = feat_outputs[:, 0]``). Rank-equivalent to ``probs[:, 1]``,
  which we expose as ``score_ai`` to match DetectZoo's convention. **This
  is the only mode that is well-defined for the public ``samo.pt`` without
  external enrollment data; expect ~5 % EER on ASVspoof 2019 LA.**
* ``"samo"`` — multi-center cosine scoring. ``score = max_k cos(embed, c_k)``
  where ``c_k`` are 20 attractor centers. Without :meth:`enroll`, the
  centers default to ``torch.eye(160)[:20]`` (upstream ``--one_hot``
  fallback): this is a *degenerate* baseline whose EER is much worse than
  ``fc`` because the released checkpoint's bonafide subspace is not
  axis-aligned for unseen speakers. Call :meth:`SAMODetector.enroll` with
  per-speaker bonafide audio to install proper attractors and recover the
  paper's protocol (upstream ``--val_sp 1``).

The legacy ``"ocsoftmax"`` mode was removed: the released ``samo.pt`` was
trained with the SAMO loss, not OCSoftmax, so projecting onto a single
arbitrary direction (``feats_n[:, 0]``) has no theoretical justification and
produced noisy / misleading scores.
"""

from __future__ import annotations

import math
import random
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir

_CKPT_URL = "https://raw.githubusercontent.com/sivannavis/samo/main/models/samo.pt"
_CKPT_NAME = "samo.pt"

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_600

_MODEL_CONFIG: dict = {
    "nb_samp": 64_600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


# ---------------------------------------------------------------------------
# Vendored upstream architecture — copied VERBATIM from
#   https://github.com/sivannavis/samo/blob/master/samo/aasist/AASIST.py
# so that pickled ``samo.pt`` instances round-trip cleanly. Do not rename these
# classes — pickle stores their qualified names.
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
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        n = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, n, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map(self, x: Tensor) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org = x.size()
        x = x.view(-1, org[-1])
        x = self.bn(x)
        return x.view(org)

    @staticmethod
    def _init_new_params(*size: int) -> nn.Parameter:
        p = nn.Parameter(torch.empty(*size))
        nn.init.xavier_normal_(p)
        return p


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
        self.temp = kwargs.get("temperature", 1.0)

    def forward(
        self, x1: Tensor, x2: Tensor, master: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n1, n2 = x1.size(1), x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, n1, n2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    @staticmethod
    def _pairwise_mul_nodes(x: Tensor) -> Tensor:
        n = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, n, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM) / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x: Tensor, n1: int, n2: int) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :n1, :n1, :] = torch.matmul(att_map[:, :n1, :n1, :], self.att_weight11)
        att_board[:, n1:, n1:, :] = torch.matmul(att_map[:, n1:, n1:, :], self.att_weight22)
        att_board[:, :n1, n1:, :] = torch.matmul(att_map[:, :n1, n1:, :], self.att_weight12)
        att_board[:, n1:, :n1, :] = torch.matmul(att_map[:, n1:, :n1, :], self.att_weight12)
        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x: Tensor, master: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org = x.size()
        x = x.view(-1, org[-1])
        x = self.bn(x)
        return x.view(org)

    @staticmethod
    def _init_new_params(*size: int) -> nn.Parameter:
        p = nn.Parameter(torch.empty(*size))
        nn.init.xavier_normal_(p)
        return p


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]) -> None:
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: Tensor) -> Tensor:
        z = self.drop(h)
        scores = self.sigmoid(self.proj(z))
        return self._top_k_graph(scores, h, self.k)

    @staticmethod
    def _top_k_graph(scores: Tensor, h: Tensor, k: float) -> Tensor:
        _, n_nodes, n_feat = h.size()
        n_keep = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class CONV(nn.Module):
    """SincNet-style learnable band-pass front-end (Mel-spaced filterbank init)."""

    @staticmethod
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: np.ndarray) -> np.ndarray:
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
            raise ValueError(f"SincConv only supports one input channel (got {in_channels})")
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

        nfft = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = self.to_mel(f)
        bands_mel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        bands_hz = self.to_hz(bands_mel)

        self.mel = bands_hz
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            h_high = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            h_low = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(h_high - h_low)

    def forward(self, x: Tensor, mask: bool = False) -> Tensor:
        bp = self.band_pass.clone().to(x.device)
        if mask:
            a = int(np.random.uniform(0, 20))
            a0 = random.randint(0, bp.shape[0] - a)
            bp[a0 : a0 + a, :] = 0
        self.filters = bp.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class Residual_block(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not self.first:
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

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


class Model(nn.Module):
    """Raw-waveform AASIST backbone used by SAMO (upstream class name ``Model``).

    Forward returns ``(embedding[B, 160], logits[B, 2])``. Index 0 of the logits
    corresponds to the **bonafide** class under SAMO's training convention
    (see :mod:`samo.main` — ``score = feat_outputs[:, 0]`` for SAMO-pretrained).
    """

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
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
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

    def forward(self, x: Tensor, Freq_aug: bool = False) -> Tuple[Tensor, Tensor]:
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.encoder(x)

        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

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

        t_max, _ = torch.max(torch.abs(out_T), dim=1)
        t_avg = torch.mean(out_T, dim=1)
        s_max, _ = torch.max(torch.abs(out_S), dim=1)
        s_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        return last_hidden, self.out_layer(last_hidden)


# ---------------------------------------------------------------------------
# Pickle compatibility shim — samo.pt was saved from module path
# ``aasist.AASIST`` so unpickling requires the classes to resolve there.
# ---------------------------------------------------------------------------
def _register_pickle_shim() -> None:
    """Expose vendored classes at ``aasist.AASIST`` so ``torch.load`` can unpickle."""
    if "aasist.AASIST" in sys.modules:
        return

    pkg = sys.modules.get("aasist") or types.ModuleType("aasist")
    mod = types.ModuleType("aasist.AASIST")
    for cls in (
        CONV,
        Residual_block,
        GraphAttentionLayer,
        HtrgGraphAttentionLayer,
        GraphPool,
        Model,
    ):
        setattr(mod, cls.__name__, cls)
    pkg.AASIST = mod  # type: ignore[attr-defined]
    sys.modules["aasist"] = pkg
    sys.modules["aasist.AASIST"] = mod


# ---------------------------------------------------------------------------
# Audio helpers (mirror other audio detectors for consistency)
# ---------------------------------------------------------------------------


def _load_audio(path: Union[str, Path], target_sr: int = _SAMPLE_RATE) -> torch.Tensor:
    """Load audio → mono float32 [T] at ``target_sr``."""
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
    return wav.squeeze(0)


def _pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    t = wav.shape[-1]
    if t < length:
        wav = wav.repeat(math.ceil(length / t))
    return wav[:length]


# ---------------------------------------------------------------------------
# DetectZoo wrapper
# ---------------------------------------------------------------------------

_ENC_DIM = 160  # SAMO embedding dim (5 * gat_dims[1] = 5 * 32)
_NUM_CENTERS = 20  # upstream default --num_centers


@register_detector("samo")
class SAMODetector(BaseDetector):
    """SAMO audio deepfake detector (Ding et al., ICASSP 2023).

    Uses the raw-waveform AASIST backbone fine-tuned with the SAMO
    one-class loss. The released ``samo.pt`` is a pickled ``nn.Module``
    and is loaded via a lightweight ``aasist.AASIST`` import shim.

    Two scoring modes are supported (see module docstring for the rationale
    behind the default):

    * ``"fc"`` (default) — softmax classification head, ``score_ai =
      probs[:, 1]``. Mirrors upstream ``--scoring fc`` on the SAMO-pretrained
      model (~5 % EER on ASVspoof 2019 LA). **The only mode that is
      well-defined for the public ``samo.pt`` without external enrollment.**
    * ``"samo"`` — max cosine similarity of the 160-d embedding to a set of
      attractor centers. Without :meth:`enroll`, falls back to upstream's
      ``--one_hot`` baseline (degenerate; expect very high EER). Call
      :meth:`enroll` with per-speaker bonafide audio to install proper
      attractors and reproduce the paper's ``--val_sp 1`` protocol
      (~0.88 % EER).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Local ``.pt`` file. Defaults to auto-download from upstream GitHub.
    scoring : str
        One of ``"fc"`` (default) or ``"samo"``.
    threshold : float
        Score threshold for ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str or Path, optional
        Override download cache root.
    asv_path : str or Path, optional
        Root of the ASVspoof 2019 LA dataset. When provided, the detector
        reads the female / male ASV enrolment protocols
        (``ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.{female,male}.trn.txt``),
        groups the referenced bonafide ``.flac`` files (under
        ``ASVspoof2019_LA_eval/flac/``) by speaker, calls :meth:`enroll`
        to install per-speaker attractors, and switches scoring to
        ``"samo"`` — reproducing the paper's ``--val_sp 1`` protocol
        (~0.88 % EER on ASVspoof 2019 LA).
    """

    modality = "audio"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        scoring: str = "fc",
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        asv_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if scoring not in {"samo", "fc"}:
            raise ValueError(
                f"scoring must be one of 'fc', 'samo' — got {scoring!r}. "
                "(The legacy 'ocsoftmax' mode was removed: it is not "
                "applicable to the SAMO-trained release checkpoint.)"
            )
        self._scoring = scoring

        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            cache = get_cache_dir("samo", cache_dir)
            self._weight_path = cache / _CKPT_NAME
            download_file(_CKPT_URL, self._weight_path)

        self._model = self._load_checkpoint()
        self._model.to(self._device).eval()

        # Default SAMO centers: first 20 one-hot basis vectors (upstream
        # --one_hot fallback). This is a degenerate baseline; ``enroll()``
        # replaces it with per-speaker attractors for the paper protocol.
        self._centers = torch.eye(_ENC_DIM)[:_NUM_CENTERS].to(self._device)
        self._enrolled = False
        self._warned_unenrolled_samo = False

        if asv_path is not None:
            speaker_audio = self._collect_asvspoof_enrollment(Path(asv_path))
            self.enroll(speaker_audio)
            self._scoring = "samo"

    @staticmethod
    def _collect_asvspoof_enrollment(
        asv_path: Path,
    ) -> dict:
        """Build ``{speaker_id: [flac_path, ...]}`` from ASVspoof 2019 LA protocols.

        Reads the female and male ``*.asv.eval.*.trn.txt`` enrolment protocols
        under ``ASVspoof2019_LA_asv_protocols/`` and resolves each referenced
        utterance to its bonafide ``.flac`` under ``ASVspoof2019_LA_eval/flac/``.
        """
        asv_path = Path(asv_path).expanduser().resolve()
        protocol_dir = asv_path / "ASVspoof2019_LA_asv_protocols"
        flac_dir = asv_path / "ASVspoof2019_LA_eval" / "flac"

        protocols = [
            protocol_dir / "ASVspoof2019.LA.asv.eval.female.trn.txt",
            protocol_dir / "ASVspoof2019.LA.asv.eval.male.trn.txt",
        ]

        speaker_audio: dict[str, list[Path]] = {}
        for protocol in protocols:
            if not protocol.is_file():
                raise FileNotFoundError(f"ASVspoof enrolment protocol not found: {protocol}")
            with open(protocol, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    spk_id, filename = parts[0], parts[1]
                    # Some protocol variants pack comma-separated filenames
                    # into the second column ("LA_E_xxx,LA_E_yyy,...").
                    for fname in filename.split(","):
                        fname = fname.strip()
                        if not fname:
                            continue
                        flac_path = flac_dir / f"{fname}.flac"
                        speaker_audio.setdefault(spk_id, []).append(flac_path)

        if not speaker_audio:
            raise RuntimeError(f"No enrolment utterances found under {protocol_dir}")
        return speaker_audio

    def _load_checkpoint(self) -> nn.Module:
        """Load ``samo.pt`` — either a state-dict or a pickled full Module."""
        _register_pickle_shim()

        try:
            obj = torch.load(self._weight_path, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(self._weight_path, map_location="cpu", weights_only=False)

        if isinstance(obj, nn.Module):
            return obj

        if isinstance(obj, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
            obj = {k.replace("module.", ""): v for k, v in obj.items()}
            model = Model(_MODEL_CONFIG)
            missing, unexpected = model.load_state_dict(obj, strict=False)
            if missing:
                raise RuntimeError(f"SAMO checkpoint missing keys: {missing[:5]} …")
            return model

        raise RuntimeError(f"Unexpected SAMO checkpoint type: {type(obj).__name__}")

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        """Accept path / numpy / tensor → mono waveform [T] at 16 kHz."""
        if isinstance(input_data, torch.Tensor):
            wav = input_data.float()
            if wav.dim() > 1:
                wav = wav.view(-1)
        elif isinstance(input_data, np.ndarray):
            wav = torch.from_numpy(input_data.astype(np.float32))
            if wav.dim() > 1:
                wav = wav.view(-1)
        else:
            wav = _load_audio(input_data, _SAMPLE_RATE)
        return _pad_or_trim(wav, _MAX_SAMPLES)

    @torch.no_grad()
    def enroll(
        self,
        speaker_audio: dict,
        *,
        merge_with_onehot: bool = False,
    ) -> None:
        """Build speaker attractors from bonafide audio for paper-exact scoring.

        Mirrors upstream ``update_embeds`` (``samo/main.py`` L635): for each
        speaker, average L2-normalised embeddings of their bonafide samples.
        The resulting attractors replace the default one-hot centers and are
        used by ``scoring="samo"`` at inference time (upstream ``val_sp=1``).

        Parameters
        ----------
        speaker_audio : dict[str, list]
            ``{speaker_id: [audio1, audio2, ...]}``. Each entry can be a file
            path, numpy array, or torch tensor (16 kHz mono).
        merge_with_onehot : bool
            If True, keep the original 20 one-hot centers alongside the
            learned attractors (improves coverage on unseen speakers). Default
            False — matches upstream paper setup.
        """
        if not speaker_audio:
            raise ValueError("speaker_audio must contain at least one speaker")

        attractors = []
        for _spk_id, audio_list in speaker_audio.items():
            if not audio_list:
                continue
            embeds = []
            for audio in audio_list:
                wav = self._normalize_input(audio).to(self._device).view(1, -1)
                feats, _ = self._model(wav)
                embeds.append(F.normalize(feats, p=2, dim=1))
            spk_attr = torch.cat(embeds, dim=0).mean(dim=0, keepdim=True)
            attractors.append(F.normalize(spk_attr, p=2, dim=1))

        if not attractors:
            raise ValueError("No audio provided for any speaker")

        centers = torch.cat(attractors, dim=0)
        if merge_with_onehot:
            onehot = torch.eye(_ENC_DIM)[:_NUM_CENTERS].to(self._device)
            centers = torch.cat([centers, onehot], dim=0)
        self._centers = centers
        self._enrolled = True

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
            ``score=P(ai)`` in ``[0, 1]`` (higher = more likely AI),
            label ``'ai'`` / ``'human'``.
        """
        wav = self._normalize_input(input_data).to(self._device)
        wav = wav.view(1, -1)
        feats, logits = self._model(wav)
        probs = torch.softmax(logits, dim=-1)

        if self._scoring == "fc":
            # Upstream: score = feat_outputs[:, 0] for samo-pretrained (bonafide
            # likelihood). We expose the complementary spoof probability
            # directly, which is rank-equivalent for EER.
            score_ai = float(probs[0, 1])
        else:
            # Cosine-similarity scoring (upstream SAMO.forward, attractor=0):
            #   w = F.normalize(centers, dim=1)
            #   scores = F.normalize(feats) @ w.T  ->  [1, num_centers]
            #   bona_sim = max over centers (upstream `maxscores`)
            if not self._enrolled and not self._warned_unenrolled_samo:
                warnings.warn(
                    "SAMODetector(scoring='samo') is using one-hot fallback "
                    "centers (upstream `--one_hot`); EER will be much worse "
                    "than the paper's 0.88%. Either call `enroll(...)` with "
                    "per-speaker bonafide audio to install real attractors, "
                    "or use `scoring='fc'` for the ~5% EER classification "
                    "baseline.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_unenrolled_samo = True

            feats_n = F.normalize(feats, p=2, dim=1)
            w = F.normalize(self._centers, p=2, dim=1)
            sims = feats_n @ w.transpose(0, 1)
            bona_sim = float(sims.max(dim=1).values.item())
            # Map cosine similarity in [-1, 1] to AI score in [0, 1]:
            # higher similarity -> more bonafide -> lower AI score.
            score_ai = float((1.0 - bona_sim) / 2.0)
            score_ai = max(0.0, min(1.0, score_ai))

        return self._make_result(
            score_ai,
            scoring=self._scoring,
            score_bonafide=float(probs[0, 0]),
            score_spoof=float(probs[0, 1]),
            logit_bonafide=float(logits[0, 0]),
            logit_spoof=float(logits[0, 1]),
        )
