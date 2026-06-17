"""RawGAT-ST — End-to-end spectro-temporal Graph Attention Network anti-spoofing.

Reference:
    Tak et al., "End-to-End Spectro-Temporal Graph Attention Networks for Speaker
    Verification Anti-Spoofing and Speech Deepfake Detection", ASVspoof 2021
    Workshop.
    https://arxiv.org/abs/2107.12710

GitHub:  https://github.com/eurecom-asp/RawGAT-ST-antispoofing
Weights: https://github.com/eurecom-asp/RawGAT-ST-antispoofing/raw/main/
         Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth   (mul fusion, default)
         Pre_trained_models/RawGAT_ST_add/Best_epoch.pth   (add fusion)

Key idea:
    A learned sinc front-end is followed by a 2D residual encoder that turns
    the raw waveform into a spectro-temporal feature map. Two parallel Graph
    Attention (GAT) branches model spectral and temporal relations; their
    outputs are fused element-wise (``mul`` or ``add``) and passed through a
    third GAT, a top-k graph pool, and a small MLP classifier
    (2 logits → bonafide vs spoof). ~0.44 M parameters.

Architecture and module names are kept identical to the upstream ``model.py``
so the official pretrained checkpoints load without key renaming.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir

# ---------------------------------------------------------------------------
# Constants — from upstream ``model_config_RawGAT_ST.yaml``
# ---------------------------------------------------------------------------
_CKPT_URLS = {
    "mul": (
        "https://github.com/eurecom-asp/RawGAT-ST-antispoofing/raw/main/"
        "Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth"
    ),
    "add": (
        "https://github.com/eurecom-asp/RawGAT-ST-antispoofing/raw/main/"
        "Pre_trained_models/RawGAT_ST_add/Best_epoch.pth"
    ),
}
_CKPT_NAMES = {
    "mul": "RawGAT_ST_mul.pth",
    "add": "RawGAT_ST_add.pth",
}

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_600  # ``nb_samp`` in the upstream config (~4 s)

_OUT_CHANNELS = 70  # sinc filters
_FIRST_CONV = 128  # sinc kernel size (becomes 129 after odd-fix)
_FILTS = [32, [32, 32], [32, 64], [64, 64]]


# ---------------------------------------------------------------------------
# Sinc-conv front-end (upstream ``CONV``) — filters recomputed on the fly;
# there are no learnable parameters here, so nothing to load from ckpt.
# ---------------------------------------------------------------------------
class _SincConv(nn.Module):
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
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        nfft = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = self._to_mel(f)
        mel_lo, mel_hi = fmel.min(), fmel.max()
        mel_edges = np.linspace(mel_lo, mel_hi, out_channels + 1)
        self.mel = self._to_hz(mel_edges)
        self.register_buffer(
            "hsupp",
            torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        hsupp = self.hsupp.to(device)
        sr = self.sample_rate
        filters = torch.zeros(self.out_channels, self.kernel_size, device=device)
        hamming = torch.hamming_window(self.kernel_size, periodic=False, device=device)
        for i in range(len(self.mel) - 1):
            fmin = float(self.mel[i])
            fmax = float(self.mel[i + 1])
            h_high = (2 * fmax / sr) * torch.sinc(2 * fmax * hsupp / sr)
            h_low = (2 * fmin / sr) * torch.sinc(2 * fmin * hsupp / sr)
            filters[i] = hamming * (h_high - h_low)
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=0, bias=None)


# ---------------------------------------------------------------------------
# 2D residual block (upstream ``Residual_block``) — same layer names for
# checkpoint key compatibility: conv1, conv_1, bn1/bn2, conv2, conv_downsample.
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(
            nb_filts[0],
            nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.conv_1 = nn.Conv2d(
            1,
            nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(
            nb_filts[1],
            nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                nb_filts[0],
                nb_filts[1],
                kernel_size=(1, 3),
                padding=(0, 1),
                stride=1,
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.first:
            out = self.conv_1(x)
        else:
            out = self.selu(self.bn1(x))
            out = self.conv1(x)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


# ---------------------------------------------------------------------------
# Graph attention layer (upstream ``GraphAttentionLayer``) — keys: att_proj,
# att_weight, proj_with_att, proj_without_att, bn.
# ---------------------------------------------------------------------------
class _GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)

    def _pairwise_mul(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        a = x.unsqueeze(2).expand(-1, -1, n, -1)
        b = a.transpose(1, 2)
        return a * b

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        a = self._pairwise_mul(x)
        a = torch.tanh(self.att_proj(a))
        a = torch.matmul(a, self.att_weight)
        return F.softmax(a, dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        att = self._attention(x)
        out = self.proj_with_att(torch.matmul(att.squeeze(-1), x))
        out = out + self.proj_without_att(x)
        size = out.size()
        out = self.bn(out.reshape(-1, size[-1])).reshape(size)
        return self.act(out)


# ---------------------------------------------------------------------------
# Top-k graph pool (upstream ``Pool``) — key: proj (+ no-op drop).
# ---------------------------------------------------------------------------
class _GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: float) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return ``(B, k, 1, D)`` — the extra singleton dim matches upstream
        ``Pool``'s indexing so ``transpose(1, 3)`` in the outer net works."""
        z = self.drop(h)
        scores = self.sigmoid(self.proj(z))  # (B, N, 1)
        num_nodes = h.size(1)
        k = max(2, int(self.k * num_nodes))
        _, idx = torch.topk(scores, k, dim=1)  # (B, k, 1)
        weighted = h * scores  # (B, N, D)
        picked = []
        for i in range(h.size(0)):
            picked.append(weighted[i, idx[i], :])  # (k, 1, D)
        return torch.stack(picked, dim=0)  # (B, k, 1, D)


# ---------------------------------------------------------------------------
# Full RawGAT-ST network (upstream ``RawGAT_ST``) — keys / layer names match
# the official checkpoint exactly.
# ---------------------------------------------------------------------------
class _RawGATST(nn.Module):
    """Same module layout as upstream ``RawGAT_ST`` → checkpoint keys match."""

    def __init__(self, fusion: str = "mul") -> None:
        super().__init__()
        if fusion not in {"mul", "add"}:
            raise ValueError(f"fusion must be 'mul' or 'add', got {fusion!r}")
        self._fusion = fusion

        self.conv_time = _SincConv(
            out_channels=_OUT_CHANNELS,
            kernel_size=_FIRST_CONV,
            sample_rate=_SAMPLE_RATE,
        )
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)

        def _mk_encoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Sequential(_ResidualBlock(_FILTS[1], first=True)),
                nn.Sequential(_ResidualBlock(_FILTS[1])),
                nn.Sequential(_ResidualBlock(_FILTS[2])),
                nn.Sequential(_ResidualBlock(_FILTS[3])),
                nn.Sequential(_ResidualBlock(_FILTS[3])),
                nn.Sequential(_ResidualBlock(_FILTS[3])),
            )

        self.encoder1 = _mk_encoder()
        self.encoder2 = _mk_encoder()

        feat_dim = _FILTS[-1][-1]  # 64
        self.GAT_layer1 = _GraphAttentionLayer(feat_dim, 32)
        self.pool1 = _GraphPool(0.64, 32, 0.3)

        self.GAT_layer2 = _GraphAttentionLayer(feat_dim, 32)
        self.pool2 = _GraphPool(0.81, 32, 0.3)

        self.GAT_layer3 = _GraphAttentionLayer(32, 16)
        self.pool3 = _GraphPool(0.64, 16, 0.3)

        self.proj1 = nn.Linear(14, 12)
        self.proj2 = nn.Linear(23, 12)
        self.proj = nn.Linear(16, 1)
        self.proj_node = nn.Linear(7, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, T]`` raw waveform at 16 kHz (T = 64 600)."""
        b, t = x.shape
        x = x.view(b, 1, t)
        x = self.conv_time(x)
        x = x.unsqueeze(1)  # (B, 1, 70, T')
        x = F.max_pool2d(torch.abs(x), (3, 3))  # (B, 1, 23, T'')
        x = self.selu(self.first_bn(x))

        e1 = self.encoder1(x)  # (B, 64, 23, 29)
        s_max, _ = torch.max(torch.abs(e1), dim=3)  # (B, 64, 23)
        g1 = self.GAT_layer1(s_max.transpose(1, 2))  # (B, 23, 32)
        p1 = self.pool1(g1)
        o1 = self.proj1(p1.transpose(1, 3))
        o1 = o1.view(o1.size(0), o1.size(1), o1.size(3))  # (B, 32, 12)

        e2 = self.encoder2(x)
        t_max, _ = torch.max(torch.abs(e2), dim=2)  # (B, 64, 29)
        g2 = self.GAT_layer2(t_max.transpose(1, 2))  # (B, 29, 32)
        p2 = self.pool2(g2)
        o2 = self.proj2(p2.transpose(1, 3))
        o2 = o2.view(o2.size(0), o2.size(1), o2.size(3))  # (B, 32, 12)

        fused = torch.mul(o1, o2) if self._fusion == "mul" else (o1 + o2)
        g3 = self.GAT_layer3(fused.transpose(1, 2))  # (B, 12, 16)
        p3 = self.pool3(g3)
        nodes = self.proj(p3).flatten(1)  # (B, 7)
        return self.proj_node(nodes)  # (B, 2)


# ---------------------------------------------------------------------------
# Audio helpers (shared style with rawnet2.py)
# ---------------------------------------------------------------------------


def _load_audio(path: Union[str, Path], target_sr: int = _SAMPLE_RATE) -> torch.Tensor:
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
    return wav


def _pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    T = wav.shape[-1]
    if T < length:
        wav = wav.repeat(1, math.ceil(length / T))
    return wav[:, :length]


# ---------------------------------------------------------------------------
# DetectZoo detector wrapper
# ---------------------------------------------------------------------------


@register_detector("rawgat_st", aliases=["rawgat", "rawgatst"])
class RawGATSTDetector(BaseDetector):
    """RawGAT-ST audio deepfake detector (Tak et al., ASVspoof 2021 Workshop).

    End-to-end spectro-temporal Graph Attention Network trained on ASVspoof
    2019 LA. Raw waveform in → bonafide / spoof logits out. Reported ~1.06 %
    EER on LA eval (``mul`` fusion variant).

    Parameters
    ----------
    fusion : {"mul", "add"}
        Graph-fusion strategy, selects the matching pretrained checkpoint.
        Default ``"mul"``.
    checkpoint_path : str or Path, optional
        Local ``.pth`` file. When omitted the official weights for the chosen
        ``fusion`` are downloaded and cached automatically.
    threshold : float
        Score threshold for the ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("rawgat_st")         # downloads ~1.86 MB .pth
    >>> result = det.predict("path/to/audio.wav")
    >>> print(result.label, result.score)
    """

    modality = "audio"

    def __init__(
        self,
        fusion: str = "mul",
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        fusion = fusion.strip().lower()
        if fusion not in _CKPT_URLS:
            raise ValueError(f"fusion must be 'mul' or 'add', got {fusion!r}")
        self._fusion = fusion

        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            cache = get_cache_dir("rawgat_st", cache_dir)
            self._weight_path = cache / _CKPT_NAMES[fusion]
            if not self._weight_path.exists():
                download_file(_CKPT_URLS[fusion], self._weight_path)

        self._model = _RawGATST(fusion=fusion)
        self._load_weights()
        self._model.to(self._device).eval()

    def _load_weights(self) -> None:
        state = torch.load(self._weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        if isinstance(input_data, torch.Tensor):
            wav = input_data.float()
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
        elif isinstance(input_data, np.ndarray):
            wav = torch.from_numpy(input_data.astype(np.float32))
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
        else:
            wav = _load_audio(input_data, _SAMPLE_RATE).squeeze(0)
        wav = _pad_or_trim(wav.unsqueeze(0), _MAX_SAMPLES).squeeze(0)
        return wav

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
            score=P(spoof), label='ai'/'human', confidence in [0, 1].
        """
        wav = self._normalize_input(input_data).unsqueeze(0).to(self._device)
        logits = self._model(wav)
        probs = torch.softmax(logits, dim=-1)

        score_ai = float(probs[0, 0])

        return self._make_result(
            score_ai,
            score_bonafide=float(probs[0, 1]),
            score_spoof=float(probs[0, 0]),
            logit_bonafide=float(logits[0, 1]),
            logit_spoof=float(logits[0, 0]),
            fusion=self._fusion,
        )
