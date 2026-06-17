"""RawNet2 — End-to-end anti-spoofing from raw waveforms.

Reference:
    Tak et al., "End-to-end anti-spoofing with RawNet2", ICASSP 2021.
    https://arxiv.org/abs/2011.01108

GitHub:  https://github.com/eurecom-asp/rawnet2-antispoofing
         https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2
Weights: https://www.asvspoof.org/asvspoof2021/pre_trained_LA_RawNet2.zip

Key idea:
    RawNet2 operates directly on raw waveforms via a fixed sinc filter
    front-end, followed by 6 residual blocks with filter-wise feature map
    scaling (FMS) attention, a GRU layer, and a linear classifier.
    No hand-crafted features needed.
"""

from __future__ import annotations

import math
import shutil
import zipfile
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
# Constants — dims verified from official pretrained checkpoint
# ---------------------------------------------------------------------------
_CKPT_URL = "https://www.asvspoof.org/asvspoof2021/pre_trained_LA_RawNet2.zip"
_CKPT_NAME = "pre_trained_LA_RawNet2.pth"
_CKPT_ZIP = "pre_trained_LA_RawNet2.zip"

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_600

_NB_SINC_FILTERS = 20
_SINC_FILTER_LEN = 1024
_NB_FILTS = [20, 20, 20, 128, 128, 128, 128]
_GRU_NODE = 1024
_NB_FC_NODE = 1024
_NB_CLASSES = 2


# ---------------------------------------------------------------------------
# SincConv front-end
# Direct port of asvspoof-challenge/2021/LA/Baseline-RawNet2/model.py::SincConv.
# Deterministic Mel-spaced band-pass filterbank, recomputed at every forward.
# Carries NO learnable parameters and NO registered buffers, which is why
# the upstream checkpoint contains no ``Sinc_conv.*`` keys.
# ---------------------------------------------------------------------------
class _SincConv(nn.Module):
    @staticmethod
    def _to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        in_channels: int = 1,
        sample_rate: int = 16_000,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports in_channels=1")

        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd (symmetric filter)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        n_fft = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(n_fft / 2) + 1)
        fmel = self._to_mel(f)
        band_edges_mel = np.linspace(fmel.min(), fmel.max(), out_channels + 1)
        self.mel = self._to_hz(band_edges_mel)
        self.hsupp = torch.arange(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0 + 1)
        self.band_pass = torch.zeros(out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp.numpy() / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp.numpy() / self.sample_rate
            )
            hideal = hHigh - hLow
            self.band_pass[i, :] = torch.from_numpy(
                np.hamming(self.kernel_size).astype(np.float32) * hideal.astype(np.float32)
            )

        filters = self.band_pass.to(x.device).view(self.out_channels, 1, self.kernel_size)
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
# Residual block — faithful port of upstream ``Residual_block``.
# Notable quirks preserved from upstream (part of the trained behaviour):
#   * LeakyReLU(0.3), not SELU.
#   * ``conv1(x)`` receives the ORIGINAL ``x`` rather than the bn1/lrelu-ed
#     tensor (upstream discards the bn1 output before conv1).
#   * All convs use ``bias=True`` (default); the checkpoint stores these.
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, nb_filts: list, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], kernel_size=3, padding=1, stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                nb_filts[0], nb_filts[1], kernel_size=1, padding=0, stride=1
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)  # upstream quirk: uses x, not bn1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


# ---------------------------------------------------------------------------
# Full RawNet2 model — matches d_args used for the ASVspoof 2019 LA / 2021
# pretrained checkpoints (model_config_RawNet.yaml):
#     first_conv=1024, filts=[20, [20,20], [20,128], [128,128]],
#     gru_node=1024, nb_gru_layer=3, nb_fc_node=1024, nb_classes=2
# ---------------------------------------------------------------------------
class _RawNet2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.Sinc_conv = _SincConv(
            out_channels=_NB_SINC_FILTERS,
            kernel_size=_SINC_FILTER_LEN,
            in_channels=1,
            sample_rate=_SAMPLE_RATE,
        )
        self.first_bn = nn.BatchNorm1d(_NB_SINC_FILTERS)
        self.selu = nn.SELU(inplace=True)
        self.sig = nn.Sigmoid()

        # Residual blocks  (channel schedule: 20→20→20→128→128→128→128)
        self.block0 = nn.Sequential(_ResBlock([_NB_FILTS[0], _NB_FILTS[1]], first=True))
        self.block1 = nn.Sequential(_ResBlock([_NB_FILTS[1], _NB_FILTS[2]]))
        self.block2 = nn.Sequential(_ResBlock([_NB_FILTS[2], _NB_FILTS[3]]))
        self.block3 = nn.Sequential(_ResBlock([_NB_FILTS[3], _NB_FILTS[4]]))
        self.block4 = nn.Sequential(_ResBlock([_NB_FILTS[4], _NB_FILTS[5]]))
        self.block5 = nn.Sequential(_ResBlock([_NB_FILTS[5], _NB_FILTS[6]]))

        # Upstream wraps each FMS in nn.Sequential → keys are
        #   ``fc_attention{i}.0.weight/bias`` (note the ``.0.``).
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_attention0 = self._make_attention_fc(_NB_FILTS[1], _NB_FILTS[1])
        self.fc_attention1 = self._make_attention_fc(_NB_FILTS[2], _NB_FILTS[2])
        self.fc_attention2 = self._make_attention_fc(_NB_FILTS[3], _NB_FILTS[3])
        self.fc_attention3 = self._make_attention_fc(_NB_FILTS[4], _NB_FILTS[4])
        self.fc_attention4 = self._make_attention_fc(_NB_FILTS[5], _NB_FILTS[5])
        self.fc_attention5 = self._make_attention_fc(_NB_FILTS[6], _NB_FILTS[6])

        self.bn_before_gru = nn.BatchNorm1d(_NB_FILTS[-1])
        self.gru = nn.GRU(
            input_size=_NB_FILTS[-1],
            hidden_size=_GRU_NODE,
            num_layers=3,  # upstream: nb_gru_layer=3
            batch_first=True,
        )
        self.fc1_gru = nn.Linear(_GRU_NODE, _NB_FC_NODE)
        self.fc2_gru = nn.Linear(_NB_FC_NODE, _NB_CLASSES, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    @staticmethod
    def _make_attention_fc(in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] raw waveform at 16 kHz."""
        B, T = x.shape
        x = x.view(B, 1, T)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)
        x = x0 * y0 + y0

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1)
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)
        x = x1 * y1 + y1

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)
        x = x2 * y2 + y2

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)
        x = x3 * y3 + y3

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)
        x = x4 * y4 + y4

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)
        x = x5 * y5 + y5

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (B, F, T) → (B, T, F)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        return self.logsoftmax(x)


# ---------------------------------------------------------------------------
# Audio helpers
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


@register_detector("rawnet2", aliases=["rawnet2_audio"])
class RawNet2Detector(BaseDetector):
    """RawNet2 audio deepfake detector (Tak et al., ICASSP 2021).

    Detects AI-generated (spoofed) speech directly from raw waveforms using
    a fixed sinc filter front-end, 6 residual blocks with filter-wise feature
    map scaling (FMS) attention, and a GRU back-end classifier.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Local ``.pth`` file. When omitted the official pretrained weights
        are downloaded and cached automatically.
    threshold : float
        Score threshold for ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("rawnet2")
    >>> result = det.predict("path/to/audio.wav")
    >>> print(result.label, result.score)
    """

    modality = "audio"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if checkpoint_path is not None:
            self._weight_path = Path(checkpoint_path).expanduser().resolve()
        else:
            cache = get_cache_dir("rawnet2", cache_dir)
            self._weight_path = cache / _CKPT_NAME
            if not self._weight_path.exists():
                zip_path = cache / _CKPT_ZIP
                download_file(_CKPT_URL, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    pth_names = [n for n in zf.namelist() if n.endswith(".pth")]
                    if not pth_names:
                        raise RuntimeError("No .pth found inside RawNet2 zip")
                    with zf.open(pth_names[0]) as f, open(self._weight_path, "wb") as out:
                        shutil.copyfileobj(f, out, length=1 << 20)

        self._model = _RawNet2Model()
        self._load_weights()
        self._model.to(self._device).eval()

    def _load_weights(self) -> None:
        state = torch.load(self._weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # strict=False — upstream checkpoint has no ``Sinc_conv.*`` keys
        # because ``SincConv`` stores its band-pass filter as a plain tensor
        # (not a buffer) and has no learnable params. Any *other* missing or
        # unexpected key means our architecture has drifted from upstream and
        # will quietly poison EER numbers — so we surface them loudly.
        result = self._model.load_state_dict(state, strict=False)
        unexpected = [k for k in result.unexpected_keys]
        missing = [k for k in result.missing_keys if not k.startswith("Sinc_conv.")]
        if unexpected or missing:
            from detectzoo.utils.logger import get_logger

            get_logger(__name__).warning(
                "RawNet2 checkpoint key mismatch -- EER will be degraded.\n"
                "  missing   : %s\n  unexpected: %s",
                missing,
                unexpected,
            )

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
            score=P(ai), label='ai'/'human', confidence in [0,1].
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
        )
