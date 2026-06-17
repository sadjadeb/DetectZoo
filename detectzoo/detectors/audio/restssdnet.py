"""Res-TSSDNet — End-to-end synthetic speech detection from raw waveforms.

Reference:
    Hua et al., "Towards End-to-End Synthetic Speech Detection", IEEE SPL 2021.
    https://arxiv.org/abs/2106.06341

GitHub:  https://github.com/ghua-ac/end-to-end-synthetic-speech-detection
Weights: https://github.com/ghua-ac/end-to-end-synthetic-speech-detection/
         tree/main/pretrained (Res-TSSDNet, eEER 1.64% on ASVspoof2019 LA eval)

Key idea:
    Res-TSSDNet is a tiny (~0.35M params) 1-D ResNet that operates directly
    on the raw 6-second waveform (96 000 samples @ 16 kHz) and outputs
    two logits (bonafide vs spoof). Pure PyTorch, no SSL front-end, no RNN.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir

# ---------------------------------------------------------------------------
# Constants — taken from official Res-TSSDNet checkpoint / upstream ``models.py``
# ---------------------------------------------------------------------------
_CKPT_FILE = "Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth"
_CKPT_URL = (
    "https://github.com/ghua-ac/end-to-end-synthetic-speech-detection/raw/"
    "main/pretrained/" + quote(_CKPT_FILE)
)
_CKPT_NAME = "Res_TSSDNet_ASVspoof2019_LA.pth"

_SAMPLE_RATE = 16_000
# Fixed-length 6-second crop used by the authors; the final ``max_pool1d`` in
# ``SSDNet1D`` hard-codes ``kernel_size=375`` (= 96_000 / 4**4), so this value
# must match.
_MAX_SAMPLES = 96_000
_FINAL_POOL = 375

_NB_CLASSES = 2


# ---------------------------------------------------------------------------
# ResNet-style 1-D block (upstream ``RSM1D``)
# ---------------------------------------------------------------------------
class _RSM1D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)
        self.nin = nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.conv3(y)
        x = self.nin(x)
        return F.relu(self.bn3(x + y))


# ---------------------------------------------------------------------------
# Full Res-TSSDNet model (upstream ``SSDNet1D``)
# ---------------------------------------------------------------------------
class _ResTSSDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        self.RSM1 = _RSM1D(16, 32)
        self.RSM2 = _RSM1D(32, 64)
        self.RSM3 = _RSM1D(64, 128)
        self.RSM4 = _RSM1D(128, 128)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, _NB_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, 1, T]`` raw waveform, T=96 000 @ 16 kHz."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        x = F.max_pool1d(self.RSM1(x), kernel_size=4)
        x = F.max_pool1d(self.RSM2(x), kernel_size=4)
        x = F.max_pool1d(self.RSM3(x), kernel_size=4)
        x = self.RSM4(x)
        x = F.max_pool1d(x, kernel_size=_FINAL_POOL)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


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


@register_detector("res_tssdnet", aliases=["restssdnet", "tssdnet"])
class ResTSSDNetDetector(BaseDetector):
    """Res-TSSDNet audio deepfake detector (Hua et al., IEEE SPL 2021).

    A lightweight end-to-end anti-spoofing model (~350 K params) that takes a
    6-second raw waveform at 16 kHz and outputs bonafide/spoof logits. The
    pretrained weights in this repo were trained on ASVspoof2019 LA and report
    **eEER = 1.64 %** on the LA eval partition.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Local ``.pth`` file (``{"model_state_dict": ...}`` format). When
        omitted the official pretrained weights are downloaded and cached.
    threshold : float
        Score threshold for the ``"ai"`` label. Default ``0.5``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("res_tssdnet")
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
            cache = get_cache_dir("res_tssdnet", cache_dir)
            self._weight_path = cache / _CKPT_NAME
            if not self._weight_path.exists():
                download_file(_CKPT_URL, self._weight_path)

        self._model = _ResTSSDNet()
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
        wav = self._normalize_input(input_data).to(self._device)
        wav = wav.view(1, 1, -1)
        logits = self._model(wav)
        probs = torch.softmax(logits, dim=-1)

        score_ai = float(probs[0, 1])

        return self._make_result(
            score_ai,
            score_bonafide=float(probs[0, 0]),
            score_spoof=float(probs[0, 1]),
            logit_bonafide=float(logits[0, 0]),
            logit_spoof=float(logits[0, 1]),
        )
