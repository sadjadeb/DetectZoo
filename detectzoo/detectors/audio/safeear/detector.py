"""SafeEar: decoupled codec + transformer detector for spoof detection.

Reference:
    Li et al., "SafeEar: Content Privacy-Preserving Audio Deepfake Detection".
    https://github.com/LetterLiGo/SafeEar

Weights
-------
    ``model.ckpt`` and ``SpeechTokenizer.pt`` download automatically on first use
    from HuggingFace ``TEC2004/SafeEar-ASV19-spoof-detection`` (~1.3 GB total).
    Cached under ``<cache_dir>/safeear/``. Override with ``repo_id=`` or local
    ``detect_checkpoint_path`` / ``codec_checkpoint_path``.

Inference path (test time)
--------------------------
    Matches ``SafeEarTrainer.forward`` at ``is_train=False``:

    1. Raw waveform ``[B, 1, T]`` @ 16 kHz (64600 samples, zero-pad / truncate)
    2. Frozen ``SpeechTokenizer`` RVQ layers 0-7 -> acoustic token list
    3. ``SafeEar1s`` -> logits ``[B, 2]``
    4. Score = ``softmax(logits)[:, 0]`` (P(bonafide); labels: bonafide=0, spoof=1)

    DetectZoo exposes ``score = 1 - P(bonafide)`` (higher = more likely AI / spoof).

    HuBERT / fairseq features are **not** used at inference (training-only pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.detectors.audio._anti_deepfake_common import normalize_input
from detectzoo.detectors.audio.safeear._loader import (
    build_safeear_models,
    resolve_weight_paths,
    safeear_forward,
)
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_600
_CACHE_NAMESPACE = "safeear"
_DEFAULT_REPO_ID = "TEC2004/SafeEar-ASV19-spoof-detection"


def _prepare_waveform(wav: np.ndarray, max_len: int) -> np.ndarray:
    """Zero-pad or truncate (upstream ASVspoof19 eval path, ``is_train=False``)."""
    x_len = wav.shape[0]
    if x_len >= max_len:
        return wav[:max_len]
    out = np.zeros(max_len, dtype=np.float32)
    out[:x_len] = wav
    return out


@register_detector("safeear", aliases=["safe_ear", "safeear_asv19", "safeear_audio"])
class SafeEarDetector(BaseDetector):
    """SafeEar audio deepfake / spoofing detector.

    Parameters
    ----------
    repo_id : str, optional
        HuggingFace repo for ``model.ckpt`` and ``SpeechTokenizer.pt`` (default
        ``TEC2004/SafeEar-ASV19-spoof-detection``; auto-download on first use).
    detect_checkpoint_path, codec_checkpoint_path : str or Path, optional
        Local checkpoint overrides (skip auto-download for the given file).
    deterministic : bool, optional
        When True (default), fixes the Torch RNG seed before SafeEar1s block
        shuffling so repeated calls on the same input are reproducible.
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
    """

    modality = "audio"

    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO_ID,
        detect_checkpoint_path: Optional[Union[str, Path]] = None,
        codec_checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        deterministic: bool = True,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self.repo_id = repo_id
        self.deterministic = bool(deterministic)
        self._detect_checkpoint_path = (
            Path(detect_checkpoint_path).expanduser().resolve()
            if detect_checkpoint_path
            else None
        )
        self._codec_checkpoint_path = (
            Path(codec_checkpoint_path).expanduser().resolve()
            if codec_checkpoint_path
            else None
        )
        self._cache_root = get_cache_dir(_CACHE_NAMESPACE, cache_dir)
        self._sample_rate = _SAMPLE_RATE

        self._codec: Optional[torch.nn.Module] = None
        self._detect: Optional[torch.nn.Module] = None
        self._load_report: Optional[Dict[str, Any]] = None

    def _ensure_loaded(self) -> None:
        if self._codec is not None and self._detect is not None:
            return
        detect_path, codec_path = resolve_weight_paths(
            self._cache_root,
            repo_id=self.repo_id,
            detect_checkpoint_path=self._detect_checkpoint_path,
            codec_checkpoint_path=self._codec_checkpoint_path,
        )
        self._codec, self._detect, self._load_report = build_safeear_models(
            detect_path,
            codec_path,
            device=self._device,
        )
        report = self._load_report or {}
        _LOGGER.info(
            "SafeEar load report: codec matched=%s, codec missing=%s; "
            "detect matched=%s, detect missing=%s",
            report.get("codec_matched", "?"),
            report.get("codec_missing", "?"),
            report.get("detect_matched", "?"),
            report.get("detect_missing", "?"),
        )

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        wav = normalize_input(input_data, self._sample_rate)
        wav = _prepare_waveform(wav, _MAX_SAMPLES)
        return torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return the spoof / AI score for a single audio input."""
        self._ensure_loaded()
        assert self._codec is not None and self._detect is not None

        wav = self._normalize_input(input_data).to(self._device)
        seed = 0 if self.deterministic else None
        probs = safeear_forward(
            self._codec,
            self._detect,
            wav,
            seed=seed,
        )
        p_bonafide = float(probs[0, 0].item())
        p_spoof = float(probs[0, 1].item())
        score_ai = float(1.0 - p_bonafide)

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=p_bonafide,
            prob_bonafide=p_bonafide,
            prob_spoof=p_spoof,
            repo_id=self.repo_id,
            deterministic=self.deterministic,
            load_report=self._load_report,
        )

    def unload(self) -> None:
        super().unload()
        self._codec = None
        self._detect = None
        self._load_report = None
