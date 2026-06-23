"""TCM: Temporal-Channel Modeling in Multi-head Self-Attention for spoof detection.

Reference:
    Truong et al., "Temporal-Channel Modeling in Multi-head Self-Attention for
    Synthetic Speech Detection", Interspeech 2024.
    https://github.com/ductuantruong/tcm_add

Weights
-------
Official pretrained checkpoints are **OneDrive only** (see upstream README).
They are **not** hardcoded here. Mirror LA/DF ``.pth`` files to the lab
HuggingFace org and set ``_HF_REPO_IDS`` in ``tcm/_loader.py``, or pass
``checkpoint_path=`` for a local copy.

Score convention
----------------
Upstream ``inference.py`` uses the raw bonafide logit ``out[:, 1]`` and
declares bonafide when ``logit_bonafide > score_threshold`` (default
``-3.73`` on the DF checkpoint). DetectZoo maps this to a continuous
``score`` with higher values meaning more likely AI / spoof:

    score = sigmoid(score_threshold - logit_bonafide)

so ``score_threshold`` acts as the upstream decision boundary and
``threshold`` (default ``0.5``) is applied to ``score`` for the label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.detectors.audio._anti_deepfake_common import normalize_input
from detectzoo.detectors.audio.tcm._loader import build_tcm_model, resolve_checkpoint_path
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 66_800
_CACHE_NAMESPACE = "tcm"

# Upstream ``inference.py`` default (DF EER threshold).
_DEFAULT_SCORE_THRESHOLDS: Dict[str, float] = {
    "LA": -3.73,
    "DF": -3.73,
}


def _pad_waveform(wav: np.ndarray, max_len: int) -> np.ndarray:
    """Repeat-pad or trim to ``max_len`` (upstream ``utils.pad``)."""
    x_len = wav.shape[0]
    if x_len >= max_len:
        return wav[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(wav, num_repeats)[:max_len]
    return padded.astype(np.float32, copy=False)


@register_detector("tcm", aliases=["tcm_add", "tcm_audio", "conformer_tcm"])
class TCMDetector(BaseDetector):
    """TCM (Conformer + XLSR) audio deepfake / spoofing detector.

    Parameters
    ----------
    variant : str
        ``"LA"`` or ``"DF"`` (selects which mirrored checkpoint to download
        once ``_HF_REPO_IDS`` is configured in ``tcm/_loader.py``).
    checkpoint_path : str or Path, optional
        Local ``.pth`` state dict. When supplied, auto-download is skipped.
    score_threshold : float, optional
        Raw bonafide logit threshold from upstream ``inference.py`` (default
        ``-3.73`` for both variants until you set track-specific values).
    emb_size, heads, kernel_size, num_encoders : int, optional
        Architecture hyper-parameters (defaults match upstream ``main.py``).
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
        Weights are lazy-loaded on the first :meth:`predict` call.
    """

    modality = "audio"

    def __init__(
        self,
        variant: str = "LA",
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        score_threshold: Optional[float] = None,
        emb_size: int = 144,
        heads: int = 4,
        kernel_size: int = 31,
        num_encoders: int = 4,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        variant_key = variant.upper()
        if variant_key not in _DEFAULT_SCORE_THRESHOLDS:
            raise ValueError(
                f"variant must be one of {sorted(_DEFAULT_SCORE_THRESHOLDS)}, "
                f"got {variant!r}"
            )
        self.variant = variant_key
        self.score_threshold = (
            float(score_threshold)
            if score_threshold is not None
            else _DEFAULT_SCORE_THRESHOLDS[variant_key]
        )
        self.emb_size = int(emb_size)
        self.heads = int(heads)
        self.kernel_size = int(kernel_size)
        self.num_encoders = int(num_encoders)
        self._checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        )
        self._cache_root = get_cache_dir(_CACHE_NAMESPACE, cache_dir)
        self._sample_rate = _SAMPLE_RATE

        self._model: Optional[torch.nn.Module] = None
        self._load_report: Optional[Dict[str, Any]] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        weight_path = resolve_checkpoint_path(
            self.variant,
            self._cache_root,
            checkpoint_path=self._checkpoint_path,
        )
        self._model, self._load_report = build_tcm_model(
            weight_path,
            device=self._device,
            emb_size=self.emb_size,
            heads=self.heads,
            kernel_size=self.kernel_size,
            num_encoders=self.num_encoders,
        )
        self._model.to(self._device).eval()
        report = self._load_report or {}
        _LOGGER.info(
            "TCM load report (%s): matched=%s, missing=%s, unexpected=%s",
            weight_path.name,
            report.get("matched", "?"),
            report.get("missing", "?"),
            report.get("unexpected", "?"),
        )

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        wav = normalize_input(input_data, self._sample_rate)
        wav = _pad_waveform(wav, _MAX_SAMPLES)
        return torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return the spoof / AI score for a single audio input."""
        self._ensure_loaded()
        assert self._model is not None

        wav = self._normalize_input(input_data).to(self._device, dtype=torch.float32)
        logits, _ = self._model(wav)
        logits = logits.view(-1)

        logit_spoof = float(logits[0].item())
        logit_bonafide = float(logits[1].item())
        score_ai = float(torch.sigmoid(torch.tensor(self.score_threshold - logit_bonafide)).item())
        score_human = float(1.0 - score_ai)

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=score_human,
            logit_spoof=logit_spoof,
            logit_bonafide=logit_bonafide,
            raw_bonafide_logit=logit_bonafide,
            score_threshold=self.score_threshold,
            upstream_bonafide=(logit_bonafide > self.score_threshold),
            variant=self.variant,
            load_report=self._load_report,
        )

    def unload(self) -> None:
        super().unload()
        self._model = None
        self._load_report = None
