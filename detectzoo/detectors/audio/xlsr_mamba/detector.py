"""XLSR-Mamba: dual-column bidirectional Mamba on XLSR features for spoof detection.

Reference:
    Xiao & Das, "XLSR-Mamba: A Dual-Column Bidirectional State Space Model for
    Spoofing Attack Detection", arXiv:2411.10027.
    https://github.com/swagshaw/XLSR-Mamba

Weights (HuggingFace Hub):
    ``AustinXiao/XLSR-Mamba-LA``  -- ASVspoof 2021 LA
    ``AustinXiao/XLSR-Mamba-DF``  -- ASVspoof 2021 DF

Score convention
----------------
Upstream training uses ``bonafide=1, spoof=0`` and writes CM scores as
``batch_out[:, 1]`` (bonafide logit) in ``main.py::produce_evaluation_file``.
DetectZoo exposes ``score = softmax(logits)[0]`` (spoof / AI probability).

Dependencies
------------
Uses vendored Mamba blocks when ``mamba-ssm`` is not installed (pure PyTorch,
slower on CPU). Install ``mamba-ssm`` separately after PyTorch for faster
CUDA inference; it is not listed in DetectZoo's ``pyproject.toml``.
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
from detectzoo.detectors.audio.xlsr_mamba._loader import build_xlsr_mamba_model
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_SAMPLE_RATE = 16_000
# Upstream ``Dataset_eval.cut`` for LA/DF (``data_utils.py``).
_MAX_SAMPLES = 66_800

_VARIANTS: Dict[str, str] = {
    "LA": "AustinXiao/XLSR-Mamba-LA",
    "DF": "AustinXiao/XLSR-Mamba-DF",
}
_CACHE_NAMESPACE = "xlsr_mamba"


def _pad_waveform(wav: np.ndarray, max_len: int) -> np.ndarray:
    """Repeat-pad or trim to ``max_len`` (upstream ``utils.pad``)."""
    x_len = wav.shape[0]
    if x_len >= max_len:
        return wav[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(wav, num_repeats)[:max_len]
    return padded.astype(np.float32, copy=False)


@register_detector(
    "xlsr_mamba",
    aliases=["xlsr-mamba", "xlsr_mamba_audio", "xlsr_mamba_la"],
)
class XLSRMambaDetector(BaseDetector):
    """XLSR-Mamba audio deepfake / spoofing detector.

    Parameters
    ----------
    variant : str
        ``"LA"`` (ASVspoof 2021 LA checkpoint) or ``"DF"`` (ASVspoof 2021 DF).
    model_name : str, optional
        Override the HuggingFace Hub repo id. When set, ``variant`` is ignored.
    emb_size, num_encoders : int, optional
        Architecture hyper-parameters (defaults match upstream ``main.py``:
        ``emb_size=144``, ``num_encoders=12``).
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
        Weights are lazy-loaded on the first :meth:`predict` call into
        ``<cache_dir>/xlsr_mamba/`` via ``hf_hub_download``.

    Notes
    -----
    Input is a raw 16 kHz mono waveform, repeat-padded or trimmed to
    66 800 samples (~4.2 s), matching upstream eval preprocessing.
    The XLSR frontend is served through HuggingFace ``Wav2Vec2Model`` with
    fairseq checkpoint keys translated on load (no ``pip install fairseq``).
    """

    modality = "audio"

    def __init__(
        self,
        variant: str = "LA",
        model_name: Optional[str] = None,
        *,
        emb_size: int = 144,
        num_encoders: int = 12,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        variant_key = variant.upper()
        if model_name is None:
            if variant_key not in _VARIANTS:
                raise ValueError(
                    f"variant must be one of {sorted(_VARIANTS)}, got {variant!r}"
                )
            model_name = _VARIANTS[variant_key]
        self.variant = variant_key if model_name in _VARIANTS.values() else variant
        self.model_name = model_name
        self.emb_size = int(emb_size)
        self.num_encoders = int(num_encoders)
        self._cache_root = get_cache_dir(_CACHE_NAMESPACE, cache_dir)
        self._sample_rate = _SAMPLE_RATE

        self._model: Optional[torch.nn.Module] = None
        self._load_report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model, self._load_report = build_xlsr_mamba_model(
            repo_id=self.model_name,
            cache_dir=self._cache_root,
            emb_size=self.emb_size,
            num_encoders=self.num_encoders,
        )
        self._model.to(self._device).eval()
        report = self._load_report or {}
        _LOGGER.info(
            "XLSR-Mamba load report (%s): matched=%s, missing=%s, unexpected=%s",
            self.model_name,
            report.get("matched", "?"),
            report.get("missing", "?"),
            report.get("unexpected", "?"),
        )

    def _normalize_input(self, input_data: Any) -> torch.Tensor:
        """Accept path / numpy / tensor -> raw waveform ``[1, T]``."""
        wav = normalize_input(input_data, self._sample_rate)
        wav = _pad_waveform(wav, _MAX_SAMPLES)
        return torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return the spoof / AI probability for a single audio input."""
        self._ensure_loaded()
        assert self._model is not None

        wav = self._normalize_input(input_data).to(self._device, dtype=torch.float32)
        logits = self._model(wav).view(-1)
        probs = F.softmax(logits, dim=-1)

        # Upstream: class 0 = spoof, class 1 = bonafide (data_utils label map).
        score_ai = float(probs[0].item())
        score_human = float(probs[1].item())

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=score_human,
            logit_spoof=float(logits[0].item()),
            logit_bonafide=float(logits[1].item()),
            variant=self.variant,
            model_name=self.model_name,
            load_report=self._load_report,
        )

    def unload(self) -> None:
        super().unload()
        self._model = None
        self._load_report = None
