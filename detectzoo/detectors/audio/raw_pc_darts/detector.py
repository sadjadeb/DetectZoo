"""Raw-PC-DARTS: raw differentiable architecture search for spoof detection.

Reference:
    Ge et al., "Raw Differentiable Architecture Search for Speech Deepfake and
    Spoofing Detection", ASVspoof 2021 workshop.
    https://github.com/eurecom-asp/raw-pc-darts-anti-spoofing

Weights
-------
    Official ``fix_mel.pth`` (paper mel-scale architecture) is downloaded
    automatically on first use from the EURECOM Nextcloud share linked in the
    upstream README. Cached under ``<cache_dir>/raw_pc_darts/``.

    Optional: set ``repo_id=`` to a HuggingFace mirror, or pass
    ``checkpoint_path=`` for a local copy.

Score convention
----------------
Upstream ``evaluate.py`` writes CM scores as ``output[:, 1]`` (bonafide cosine
from the P2SGrad head). Labels use ``bonafide=1, spoof=0``. DetectZoo maps the
cosine score to a continuous AI probability with higher values meaning spoof:

    score_ai = (1 - bonafide_cos) / 2
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
from detectzoo.detectors.audio.raw_pc_darts._loader import (
    build_raw_pc_darts_model,
    resolve_checkpoint_path,
)
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_SAMPLE_RATE = 16_000
_MAX_SAMPLES = 64_000
_CACHE_NAMESPACE = "raw_pc_darts"


def _pad_waveform(wav: np.ndarray, max_len: int) -> np.ndarray:
    """Repeat-pad or trim to ``max_len`` (upstream ``ASVRawDataset.load_feature``)."""
    x_len = wav.shape[0]
    if x_len >= max_len:
        return wav[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(wav, num_repeats)[:max_len]
    return padded.astype(np.float32, copy=False)


@register_detector(
    "raw_pc_darts",
    aliases=["raw-pc-darts", "raw_pc_darts_la", "rpcd", "pc_darts"],
)
class RawPCDartsDetector(BaseDetector):
    """Raw-PC-DARTS audio deepfake / spoofing detector.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Local ``.pth`` state dict. When omitted, official ``fix_mel.pth`` weights
        are downloaded and cached automatically.
    repo_id : str, optional
        HuggingFace repo id override (optional mirror; default uses Nextcloud).
    layers, init_channels, gru_hsize, gru_layers : int, optional
        Architecture hyper-parameters (defaults match upstream README / evaluate.py).
    sinc_scale, sinc_kernel : str / int, optional
        Sinc front-end settings (default mel scale, kernel 128).
    is_mask, is_trainable : bool, optional
        Sinc layer options (defaults: no masking, fixed filters).
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
    """

    modality = "audio"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        repo_id: Optional[str] = None,
        layers: int = 8,
        init_channels: int = 64,
        gru_hsize: int = 1024,
        gru_layers: int = 3,
        sinc_scale: str = "mel",
        sinc_kernel: int = 128,
        is_mask: bool = False,
        is_trainable: bool = False,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self.layers = int(layers)
        self.init_channels = int(init_channels)
        self.gru_hsize = int(gru_hsize)
        self.gru_layers = int(gru_layers)
        self.sinc_scale = str(sinc_scale)
        self.sinc_kernel = int(sinc_kernel)
        self.is_mask = bool(is_mask)
        self.is_trainable = bool(is_trainable)
        self._checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        )
        self._repo_id = repo_id
        self._cache_root = get_cache_dir(_CACHE_NAMESPACE, cache_dir)
        self._sample_rate = _SAMPLE_RATE

        self._model: Optional[torch.nn.Module] = None
        self._load_report: Optional[Dict[str, Any]] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        weight_path = resolve_checkpoint_path(
            self._cache_root,
            checkpoint_path=self._checkpoint_path,
            repo_id=self._repo_id,
        )
        self._model, self._load_report = build_raw_pc_darts_model(
            weight_path,
            device=self._device,
            init_channels=self.init_channels,
            layers=self.layers,
            gru_hsize=self.gru_hsize,
            gru_layers=self.gru_layers,
            sinc_scale=self.sinc_scale,
            sinc_kernel=self.sinc_kernel,
            is_mask=self.is_mask,
            is_trainable=self.is_trainable,
        )
        report = self._load_report or {}
        print(
            f"Raw-PC-DARTS load report ({weight_path.name}): "
            f"matched={report.get('matched', '?')}, "
            f"missing={report.get('missing', '?')}, "
            f"unexpected={report.get('unexpected', '?')}"
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
        embeddings = self._model(wav)
        logits = self._model.forward_classifier(embeddings)
        logits = logits.view(-1)

        cos_spoof = float(logits[0].item())
        cos_bonafide = float(logits[1].item())
        score_bonafide = float(np.clip((cos_bonafide + 1.0) / 2.0, 0.0, 1.0))
        score_ai = float(1.0 - score_bonafide)

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=score_bonafide,
            cos_spoof=cos_spoof,
            cos_bonafide=cos_bonafide,
            raw_bonafide_cos=cos_bonafide,
            load_report=self._load_report,
        )

    def unload(self) -> None:
        super().unload()
        self._model = None
        self._load_report = None
