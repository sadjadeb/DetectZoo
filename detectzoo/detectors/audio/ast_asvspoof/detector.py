"""DetectZoo wrapper: AST-ASVspoof — Audio Spectrogram Transformer for spoof detection.

References
----------
Backbone:
    Yuan Gong, Yu-An Chung, James Glass,
    "AST: Audio Spectrogram Transformer", Interspeech 2021.
    https://arxiv.org/abs/2104.01778
    Code:    https://github.com/YuanGongND/ast
    HF base: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

Anti-spoofing fine-tune (default checkpoint used here):
    https://huggingface.co/MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection
    Fine-tuned on the ``LanceaKing/asvspoof2019`` HF dataset for binary
    bonafide-vs-spoof classification (3 epochs, lr 5e-5, batch size 8).

Architecture (MIT/AST)
----------------------
* Input: 128-bin log-mel filterbank, 1024 frames (~10.24 s @ 16 kHz),
  patched into 16x16 patches with 10-frame overlap → ~1212 patches.
* Backbone: DeiT-Base (12 transformer layers, hidden=768, heads=12).
* Head: a 2-class linear classifier on top of the [CLS] / pooled features.
* ~86 M parameters.

This is the first **pure-transformer-on-spectrogram** detector in DetectZoo —
it complements the existing graph-attention (AASIST/RawGAT-ST), residual-CNN
(Res-TSSDNet, RawNet2), and embedding-margin (SAMO) approaches with a
fundamentally different inductive bias.

Score convention
----------------
Higher score ⇒ more likely AI / spoof, matching the rest of DetectZoo.
The class index for "spoof"/"fake" is resolved at load time from
``model.config.id2label`` so the wrapper transparently supports both
``{0: bonafide, 1: spoof}`` and ``{0: spoof, 1: bonafide}`` orderings,
as well as ``real``/``fake`` synonyms used by some sibling checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Default HF Hub model id. The MattyB95 collection has several siblings
# ("Jabberjay") fine-tuned on different spoof datasets — a user can switch
# by passing ``model_name=...`` to the constructor.
_DEFAULT_MODEL_NAME = "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"

# AST always works at 16 kHz. The feature extractor's default
# ``max_length`` of 1024 frames corresponds to ~10.24 s of audio.
_SAMPLE_RATE = 16_000

# Synonyms used across MattyB95's checkpoints + the LanceaKing/asvspoof2019
# dataset card. We map any of these to "AI / spoof" vs "human / bonafide".
_SPOOF_LABEL_SYNONYMS: Tuple[str, ...] = ("spoof", "fake", "ai", "synthetic", "deepfake")
_BONAFIDE_LABEL_SYNONYMS: Tuple[str, ...] = ("bonafide", "real", "human", "genuine")


# ---------------------------------------------------------------------------
# Audio I/O helpers — mirror the conventions used by the other audio
# detectors in DetectZoo (aasist, rawnet2, …).
# ---------------------------------------------------------------------------

def _load_audio_to_numpy(
    path: Union[str, Path], target_sr: int = _SAMPLE_RATE
) -> np.ndarray:
    """Load an audio file -> mono float32 numpy array at ``target_sr``."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        wav = data.astype(np.float32)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1, keepdims=True)
        wav = wav[:, 0]
        if sr != target_sr:
            import torchaudio

            wav_t = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, target_sr
            )
            wav = wav_t.squeeze(0).numpy().astype(np.float32)
        return wav


def _resolve_label_indices(id2label: Dict[int, str]) -> Tuple[int, int]:
    """Map ``model.config.id2label`` to ``(spoof_idx, bonafide_idx)``.

    Falls back to the conventional ``{0: bonafide, 1: spoof}`` ordering if
    the labels don't contain any recognised synonyms (with a warning).
    """
    spoof_idx: Optional[int] = None
    bonafide_idx: Optional[int] = None
    for idx, name in id2label.items():
        norm = str(name).strip().lower()
        if any(syn in norm for syn in _SPOOF_LABEL_SYNONYMS):
            spoof_idx = int(idx)
        elif any(syn in norm for syn in _BONAFIDE_LABEL_SYNONYMS):
            bonafide_idx = int(idx)

    if spoof_idx is not None and bonafide_idx is not None:
        return spoof_idx, bonafide_idx

    # Fallback: assume the canonical {bonafide=0, spoof=1} layout used by
    # MattyB95's ASVspoof checkpoints.
    _LOGGER.warning(
        "AST-ASVspoof: could not interpret id2label=%s — falling back to "
        "the conventional {bonafide: 0, spoof: 1} ordering. Pass "
        "`spoof_label_index` / `bonafide_label_index` if this is wrong.",
        id2label,
    )
    return 1, 0


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

@register_detector(
    "ast_asvspoof",
    aliases=["ast", "ast-asvspoof", "ast_audio", "ast_synthetic_voice"],
)
class ASTASVspoofDetector(BaseDetector):
    """Audio Spectrogram Transformer fine-tuned for synthetic-voice detection.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace Hub model id. Defaults to
        ``"MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection"``. Any
        other AST-style audio-classification checkpoint with bonafide/spoof
        (or real/fake) labels also works — for example sibling models from
        the ``MattyB95/jabberjay`` collection trained on ASVspoof 5,
        in-the-wild data, etc.
    checkpoint_path : str or Path, optional
        Path to a *local* directory containing a saved
        ``ASTForAudioClassification`` (i.e. produced by
        ``model.save_pretrained(...)``). Useful for offline evaluation.
        When supplied, ``model_name`` is ignored.
    spoof_label_index, bonafide_label_index : int, optional
        Override the auto-detected class indices. By default they are
        resolved from ``model.config.id2label`` using a small synonym
        table; pass these only if the labels in your checkpoint are
        non-standard.
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
        ``cache_dir`` is forwarded to ``transformers.from_pretrained`` as
        ``cache_dir`` so the HF Hub download lands inside DetectZoo's
        cache tree (``<cache_dir>/ast_asvspoof/...``).

    Notes
    -----
    The HF AST feature extractor handles all preprocessing internally:
    log-mel filterbank (128 bins) + patching + normalisation. The model
    expects mono waveforms sampled at 16 kHz; arbitrary input audio is
    resampled and padded/trimmed to the configured ``max_length`` (default
    1024 frames ≈ 10.24 s) automatically.

    No Google-Drive download path is involved — the weights live entirely
    on HuggingFace Hub, which provides proper caching, retries and
    rate-limit handling out of the box.

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("ast_asvspoof")
    >>> res = det.predict("sample.wav")
    >>> print(res.label, res.score)
    """

    modality = "audio"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        spoof_label_index: Optional[int] = None,
        bonafide_label_index: Optional[int] = None,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self.model_name = model_name
        self._cache_root = get_cache_dir("ast_asvspoof", cache_dir)

        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as exc:
            raise ImportError(
                "AST-ASVspoof requires `transformers`. Install with:\n"
                "  pip install detectzoo[ast_asvspoof]\n"
                "or:\n"
                "  pip install 'transformers>=4.30' torchaudio soundfile"
            ) from exc

        source: Union[str, Path]
        if checkpoint_path is not None:
            source = Path(checkpoint_path).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(
                    f"checkpoint_path does not exist: {source}"
                )
            _LOGGER.info("Loading AST-ASVspoof from local directory %s", source)
            source = str(source)
        else:
            _LOGGER.info(
                "Loading AST-ASVspoof from HuggingFace Hub: %s (cache=%s)",
                model_name,
                self._cache_root,
            )
            source = model_name

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            source, cache_dir=str(self._cache_root)
        )
        # ``low_cpu_mem_usage=True`` (when ``accelerate`` is available) streams
        # weights into the model layer-by-layer instead of allocating a full
        # CPU copy of the state dict up-front. On Windows hosts with a small
        # paging file this avoids ``OSError 1455`` (commit-charge exhaustion);
        # on Linux/macOS it's a strict win for peak memory usage.
        try:
            self._model = AutoModelForAudioClassification.from_pretrained(
                source,
                cache_dir=str(self._cache_root),
                low_cpu_mem_usage=True,
            )
        except (ImportError, ValueError):
            # ``accelerate`` not installed — fall back to the default loader.
            _LOGGER.info(
                "Loading AST-ASVspoof without `low_cpu_mem_usage` "
                "(install `accelerate>=0.20` to reduce peak RAM usage)."
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                source, cache_dir=str(self._cache_root)
            )
        self._model.to(self._device).eval()

        # Resolve which class index corresponds to "spoof"/"fake" (= AI).
        id2label = dict(self._model.config.id2label)
        auto_spoof, auto_bona = _resolve_label_indices(id2label)
        self._spoof_idx = (
            auto_spoof if spoof_label_index is None else int(spoof_label_index)
        )
        self._bonafide_idx = (
            auto_bona if bonafide_label_index is None else int(bonafide_label_index)
        )
        if self._spoof_idx == self._bonafide_idx:
            raise ValueError(
                "spoof_label_index and bonafide_label_index must differ; "
                f"both are {self._spoof_idx}. Check id2label={id2label!r}."
            )
        _LOGGER.debug(
            "AST-ASVspoof labels resolved: spoof=%d, bonafide=%d (id2label=%s)",
            self._spoof_idx,
            self._bonafide_idx,
            id2label,
        )

        # Sampling rate the feature extractor was configured with; AST uses
        # 16 kHz by default but we read it from the config to be safe.
        self._sample_rate = int(getattr(self._feature_extractor, "sampling_rate", _SAMPLE_RATE))

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    def _normalize_input(self, input_data: Any) -> np.ndarray:
        """Accept path / numpy / tensor → mono float32 numpy at 16 kHz."""
        if isinstance(input_data, np.ndarray):
            wav = input_data.astype(np.float32)
            if wav.ndim == 2:
                # (channels, time) → mono
                wav = wav.mean(axis=0) if wav.shape[0] < wav.shape[1] else wav.mean(axis=1)
            return wav
        if isinstance(input_data, torch.Tensor):
            wav = input_data.detach().to(torch.float32).cpu()
            if wav.dim() == 2:
                wav = wav.mean(dim=0) if wav.shape[0] < wav.shape[1] else wav.mean(dim=1)
            return wav.numpy().astype(np.float32)
        return _load_audio_to_numpy(input_data, self._sample_rate)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return the spoof / AI probability for a single audio input.

        Parameters
        ----------
        input_data
            Audio file path, 1-D / 2-D numpy array, or torch tensor.
            Multi-channel inputs are downmixed to mono and resampled to
            16 kHz internally; arbitrary lengths are accepted (the HF AST
            feature extractor pads/truncates to the model's configured
            ``max_length``).
        """
        wav = self._normalize_input(input_data)

        inputs = self._feature_extractor(
            wav,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        logits = self._model(**inputs).logits.view(-1)
        probs = torch.softmax(logits, dim=-1)
        score_ai = float(probs[self._spoof_idx].item())
        score_human = float(probs[self._bonafide_idx].item())

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=score_human,
            logit_spoof=float(logits[self._spoof_idx].item()),
            logit_bonafide=float(logits[self._bonafide_idx].item()),
            model_name=self.model_name,
        )
