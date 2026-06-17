"""DetectZoo wrapper: NII Yamagishi-Lab AntiDeepfake (Wav2Vec2-Large frontend).

Reference / Checkpoint
----------------------
HuggingFace: ``nii-yamagishilab/wav2vec-large-anti-deepfake``
    https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake
Paper: Ge et al. -- "Post-training for Deepfake Speech Detection"
    https://arxiv.org/abs/2506.21090
GitHub: https://github.com/nii-yamagishilab/AntiDeepfake

Architecture
------------
* Frontend: Wav2Vec 2.0 Large (24 transformer layers, hidden=1024,
  ~317 M parameters), initialised from
  ``facebook/wav2vec2-large-960h-lv60-self`` and post-trained on
  ~56 000 hours of bona-fide speech and ~18 000 hours of fake speech
  (29 corpora, >100 languages -- see model card).
* Backend: ``AdaptiveAvgPool1d`` over the time axis followed by a single
  fully-connected layer projecting the 1024-dim embedding to a 2-class
  logit vector.
* Sampling rate: 16 kHz mono, arbitrary length.
* Output ordering: ``logits[0]`` is the *fake* score, ``logits[1]`` is
  the *real* score (matches the inference script printed on the model
  card).

Implementation notes
--------------------
The official inference script published on the model card builds the
SSL frontend with ``fairseq``. fairseq is heavy / fragile to install,
so this wrapper re-implements the same forward pass on top of pure
HuggingFace ``transformers``: ``transformers.AutoModel`` is used to
construct a ``Wav2Vec2Model`` from the published config, and the
fairseq-style state dict that ``model.safetensors`` actually contains
is translated to HF transformers naming on the fly (see
:mod:`detectzoo.detectors.audio._anti_deepfake_common`). The 2-D
``proj_fc`` head is loaded directly. Pretraining-only weights
(``final_proj``, ``project_q``, ``quantizer.*``) are dropped since they
are unused at inference.

Reported numbers (model card)
-----------------------------
* In-the-Wild EER          : **1.91 %** (@ threshold 0.3301)
* FakeOrReal EER           : 0.67 %
* DeepVoice EER            : 4.44 %
* ADD2023 EER              : 13.25 %
* Deepfake-Eval-2024 EER   : 33.36 % (zero-shot; can be fine-tuned)

Score convention
----------------
Higher score => more likely AI / spoof, matching the rest of DetectZoo.
The fake / real ordering is fixed by the original inference script
(index 0 = fake, index 1 = real); no ``id2label`` resolution is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.detectors.audio._anti_deepfake_common import (
    SAMPLE_RATE,
    build_anti_deepfake_detector,
    normalize_input,
    run_inference,
)
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Default HF Hub model id -- Wav2Vec2-Large + AntiDeepfake post-training.
_DEFAULT_MODEL_NAME = "nii-yamagishilab/wav2vec-large-anti-deepfake"

# DetectZoo cache subdirectory name (per the user-facing spec).
_CACHE_NAMESPACE = "anti_deepfake_wav2vec"

# Hidden size of the Wav2Vec2-Large encoder (sanity check).
_HIDDEN_SIZE = 1024


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------


@register_detector(
    "anti_deepfake_wav2vec",
    aliases=["anti_deepfake", "nii_wav2vec"],
)
class AntiDeepfakeWav2VecDetector(BaseDetector):
    """NII Yamagishi-Lab AntiDeepfake (Wav2Vec2-Large) deepfake-audio detector.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace Hub model id. Defaults to
        ``"nii-yamagishilab/wav2vec-large-anti-deepfake"``. Any other
        repo following the same SSL-frontend + ``proj_fc`` head layout
        also works (e.g. the ``-nda`` "no data augmentation" variant).
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
        ``cache_dir`` is forwarded to ``huggingface_hub.hf_hub_download``
        as ``cache_dir`` so the safetensors blob (~1.27 GB) lands inside
        DetectZoo's cache tree (``<cache_dir>/anti_deepfake_wav2vec/...``).

    Notes
    -----
    The model expects mono 16 kHz waveforms; arbitrary input audio is
    resampled to 16 kHz and downmixed to mono automatically. No HF
    ``AutoFeatureExtractor`` is used -- the only preprocessing is the
    ``F.layer_norm(wav, wav.shape)`` step from the original inference
    script, which is applied internally.

    Score convention: ``score`` is the softmax probability that the
    audio is AI-generated (= ``probs[0]`` from the model card script).

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("anti_deepfake_wav2vec")
    >>> res = det.predict("sample.wav")
    >>> print(res.label, res.score)
    """

    modality = "audio"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self.model_name = model_name
        self._cache_root = get_cache_dir(_CACHE_NAMESPACE, cache_dir)

        _LOGGER.info(
            "Loading AntiDeepfake-Wav2Vec from HuggingFace Hub: %s (cache=%s)",
            model_name,
            self._cache_root,
        )
        self._model = build_anti_deepfake_detector(
            model_name=model_name,
            cache_dir=self._cache_root,
            expected_hidden_size=_HIDDEN_SIZE,
        )
        self._model.to(self._device).eval()
        self._sample_rate = SAMPLE_RATE

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    def _normalize_input(self, input_data: Any) -> np.ndarray:
        """Accept path / numpy / tensor -> mono float32 numpy at 16 kHz."""
        return normalize_input(input_data, self._sample_rate)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        """Return the spoof / AI probability for a single audio input.

        Parameters
        ----------
        input_data
            Audio file path, 1-D / 2-D numpy array, or torch tensor.
            Multi-channel inputs are downmixed to mono and resampled to
            16 kHz internally; arbitrary lengths are accepted (the SSL
            frontend processes the full waveform end-to-end and a global
            average pool collapses the time axis).
        """
        wav = self._normalize_input(input_data)
        score_ai, score_human, logits = run_inference(self._model, wav, self._device)

        return self._make_result(
            score_ai,
            score_spoof=score_ai,
            score_bonafide=score_human,
            logit_spoof=float(logits[0].item()),
            logit_bonafide=float(logits[1].item()),
            model_name=self.model_name,
        )
