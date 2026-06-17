"""DetectZoo wrapper: NII Yamagishi-Lab AntiDeepfake (HuBERT-XLarge frontend).

Reference / Checkpoint
----------------------
HuggingFace: ``nii-yamagishilab/hubert-xlarge-anti-deepfake``
    https://huggingface.co/nii-yamagishilab/hubert-xlarge-anti-deepfake
Paper: Ge et al. -- "Post-training for Deepfake Speech Detection"
    https://arxiv.org/abs/2506.21090
GitHub: https://github.com/nii-yamagishilab/AntiDeepfake

Architecture
------------
* Frontend: HuBERT XLarge (48 transformer layers, hidden=1280,
  ~1 B parameters), initialised from
  ``facebook/hubert-xlarge-ll60k`` and post-trained on ~56 000 hours of
  bona-fide speech and ~18 000 hours of fake speech (29 corpora,
  >100 languages).
* Backend: ``AdaptiveAvgPool1d`` over the time axis followed by a single
  fully-connected layer projecting the 1280-dim embedding to a 2-class
  logit vector.
* Sampling rate: 16 kHz mono, arbitrary length.
* Output ordering: ``logits[0]`` is the *fake* score, ``logits[1]`` is
  the *real* score (matches the inference script printed on the model
  card).

Implementation notes
--------------------
The official inference script published on the model card builds the
SSL frontend with ``fairseq`` (and explicitly requires a non-PyPI
fairseq commit -- the version on PyPI does not work for HuBERT). To
avoid that brittle dependency, this wrapper uses
``transformers.AutoModel`` (i.e. ``HubertModel``) for the SSL frontend
and translates the fairseq-style state dict saved on the Hub to the
HF transformers naming on the fly (see
:mod:`detectzoo.detectors.audio._anti_deepfake_common`). The HF config
that ships in the repo lists ``conv_bias=true`` even though the actual
fairseq HuBERT-XLarge feature extractor is bias-free; the wrapper
detects this from the safetensors keys and overrides ``conv_bias=False``
before instantiating the model. Pretraining-only weights
(``final_proj``, ``label_embs_concat``) are dropped at inference.

Reported numbers (model card)
-----------------------------
* In-the-Wild EER          : **5.23 %** (@ threshold 0.0098)
* FakeOrReal EER           : 2.48 %
* DeepVoice EER            : 5.68 %
* ADD2023 EER              : 18.91 %
* Deepfake-Eval-2024 EER   : 34.10 % (zero-shot; can be fine-tuned)

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

# Default HF Hub model id -- HuBERT XLarge + AntiDeepfake post-training.
_DEFAULT_MODEL_NAME = "nii-yamagishilab/hubert-xlarge-anti-deepfake"

# DetectZoo cache subdirectory name (per the user-facing spec).
_CACHE_NAMESPACE = "anti_deepfake_hubert"

# Hidden size of the HuBERT-XLarge encoder (sanity check).
_HIDDEN_SIZE = 1280


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------


@register_detector(
    "anti_deepfake_hubert",
    aliases=["nii_hubert"],
)
class AntiDeepfakeHuBERTDetector(BaseDetector):
    """NII Yamagishi-Lab AntiDeepfake (HuBERT-XLarge) deepfake-audio detector.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace Hub model id. Defaults to
        ``"nii-yamagishilab/hubert-xlarge-anti-deepfake"``. Any other
        repo following the same SSL-frontend + ``proj_fc`` head layout
        also works (e.g. the ``-nda`` variant).
    threshold, device, cache_dir, **kwargs
        Standard :class:`~detectzoo.core.base.BaseDetector` options.
        ``cache_dir`` is forwarded to ``huggingface_hub.hf_hub_download``
        as ``cache_dir`` so the safetensors blob (~3.86 GB) lands inside
        DetectZoo's cache tree (``<cache_dir>/anti_deepfake_hubert/...``).

    Notes
    -----
    The model expects mono 16 kHz waveforms; arbitrary input audio is
    resampled to 16 kHz and downmixed to mono automatically. The only
    preprocessing applied internally is the
    ``F.layer_norm(wav, wav.shape)`` step from the original inference
    script.

    Score convention: ``score`` is the softmax probability that the
    audio is AI-generated (= ``probs[0]`` from the model card script).

    Memory footprint
    ----------------
    HuBERT-XLarge has ~1 B parameters; the safetensors checkpoint is
    ~3.86 GB and peak host RAM during loading is roughly 2x that. A GPU
    with ~6 GB of free memory is recommended for inference; CPU
    inference works but is slow.

    Examples
    --------
    >>> from detectzoo import load_detector
    >>> det = load_detector("anti_deepfake_hubert")
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
            "Loading AntiDeepfake-HuBERT from HuggingFace Hub: %s (cache=%s)",
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
            16 kHz internally; arbitrary lengths are accepted.
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
