"""RADAR — Robust AI-Text Detection via Adversarial Learning.

Reference:
    Hu et al., "RADAR: Robust AI-Text Detection via Adversarial
    Learning", NeurIPS 2023.

RADAR is a RoBERTa-large classifier that was trained *adversarially*
alongside a paraphraser, making it robust against paraphrase attacks.
Pre-trained checkpoints are available on HuggingFace under the
``TrustSafeAI/`` namespace (e.g. ``TrustSafeAI/RADAR-Vicuna-7B``).

Note: RADAR's convention is label-0 = AI, label-1 = human (opposite
of :class:`SupervisedDetector`).  This class handles the inversion
automatically.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("radar")
class RADARDetector(BaseTextDetector):
    """RADAR adversarially-robust text detector.

    Parameters:
        model_name: HuggingFace checkpoint (default
            ``"TrustSafeAI/RADAR-Vicuna-7B"``).
        threshold: Decision boundary on the AI probability.
        max_length: Max token length.
        device: ``"cpu"`` or ``"cuda"``.
    """

    modality = "text"

    def __init__(
        self,
        model_name: str = "TrustSafeAI/RADAR-Vicuna-7B",
        threshold: float = 0.5,
        max_length: int = 512,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self._cls_model: torch.nn.Module | None = None
        self._cls_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading RADAR model '%s' …", self.model_name)
        self._cls_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        ).to(self._device)
        self._cls_model.eval()

    @property
    def cls_model(self) -> torch.nn.Module:
        if self._cls_model is None:
            self._load_model()
        return self._cls_model  # type: ignore[return-value]

    @property
    def cls_tokenizer(self):
        if self._cls_tokenizer is None:
            self._load_model()
        return self._cls_tokenizer

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        enc = self.cls_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self._device)

        logits = self.cls_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # RADAR convention: label 0 = AI-generated
        ai_prob = float(probs[0]) if probs.numel() > 1 else float(probs[0])

        return self._make_result(
            ai_prob,
            ai_prob=ai_prob,
            human_prob=1.0 - ai_prob,
        )
