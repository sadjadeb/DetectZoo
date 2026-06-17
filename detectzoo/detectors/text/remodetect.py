"""ReMoDetect — Reward Model based AI-text detector.

Reference:
    Lee et al., "ReMoDetect: Reward Models Recognize Aligned LLM's
    Generations", NeurIPS 2024.

Aligned LLMs are optimised to maximise human-preference scores, which
causes them to generate text with higher predicted reward than actual
human-written text.  A reward model can therefore distinguish
LLM-generated text by thresholding on the reward score.

The default model is the official fine-tuned checkpoint
``hyunseoki/ReMoDetect-deberta`` (DeBERTa-v3-Large, ~400 M params),
which applies continual preference fine-tuning and reward modelling of
human/LLM mixed texts as described in the paper.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("remodetect")
class ReMoDetectDetector(BaseTextDetector):
    """ReMoDetect reward-model detector.

    Parameters:
        model_name: HuggingFace reward-model checkpoint (default
            ``"hyunseoki/ReMoDetect-deberta"``).
        threshold: Decision boundary on the reward score.
        max_length: Max token length.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "hyunseoki/ReMoDetect-deberta",
        threshold: float = 0.0,
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
        self._rm_model: torch.nn.Module | None = None
        self._rm_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading ReMoDetect reward model '%s' …", self.model_name)
        self._rm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._rm_model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(
            self._device
        )
        self._rm_model.eval()

    @property
    def rm_model(self) -> torch.nn.Module:
        if self._rm_model is None:
            self._load_model()
        return self._rm_model  # type: ignore[return-value]

    @property
    def rm_tokenizer(self):
        if self._rm_tokenizer is None:
            self._load_model()
        return self._rm_tokenizer

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        enc = self.rm_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self._device)

        reward_score = float(self.rm_model(**enc).logits[0].squeeze())

        return self._make_result(
            reward_score,
            reward=reward_score,
        )
