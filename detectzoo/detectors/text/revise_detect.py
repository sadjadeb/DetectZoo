"""Revise-Detect — revision-based AI-text detector.

Reference:
    Zhu et al., "Beat LLMs at Their Own Game: Zero-Shot
    LLM-Generated Text Detection via Querying ChatGPT", EMNLP 2023.

The intuition: when an LLM is asked to *revise* a piece of text,
AI-generated text changes less (it is already close to the model's
preferred output).  The similarity between the original and the
revision serves as a detection signal.

This implementation uses a seq2seq model (BART by default) both to
produce the revision and to score the similarity between original and
revised text via sequence-level log-probability (BARTScore).
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("revise_detect")
class ReviseDetector(BaseTextDetector):
    """Revision-based detector using BARTScore-style similarity.

    Steps:
      1. Feed the text through a seq2seq model with a revision prompt
         to obtain a revised version.
      2. Compute BARTScore = average log p(original | revised) under
         the seq2seq model.  Higher similarity → more likely AI.

    Parameters:
        revision_model: HuggingFace seq2seq model for revision and
            scoring (default ``"facebook/bart-large-cnn"``).
        threshold: Decision boundary on the BARTScore.
        max_length: Max token length for encoder inputs.
        device: ``"cpu"`` or ``"cuda"``.
    """

    modality = "text"

    def __init__(
        self,
        revision_model: str = "facebook/bart-large-cnn",
        threshold: float = -2.0,
        max_length: int = 512,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=revision_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.revision_model_name = revision_model
        self._seq2seq_model: torch.nn.Module | None = None
        self._seq2seq_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy loading (overrides BaseTextDetector)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading revision model '%s' …", self.revision_model_name)
        self._seq2seq_tokenizer = AutoTokenizer.from_pretrained(self.revision_model_name)
        self._seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(self.revision_model_name).to(
            self._device
        )
        self._seq2seq_model.eval()

    @property
    def seq2seq_model(self) -> torch.nn.Module:
        if self._seq2seq_model is None:
            self._load_model()
        return self._seq2seq_model  # type: ignore[return-value]

    @property
    def seq2seq_tokenizer(self):
        if self._seq2seq_tokenizer is None:
            self._load_model()
        return self._seq2seq_tokenizer

    # ------------------------------------------------------------------
    # Revision
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _revise(self, text: str) -> str:
        """Generate a revised version of *text* using the seq2seq model."""
        enc = self.seq2seq_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        out = self.seq2seq_model.generate(
            **enc,
            max_new_tokens=self.max_length,
            num_beams=4,
            length_penalty=1.0,
        )
        return self.seq2seq_tokenizer.decode(out[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # BARTScore — log p(target | source) under the seq2seq model
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _bart_score(self, source: str, target: str) -> float:
        """Compute average token log-prob of *target* conditioned on *source*."""
        src_enc = self.seq2seq_tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        tgt_enc = self.seq2seq_tokenizer(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        labels = tgt_enc["input_ids"].to(self._device)

        outputs = self.seq2seq_model(
            input_ids=src_enc["input_ids"],
            attention_mask=src_enc["attention_mask"],
            labels=labels,
        )
        # outputs.loss is mean cross-entropy; negate to get mean log-prob
        return -float(outputs.loss)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        revised = self._revise(text)
        if not revised.strip():
            return self._make_result(0.0, reason="empty revision")

        similarity = self._bart_score(revised, text)

        return self._make_result(
            similarity,
            revised_text=revised[:200],
            bart_score=similarity,
        )
