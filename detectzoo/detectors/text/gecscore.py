"""GECScore — Grammar Error Correction based zero-shot detector.

Reference:
    Wu et al., "Who Wrote This? The Key to Zero-Shot LLM-Generated
    Text Detection Is GECScore", COLING 2025.

The key insight: LLM-generated text contains fewer grammatical errors
than human-written text.  After correcting grammar with a GEC model,
the similarity between the original and corrected text is measured.
High similarity → few corrections → likely AI-generated.

The default GEC model is ``grammarly/coedit-large`` (Flan-T5-Large
based).  The similarity metric is ROUGE-2 F-score.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("gecscore")
class GECScoreDetector(BaseTextDetector):
    """GECScore detector — grammar correction similarity.

    Parameters:
        gec_model: HuggingFace seq2seq model for grammar correction
            (default ``"grammarly/coedit-large"``).
        threshold: Decision boundary on the ROUGE-2 F-score.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        gec_model: str = "grammarly/coedit-large",
        threshold: float = 0.9243697428995128,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=gec_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.gec_model_name = gec_model
        self._gec_model: torch.nn.Module | None = None
        self._gec_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading GEC model '%s' …", self.gec_model_name)
        self._gec_tokenizer = AutoTokenizer.from_pretrained(self.gec_model_name)
        self._gec_model = AutoModelForSeq2SeqLM.from_pretrained(self.gec_model_name).to(
            self._device
        )
        self._gec_model.eval()

    @property
    def gec_model(self) -> torch.nn.Module:
        if self._gec_model is None:
            self._load_model()
        return self._gec_model  # type: ignore[return-value]

    @property
    def gec_tokenizer(self):
        if self._gec_tokenizer is None:
            self._load_model()
        return self._gec_tokenizer

    @torch.no_grad()
    def _correct_grammar(self, text: str) -> str:
        prompt = f"Fix grammatical errors in this sentence: {text}"
        enc = self.gec_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        out = self.gec_model.generate(
            **enc,
            max_new_tokens=self.max_length,
            num_beams=4,
            length_penalty=1.0,
        )
        return self.gec_tokenizer.decode(out[0], skip_special_tokens=True)

    @staticmethod
    def _rouge2_f(hypothesis: str, reference: str) -> float:
        """Compute ROUGE-2 F-score between two texts using py-rouge."""
        from rouge import Rouge

        rouge = Rouge()
        try:
            scores = rouge.get_scores(hypothesis, reference, avg=True)
            return scores["rouge-2"]["f"]
        except ValueError:
            return 1.0 if hypothesis.strip() == reference.strip() else 0.0

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        corrected = self._correct_grammar(text)

        if not corrected.strip():
            return self._make_result(1.0, reason="empty correction")

        score = self._rouge2_f(text, corrected)

        return self._make_result(
            score,
            corrected_text=corrected[:200],
            rouge2_f=score,
        )
