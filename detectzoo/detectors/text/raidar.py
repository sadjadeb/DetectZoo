"""Raidar — Rewriting-based AI-text detector.

Reference:
    Mao et al., "RAIDAR: geneRative AI Detection viA Rewriting",
    ICLR 2024.

The core observation: LLMs modify human-written text more than
AI-generated text when asked to rewrite it.  The method:

1. Rewrites the input text using a seq2seq model.
2. Computes the edit-distance (Levenshtein-based) similarity between
   the original and rewritten text.
3. Higher similarity → AI-generated (less change under rewriting).

This implementation uses a local seq2seq model (BART or FLAN-T5)
instead of the original GPT-3.5-turbo API.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalised Levenshtein similarity: 1 − edit_dist / max(len)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp

    edit_dist = dp[n]
    return 1.0 - edit_dist / max(m, n)


def _ngram_overlap(text1: str, text2: str, n: int) -> float:
    """Normalised n-gram overlap between two texts."""

    def _ngrams(text: str, n: int) -> dict[tuple[str, ...], int]:
        tokens = text.lower().split()
        ng: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            key = tuple(tokens[i : i + n])
            ng[key] = ng.get(key, 0) + 1
        return ng

    ng1 = _ngrams(text1, n)
    ng2 = _ngrams(text2, n)
    if not ng1 or not ng2:
        return 1.0 if text1.strip() == text2.strip() else 0.0

    overlap = sum(min(ng1[k], ng2.get(k, 0)) for k in ng1)
    return overlap / max(sum(ng1.values()), 1)


_REWRITE_PROMPTS = [
    "Revise this with your best effort: ",
    "Help me polish this: ",
    "Rewrite this for me: ",
]


@register_detector("raidar")
class RaidarDetector(BaseTextDetector):
    """Raidar rewriting-invariance detector.

    Parameters:
        rewrite_model: HuggingFace seq2seq model for rewriting
            (default ``"facebook/bart-large-cnn"``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        rewrite_model: str = "facebook/bart-large-cnn",
        threshold: float = 0.5,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=rewrite_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.rewrite_model_name = rewrite_model
        self._rw_model: torch.nn.Module | None = None
        self._rw_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading Raidar rewrite model '%s' …", self.rewrite_model_name)
        self._rw_tokenizer = AutoTokenizer.from_pretrained(self.rewrite_model_name)
        self._rw_model = AutoModelForSeq2SeqLM.from_pretrained(self.rewrite_model_name).to(
            self._device
        )
        self._rw_model.eval()

    @property
    def rw_model(self) -> torch.nn.Module:
        if self._rw_model is None:
            self._load_model()
        return self._rw_model  # type: ignore[return-value]

    @property
    def rw_tokenizer(self):
        if self._rw_tokenizer is None:
            self._load_model()
        return self._rw_tokenizer

    @torch.no_grad()
    def _rewrite(self, text: str) -> str:
        enc = self.rw_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        out = self.rw_model.generate(
            **enc,
            max_new_tokens=self.max_length,
            num_beams=4,
            length_penalty=1.0,
        )
        return self.rw_tokenizer.decode(out[0], skip_special_tokens=True)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        sims: list[float] = []
        ngram_sims: list[float] = []
        for prompt in _REWRITE_PROMPTS:
            rewritten = self._rewrite(prompt + text)
            if not rewritten.strip():
                continue
            sims.append(_levenshtein_ratio(text, rewritten))
            for n in range(1, 5):
                ngram_sims.append(_ngram_overlap(text, rewritten, n))

        if not sims:
            return self._make_result(0.0, reason="rewriting failed")

        # Higher similarity → less change → more likely AI
        mean_sim = float(sum(sims) / len(sims))
        mean_ngram = float(sum(ngram_sims) / len(ngram_sims)) if ngram_sims else 0.0

        score = mean_sim

        return self._make_result(
            score,
            mean_levenshtein_sim=mean_sim,
            mean_ngram_overlap=mean_ngram,
            n_rewrites=len(sims),
        )
