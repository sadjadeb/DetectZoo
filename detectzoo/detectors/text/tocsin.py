"""TOCSIN — Token Cohesiveness based zero-shot detector.

Reference:
    Ma & Wang, "Zero-Shot Detection of LLM-Generated Text using
    Token Cohesiveness", EMNLP 2024.

Token cohesiveness measures how tightly bound the tokens in a passage
are.  LLM-generated text has higher cohesiveness (tokens are more
interdependent due to causal self-attention).

TOCSIN creates perturbed copies by randomly deleting a small fraction
of tokens, measures the semantic difference via BARTScore, and uses
the result to amplify a base detection signal (the conditional
probability curvature from Fast-DetectGPT).

    w(x) = e^{u(x)} * v(x)

where u(x) is the token cohesiveness and v(x) is the base detector
score (Fast-DetectGPT curvature by default).
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("tocsin")
class TOCSINDetector(BaseTextDetector):
    """TOCSIN token-cohesiveness detector.

    Combines token cohesiveness (via BARTScore after random token
    deletion) with Fast-DetectGPT curvature scoring.

    Parameters:
        model_name: Causal LM for Fast-DetectGPT scoring
            (default ``"gpt2"``).
        bart_model: BART model for BARTScore computation
            (default ``"facebook/bart-base"``).
        n_copies: Number of perturbed copies (default ``10``).
        deletion_rate: Fraction of tokens deleted (default ``0.015``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        bart_model: str = "facebook/bart-base",
        n_copies: int = 10,
        deletion_rate: float = 0.015,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.bart_model_name = bart_model
        self.n_copies = n_copies
        self.deletion_rate = deletion_rate
        self._bart_model: torch.nn.Module | None = None
        self._bart_tokenizer: Any = None

    # ------------------------------------------------------------------
    # BART model loading
    # ------------------------------------------------------------------

    def _load_bart(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading BART model '%s' for BARTScore …", self.bart_model_name)
        self._bart_tokenizer = AutoTokenizer.from_pretrained(self.bart_model_name)
        self._bart_model = AutoModelForSeq2SeqLM.from_pretrained(self.bart_model_name).to(
            self._device
        )
        self._bart_model.eval()

    @property
    def bart_model(self) -> torch.nn.Module:
        if self._bart_model is None:
            self._load_bart()
        return self._bart_model  # type: ignore[return-value]

    @property
    def bart_tokenizer(self):
        if self._bart_tokenizer is None:
            self._load_bart()
        return self._bart_tokenizer

    # ------------------------------------------------------------------
    # Token cohesiveness
    # ------------------------------------------------------------------

    def _random_delete(self, text: str) -> str:
        words = text.split()
        if len(words) < 5:
            return text
        n_delete = max(1, int(len(words) * self.deletion_rate))
        indices = set(random.sample(range(len(words)), min(n_delete, len(words))))
        return " ".join(w for i, w in enumerate(words) if i not in indices)

    @torch.no_grad()
    def _bart_score(self, source: str, target: str) -> float:
        """BARTScore: avg token log-prob of *target* given *source*."""
        src_enc = self.bart_tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)
        tgt_enc = self.bart_tokenizer(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        labels = tgt_enc["input_ids"].to(self._device)

        out = self.bart_model(
            input_ids=src_enc["input_ids"],
            attention_mask=src_enc["attention_mask"],
            labels=labels,
        )
        return -float(out.loss)

    def _token_cohesiveness(self, text: str) -> float:
        """Compute token cohesiveness u(x) = E[DIFF(x, x̃)]."""
        scores = []
        for _ in range(self.n_copies):
            perturbed = self._random_delete(text)
            bs = self._bart_score(perturbed, text)
            scores.append(-bs)  # DIFF = -BARTScore
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Fast-DetectGPT curvature (base detector)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _curvature_score(self, text: str) -> float:
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]

        if shift_labels.ndim == shift_logits.ndim - 1:
            shift_labels = shift_labels.unsqueeze(-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)

        ll = log_probs.gather(dim=-1, index=shift_labels).squeeze(-1)
        mean_ref = (probs * log_probs).sum(dim=-1)
        var_ref = (probs * log_probs.square()).sum(dim=-1) - mean_ref.square()

        ll_sum = ll.sum(dim=-1)
        mean_sum = mean_ref.sum(dim=-1)
        std_sum = var_ref.sum(dim=-1).clamp(min=1e-10).sqrt()

        return float((ll_sum - mean_sum) / std_sum)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        curvature = self._curvature_score(text)
        cohesiveness = self._token_cohesiveness(text)

        # Combine: w(x) = e^{u(x)} * v(x) (sign-aware)
        if curvature >= 0:
            score = math.exp(cohesiveness) * curvature
        else:
            score = math.exp(-cohesiveness) * curvature

        return self._make_result(
            score,
            curvature=curvature,
            cohesiveness=cohesiveness,
        )
