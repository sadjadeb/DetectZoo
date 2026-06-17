"""Binoculars detector — PPL / cross-perplexity ratio from two LMs.

Reference:
    Hans et al., "Spotting LLMs With Binoculars: Zero-Shot
    Detection of Machine-Generated Text", ICML 2024.

The detector uses an *observer* and a *performer* model.  The Binoculars
score is:

    B(s) = PPL_performer(s) / X-PPL_{observer, performer}(s)

where PPL is the standard perplexity of the *performer* and X-PPL is
the *cross-perplexity*: the cross-entropy between the observer's
softmax distribution and the performer's log-softmax, averaged over
token positions.

Low Binoculars score → likely AI.  We negate the log-ratio so that
*higher* score → more likely AI, matching the rest of DetectZoo.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("binoculars")
class BinocularsDetector(BaseTextDetector):
    """Binoculars detector (PPL / cross-perplexity).

    Parameters:
        observer_model: Observer LM name (default ``"gpt2"``).
        performer_model: Performer LM name (default ``"gpt2-medium"``).
        threshold: Decision threshold (on the negated ratio).
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "gpt2-medium",
        threshold: float = -0.9,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=observer_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.performer_model_name = performer_model
        self._performer_model: torch.nn.Module | None = None
        self._performer_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def performer_model(self) -> torch.nn.Module:
        if self._performer_model is None:
            self._load_performer()
        return self._performer_model  # type: ignore[return-value]

    @property
    def performer_tokenizer(self):
        if self._performer_tokenizer is None:
            self._load_performer()
        return self._performer_tokenizer

    def _load_performer(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading performer model '%s' …", self.performer_model_name)
        self._performer_tokenizer = AutoTokenizer.from_pretrained(self.performer_model_name)
        if self._performer_tokenizer.pad_token is None:
            self._performer_tokenizer.pad_token = self._performer_tokenizer.eos_token
        self._performer_model = AutoModelForCausalLM.from_pretrained(self.performer_model_name).to(
            self._device
        )
        self._performer_model.eval()

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_both_logits(self, text: str):
        """Tokenise *text* once and get logits from both models."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        observer_logits = self.model(**enc).logits  # [1, T, V]
        performer_logits = self.performer_model(**enc).logits  # [1, T, V]
        return observer_logits, performer_logits, enc

    @staticmethod
    def _performer_ppl(performer_logits: torch.Tensor, input_ids: torch.Tensor) -> float:
        """Standard perplexity of the performer on the token sequence."""
        shift_logits = performer_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return float(loss)  # log-perplexity (we keep in log-space)

    @staticmethod
    def _cross_perplexity(
        observer_logits: torch.Tensor,
        performer_logits: torch.Tensor,
        input_ids: torch.Tensor,
        pad_token_id: int | None = None,
    ) -> float:
        """Cross-perplexity: CE(observer_softmax, performer_log_softmax).

        For each token position, treat the observer's softmax output as the
        target distribution and compute the cross-entropy against the
        performer's log-softmax — then average over positions.
        """
        # Use all token positions (not shifted) following the official code
        observer_probs = F.softmax(observer_logits, dim=-1).view(-1, observer_logits.size(-1))
        performer_scores = performer_logits.view(-1, performer_logits.size(-1))

        # F.cross_entropy with soft targets (probabilities as target)
        ce = F.cross_entropy(
            input=performer_scores,
            target=observer_probs,
            reduction="none",
        )  # [T_total]

        total_tokens = observer_logits.shape[-2]
        ce = ce.view(-1, total_tokens)

        # Build a padding mask if needed
        if pad_token_id is not None:
            mask = (input_ids != pad_token_id).float()
            x_ppl = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            x_ppl = ce.mean(dim=1)

        return float(x_ppl.squeeze(0))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        observer_logits, performer_logits, enc = self._get_both_logits(text)

        log_ppl = self._performer_ppl(performer_logits, enc["input_ids"])
        x_ppl = self._cross_perplexity(
            observer_logits,
            performer_logits,
            enc["input_ids"],
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Binoculars score = ppl / x_ppl (in log-space: log_ppl / x_ppl_val)
        binoculars_score = log_ppl / max(x_ppl, 1e-8)

        # Low Binoculars score → likely AI.
        # Negate so higher score → more likely AI (DetectZoo convention).
        score = -binoculars_score

        return self._make_result(
            score,
            log_perplexity=log_ppl,
            cross_perplexity=x_ppl,
            binoculars_raw=binoculars_score,
        )
