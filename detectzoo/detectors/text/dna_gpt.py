"""DNA-GPT — Divergent N-Gram Analysis for GPT-generated text.

Reference:
    Yang et al., "DNA-GPT: Divergent N-Gram Analysis with LLM Probability
    for Training-Free Detection of GPT-Generated Text", ICLR 2024.

The core idea: truncate the text at a midpoint, use a language model to
re-generate the continuation multiple times, then compare the
log-probability of the *original* full text against the
log-probabilities of the re-generated versions.

Machine text was drawn from a model distribution, so its continuation
is hard to reproduce exactly — the original will have notably higher
log-prob than the re-generations.  For human text the gap is smaller.

    DNA-GPT(x) = log p(x) − mean_i( log p(trim(x̃_i)) )

where x̃_i are re-generated versions with different continuations,
each trimmed to ``min(len(x), len(x̃_i))`` words.  The original text
is scored at full length (not trimmed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


@register_detector("dna_gpt")
class DNAGPTDetector(BaseTextDetector):
    """DNA-GPT detector (truncation + re-generation).

    A *single* causal LM is used both for scoring (log-prob) and for
    re-generating continuations.  You may optionally specify a
    separate ``regen_model`` for generation.

    The re-generation procedure is as follows:

    * prefix = first ``truncate_ratio`` fraction of words
    * sample all ``n_regens`` completions in a single batch
    * retry the whole batch if the shortest completion is below ``min_words``
    * trim both the original and each regen to the shorter of the two
      before scoring, so log-prob means are comparable

    Parameters:
        model_name: Scoring model (default ``"gpt2"``).
        regen_model: Regeneration model.  Defaults to *model_name*.
        truncate_ratio: Fraction of text (by words) kept as prompt (default 0.5).
        n_regens: Number of re-generated continuations (default 10).
        min_words: Minimum word count for each regen; the batch is
            re-sampled until this is reached (default 55).
        min_length: ``min_length`` passed to ``generate`` (default 150).
        max_length: ``max_length`` passed to ``generate`` (default 200).
        top_p: top-p for nucleus sampling (default 0.96).
        temperature: Sampling temperature (default 1.0).
        max_retries: Cap on regeneration retries (default 10).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        regen_model: str | None = None,
        truncate_ratio: float = 0.5,
        n_regens: int = 10,
        min_words: int = 55,
        min_length: int = 150,
        max_length: int = 200,
        top_p: float = 0.96,
        temperature: float = 1.0,
        max_retries: int = 10,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.regen_model_name = regen_model or model_name
        self.truncate_ratio = truncate_ratio
        self.n_regens = n_regens
        self.min_words = min_words
        self.gen_min_length = min_length
        self.gen_max_length = max_length
        self.top_p = top_p
        self.temperature = temperature
        self.max_retries = max_retries
        self._regen_model: torch.nn.Module | None = None
        self._regen_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Regen model (may be the same as scoring model)
    # ------------------------------------------------------------------

    @property
    def regen_model(self) -> torch.nn.Module:
        if self.regen_model_name == self.model_name:
            return self.model
        if self._regen_model is None:
            self._load_regen_model()
        return self._regen_model  # type: ignore[return-value]

    @property
    def regen_tokenizer(self):
        if self.regen_model_name == self.model_name:
            return self.tokenizer
        if self._regen_tokenizer is None:
            self._load_regen_model()
        return self._regen_tokenizer

    def _load_regen_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading regen model '%s' …", self.regen_model_name)
        self._regen_tokenizer = AutoTokenizer.from_pretrained(self.regen_model_name)
        if self._regen_tokenizer.pad_token is None:
            self._regen_tokenizer.pad_token = self._regen_tokenizer.eos_token
        self._regen_model = AutoModelForCausalLM.from_pretrained(self.regen_model_name).to(
            self._device
        )
        self._regen_model.eval()

    # ------------------------------------------------------------------
    # Regeneration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_batch(self, prompt: str) -> list[str]:
        """Return ``n_regens`` completions of *prompt*, retrying until they are long enough."""
        prompts = [prompt] * self.n_regens
        enc = self.regen_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        eos_id = self.regen_tokenizer.eos_token_id
        pad_id = self.regen_tokenizer.pad_token_id or eos_id

        decoded: list[str] = []
        for attempt in range(self.max_retries):
            out = self.regen_model.generate(
                **enc,
                min_length=self.gen_min_length,
                max_length=self.gen_max_length,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            decoded = self.regen_tokenizer.batch_decode(out, skip_special_tokens=True)
            m = min(len(x.split()) for x in decoded) if decoded else 0
            if m >= self.min_words:
                return decoded
            logger.debug(
                "DNA-GPT regeneration attempt %d: shortest %d < %d words, retrying.",
                attempt + 1,
                m,
                self.min_words,
            )
        return decoded  # last attempt, even if still too short

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_shorter_length(a: str, b: str) -> tuple[str, str]:
        """Truncate *a* and *b* to the shorter of their two word lengths."""
        wa, wb = a.split(" "), b.split(" ")
        k = min(len(wa), len(wb))
        return " ".join(wa[:k]), " ".join(wb[:k])

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        words = text.split()
        if len(words) < 6:
            return self._make_result(0.0, reason="text too short")

        split_idx = max(3, int(len(words) * self.truncate_ratio))
        prompt = " ".join(words[:split_idx])

        regens_raw = self._sample_batch(prompt)
        if not regens_raw:
            return self._make_result(0.0, reason="regeneration failed")

        # The full original text is scored once, and each regen
        # is trimmed to min(len(original), len(regen)) words.
        original_ll = self._mean_log_prob(text)

        regen_lls: list[float] = []
        for regen in regens_raw:
            if not regen.strip():
                continue
            _, r_trim = self._trim_to_shorter_length(text, regen)
            if not r_trim.strip():
                continue
            regen_lls.append(self._mean_log_prob(r_trim))

        if not regen_lls:
            return self._make_result(0.0, reason="regeneration failed")

        mean_regen_ll = float(np.mean(regen_lls))
        score = original_ll - mean_regen_ll

        return self._make_result(
            score,
            original_ll=original_ll,
            mean_regen_ll=mean_regen_ll,
            n_regens_used=len(regen_lls),
        )
