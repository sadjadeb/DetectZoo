"""IRM — Implicit Reward Model for zero-shot detection.

Reference:
    Liu et al., "Zero-Shot Detection of LLM-Generated Text via
    Implicit Reward Model", NeurIPS 2025.

The key insight: any instruction-tuned LLM paired with its base model
implicitly defines a reward model via the DPO framework.  The log-
likelihood ratio between the instruction-tuned model and its base
model separates AI-generated text (higher ratio) from human text
(lower ratio).

    IRM(y) = sum_t [ log π_θ(y_t | y_{<t}) − log π_ref(y_t | y_{<t}) ]

where π_θ is the instruction-tuned model and π_ref is the base model.
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


@register_detector("irm")
class IRMDetector(BaseTextDetector):
    """Implicit Reward Model detector.

    Computes the log-probability ratio between an instruction-tuned
    model and its base counterpart.

    Parameters:
        instruct_model: Instruction-tuned model (default
            ``"meta-llama/Llama-3.2-1B-Instruct"``).
        base_model: Corresponding base model (default
            ``"meta-llama/Llama-3.2-1B"``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        instruct_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        base_model: str = "meta-llama/Llama-3.2-1B",
        threshold: float = 0.0,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=instruct_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.instruct_model_name = instruct_model
        self.base_model_name = base_model
        self._instruct_model: torch.nn.Module | None = None
        self._base_model_obj: torch.nn.Module | None = None
        self._shared_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading IRM instruct model '%s' …", self.instruct_model_name)
        self._shared_tokenizer = AutoTokenizer.from_pretrained(self.instruct_model_name)
        if self._shared_tokenizer.pad_token is None:
            self._shared_tokenizer.pad_token = self._shared_tokenizer.eos_token

        self._instruct_model = AutoModelForCausalLM.from_pretrained(self.instruct_model_name).to(
            self._device
        )
        self._instruct_model.eval()

        logger.info("Loading IRM base model '%s' …", self.base_model_name)
        self._base_model_obj = AutoModelForCausalLM.from_pretrained(self.base_model_name).to(
            self._device
        )
        self._base_model_obj.eval()

        self._model = self._instruct_model
        self._tokenizer = self._shared_tokenizer

    @property
    def instruct_model(self) -> torch.nn.Module:
        if self._instruct_model is None:
            self._load_model()
        return self._instruct_model  # type: ignore[return-value]

    @property
    def base_model_ref(self) -> torch.nn.Module:
        if self._base_model_obj is None:
            self._load_model()
        return self._base_model_obj  # type: ignore[return-value]

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        ids = enc["input_ids"]
        shift_labels = ids[:, 1:]

        instruct_logits = self.instruct_model(**enc).logits[:, :-1, :]
        base_logits = self.base_model_ref(**enc).logits[:, :-1, :]

        instruct_lp = F.log_softmax(instruct_logits, dim=-1)
        base_lp = F.log_softmax(base_logits, dim=-1)

        instruct_token_lp = instruct_lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        base_token_lp = base_lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # IRM score: sum of per-token log-prob differences
        score = float((instruct_token_lp - base_token_lp).sum())

        return self._make_result(
            score,
            instruct_ll=float(instruct_token_lp.sum()),
            base_ll=float(base_token_lp.sum()),
            n_tokens=int(shift_labels.shape[1]),
        )
