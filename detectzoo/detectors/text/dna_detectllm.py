"""DNA-DetectLLM — mutation-repair paradigm for zero-shot detection.

Reference:
    Zhu et al., "DNA-DetectLLM: Unveiling AI-Generated Text via a
    DNA-Inspired Mutation-Repair Paradigm", NeurIPS 2025.

The method constructs an "ideal AI sequence" by greedily selecting
the most probable token at each position under a performer model,
then measures the repair effort needed to transform the input into
this ideal sequence.  The score combines perplexity of the original
text, perplexity of the ideal sequence, and cross-perplexity between
an observer and performer model:

    score = (ppl_std + ppl_max) / (2 * x_ppl)

Low score → AI-generated; high score → human-written.
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


@register_detector("dna_detectllm")
class DNADetectLLMDetector(BaseTextDetector):
    """DNA-DetectLLM detector — mutation-repair paradigm.

    Requires a performer (instruction-tuned) and observer (base) model
    from the same family.

    Parameters:
        performer_model: Instruction-tuned LM (default
            ``"tiiuae/falcon-7b-instruct"``).
        observer_model: Base LM from the same family (default
            ``"tiiuae/falcon-7b"``).
        threshold: Decision boundary (lower score → AI).  Negated
            internally so higher → more likely AI.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        performer_model: str = "tiiuae/falcon-7b-instruct",
        observer_model: str = "tiiuae/falcon-7b",
        threshold: float = 0.0,
        device: str = "cpu",
        max_length: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=performer_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.performer_model_name = performer_model
        self.observer_model_name = observer_model
        self._performer_model: torch.nn.Module | None = None
        self._observer_model: torch.nn.Module | None = None
        self._shared_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading DNA-DetectLLM performer '%s' …", self.performer_model_name)
        self._shared_tokenizer = AutoTokenizer.from_pretrained(self.performer_model_name)
        if self._shared_tokenizer.pad_token is None:
            self._shared_tokenizer.pad_token = self._shared_tokenizer.eos_token

        self._performer_model = AutoModelForCausalLM.from_pretrained(self.performer_model_name).to(
            self._device
        )
        self._performer_model.eval()

        logger.info("Loading DNA-DetectLLM observer '%s' …", self.observer_model_name)
        self._observer_model = AutoModelForCausalLM.from_pretrained(self.observer_model_name).to(
            self._device
        )
        self._observer_model.eval()

        self._model = self._performer_model
        self._tokenizer = self._shared_tokenizer

    @property
    def performer(self) -> torch.nn.Module:
        if self._performer_model is None:
            self._load_model()
        return self._performer_model  # type: ignore[return-value]

    @property
    def observer(self) -> torch.nn.Module:
        if self._observer_model is None:
            self._load_model()
        return self._observer_model  # type: ignore[return-value]

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
        mask = enc.get("attention_mask", torch.ones_like(ids))

        performer_logits = self.performer(**enc).logits
        observer_logits = self.observer(**enc).logits

        shifted_logits = performer_logits[:, :-1, :]
        labels_std = ids[:, 1:]
        labels_max = shifted_logits.argmax(dim=-1)

        shift_mask = mask[:, 1:].float()
        n_tokens = shift_mask.sum().clamp(min=1.0)

        ce_std = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            labels_std.reshape(-1),
            reduction="none",
        ).reshape(shift_mask.shape)

        ce_max = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            labels_max.reshape(-1),
            reduction="none",
        ).reshape(shift_mask.shape)

        ppl_std = float((ce_std * shift_mask).sum() / n_tokens)
        ppl_max = float((ce_max * shift_mask).sum() / n_tokens)
        ppl = ppl_std + ppl_max

        # Cross-perplexity: CE(observer_softmax, performer_logits)
        observer_probs = F.softmax(observer_logits, dim=-1)
        padding_mask = mask.float()
        total_tokens = padding_mask.sum().clamp(min=1.0)

        x_ce = F.cross_entropy(
            performer_logits.reshape(-1, performer_logits.size(-1)),
            observer_probs.reshape(-1, observer_probs.size(-1)),
            reduction="none",
        )
        x_ce = x_ce.reshape(padding_mask.shape)
        x_ppl = float((x_ce * padding_mask).sum() / total_tokens)

        # DNA-DetectLLM score: low = AI, high = human
        # Negate so higher → more likely AI (DetectZoo convention)
        raw_score = ppl / max(2.0 * x_ppl, 1e-8)
        score = -raw_score

        return self._make_result(
            score,
            ppl_std=ppl_std,
            ppl_max=ppl_max,
            cross_ppl=x_ppl,
            raw_score=raw_score,
        )
