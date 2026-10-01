"""RoBERTa supervised detectors — OpenAI Detector and ChatGPT-Detector.

References:
    Solaiman et al., "Release strategies and the social impacts
    of language models." arXiv 2019.
    Guo et al., "How Close is ChatGPT to Human Experts? Comparison
    Corpus, Evaluation, and Detection." arXiv 2023.

HuggingFace checkpoints:

* **base** — ``openai-community/roberta-base-openai-detector`` (125 M params)
* **large** — ``openai-community/roberta-large-openai-detector`` (355 M params)
* **chatgpt** — ``Hello-SimpleAI/chatgpt-detector-roberta`` (RoBERTa-base
  trained on HC3; the ChatGPT-Detector baseline in ReMoDetect)

OpenAI variants distinguish WebText (human) from GPT-2 outputs.
ChatGPT-Detector distinguishes human vs ChatGPT answers on HC3.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_VARIANTS: dict[str, str] = {
    "base": "openai-community/roberta-base-openai-detector",
    "large": "openai-community/roberta-large-openai-detector",
    "chatgpt": "Hello-SimpleAI/chatgpt-detector-roberta",
}

_AI_LABEL_TOKENS = frozenset(
    {"fake", "ai", "chatgpt", "machine", "llm", "generated", "gpt"}
)


class _RobertaOpenAIBase(BaseTextDetector):
    """Shared implementation for RoBERTa sequence-classification detectors.

    Parameters:
        threshold: Decision boundary on the AI-class probability.
        max_length: Maximum token length for the tokenizer.
        device: ``"cpu"`` or ``"cuda"``.
    """

    modality = "text"
    _variant: str = "base"

    def __init__(
        self,
        threshold: float = 0.5,
        max_length: int = 512,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        model_name = _VARIANTS[self._variant]
        super().__init__(
            model_name=model_name,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self._cls_model: torch.nn.Module | None = None
        self._cls_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Model loading — sequence-classification head
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading RoBERTa classifier (%s) '%s' …", self._variant, self.model_name)
        self._cls_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._cls_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
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

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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

        ai_idx = 1
        id2label = getattr(self.cls_model.config, "id2label", None) or {}
        for idx, label in id2label.items():
            if str(label).lower() in _AI_LABEL_TOKENS:
                ai_idx = int(idx)
                break
        human_idx = 0 if ai_idx != 0 else 1

        real_prob = float(probs[human_idx])
        fake_prob = float(probs[ai_idx])

        return self._make_result(
            fake_prob,
            real_prob=real_prob,
            fake_prob=fake_prob,
            variant=self._variant,
        )


# ------------------------------------------------------------------
# Registered variants
# ------------------------------------------------------------------


@register_detector("roberta_base", aliases=["roberta_openai_base"])
class RobertaBaseDetector(_RobertaOpenAIBase):
    """RoBERTa Base OpenAI Detector (125 M parameters).

    Uses ``openai-community/roberta-base-openai-detector``.
    """

    _variant = "base"


@register_detector("roberta_large", aliases=["roberta_openai_large"])
class RobertaLargeDetector(_RobertaOpenAIBase):
    """RoBERTa Large OpenAI Detector (355 M parameters).

    Uses ``openai-community/roberta-large-openai-detector``.
    """

    _variant = "large"


@register_detector("chatgpt_detector", aliases=["chatgpt_roberta", "chat_d"])
class ChatGPTDetector(_RobertaOpenAIBase):
    """ChatGPT-Detector (Guo et al., 2023) — ReMoDetect Table 2 Chat-D.

    Uses ``Hello-SimpleAI/chatgpt-detector-roberta`` (RoBERTa-base trained
    on HC3). Label 0 = Human, label 1 = ChatGPT.
    """

    _variant = "chatgpt"
