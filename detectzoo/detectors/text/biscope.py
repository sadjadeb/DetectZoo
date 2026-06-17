"""BiScope — bidirectional cross-entropy zero-shot detector.

Reference:
    Guo et al., "BiScope: AI-generated Text Detection by Checking
    Memorization of Preceding Tokens", NeurIPS 2024.

BiScope exploits two kinds of information in a causal LM's logits
**conditioned on a completion prompt**:

1. **Forward CE (FCE):** Standard next-token cross-entropy — logits at
   position *i−1* predict the token at position *i* (within the text
   region only; the first text token is predicted by the last prompt
   token).
2. **Backward CE (BCE):** Logits at position *i* predict the token at
   position *i itself* (memorisation signal).

Human text → poor next-token prediction, strong memorisation.
Machine text → good prediction, weak memorisation.

The prompt is ``"Given the summary:\\n{summary}\\n Complete the
following text: "`` when a summary model is provided, otherwise
``"Complete the following text: "``.  Losses are computed **only** on
the text portion after the prompt.

Per-token FCE and BCE are split into suffix segments and summarised
with (mean, max, min, std) statistics — yielding a 72-d feature
vector (9 segments × 8 stats).  These features are intended for a
downstream classifier.  As a zero-shot proxy the detector returns
``−(mean_FCE − mean_BCE)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_COMPLETION_PROMPT_ONLY = "Complete the following text: "
_COMPLETION_PROMPT_TPL = "Given the summary:\n{summary}\n Complete the following text: "


@register_detector("biscope")
class BiScopeDetector(BaseTextDetector):
    """BiScope bidirectional cross-entropy detector.

    Parameters:
        model_name: HuggingFace causal LM for detection
            (default ``"meta-llama/Llama-2-7b-chat-hf"``).
        summary_model: Optional HuggingFace causal LM used to
            generate a short summary/title that is embedded in the
            completion prompt.  If ``None``, falls back to the
            generic ``"Complete the following text: "`` prompt.
        sample_clip: Max tokens for the text portion
            (default ``512``).
        n_segments: Number of segments for multi-point feature
            extraction (default ``10``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        summary_model: str | None = None,
        sample_clip: int = 512,
        n_segments: int = 10,
        threshold: float = 6.9,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.summary_model_name = summary_model
        self.sample_clip = sample_clip
        self.n_segments = n_segments
        self._summary_model: torch.nn.Module | None = None
        self._summary_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Summary model (optional)
    # ------------------------------------------------------------------

    def _load_summary_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading BiScope summary model '%s' …", self.summary_model_name)
        self._summary_tokenizer = AutoTokenizer.from_pretrained(
            self.summary_model_name,
            padding_side="left",
        )
        if self._summary_tokenizer.pad_token is None:
            self._summary_tokenizer.pad_token = self._summary_tokenizer.eos_token
        self._summary_model = AutoModelForCausalLM.from_pretrained(
            self.summary_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._summary_model.eval()

    @property
    def summary_model(self) -> torch.nn.Module | None:
        if self.summary_model_name is None:
            return None
        if self._summary_model is None:
            self._load_summary_model()
        return self._summary_model

    @property
    def summary_tokenizer(self):
        if self.summary_model_name is None:
            return None
        if self._summary_tokenizer is None:
            self._load_summary_model()
        return self._summary_tokenizer

    # ------------------------------------------------------------------
    # Summary / prompt generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_summary(self, text: str) -> str:
        """Generate a short title/summary for the text."""
        prompt = f"Write a title for this text: {text}\nJust output the title:"
        ids = self.summary_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.sample_clip,
            truncation=True,
        ).input_ids.to(self._device)
        ids = ids[:, 1:]  # remove start token
        trigger_len = ids.shape[1]
        config = self.summary_model.generation_config
        config.max_new_tokens = 64
        attn = torch.ones_like(ids)
        out = self.summary_model.generate(
            ids,
            attention_mask=attn,
            generation_config=config,
            pad_token_id=self.summary_tokenizer.pad_token_id,
        )[0]
        gen_ids = out[trigger_len:]
        summary = self.summary_tokenizer.decode(gen_ids, skip_special_tokens=True)
        return summary.strip().split("\n")[0]

    def _build_prompt(self, text: str) -> str:
        """Return the completion prompt (with or without summary)."""
        if self.summary_model is not None:
            summary = self._generate_summary(text)
            return _COMPLETION_PROMPT_TPL.format(summary=summary)
        return _COMPLETION_PROMPT_ONLY

    # ------------------------------------------------------------------
    # FCE / BCE computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_biscope_losses(
        self,
        text: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-token FCE and BCE loss arrays on the text region.

        The text is prepended with a completion prompt.  Losses are
        computed only over the text tokens (not the prompt).
        """
        prompt_text = self._build_prompt(text)

        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        ).input_ids.to(self._device)

        text_ids = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.sample_clip,
            truncation=True,
        ).input_ids.to(self._device)

        combined_ids = torch.cat([prompt_ids, text_ids], dim=1)  # [1, P+T]
        prompt_len = prompt_ids.shape[1]
        total_len = combined_ids.shape[1]

        if total_len - prompt_len < 3:
            return np.array([0.0]), np.array([0.0])

        logits = self.model(input_ids=combined_ids).logits  # [1, P+T, V]
        targets = combined_ids[0, prompt_len:total_len]  # [T]

        # FCE: logits at [prompt_len-1 .. total_len-2] predict targets [0..T-1]
        fce_logits = logits[0, prompt_len - 1 : total_len - 1, :]
        fce = F.cross_entropy(fce_logits, targets, reduction="none")

        # BCE: logits at [prompt_len .. total_len-1] predict targets [0..T-1]
        bce_logits = logits[0, prompt_len:total_len, :]
        bce = F.cross_entropy(bce_logits, targets, reduction="none")

        return fce.cpu().numpy(), bce.cpu().numpy()

    def _extract_features(self, fce: np.ndarray, bce: np.ndarray) -> np.ndarray:
        """Extract 72-d multi-point statistical features from loss arrays.

        9 suffix splits × (mean, max, min, std) × 2 loss types.
        """
        features: list[float] = []
        n = len(fce)
        for p in range(1, self.n_segments):
            split = n * p // self.n_segments
            fce_suffix = fce[split:]
            bce_suffix = bce[split:]
            if len(fce_suffix) == 0:
                features.extend([0.0] * 8)
                continue
            features.extend(
                [
                    float(np.mean(fce_suffix)),
                    float(np.max(fce_suffix)),
                    float(np.min(fce_suffix)),
                    float(np.std(fce_suffix)),
                    float(np.mean(bce_suffix)),
                    float(np.max(bce_suffix)),
                    float(np.min(bce_suffix)),
                    float(np.std(bce_suffix)),
                ]
            )
        return np.array(features)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        fce, bce = self._compute_biscope_losses(text)

        if len(fce) < 2:
            return self._make_result(0.0, reason="text too short")

        features = self._extract_features(fce, bce)

        mean_fce = float(np.mean(fce))
        mean_bce = float(np.mean(bce))
        score = -(mean_fce - mean_bce)

        return self._make_result(
            score,
            mean_fce=mean_fce,
            mean_bce=mean_bce,
            feature_dim=len(features),
        )
