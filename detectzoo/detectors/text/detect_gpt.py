"""DetectGPT — perturbation-based LLM-text detector.

Reference:
    Mitchell et al., "DetectGPT: Zero-Shot Machine-Generated Text
    Detection using Probability Curvature", ICML 2023.

The key idea: machine-generated text sits near local maxima of the
log-probability surface of the source model.  By comparing the original
log-prob to those of minor perturbations (produced by a mask-filling
model such as T5) one obtains a curvature score that separates human
from machine text.
"""

from __future__ import annotations

import random
import re
from typing import Any, List

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

# Regex that matches T5 sentinel tokens in decoded output.
_SENTINEL_RE = re.compile(r"<extra_id_\d+>")


@register_detector("detectgpt")
class DetectGPTDetector(BaseTextDetector):
    """DetectGPT detector using perturbation-based probability curvature.

    Parameters:
        model_name: Scoring model (causal LM, default ``"EleutherAI/gpt-neo-2.7B"``).
        perturbation_model: Mask-filling model (default ``"t5-3b"``).
        n_perturbations: Number of perturbations per input.
        pct_words_masked: Fraction of words targeted for masking.
        span_length: Number of consecutive words per masked span.
        buffer_size: Minimum gap (in words) between two masked spans.
        mask_top_p: top-p for the mask-filling T5 sampler.
        threshold: Decision threshold on the z-score.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-2.7B",
        perturbation_model: str = "t5-3b",
        n_perturbations: int = 10,
        pct_words_masked: float = 0.3,
        span_length: int = 2,
        buffer_size: int = 1,
        mask_top_p: float = 1.0,
        threshold: float = 0.0,
        device: str = "cpu",
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.perturbation_model_name = perturbation_model
        self.n_perturbations = n_perturbations
        self.pct_words_masked = pct_words_masked
        self.span_length = span_length
        self.buffer_size = buffer_size
        self.mask_top_p = mask_top_p
        self._pmodel: torch.nn.Module | None = None
        self._ptokenizer: Any = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Perturbation model loading
    # ------------------------------------------------------------------

    @property
    def perturbation_model(self) -> torch.nn.Module:
        if self._pmodel is None:
            self._load_perturbation_model()
        return self._pmodel  # type: ignore[return-value]

    @property
    def perturbation_tokenizer(self):
        if self._ptokenizer is None:
            self._load_perturbation_model()
        return self._ptokenizer

    def _load_perturbation_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading perturbation model '%s' …", self.perturbation_model_name)
        self._ptokenizer = AutoTokenizer.from_pretrained(self.perturbation_model_name)
        self._pmodel = AutoModelForSeq2SeqLM.from_pretrained(self.perturbation_model_name).to(
            self._device
        )
        self._pmodel.eval()

    # ------------------------------------------------------------------
    # Perturbation generation
    # ------------------------------------------------------------------

    def _tokenize_and_mask(self, text: str, ceil_pct: bool = False) -> str:
        """Replace random fixed-length word spans with ``<extra_id_*>`` tokens."""
        span_length = self.span_length
        buffer_size = self.buffer_size
        pct = self.pct_words_masked
        tokens = text.split(" ")
        mask_string = "<<<mask>>>"

        if len(tokens) <= span_length:
            return text

        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        n_spans = int(np.ceil(n_spans) if ceil_pct else n_spans)
        if n_spans <= 0:
            return text

        n_masks = 0
        tries = 0
        max_tries = 10 * n_spans
        while n_masks < n_spans and tries < max_tries:
            tries += 1
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # Replace each <<<mask>>> with <extra_id_k> for increasing k.
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # T5 mask filling
    # ------------------------------------------------------------------

    @staticmethod
    def _count_masks(text: str) -> int:
        return sum(1 for t in text.split() if t.startswith("<extra_id_"))

    @torch.no_grad()
    def _replace_masks(self, masked_text: str) -> str:
        """Ask T5 to fill in all ``<extra_id_*>`` tokens and return raw decoded output."""
        n_expected = self._count_masks(masked_text)
        if n_expected == 0:
            return ""
        stop_token = f"<extra_id_{n_expected}>"
        stop_ids = self.perturbation_tokenizer.encode(stop_token, add_special_tokens=False)
        stop_id = stop_ids[0] if stop_ids else self.perturbation_tokenizer.eos_token_id

        enc = self.perturbation_tokenizer(
            masked_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        out_ids = self.perturbation_model.generate(
            **enc,
            max_length=150,
            do_sample=True,
            top_p=self.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return self.perturbation_tokenizer.decode(out_ids[0], skip_special_tokens=False)

    @staticmethod
    def _extract_fills(raw: str) -> List[str]:
        cleaned = raw.replace("<pad>", "").replace("</s>", "").strip()
        parts = _SENTINEL_RE.split(cleaned)
        return [p.strip() for p in parts[1:-1]] if len(parts) >= 3 else []

    @staticmethod
    def _apply_extracted_fills(masked_text: str, fills: List[str]) -> str:
        tokens = masked_text.split(" ")
        n_expected = sum(1 for t in tokens if t.startswith("<extra_id_"))
        if len(fills) < n_expected:
            return ""
        for k in range(n_expected):
            sentinel = f"<extra_id_{k}>"
            try:
                idx = tokens.index(sentinel)
            except ValueError:
                continue
            tokens[idx] = fills[k]
        return " ".join(tokens)

    def _perturb_once(self, text: str, ceil_pct: bool = False) -> str:
        """Produce a single perturbation of *text* (retrying the fill up to a few times)."""
        masked = self._tokenize_and_mask(text, ceil_pct=ceil_pct)
        if "<extra_id_" not in masked:
            return text  # nothing to mask

        for _ in range(5):  # a few attempts per input, matching reference behaviour
            raw = self._replace_masks(masked)
            fills = self._extract_fills(raw)
            out = self._apply_extracted_fills(masked, fills)
            if out:
                return out
        return ""  # give up — caller will skip

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        original_ll = self._mean_log_prob(text)

        perturbation_lls: list[float] = []
        for _ in range(self.n_perturbations):
            perturbed = self._perturb_once(text)
            if perturbed and perturbed.strip():
                perturbation_lls.append(self._mean_log_prob(perturbed))

        if len(perturbation_lls) < 2:
            return self._make_result(0.0, reason="insufficient perturbations")

        mean_pert = float(np.mean(perturbation_lls))
        std_pert = float(np.std(perturbation_lls))

        if std_pert < 1e-8:
            std_pert = 1.0
        score = (original_ll - mean_pert) / std_pert

        return self._make_result(
            score,
            original_ll=original_ll,
            mean_perturbation_ll=mean_pert,
            std_perturbation_ll=std_pert,
            n_perturbations_used=len(perturbation_lls),
        )
