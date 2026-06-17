"""Log-Likelihood baseline detector.

Reference:
    Gehrmann et al., "GLTR: Statistical Detection and
    Visualization of Generated Text", ACL 2019.

Scores text by average token log-probability under a causal LM.
Machine-generated text tends to have *higher* average log-prob (lower
perplexity) than human text, so we use the mean log-probability:
higher score → more likely AI.
"""

from __future__ import annotations

from typing import Any

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("log_likelihood")
class LogLikelihoodDetector(BaseTextDetector):
    """Detect AI text via average log-likelihood under a language model.

    Score = mean_i log p(x_i | x_{<i}).  Typically a negative number
    (since probabilities are in (0, 1)); values closer to 0 (higher)
    indicate machine-generated text.

    There is no universal threshold for log-likelihood — it depends
    on the scoring model, tokenizer and domain.  The ImBD / DetectGPT
    / Fast-DetectGPT papers report only AUROC / AUPRC (threshold-free).
    The default ``-3.0`` is a rough rule-of-thumb for ``gpt2`` on
    English news text; calibrate it per (model, dataset) via
    or passing your own ``threshold=...``.

    Parameters:
        model_name: HuggingFace model identifier (default ``"gpt2"``).
        threshold: Decision boundary on the score (default ``-3.0``).
            Texts with score >= threshold are labelled ``"ai"``.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        threshold: float = -3.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        avg_ll = self._mean_log_prob(text)
        score = avg_ll
        return self._make_result(score, avg_log_likelihood=avg_ll)
