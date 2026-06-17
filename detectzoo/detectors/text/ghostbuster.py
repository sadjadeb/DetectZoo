"""Ghostbuster — structured-feature LLM-text detector.

Reference:
    Verma et al., "Ghostbuster: Detecting Text Ghostwritten by Large
    Language Models", NAACL 2024.

Ghostbuster uses *multiple weaker language models* to compute per-token
probability vectors, then builds structured features from these
vectors via combinatorial search.  The features are classified with
logistic regression.

The original implementation uses GPT-3 ada/davinci (now deprecated)
plus n-gram models.  This implementation substitutes with locally
available causal LMs: a small model (default ``gpt2``) and a larger
model (default ``gpt2-large``), combined with simple n-gram baselines.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


def _unigram_probs(tokens: List[str]) -> np.ndarray:
    """Cheap unigram probability proxy based on token frequency."""
    from collections import Counter

    if not tokens:
        return np.array([1.0])
    counts = Counter(tokens)
    total = len(tokens)
    return np.array([counts[t] / total for t in tokens], dtype=np.float64)


@register_detector("ghostbuster")
class GhostbusterDetector(BaseTextDetector):
    """Ghostbuster detector — multi-model probability features.

    Uses a small and a large causal LM to extract per-token probability
    vectors, computes structured features, and outputs a composite
    detection score.

    Parameters:
        small_model: Smaller scoring LM (default ``"gpt2"``).
        large_model: Larger scoring LM (default ``"gpt2-large"``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        small_model: str = "gpt2",
        large_model: str = "gpt2-large",
        threshold: float = 0.0,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=small_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.small_model_name = small_model
        self.large_model_name = large_model
        self._large_model: torch.nn.Module | None = None
        self._large_tokenizer: Any = None

    def _load_large_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Ghostbuster large model '%s' …", self.large_model_name)
        self._large_tokenizer = AutoTokenizer.from_pretrained(self.large_model_name)
        if self._large_tokenizer.pad_token is None:
            self._large_tokenizer.pad_token = self._large_tokenizer.eos_token
        self._large_model = AutoModelForCausalLM.from_pretrained(self.large_model_name).to(
            self._device
        )
        self._large_model.eval()

    @property
    def large_model(self) -> torch.nn.Module:
        if self._large_model is None:
            self._load_large_model()
        return self._large_model  # type: ignore[return-value]

    @property
    def large_tokenizer(self):
        if self._large_tokenizer is None:
            self._load_large_model()
        return self._large_tokenizer

    @torch.no_grad()
    def _token_probs(self, text: str, use_large: bool = False) -> np.ndarray:
        """Get per-token probabilities from one of the two models."""
        model = self.large_model if use_large else self.model
        tok = self.large_tokenizer if use_large else self.tokenizer

        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        logits = model(**enc).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        ids = enc["input_ids"][:, 1:]
        token_lp = log_probs.gather(2, ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
        return torch.exp(token_lp).cpu().numpy()

    def _build_features(self, text: str) -> np.ndarray:
        """Extract Ghostbuster-style structured features."""
        words = text.lower().split()
        unigram = _unigram_probs(words)

        small_probs = self._token_probs(text, use_large=False)
        large_probs = self._token_probs(text, use_large=True)

        min_len = min(len(unigram), len(small_probs), len(large_probs))
        if min_len < 2:
            return np.zeros(15)

        u = unigram[:min_len]
        s = small_probs[:min_len]
        l_ = large_probs[:min_len]

        features = []
        # Variance of indicator: unigram > large
        features.append(np.var((u > l_).astype(float)))
        # Mean of (small - large)
        features.append(np.mean(s - l_))
        # Variance of small
        features.append(np.var(s))
        # Variance of large
        features.append(np.var(l_))
        # L2 of (small - large)
        features.append(np.linalg.norm(s - l_))
        # Mean of indicator: unigram > small
        features.append(np.mean((u > s).astype(float)))
        # Mean large
        features.append(np.mean(l_))
        # Mean small
        features.append(np.mean(s))
        # Std of large
        features.append(np.std(l_))
        # Max large
        features.append(np.max(l_))

        # Handcrafted features from the paper
        diff = l_ - s
        sorted_diff = np.sort(diff)[::-1]
        features.append(np.mean(sorted_diff[: min(25, len(sorted_diff))]))
        if len(sorted_diff) > 25:
            features.append(np.mean(sorted_diff[25:50]))
        else:
            features.append(0.0)

        # Outlier count: tokens where large prob > 0.95
        features.append(float(np.sum(l_ > 0.95)))
        # Mean of top-25 large probs
        sorted_l = np.sort(l_)[::-1]
        features.append(np.mean(sorted_l[: min(25, len(sorted_l))]))
        if len(sorted_l) > 25:
            features.append(np.mean(sorted_l[25:50]))
        else:
            features.append(0.0)

        return np.array(features)

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        features = self._build_features(text)

        # Composite score: combine feature signals
        # Higher large-model prob → more likely AI
        # Use mean large prob - variance as primary score
        score = float(features[6] - features[3]) if len(features) > 6 else 0.0

        return self._make_result(
            score,
            mean_large_prob=float(features[6]) if len(features) > 6 else 0.0,
            var_large_prob=float(features[3]) if len(features) > 3 else 0.0,
            feature_dim=len(features),
        )
