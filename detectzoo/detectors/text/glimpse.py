"""Glimpse — probability distribution estimation for zero-shot detection.

Reference:
    Bao et al., "Glimpse: Enabling White-Box Methods to Use
    Proprietary Models for Zero-Shot LLM-Generated Text Detection",
    ICLR 2025.

Glimpse estimates the full token-probability distribution from partial
observations (top-K log-probs) using a geometric distribution model,
then applies the Fast-DetectGPT conditional probability curvature
criterion on the estimated distribution.

In this implementation, when a full open-source model is available,
Glimpse still applies the geometric tail estimation to approximate
the distribution with fewer vocabulary entries (``rank_size``), which
can be useful for efficiency.  Alternatively, it can directly use the
full distribution (setting ``use_full_dist=True``).
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


def _geometric_estimate(
    top_k_probs: torch.Tensor,
    rank_size: int = 1000,
) -> torch.Tensor:
    """Estimate the full distribution from top-K probabilities using
    a geometric tail model.

    Parameters:
        top_k_probs: Sorted (descending) probabilities, shape ``[K]``.
        rank_size: Target vocabulary size for estimation.

    Returns:
        Estimated probability distribution of shape ``[rank_size]``.
    """
    K = top_k_probs.shape[0]
    if rank_size <= K:
        out = top_k_probs[:rank_size]
        return out / out.sum().clamp(min=1e-30)

    p_rest = (1.0 - top_k_probs.sum()).clamp(min=1e-30)
    p_K = top_k_probs[-1].clamp(min=1e-30)

    lam = p_rest / (p_K + p_rest)
    lam = lam.clamp(min=1e-10, max=1.0 - 1e-10)

    tail_len = rank_size - K
    exponents = torch.arange(1, tail_len + 1, device=top_k_probs.device, dtype=top_k_probs.dtype)
    tail = p_K * (lam**exponents)

    full = torch.cat([top_k_probs, tail])
    return full / full.sum().clamp(min=1e-30)


@register_detector("glimpse")
class GlimpseDetector(BaseTextDetector):
    """Glimpse detector — PDE + Fast-DetectGPT curvature.

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        top_k: Number of top probabilities to use (default ``5``).
        rank_size: Virtual vocabulary size for estimation
            (default ``1000``).
        use_full_dist: If ``True``, skip geometric estimation and use
            the full softmax distribution directly (default ``False``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        top_k: int = 5,
        rank_size: int = 1000,
        use_full_dist: bool = False,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.top_k = top_k
        self.rank_size = rank_size
        self.use_full_dist = use_full_dist

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        logits, ids = self._get_logits(text)

        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        T = shift_labels.shape[1]

        log_probs_full = F.log_softmax(shift_logits, dim=-1)

        # Observed token log-probabilities
        ll_observed = (
            log_probs_full.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        )  # [T]

        if self.use_full_dist:
            probs_full = F.softmax(shift_logits, dim=-1).squeeze(0)
            lp = log_probs_full.squeeze(0)
            mean_ref = (probs_full * lp).sum(dim=-1)
            var_ref = (probs_full * lp.square()).sum(dim=-1) - mean_ref.square()
        else:
            probs = F.softmax(shift_logits, dim=-1).squeeze(0)  # [T, V]
            mean_ref_list = []
            var_ref_list = []

            for t in range(T):
                p_t = probs[t]
                sorted_probs, _ = p_t.sort(descending=True)
                top_k_probs = sorted_probs[: self.top_k]

                est_dist = _geometric_estimate(top_k_probs, self.rank_size)
                log_est = torch.log(est_dist.clamp(min=1e-30))

                mu = (est_dist * log_est).sum()
                var = (est_dist * log_est.square()).sum() - mu.square()
                mean_ref_list.append(mu)
                var_ref_list.append(var)

            mean_ref = torch.stack(mean_ref_list)
            var_ref = torch.stack(var_ref_list)

        ll_sum = ll_observed.sum()
        mean_sum = mean_ref.sum()
        std_sum = var_ref.sum().clamp(min=1e-10).sqrt()

        score = float((ll_sum - mean_sum) / std_sum)

        return self._make_result(
            score,
            mean_log_prob=float(ll_observed.mean()),
            n_tokens=T,
        )
