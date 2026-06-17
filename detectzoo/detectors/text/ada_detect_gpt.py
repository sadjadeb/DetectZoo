"""AdaDetectGPT — adaptive probability curvature detector.

Reference:
    Zhou et al., "AdaDetectGPT: Adaptive Detection of LLM-Generated
    Text with Statistical Guarantees", NeurIPS 2025.

AdaDetectGPT generalises Fast-DetectGPT by introducing a *learned
witness function* ``w`` that non-linearly transforms log-probabilities
before computing the curvature statistic.  The witness function is
parameterised with B-spline bases and optimised to maximise detection
power (TNR lower bound).

When ``w = identity``, AdaDetectGPT degrades to vanilla Fast-DetectGPT.
Pre-trained witness coefficients (learned on GPT-4o + Claude-3.5 +
Gemini-2.5 data) are provided as defaults.
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


def _bspline_basis(
    x: torch.Tensor, start: float, end: float, n_bases: int, order: int
) -> torch.Tensor:
    """Evaluate B-spline basis functions of given order via Cox-de Boor.

    Returns tensor of shape ``(*x.shape, n_bases + order)`` containing
    each basis evaluated at every point in *x*.
    """
    n_knots = n_bases + 2 * order
    knots = torch.linspace(start, end, n_knots, device=x.device, dtype=x.dtype)
    flat = x.reshape(-1)

    # Order-0 bases: indicator functions
    bases = torch.zeros(flat.shape[0], n_knots - 1, device=x.device, dtype=x.dtype)
    for i in range(n_knots - 1):
        mask = (flat >= knots[i]) & (flat < knots[i + 1])
        bases[:, i] = mask.float()
    # Include the right endpoint in the last basis
    bases[:, -1] += (flat == knots[-1]).float()

    # Recursive evaluation up to the desired order
    for k in range(1, order + 1):
        new_bases = torch.zeros(flat.shape[0], n_knots - 1 - k, device=x.device, dtype=x.dtype)
        for i in range(n_knots - 1 - k):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]
            t1 = ((flat - knots[i]) / denom1 * bases[:, i]) if denom1 > 1e-10 else 0.0
            t2 = ((knots[i + k + 1] - flat) / denom2 * bases[:, i + 1]) if denom2 > 1e-10 else 0.0
            new_bases[:, i] = t1 + t2
        bases = new_bases

    return bases.reshape(*x.shape, -1)


def _apply_witness(
    log_probs: torch.Tensor,
    beta: torch.Tensor,
    start: float,
    end: float,
    n_bases: int,
    order: int,
    intercept: bool,
) -> torch.Tensor:
    """Apply the B-spline witness function ``w(z) = basis(z) @ beta``."""
    basis = _bspline_basis(log_probs, start, end, n_bases, order)
    if intercept:
        ones = torch.ones(*basis.shape[:-1], 1, device=basis.device, dtype=basis.dtype)
        basis = torch.cat([ones, basis], dim=-1)
    return (basis * beta).sum(dim=-1)


# Pre-trained coefficients from the official repo (GPT-4o + Claude-3.5 + Gemini-2.5)
_DEFAULT_BETA = [0.0, -0.011333, -0.037667, -0.056667, -0.281667, -0.592, 0.157833, 0.727333]


@register_detector("adadetectgpt")
class AdaDetectGPTDetector(BaseTextDetector):
    """AdaDetectGPT — adaptive probability curvature detector.

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        beta: Witness function coefficients (B-spline weights).
            If ``None``, uses the pre-trained default.
        n_bases: Number of B-spline basis functions (default ``7``).
        spline_order: B-spline order (default ``2``).
        spline_start: Left bound of spline domain (default ``-32``).
        spline_end: Right bound of spline domain (default ``0``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        beta: list[float] | None = None,
        n_bases: int = 7,
        spline_order: int = 2,
        spline_start: float = -32.0,
        spline_end: float = 0.0,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.beta_list = beta if beta is not None else list(_DEFAULT_BETA)
        self.n_bases = n_bases
        self.spline_order = spline_order
        self.spline_start = spline_start
        self.spline_end = spline_end

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        logits, ids = self._get_logits(text)

        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)

        beta = torch.tensor(self.beta_list, device=self._device, dtype=log_probs.dtype)

        # Apply witness function to log-probs
        w_lp = _apply_witness(
            log_probs,
            beta,
            self.spline_start,
            self.spline_end,
            self.n_bases,
            self.spline_order,
            intercept=True,
        )

        # Gather observed witness-transformed log-prob
        if shift_labels.ndim == w_lp.ndim - 1:
            shift_labels_exp = shift_labels.unsqueeze(-1)
        else:
            shift_labels_exp = shift_labels

        # w_lp has shape [1, T-1, V] — gather the observed token's value
        ll = w_lp.gather(dim=-1, index=shift_labels_exp).squeeze(-1)

        # Expected and variance under the model distribution
        mean_ref = (probs * w_lp).sum(dim=-1)
        var_ref = (probs * w_lp.square()).sum(dim=-1) - mean_ref.square()

        ll_sum = ll.sum(dim=-1)
        mean_sum = mean_ref.sum(dim=-1)
        std_sum = var_ref.sum(dim=-1).clamp(min=1e-10).sqrt()

        score = float((ll_sum - mean_sum) / std_sum)

        return self._make_result(
            score,
            mean_log_prob=float(ll.mean()),
            sum_variance=float(var_ref.sum()),
        )
