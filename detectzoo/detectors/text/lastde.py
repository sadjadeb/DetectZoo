"""Lastde and Lastde++ — token probability sequence mining detectors.

Reference:
    Xu et al., "Training-free LLM-generated Text Detection by Mining
    Token Probability Sequences", ICLR 2025.

**Lastde** (sample-based) computes Multiscale Distribution Entropy
(MDE) of the per-token log-probability sequence.  MDE captures the
regularity of the probability landscape: machine text tends to have
more regular (lower-entropy) patterns, yielding a lower MDE.  The
final score is ``mean(log_prob) / MDE``.

**Lastde++** (distribution-based) extends the idea with sampling from
the model's conditional distribution (like Fast-DetectGPT), computing
the Lastde score for both observed and sampled tokens and reporting
the normalised discrepancy.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Multiscale Distribution Entropy (MDE)
# ------------------------------------------------------------------


def _distribution_entropy(probs: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Compute distribution entropy: DE = -1/log(n_bins) * sum(p*log(p))."""
    log_probs = torch.log(probs + 1e-30)
    de = -1.0 / torch.log(torch.tensor(float(n_bins), device=probs.device))
    de = de * (probs * log_probs).nansum(dim=0)
    return de


def _compute_de(log_likelihood: torch.Tensor, embed_size: int, n_bins: int) -> torch.Tensor:
    """Compute Distribution Entropy of orbits built from *log_likelihood*.

    Parameters:
        log_likelihood: shape ``[1, T, S]`` — per-token log-probs (S samples).
        embed_size: orbit embedding dimension.
        n_bins: number of histogram bins (epsilon).

    Returns:
        DE values of shape ``[S]``.
    """
    seq = log_likelihood.squeeze(0)  # [T, S]
    T = seq.shape[0]
    if T < embed_size + 2:
        return torch.zeros(seq.shape[1], device=seq.device)

    orbits = seq.unfold(0, embed_size, 1)  # [T - embed_size + 1, S, embed_size]

    cos_sim = torch.nn.functional.cosine_similarity(
        orbits[:-1], orbits[1:], dim=-1
    )  # [T - embed_size, S]

    de_values = []
    for s_idx in range(cos_sim.shape[1]):
        col = cos_sim[:, s_idx]
        hist = torch.histc(col.float(), bins=n_bins, min=-1.0, max=1.0)
        probs = hist / hist.sum().clamp(min=1e-30)
        de_val = _distribution_entropy(probs, n_bins)
        de_values.append(de_val)
    return torch.stack(de_values)


def _compute_mde(
    log_likelihood: torch.Tensor,
    embed_size: int,
    n_bins: int,
    tau_prime: int,
) -> torch.Tensor:
    """Multiscale Distribution Entropy: std of DE across time scales 1..tau_prime.

    Parameters:
        log_likelihood: shape ``[1, T, S]``.
        embed_size: orbit dimension.
        n_bins: histogram bins.
        tau_prime: max time scale.

    Returns:
        MDE values of shape ``[S]``.
    """
    n_samples = log_likelihood.shape[2]
    de_list = []
    for tau in range(1, tau_prime + 1):
        seq = log_likelihood.squeeze(0)  # [T, S]
        if tau > 1:
            if seq.shape[0] < tau:
                break
            windows = seq.unfold(0, tau, 1)  # [T - tau + 1, S, tau]
            seq = windows.mean(dim=-1)  # [T - tau + 1, S]
        # _compute_de needs at least embed_size + 2 time steps
        if seq.shape[0] < embed_size + 2:
            break
        de = _compute_de(seq.unsqueeze(0), embed_size, n_bins)
        de_list.append(de)
    if len(de_list) < 2:
        return torch.zeros(n_samples, device=log_likelihood.device)
    de_stack = torch.stack(de_list, dim=0)  # [<=tau_prime, S]
    return de_stack.std(dim=0)  # [S]


# ------------------------------------------------------------------
# Lastde detector
# ------------------------------------------------------------------


@register_detector("lastde")
class LastdeDetector(BaseTextDetector):
    """Lastde detector — MDE of token log-probability sequence.

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        embed_size: Orbit embedding dimension (default ``3``).
        epsilon_scale: Multiplier for number of histogram bins
            (``n_bins = epsilon_scale * T``). Default ``10``.
        tau_prime: Maximum time scale (default ``5``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        embed_size: int = 3,
        epsilon_scale: float = 10.0,
        tau_prime: int = 5,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.embed_size = embed_size
        self.epsilon_scale = epsilon_scale
        self.tau_prime = tau_prime

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1))  # [1, T, 1]

        T = token_lp.shape[1]
        n_bins = max(2, int(self.epsilon_scale * T))
        mean_lp = float(token_lp.mean())

        mde = _compute_mde(token_lp, self.embed_size, n_bins, self.tau_prime)
        mde_val = float(mde[0])

        if abs(mde_val) < 1e-10:
            score = mean_lp
        else:
            score = mean_lp / mde_val

        return self._make_result(
            score,
            mean_log_prob=mean_lp,
            mde=mde_val,
            n_tokens=T,
        )


# ------------------------------------------------------------------
# Lastde++ detector
# ------------------------------------------------------------------


@register_detector("lastde_pp")
class LastdePPDetector(BaseTextDetector):
    """Lastde++ detector — distribution-based Lastde with sampling.

    Extends Lastde with the Fast-DetectGPT sampling strategy: draw
    alternative tokens from the model's own distribution and compare
    the Lastde score of the observed sequence to those of the samples.

    Parameters:
        model_name: HuggingFace causal LM (default ``"gpt2"``).
        embed_size: Orbit embedding dimension (default ``4``).
        epsilon_scale: Bins multiplier (default ``8``).
        tau_prime: Max time scale (default ``10``).
        n_samples: Number of sampled alternatives per position.
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        embed_size: int = 4,
        epsilon_scale: float = 8.0,
        tau_prime: int = 10,
        n_samples: int = 100,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.embed_size = embed_size
        self.epsilon_scale = epsilon_scale
        self.tau_prime = tau_prime
        self.n_samples = n_samples

    def _lastde_score(self, log_likelihood: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Compute Lastde = mean(log_prob) / MDE for each sample column."""
        mean_lp = log_likelihood.mean(dim=1)  # [1, S]
        mde = _compute_mde(log_likelihood, self.embed_size, n_bins, self.tau_prime)
        mde = mde.clamp(min=1e-10)
        return mean_lp.squeeze(0) / mde  # [S]

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        logits, ids = self._get_logits(text)
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        T = shift_labels.shape[1]
        n_bins = max(2, int(self.epsilon_scale * T))

        # Log-likelihood of observed tokens — [1, T, 1]
        ll_x = log_probs.gather(2, shift_labels.unsqueeze(-1))

        # Sample alternative tokens — [1, T, n_samples]
        dist = torch.distributions.Categorical(logits=shift_logits)
        samples = dist.sample([self.n_samples]).permute(1, 2, 0)  # [1, T, n_samples]
        ll_tilde = log_probs.gather(2, samples)  # [1, T, n_samples]

        lastde_x = self._lastde_score(ll_x, n_bins)  # [1]
        lastde_tilde = self._lastde_score(ll_tilde, n_bins)  # [n_samples]

        mu = float(lastde_tilde.mean())
        sigma = float(lastde_tilde.std())
        if sigma < 1e-10:
            score = 0.0
        else:
            score = (float(lastde_x[0]) - mu) / sigma

        return self._make_result(
            score,
            lastde_original=float(lastde_x[0]),
            lastde_sampled_mean=mu,
            lastde_sampled_std=sigma,
        )
