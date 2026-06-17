"""Fast-DetectGPT — perturbation-free probability-curvature detector.

Reference:
    Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection
    of Machine-Generated Text via Conditional Probability Curvature",
    ICLR 2024.

Instead of generating explicit perturbations (expensive), Fast-DetectGPT
estimates the curvature of the log-probability surface by comparing
observed token log-probabilities under a *scoring* model against the
*expected* log-prob taken under a *reference (sampling)* model's
conditional distribution.

    discrepancy = ( sum_i log p_score(x_i | x_{<i})
                   - sum_i  E_{p_ref}[ log p_score(· | x_{<i}) ] )
                  / sqrt( sum_i Var_{p_ref}[ log p_score(· | x_{<i}) ] )

When the reference and scoring models are identical the expectation
term reduces to the model's own negative entropy, recovering the
original single-model variant.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector


@register_detector("fast_detectgpt")
class FastDetectGPTDetector(BaseTextDetector):
    """Probability Curvature Detector.

    Parameters:
        model_name: Scoring model — HuggingFace causal LM used to compute
            log p(x_i | x_{<i}). Default ``"EleutherAI/gpt-neo-2.7B"``.
        reference_model_name: Reference (sampling) model used to form the
            expectation / variance. Default ``"EleutherAI/gpt-j-6B"``.
            If equal to ``model_name``, a single model is loaded and the
            original Fast-DetectGPT analytic form (negative entropy) is
            used.
        threshold: Decision threshold on the curvature score.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-2.7B",
        reference_model_name: str = "EleutherAI/gpt-j-6B",
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.reference_model_name = reference_model_name
        self._ref_model: torch.nn.Module | None = None
        self._ref_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Reference (sampling) model — lazy loading
    # ------------------------------------------------------------------

    @property
    def reference_model(self) -> torch.nn.Module:
        if self.reference_model_name == self.model_name:
            return self.model
        if self._ref_model is None:
            self._load_reference_model()
        return self._ref_model  # type: ignore[return-value]

    @property
    def reference_tokenizer(self):
        if self.reference_model_name == self.model_name:
            return self.tokenizer
        if self._ref_tokenizer is None:
            self._load_reference_model()
        return self._ref_tokenizer

    def _load_reference_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._ref_tokenizer = AutoTokenizer.from_pretrained(self.reference_model_name)
        if self._ref_tokenizer.pad_token is None:
            self._ref_tokenizer.pad_token = self._ref_tokenizer.eos_token
        self._ref_model = AutoModelForCausalLM.from_pretrained(self.reference_model_name).to(
            self._device
        )

        self._ref_model.eval()

    # ------------------------------------------------------------------
    # Analytic sampling discrepancy
    # ------------------------------------------------------------------

    @staticmethod
    def _sampling_discrepancy_analytic(
        logits_ref: torch.Tensor,
        logits_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Eq. (4) in Bao et al. — analytic form of the sampling discrepancy."""
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1

        # Align vocab if the two models happen to have slightly different
        # vocab sizes (e.g. GPT-J vs GPT-Neo both share GPT-2 BPE but sizes
        # can differ by a handful of added tokens).
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        if labels.ndim == logits_score.ndim - 1:
            labels = labels.unsqueeze(-1)

        lprobs_score = F.log_softmax(logits_score, dim=-1)
        probs_ref = F.softmax(logits_ref, dim=-1)

        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)  # [1, T-1]
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # [1, T-1]
        var_ref = (probs_ref * lprobs_score.square()).sum(dim=-1) - mean_ref.square()

        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(
            dim=-1
        ).clamp(min=1e-10).sqrt()
        return float(discrepancy.mean())

    # ------------------------------------------------------------------
    # Per-text scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _curvature_score(self, text: str) -> tuple[float, dict]:
        # Scoring-model pass
        score_enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self._device)
        labels = score_enc["input_ids"][:, 1:]
        logits_score = self.model(**score_enc).logits[:, :-1, :]

        # Reference-model pass
        if self.reference_model_name == self.model_name:
            logits_ref = logits_score
        else:
            ref_enc = self.reference_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            ).to(self._device)
            # Both models must agree on the tokenisation of the observed
            # sequence, otherwise the scores do not refer to the same
            # token positions.
            if not torch.equal(ref_enc["input_ids"][:, 1:], labels):
                raise ValueError(
                    "Reference and scoring tokenizers produced different "
                    "token IDs for the input text; Fast-DetectGPT with two "
                    "models requires tokenizer-compatible models (e.g. "
                    "models sharing the GPT-2 BPE vocabulary)."
                )
            logits_ref = self.reference_model(**ref_enc).logits[:, :-1, :]

        score = self._sampling_discrepancy_analytic(logits_ref, logits_score, labels)

        # Lightweight diagnostics (recomputed cheaply on aligned vocab).
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            lr = logits_ref[:, :, :vocab_size]
            ls = logits_score[:, :, :vocab_size]
        else:
            lr, ls = logits_ref, logits_score
        lprobs_score = F.log_softmax(ls, dim=-1)
        probs_ref = F.softmax(lr, dim=-1)
        log_likelihood = lprobs_score.gather(
            dim=-1,
            index=labels if labels.ndim == ls.ndim else labels.unsqueeze(-1),
        ).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)

        return score, {
            "mean_log_prob": float(log_likelihood.mean()),
            "mean_cross_entropy": float(-mean_ref.mean()),
        }

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        score, meta = self._curvature_score(text)
        return self._make_result(score, **meta)
