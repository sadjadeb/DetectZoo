"""Base classes for all DetectZoo detectors."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Union

import torch


@dataclass
class DetectionResult:
    """Standardized output from any detector.

    Attributes:
        score: Detection score. Higher values indicate higher likelihood
            that the input is AI-generated.
        label: Binary label — ``"ai"`` or ``"human"``.
        confidence: Confidence in the assigned label (0–1).
        metadata: Arbitrary extra information from the detector.
    """

    score: float
    label: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DetectionResult(score={self.score:.4f}, label='{self.label}', "
            f"confidence={self.confidence:.4f})"
        )


class BaseDetector(ABC):
    """Abstract base class that every detector must inherit from.

    Subclasses must set ``name`` and ``modality`` and implement
    :meth:`predict`.
    """

    name: str = ""
    modality: str = ""  # "text", "image", "audio", or "video"

    def __init__(self, threshold: float = 0.5, device: str = "cpu", **kwargs: Any) -> None:
        self.threshold = threshold
        self._device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, input_data: Any) -> DetectionResult:
        """Run detection on a single input and return a :class:`DetectionResult`."""

    def predict_batch(
        self, inputs: Union[Sequence[Any], List[Any]], **kwargs: Any
    ) -> list[DetectionResult]:
        """Run detection on a batch of inputs.

        The default implementation loops over inputs; subclasses may
        override for vectorised inference.
        """
        return [self.predict(inp, **kwargs) for inp in inputs]

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str) -> "BaseDetector":
        """Move detector (and its models) to *device*."""
        self._device = torch.device(device)
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, torch.nn.Module):
                attr.to(self._device)
        return self

    def unload(self) -> None:
        """Release GPU memory held by this detector.

        Walks the detector's instance attributes and, for every
        :class:`torch.nn.Module` found, moves it to CPU and drops the
        reference so the weights can be garbage-collected.  After
        calling :meth:`unload`, the detector can still be used: the
        lazy-loading properties (``model``, ``tokenizer`` and any
        subclass-specific ones) will simply reload the weights on next
        access.

        Subclasses may override this to release additional state, but
        should call ``super().unload()``.
        """
        for name in list(self.__dict__.keys()):
            attr = self.__dict__[name]
            if isinstance(attr, torch.nn.Module):
                try:
                    attr.to("cpu")
                except Exception:
                    pass
                self.__dict__[name] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_result(self, score: float, **extra_metadata: Any) -> DetectionResult:
        """Build a :class:`DetectionResult` from a raw score."""
        label = "ai" if score >= self.threshold else "human"
        confidence = abs(score - self.threshold) / max(abs(score) + abs(self.threshold), 1e-8)
        confidence = min(confidence, 1.0)
        return DetectionResult(
            score=score,
            label=label,
            confidence=confidence,
            metadata=extra_metadata,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', modality='{self.modality}')"
