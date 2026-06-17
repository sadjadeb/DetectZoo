#!/usr/bin/env python
"""Show how to create, register, and use a custom detector.

This example builds a trivial text detector that scores text by
average word length (longer average → more likely AI).  It
demonstrates the full extension workflow: subclass, register, load.

Usage:
    python examples/custom_detector.py
"""

from __future__ import annotations

from typing import Any

from detectzoo import list_detectors, load_detector
from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector


@register_detector("word_length")
class WordLengthDetector(BaseDetector):
    """Toy detector: average word length as an AI-ness proxy.

    AI-generated text sometimes uses longer, more formal words.
    This is just a demonstration — not a serious detector.
    """

    modality = "text"

    def __init__(self, threshold: float = 5.0, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

    def predict(self, input_data: Any) -> DetectionResult:
        text = str(input_data)
        words = text.split()
        if not words:
            return self._make_result(0.0)
        avg_len = sum(len(w) for w in words) / len(words)
        return self._make_result(avg_len, n_words=len(words), avg_word_length=avg_len)


def main() -> None:
    print("Registered detectors (including our custom one):")
    print(f"  {list_detectors()}\n")

    detector = load_detector("word_length")
    print(f"Loaded: {detector}\n")

    samples = [
        ("Casual", "I went to the store and got some milk."),
        ("Formal", "The implementation of transformer architectures has revolutionized AI."),
        (
            "Technical",
            "Backpropagation through stochastic computational graphs "
            "enables differentiable sampling.",
        ),
    ]

    for label, text in samples:
        result = detector.predict(text)
        print(f'  [{label}] "{text[:60]}…"')
        print(
            f"    score={result.score:.2f}  label={result.label}  "
            f"avg_word_length={result.metadata['avg_word_length']:.2f}\n"
        )


if __name__ == "__main__":
    main()
