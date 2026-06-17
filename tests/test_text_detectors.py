"""Tests for text-modality detectors.

Tests marked ``@pytest.mark.slow`` download HuggingFace models (``gpt2``)
and are skipped by default.  Run with ``pytest -m slow`` to include them.
All slow tests pin the (tiny) ``gpt2`` model on CPU to stay practical —
detectors that use a separate reference model default to multi-billion
parameter models, so those must be overridden explicitly.
"""

from __future__ import annotations

import pytest

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import _REGISTRY, list_detectors

_SAMPLE = "The quick brown fox jumps over the lazy dog."


class TestTextRegistryQuick:
    """Lightweight checks that need no model download."""

    def test_zero_shot_detectors_registered(self):
        names = set(list_detectors("text"))
        for n in ("log_likelihood", "log_rank", "entropy", "fast_detectgpt"):
            assert n in names

    def test_classes_are_text_detectors(self):
        for n in ("log_likelihood", "log_rank", "entropy"):
            assert _REGISTRY[n].modality == "text"


@pytest.mark.slow
class TestLogLikelihoodDetector:
    def test_predict(self):
        from detectzoo.detectors.text.log_likelihood import LogLikelihoodDetector

        det = LogLikelihoodDetector(model_name="gpt2", device="cpu")
        result = det.predict(_SAMPLE)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.score, float)


@pytest.mark.slow
class TestLogRankDetector:
    def test_predict(self):
        from detectzoo.detectors.text.log_rank import LogRankDetector

        det = LogRankDetector(model_name="gpt2", device="cpu")
        result = det.predict(_SAMPLE)
        assert isinstance(result, DetectionResult)


@pytest.mark.slow
class TestEntropyDetector:
    def test_predict(self):
        from detectzoo.detectors.text.entropy import EntropyDetector

        det = EntropyDetector(model_name="gpt2", device="cpu")
        result = det.predict(_SAMPLE)
        assert isinstance(result, DetectionResult)


@pytest.mark.slow
class TestFastDetectGPT:
    def test_predict_single_model(self):
        from detectzoo.detectors.text.fast_detect_gpt import FastDetectGPTDetector

        # Override BOTH models to gpt2 — the default reference model is
        # gpt-j-6B which is impractical to download for a test.
        det = FastDetectGPTDetector(
            model_name="gpt2",
            reference_model_name="gpt2",
            device="cpu",
        )
        result = det.predict(_SAMPLE)
        assert isinstance(result, DetectionResult)
        assert "mean_log_prob" in result.metadata
