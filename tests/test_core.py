"""Tests for the core infrastructure (registry, base classes, results)."""

from __future__ import annotations

import pytest
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import (
    _ALIASES,
    _REGISTRY,
    list_detectors,
    load_detector,
)

VALID_MODALITIES = {"text", "image", "audio"}


class TestDetectionResult:
    def test_fields(self):
        r = DetectionResult(score=0.75, label="ai", confidence=0.6)
        assert r.score == 0.75
        assert r.label == "ai"
        assert r.confidence == 0.6
        assert r.metadata == {}

    def test_default_confidence_and_metadata(self):
        r = DetectionResult(score=0.5, label="human")
        assert r.confidence == 0.0
        assert r.metadata == {}

    def test_metadata_is_independent_per_instance(self):
        a = DetectionResult(score=0.1, label="human")
        b = DetectionResult(score=0.9, label="ai")
        a.metadata["k"] = 1
        assert b.metadata == {}

    def test_repr(self):
        r = DetectionResult(score=1.0, label="human", confidence=0.5)
        assert "DetectionResult" in repr(r)
        assert "1.0000" in repr(r)


class TestRegistry:
    def test_detectors_registered(self):
        names = list_detectors()
        assert len(names) >= 24, f"Expected >=24 detectors, got {len(names)}: {names}"

    def test_registry_invariants(self):
        """Every registered class must expose its registry name and a valid modality."""
        for name, cls in _REGISTRY.items():
            assert issubclass(cls, BaseDetector), f"{name} is not a BaseDetector"
            assert cls.name == name, f"{name}: cls.name={cls.name!r} mismatches key"
            assert cls.modality in VALID_MODALITIES, f"{name}: bad modality {cls.modality!r}"

    def test_text_detectors_present(self):
        # Text detectors have no heavy optional deps, so they always load.
        text = set(list_detectors("text"))
        assert len(text) >= 18, f"Expected >=18 text detectors, got {sorted(text)}"
        # A representative, stable subset that should always exist.
        expected = {
            "log_likelihood",
            "log_rank",
            "rank",
            "entropy",
            "detectgpt",
            "fast_detectgpt",
            "binoculars",
            "lrr",
            "npr",
            "dna_gpt",
            "revise_detect",
            "imbd",
            "lastde",
            "lastde_pp",
            "radar",
            "text_fluoroscopy",
            "coco",
            "roberta_base",
            "roberta_large",
        }
        missing = expected - text
        assert not missing, f"Missing expected text detectors: {missing}"

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            load_detector("nonexistent_detector_xyz")

    def test_alias_resolution(self):
        # roberta aliases are pure-text and resolve without any download.
        assert _ALIASES.get("roberta_openai_base") == "roberta_base"
        assert _ALIASES.get("roberta_openai_large") == "roberta_large"
        # Every alias must point at a real, registered detector.
        for alias, target in _ALIASES.items():
            assert target in _REGISTRY, f"Alias {alias!r} -> unknown target {target!r}"

    def test_list_by_modality_filters(self):
        for name in list_detectors("text"):
            assert _REGISTRY[name].modality == "text"

    def test_list_detectors_sorted(self):
        names = list_detectors()
        assert names == sorted(names)


class TestBaseDetector:
    def _dummy(self, score: float, threshold: float = 0.5):
        class _Dummy(BaseDetector):
            name = "dummy_core"
            modality = "text"

            def predict(self, input_data):
                return self._make_result(score)

        return _Dummy(threshold=threshold)

    def test_make_result_above_threshold(self):
        r = self._dummy(0.8).predict("hello")
        assert r.label == "ai"
        assert r.score == 0.8

    def test_make_result_at_threshold_is_ai(self):
        # label uses score >= threshold.
        r = self._dummy(0.5, threshold=0.5).predict("x")
        assert r.label == "ai"

    def test_make_result_below_threshold(self):
        r = self._dummy(0.2).predict("hello")
        assert r.label == "human"

    def test_confidence_in_unit_interval(self):
        r = self._dummy(0.8).predict("hello")
        assert 0.0 <= r.confidence <= 1.0
        assert r.confidence > 0.0

    def test_make_result_passes_metadata(self):
        class _Dummy(BaseDetector):
            name = "dummy_meta"
            modality = "text"

            def predict(self, input_data):
                return self._make_result(0.9, extra="info", n=3)

        r = _Dummy().predict("x")
        assert r.metadata == {"extra": "info", "n": 3}

    def test_predict_batch(self):
        class _LenDummy(BaseDetector):
            name = "dummy_len"
            modality = "text"

            def predict(self, input_data):
                return self._make_result(float(len(str(input_data))) / 100.0)

        results = _LenDummy().predict_batch(["a", "bb", "ccc"])
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_device_property_and_to(self):
        d = self._dummy(0.5)
        assert d.device == torch.device("cpu")
        d.to("cpu")
        assert d.device == torch.device("cpu")

    def test_unload_clears_modules(self):
        class _ModelDummy(BaseDetector):
            name = "dummy_model"
            modality = "text"

            def __init__(self, **kw):
                super().__init__(**kw)
                self.net = torch.nn.Linear(2, 2)

            def predict(self, input_data):
                return self._make_result(0.5)

        d = _ModelDummy()
        assert isinstance(d.net, torch.nn.Module)
        d.unload()
        assert d.net is None

    def test_repr(self):
        assert "dummy_core" in repr(self._dummy(0.5))
