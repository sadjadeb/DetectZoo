"""Tests for audio-modality detectors.

Audio detectors load pretrained checkpoints at construction time, so an
actual prediction requires a network download and is marked
``@pytest.mark.slow``.  The non-slow tests verify registration and
interface invariants only, skipping when the audio subpackage cannot be
imported.
"""

from __future__ import annotations

import numpy as np
import pytest

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import _REGISTRY, list_detectors, load_detector

from .conftest import require_modality


class TestAudioRegistry:
    def test_audio_detectors_registered(self):
        require_modality("audio")
        names = set(list_detectors("audio"))
        assert names, "No audio detectors registered"
        expected = {"aasist", "rawnet2", "res_tssdnet", "samo"}
        missing = expected - names
        assert not missing, f"Missing expected audio detectors: {missing}"

    def test_audio_detector_invariants(self):
        require_modality("audio")
        for name in list_detectors("audio"):
            cls = _REGISTRY[name]
            assert issubclass(cls, BaseDetector)
            assert cls.modality == "audio"

    def test_rawnet2_alias(self):
        require_modality("audio")
        from detectzoo.core.registry import _ALIASES

        assert _ALIASES.get("rawnet2_audio") == "rawnet2"


@pytest.mark.slow
class TestAASISTDetector:
    def test_predict_with_synthetic_audio(self):
        require_modality("audio")

        det = load_detector("aasist", device="cpu")
        rng = np.random.default_rng(0)
        waveform = rng.standard_normal(16000).astype(np.float32)
        result = det.predict(waveform)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert "score_spoof" in result.metadata
