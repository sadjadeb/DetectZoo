"""Tests for audio-modality detectors.

Audio detectors load pretrained checkpoints at construction time, so an
actual prediction requires a network download and is marked
``@pytest.mark.slow``.  The non-slow tests verify registration and
interface invariants only, skipping when the audio subpackage cannot be
imported.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import _ALIASES, _REGISTRY, list_detectors, load_detector

from .conftest import require_modality

_NEW_AUDIO_DETECTORS = ("raw_pc_darts", "safeear", "tcm", "xlsr_mamba")


class TestAudioRegistry:
    def test_audio_detectors_registered(self):
        require_modality("audio")
        names = set(list_detectors("audio"))
        assert names, "No audio detectors registered"
        expected = {"aasist", "rawnet2", "res_tssdnet", "samo"}
        missing = expected - names
        assert not missing, f"Missing expected audio detectors: {missing}"

    def test_new_audio_detectors_registered(self):
        require_modality("audio")
        names = set(list_detectors("audio"))
        missing = set(_NEW_AUDIO_DETECTORS) - names
        assert not missing, f"Missing new audio detectors: {missing}"

    def test_new_audio_detector_aliases(self):
        require_modality("audio")
        assert _ALIASES.get("xlsr-mamba") == "xlsr_mamba"
        assert _ALIASES.get("raw-pc-darts") == "raw_pc_darts"
        assert _ALIASES.get("safe_ear") == "safeear"
        assert _ALIASES.get("tcm_add") == "tcm"

    def test_audio_detector_invariants(self):
        require_modality("audio")
        for name in list_detectors("audio"):
            cls = _REGISTRY[name]
            assert issubclass(cls, BaseDetector)
            assert cls.modality == "audio"

    def test_rawnet2_alias(self):
        require_modality("audio")
        assert _ALIASES.get("rawnet2_audio") == "rawnet2"

    def test_tcm_requires_checkpoint_when_no_mirror(self):
        require_modality("audio")
        det = load_detector("tcm", device="cpu")
        rng = np.random.default_rng(0)
        waveform = rng.standard_normal(16000).astype(np.float32)
        with pytest.raises(RuntimeError, match="no HuggingFace mirror configured"):
            det.predict(waveform)


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


@pytest.mark.slow
class TestRawPCDartsDetector:
    def test_predict_with_synthetic_audio(self):
        require_modality("audio")

        det = load_detector("raw_pc_darts", device="cpu")
        rng = np.random.default_rng(0)
        waveform = rng.standard_normal(16000).astype(np.float32)
        result = det.predict(waveform)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert "cos_bonafide" in result.metadata


@pytest.mark.slow
class TestSafeEarDetector:
    def test_predict_with_synthetic_audio(self):
        require_modality("audio")

        det = load_detector("safeear", device="cpu")
        rng = np.random.default_rng(0)
        waveform = rng.standard_normal(16000).astype(np.float32)
        result = det.predict(waveform)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert "prob_bonafide" in result.metadata


@pytest.mark.slow
class TestXLSRMambaDetector:
    def test_checkpoint_loads(self):
        require_modality("audio")
        from detectzoo.datasets._download import get_cache_dir
        from detectzoo.detectors.audio.xlsr_mamba._loader import build_xlsr_mamba_model

        cache = get_cache_dir("xlsr_mamba_test")
        model, report = build_xlsr_mamba_model("AustinXiao/XLSR-Mamba-LA", cache)
        assert report["matched"] > 0
        assert report["unexpected"] == 0
        assert not report.get("silently_random")
        del model

    @pytest.mark.skipif(
        importlib.util.find_spec("mamba_ssm") is None,
        reason="Full forward pass requires optional mamba-ssm (CUDA); load test covers weights.",
    )
    def test_predict_with_synthetic_audio(self):
        require_modality("audio")

        det = load_detector("xlsr_mamba", device="cpu", variant="LA")
        rng = np.random.default_rng(0)
        waveform = rng.standard_normal(66800).astype(np.float32)
        result = det.predict(waveform)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert "logit_spoof" in result.metadata
