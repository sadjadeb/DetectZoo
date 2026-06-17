"""Tests for image-modality detectors.

The current image detectors all load pretrained checkpoints at
construction time (see e.g. ``CNNSpotDetector``), so running an actual
prediction requires a network download and is marked ``@pytest.mark.slow``.
The non-slow tests verify registration and interface invariants only and
skip automatically when the image subpackage cannot be imported (missing
optional deps such as ``diffusers`` / ``timm`` / ``open_clip``).
"""

from __future__ import annotations

import numpy as np
import pytest

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import _REGISTRY, list_detectors, load_detector

from .conftest import require_modality


class TestImageRegistry:
    def test_image_detectors_registered(self):
        require_modality("image")
        names = set(list_detectors("image"))
        assert names, "No image detectors registered"
        expected = {"cnnspot", "univfd", "aide", "freqnet", "patchcraft"}
        missing = expected - names
        assert not missing, f"Missing expected image detectors: {missing}"

    def test_image_detector_invariants(self):
        require_modality("image")
        for name in list_detectors("image"):
            cls = _REGISTRY[name]
            assert issubclass(cls, BaseDetector)
            assert cls.modality == "image"

    def test_cnn_spot_alias(self):
        require_modality("image")
        from detectzoo.core.registry import _ALIASES

        assert _ALIASES.get("cnn_spot") == "cnnspot"


@pytest.mark.slow
class TestCNNSpotDetector:
    def test_predict_on_random_image(self):
        require_modality("image")
        from PIL import Image

        det = load_detector("cnnspot", device="cpu")
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        result = det.predict(img)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert result.label in ("ai", "human")
