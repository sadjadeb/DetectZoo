"""Tests for video-modality detectors.

The non-slow tests cover registration and the ``BaseVideoDetector`` helpers
(input normalisation, frame sampling, score aggregation) with a model-free
dummy detector.  Running WaveRep requires downloading its checkpoint and is
marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import _ALIASES, _REGISTRY, list_detectors, load_detector

from .conftest import require_modality


def _dummy_detector_cls():
    from detectzoo.detectors.video.base import BaseVideoDetector

    class _MeanBrightness(BaseVideoDetector):
        name = "dummy_video"

        def _load_model(self) -> None:
            self._model = torch.nn.Identity()

        def predict(self, input_data):
            frames = self._sample_frames(self._normalise_input(input_data))
            scores = self.model(frames.float().mean(dim=(1, 2, 3)) / 255.0)
            return self._make_result(self._aggregate_scores(scores), num_frames=frames.shape[0])

    return _MeanBrightness


class TestVideoRegistry:
    def test_video_detectors_registered(self):
        require_modality("video")
        assert "waverep" in set(list_detectors("video"))

    def test_video_detector_invariants(self):
        require_modality("video")
        for name in list_detectors("video"):
            cls = _REGISTRY[name]
            assert issubclass(cls, BaseDetector)
            assert cls.modality == "video"

    def test_waverep_alias(self):
        require_modality("video")
        assert _ALIASES.get("wave_rep") == "waverep"

    def test_waverep_unknown_variant_raises(self):
        require_modality("video")
        with pytest.raises(ValueError, match="variant"):
            load_detector("waverep", variant="G7")


class TestBaseVideoDetector:
    def test_normalise_thwc_and_tchw(self):
        require_modality("video")
        cls = _dummy_detector_cls()
        thwc = np.zeros((5, 32, 24, 3), dtype=np.uint8)
        assert cls._normalise_input(thwc).shape == (5, 3, 32, 24)
        tchw = torch.zeros((5, 3, 32, 24), dtype=torch.uint8)
        assert cls._normalise_input(tchw).shape == (5, 3, 32, 24)

    def test_normalise_list_of_pil(self):
        require_modality("video")
        from PIL import Image

        cls = _dummy_detector_cls()
        frames = [Image.new("RGB", (16, 8))] * 4
        assert cls._normalise_input(frames).shape == (4, 3, 8, 16)

    def test_normalise_rejects_bad_input(self):
        require_modality("video")
        cls = _dummy_detector_cls()
        with pytest.raises(TypeError):
            cls._normalise_input(42)
        with pytest.raises(ValueError):
            cls._normalise_input(np.zeros((32, 32, 3), dtype=np.uint8))
        with pytest.raises(FileNotFoundError):
            cls._normalise_input("does_not_exist.mp4")

    def test_frame_sampling(self):
        require_modality("video")
        cls = _dummy_detector_cls()
        assert cls(num_frames=4)._sample_frame_indices(10).tolist() == [0, 3, 6, 9]
        assert cls(num_frames=4, sampling="first")._sample_frame_indices(10).tolist() == [
            0,
            1,
            2,
            3,
        ]
        assert cls(num_frames=None)._sample_frame_indices(3).tolist() == [0, 1, 2]
        assert cls(num_frames=8)._sample_frame_indices(3).tolist() == [0, 1, 2]
        with pytest.raises(ValueError):
            cls(sampling="random")._sample_frame_indices(10)

    def test_aggregation(self):
        require_modality("video")
        cls = _dummy_detector_cls()
        scores = [0.1, 0.2, 0.9]
        assert cls(aggregation="mean")._aggregate_scores(scores) == pytest.approx(0.4)
        assert cls(aggregation="max")._aggregate_scores(scores) == pytest.approx(0.9)
        assert cls(aggregation="median")._aggregate_scores(scores) == pytest.approx(0.2)
        with pytest.raises(ValueError):
            cls(aggregation="sum")._aggregate_scores(scores)

    def test_predict_from_video_file(self, tmp_path):
        require_modality("video")
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        for i in range(12):
            writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
        writer.release()

        result = _dummy_detector_cls()(num_frames=6).predict(path)
        assert isinstance(result, DetectionResult)
        assert result.metadata["num_frames"] == 6
        assert result.label in ("ai", "human")


@pytest.mark.slow
class TestWaveRepDetector:
    def test_predict_on_random_frames(self):
        require_modality("video")
        pytest.importorskip("timm")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        det = load_detector("waverep", device=device, num_frames=4)
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (6, 256, 320, 3), dtype=np.uint8)
        result = det.predict(frames)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert result.label in ("ai", "human")
        assert result.metadata["num_frames"] == 4
        assert len(result.metadata["frame_logits"]) == 4
