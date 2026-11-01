"""Tests for video-modality detectors.

The non-slow tests cover registration, the ``BaseVideoDetector`` helpers
(input normalisation, frame sampling, score aggregation) with a model-free
dummy detector, and BusterX's model-free helpers (frame sampling, resizing,
answer parsing).  Running WaveRep or BusterX requires downloading checkpoints
(BusterX's is gated) and is marked ``@pytest.mark.slow``.
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


class TestBusterXHelpers:
    def test_registered_with_aliases(self):
        require_modality("video")
        assert "busterx" in set(list_detectors("video"))
        assert _ALIASES.get("busterx_plusplus") == "busterx"
        assert _ALIASES.get("busterx++") == "busterx"

    def test_invalid_sample_fps(self):
        require_modality("video")
        with pytest.raises(ValueError, match="sample_fps"):
            load_detector("busterx", sample_fps=0)

    def test_frame_indices_follow_upstream(self):
        require_modality("video")
        from detectzoo.detectors.video.busterx import _frame_indices

        # 3 s at 30 fps sampled at 2 fps -> t = 0, 0.5, ..., 2.5
        assert _frame_indices(90, 30.0, 2.0) == [0, 15, 30, 45, 60, 75]
        assert _frame_indices(1, 24.0, 2.0) == [0]  # shorter than one step
        assert _frame_indices(0, 24.0, 2.0) == []

    def test_smart_resize_matches_64_tokens(self):
        require_modality("video")
        from detectzoo.detectors.video.busterx import _smart_resize

        pixels = 64 * 32**2
        assert _smart_resize(1024, 1024, 32, pixels) == (256, 256)
        h, w = _smart_resize(720, 1280, 32, pixels)
        assert h % 32 == 0 and w % 32 == 0 and h * w <= pixels and w > h

    @pytest.mark.parametrize(
        "response, expected",
        [
            ("<think>\nreasoning \\boxed{A}</think>\n\nSo \\boxed{B}", "B"),
            ("<think>x</think>\n\\boxed{A}", "A"),
            ("<think>x</think>\\boxed{\\text{B}}", "B"),
            ("<think>x</think>\\boxed{B) fake}", "B"),
            ("<think>x</think>\\boxed{real}", "A"),
            ("<think>never finished", None),
            ("<think>x</think> The answer is B.", None),
        ],
    )
    def test_parse_answer(self, response, expected):
        require_modality("video")
        from detectzoo.detectors.video.busterx import _parse_answer

        assert _parse_answer(response) == expected

    def test_answer_step_finds_final_boxed_letter(self):
        require_modality("video")
        from detectzoo.detectors.video.busterx import BusterXDetector

        vocab = ["\\", "boxed", "{", "A", "B", "}", "</think>", "x"]

        class _Tok:
            def convert_tokens_to_ids(self, toks):
                return (
                    [vocab.index(t) for t in toks] if isinstance(toks, list) else vocab.index(toks)
                )

            def decode(self, ids):
                return "".join(vocab[i] for i in ids)

        class _Proc:
            tokenizer = _Tok()

        det = BusterXDetector()
        det._processor = _Proc()
        ids = lambda *toks: [vocab.index(t) for t in toks]  # noqa: E731
        # boxed{A} inside the reasoning is ignored; the answer after </think> counts.
        gen = ids("\\", "boxed", "{", "A", "}", "</think>", "x", "\\", "boxed", "{", "B", "}")
        assert det._answer_step(gen) == 10
        assert det._answer_step(ids("x", "A", "x")) is None

    def test_frames_from_array_get_sample_fps_timestamps(self):
        require_modality("video")
        from detectzoo.detectors.video.busterx import BusterXDetector

        frames, ts = BusterXDetector(sample_fps=2.0)._load_frames(
            np.zeros((3, 8, 8, 3), dtype=np.uint8)
        )
        assert len(frames) == 3 and frames[0].shape == (8, 8, 3)
        assert ts == [0.0, 0.5, 1.0]


@pytest.mark.slow
class TestBusterXDetector:
    def test_predict_on_synthetic_video(self, tmp_path):
        require_modality("video")
        if not torch.cuda.is_available():
            pytest.skip("BusterX (4B MLLM) needs a GPU")
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (256, 256))
        rng = np.random.default_rng(0)
        for _ in range(10):
            writer.write(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        writer.release()

        det = load_detector("busterx", device="cuda", max_new_tokens=1024)
        result = det.predict(path)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["num_frames"] == 2  # 1 s at 2 fps
        assert result.metadata["response"].startswith("<think>")


class TestSPLITHelpers:
    def test_registered_and_validates_encoder(self):
        require_modality("video")
        assert "split" in set(list_detectors("video"))
        assert _ALIASES.get("split_eccv") == "split"
        with pytest.raises(ValueError, match="encoder"):
            load_detector("split", encoder="ResNet-18")

    def test_window_sampling_and_frame_count(self):
        require_modality("video")
        from detectzoo.detectors.video.split import _keep_count, _window_frame_indices

        # 30 fps, window [2 s, 5 s) at 8 fps -> 24 frames starting at frame 60
        idx = _window_frame_indices(300, 30.0, 2.0)
        assert len(idx) == 24 and idx[0] == 60 and idx[1] == 64
        # a 1.5 s clip only yields 12 frames
        assert len(_window_frame_indices(45, 30.0, 0.0)) == 12
        assert _keep_count(24) == 16 and _keep_count(12) == 8
        with pytest.raises(ValueError, match="at least 8"):
            _keep_count(7)

    def test_crop_center_trims_longer_side(self):
        require_modality("video")
        from detectzoo.detectors.video.split import _crop_center

        assert _crop_center(np.zeros((100, 200, 3))).shape == (100, 160, 3)
        assert _crop_center(np.zeros((200, 100, 3))).shape == (160, 100, 3)

    def test_ttr_smooth_vs_jittery(self):
        require_modality("video")
        from detectzoo.detectors.video.split import _ttr

        t = torch.arange(8, dtype=torch.float32)[None, :, None]
        smooth = torch.cat([t, 2 * t], dim=-1)  # constant velocity: 1-step == 2-step / 2
        jitter = torch.cat([t % 2, t % 2], dim=-1)  # back and forth: 2-step change is 0
        assert _ttr(smooth).item() == pytest.approx(0.0, abs=1e-5)
        assert _ttr(jitter).item() > 5

    def test_split_components_static_video_has_zero_lsmi(self):
        require_modality("video")
        from detectzoo.detectors.video.split import split_components

        tokens = torch.randn(1, 196, 8).repeat(16, 1, 1)  # identical frames, 14x14 patches
        _ttr_val, lsmi = split_components(tokens)
        assert lsmi.item() == pytest.approx(0.0, abs=1e-6)

    def test_array_input_is_bgr_and_truncated(self):
        require_modality("video")
        from detectzoo.detectors.video.split import SPLITDetector

        rgb = np.zeros((20, 8, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 255  # red
        frames = SPLITDetector()._load_frames(rgb)
        assert len(frames) == 16
        assert frames[0][0, 0].tolist() == [0, 0, 255]  # red in BGR order

    def test_calibrate_uses_real_score_quantile(self, monkeypatch):
        require_modality("video")
        from detectzoo.detectors.video.split import SPLITDetector

        det = SPLITDetector()
        monkeypatch.setattr(det, "score_components", lambda x: {"score": float(x)})
        reals = list(range(1, 101))  # scores 1..100
        assert det.calibrate(reals, target_fpr=0.05) == 96.0
        assert det.threshold == 96.0
        with pytest.raises(ValueError):
            det.calibrate(reals, target_fpr=0.0)


@pytest.mark.slow
class TestSPLITDetector:
    def test_predict_on_synthetic_video(self, tmp_path):
        require_modality("video")
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (320, 240))
        rng = np.random.default_rng(0)
        for _ in range(96):  # 4 s
            writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
        writer.release()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = load_detector("split", device=device).predict(path)
        assert isinstance(result, DetectionResult)
        assert np.isfinite(result.score) and result.score >= 0
        assert result.metadata["num_frames"] == 16


class TestVideoVeritasHelpers:
    def test_registered_and_validates_args(self):
        require_modality("video")
        assert "videoveritas" in set(list_detectors("video"))
        assert _ALIASES.get("video_veritas") == "videoveritas"
        with pytest.raises(ValueError, match="fps"):
            load_detector("videoveritas", fps=0)
        with pytest.raises(ValueError, match="temperature"):
            load_detector("videoveritas", temperature=-1)

    def test_frame_indices_follow_upstream(self):
        require_modality("video")
        from detectzoo.detectors.video.videoveritas import _frame_indices

        # 10 s at 24 fps, first 5 s at 3 fps: frames 0..120 -> 15.1 -> 14 frames
        idx = _frame_indices(240, 24.0, 3.0, 5.0)
        assert len(idx) == 14 and idx[0] == 0 and idx[-1] == 120
        # a 1 s clip still yields the minimum of 4 frames
        assert len(_frame_indices(24, 24.0, 3.0, 5.0)) == 4
        # frame counts are always even
        assert all(len(_frame_indices(n, 30.0, 3.0, 5.0)) % 2 == 0 for n in range(30, 200, 7))

    def test_resize_caps_tokens_per_frame(self):
        require_modality("video")
        from detectzoo.detectors.video.videoveritas import _resized_hw

        assert _resized_hw(1024, 1024, 14, 32) == (864, 864)  # <= 768 tokens of 32x32
        h, w = _resized_hw(1080, 1920, 14, 32)
        assert h % 32 == 0 and w % 32 == 0 and h * w <= 768 * 32 * 32
        h, w = _resized_hw(64, 64, 14, 32)  # tiny frames are upscaled to >= 128 tokens
        assert h * w >= 128 * 32 * 32

    @pytest.mark.parametrize(
        "response, expected",
        [
            ("Some reasoning. <answer>fake</answer>", "fake"),
            ("<answer> Real </answer>", "real"),
            ("<answer>The video is FAKE.</answer>", "fake"),
            ("<answer>real or fake?</answer>", None),
            ("no tags, it is fake", None),
        ],
    )
    def test_parse_answer(self, response, expected):
        require_modality("video")
        from detectzoo.detectors.video.videoveritas import _parse_answer

        assert _parse_answer(response) == expected

    def test_answer_step_after_last_tag(self):
        require_modality("video")
        from detectzoo.detectors.video.videoveritas import VideoVeritasDetector

        vocab = ["<answer>", " ", "fake", "real", "</answer>", "x", "Fake", " real", "\n"]

        class _Tok:
            def decode(self, ids, **kwargs):
                return "".join(vocab[i] if i < len(vocab) else "" for i in ids)

            def encode(self, text, add_special_tokens=False):
                return [vocab.index(text)] if text in vocab else [len(vocab)]  # unused id

        class _Proc:
            tokenizer = _Tok()

        det = VideoVeritasDetector()
        det._processor = _Proc()
        ids = lambda *toks: [vocab.index(t) for t in toks]  # noqa: E731
        gen = ids("<answer>", "real", "</answer>", "x", "<answer>", " ", "fake", "</answer>")
        assert det._answer_step(gen) == 6  # whitespace skipped, last tag wins
        assert det._answer_step(ids("x", "fake")) is None
        assert det._answer_step(ids("<answer>", "x", "</answer>")) is None

    def test_array_input_padded_to_even_frames(self):
        require_modality("video")
        from detectzoo.detectors.video.videoveritas import VideoVeritasDetector

        frames, meta = VideoVeritasDetector(fps=3.0)._load_video(
            np.zeros((5, 8, 8, 3), dtype=np.uint8)
        )
        assert frames.shape[0] == 6 and meta["frames_indices"] == list(range(6))
        assert meta["fps"] == 3.0


@pytest.mark.slow
class TestVideoVeritasDetector:
    def test_predict_on_synthetic_video(self, tmp_path):
        require_modality("video")
        if not torch.cuda.is_available():
            pytest.skip("VideoVeritas (8B MLLM) needs a GPU")
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (256, 256))
        rng = np.random.default_rng(0)
        for _ in range(24):  # 2 s
            writer.write(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        writer.release()

        det = load_detector("videoveritas", device="cuda", max_new_tokens=1024)
        result = det.predict(path)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["num_frames"] == 6  # 2 s at 3 fps
        assert result.metadata["answer"] in ("real", "fake", None)
