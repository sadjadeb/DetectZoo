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


class TestSTALLHelpers:
    def test_registered_and_validates_duration(self):
        require_modality("video")
        assert "stall" in set(list_detectors("video"))
        assert _ALIASES.get("stall_cvpr2026") == "stall"
        with pytest.raises(ValueError, match="duration"):
            load_detector("stall", duration=5)

    def test_downsample_frames_follow_upstream(self):
        require_modality("video")
        from detectzoo.detectors.video.stall import downsample_frames

        # 30 fps -> 8 fps: ratio 3.75, Python round() as upstream
        assert downsample_frames(30, 30.0)[:6] == [0, 4, 8, 11, 15, 19]
        assert downsample_frames(16, 8.0) == list(range(16))
        with pytest.raises(ValueError, match="at least 8"):
            downsample_frames(30, 6.0)

    def test_windows_use_one_seeded_rng_in_order(self):
        require_modality("video")
        from detectzoo.detectors.video.stall import compute_windows

        idxs = list(range(0, 400, 5))  # 80 downsampled frames
        windows = compute_windows(idxs)
        rng = np.random.RandomState(42)
        starts = [rng.randint(0, len(idxs) - n + 1) for n in (8, 16, 24, 32)]
        for (dur, n), start in zip(((1, 8), (2, 16), (3, 24), (4, 32)), starts):
            assert windows[dur] == idxs[start : start + n]
        assert compute_windows(list(range(10)))[2] is None  # too short for 16 frames

    def _params(self, d=4):
        from detectzoo.detectors.video.stall import load_params

        return (
            d,
            {
                "mu_spat": np.zeros(d, np.float32),
                "W_spat": np.eye(d, dtype=np.float32),
                "mu_temp": np.zeros(d, np.float32),
                "W_temp": np.eye(d, dtype=np.float32),
            },
            load_params,
        )

    def test_stall_scores_percentiles_and_zero_diff(self, tmp_path):
        require_modality("video")
        from detectzoo.detectors.video.stall import stall_scores

        d, base, load_params = self._params()
        calib_spat = np.linspace(-20, 0, 100)[:, None].repeat(3, 1)
        calib_temp = np.linspace(-20, 0, 100)[:, None].repeat(2, 1)
        np.savez(tmp_path / "p.npz", **base, calib_ll_spat=calib_spat, calib_ll_temp=calib_temp)
        params = load_params(tmp_path / "p.npz")

        embs = np.zeros((4, d), np.float32)
        embs[2, 0] = 1.0  # one frame moves away and back
        embs[3] = embs[2]  # last pair identical -> excluded from the temporal min
        s = stall_scores(embs, params)
        ll0 = -0.5 * d * np.log(2 * np.pi)
        assert s["spat_ll_agg"] == pytest.approx(ll0)  # max over frames: an all-zero frame
        assert np.isfinite(s["temp_ll_agg"])
        expected_spat = np.searchsorted(params["calib_spat_sorted"], ll0, side="right") / 100
        assert s["spat_percentile"] == pytest.approx(expected_spat)
        assert s["stall_score"] == pytest.approx(
            0.5 * (s["spat_percentile"] + s["temp_percentile"])
        )

    def test_predict_flips_score_direction(self, monkeypatch, tmp_path):
        require_modality("video")
        from detectzoo.detectors.video.stall import STALLDetector

        d, base, _ = self._params()
        rng = np.random.default_rng(0)
        np.savez(
            tmp_path / "p.npz",
            **base,
            calib_ll_spat=rng.normal(-5, 1, (50, 3)),
            calib_ll_temp=rng.normal(-5, 1, (50, 2)),
        )
        det = STALLDetector(params_path=tmp_path / "p.npz")
        monkeypatch.setattr(det, "embed", lambda frames: rng.normal(0, 0.1, (len(frames), d)))
        r = det.predict(np.zeros((4, 8, 8, 3), dtype=np.uint8))
        assert r.score == pytest.approx(1.0 - r.metadata["stall_score"])
        assert r.metadata["num_frames"] == 4
        assert 0.0 <= r.score <= 1.0


@pytest.mark.slow
class TestSTALLDetector:
    def test_predict_on_synthetic_video(self, tmp_path):
        require_modality("video")
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 120))
        rng = np.random.default_rng(0)
        for _ in range(72):  # 3 s at 24 fps
            writer.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
        writer.release()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = load_detector("stall", device=device).predict(path)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["num_frames"] == 16
