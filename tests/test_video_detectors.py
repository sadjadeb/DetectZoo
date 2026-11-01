"""Tests for video-modality detectors.

The non-slow tests cover registration, the ``BaseVideoDetector`` helpers
(input normalisation, frame sampling, score aggregation) with a model-free
dummy detector, and BusterX's model-free helpers (frame sampling, resizing,
answer parsing).  Running WaveRep or BusterX requires downloading checkpoints
(BusterX's is gated) and is marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

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


class TestNSGVDHelpers:
    def test_registered_and_validates_checkpoint(self):
        require_modality("video")
        assert "nsgvd" in set(list_detectors("video"))
        assert _ALIASES.get("nsg_vd") == "nsgvd"
        with pytest.raises(ValueError, match="checkpoint"):
            load_detector("nsgvd", checkpoint="standard-Sora-mp")
        with pytest.raises(ValueError, match="frame_backend"):
            load_detector("nsgvd", frame_backend="decord")

    def test_ffmpeg_backend_needs_ffmpeg(self, monkeypatch):
        require_modality("video")
        import detectzoo.detectors.video.nsgvd.detector as nsg

        monkeypatch.setattr(nsg.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            nsg.NSGVDDetector()
        nsg.NSGVDDetector(frame_backend="opencv")  # opting out works without ffmpeg

    def test_frame_indices_follow_upstream(self):
        require_modality("video")
        from detectzoo.detectors.video.nsgvd.detector import _frame_indices

        assert _frame_indices(81) == [0, 10, 20, 30, 40, 50, 60, 70]  # every 81 // 8 = 10th
        assert _frame_indices(8) == list(range(8))
        with pytest.raises(ValueError, match="at least 8"):
            _frame_indices(7)

    def test_vp_sde_std(self):
        require_modality("video")
        from detectzoo.detectors.video.nsgvd.detector import _vp_std

        t = torch.tensor([0.006])
        expected = math.sqrt(1 - math.exp(2 * (-0.25 * 0.006**2 * 19.9 - 0.5 * 0.006 * 0.1)))
        assert _vp_std(t).item() == pytest.approx(expected, rel=1e-5)

    def test_nsg_normalisation_and_edge_scaling(self):
        require_modality("video")
        from detectzoo.detectors.video.nsgvd.detector import nsg_from_score

        frames = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(8, 3, 2, 2).clone()
        score = torch.ones(8, 3, 2, 2)  # pixel change is 1 everywhere -> denominator = 12
        nsg = nsg_from_score(score, frames)
        assert nsg.shape == (8, 3, 2, 2)
        assert torch.allclose(nsg[1:-1], torch.full_like(nsg[1:-1], 1 / 12))
        assert torch.allclose(nsg[0], torch.full_like(nsg[0], 100 / 12))
        assert torch.allclose(nsg[-1], torch.full_like(nsg[-1], 100 / 12))

    def test_preprocess_resizes_to_unit_range(self):
        require_modality("video")
        from detectzoo.detectors.video.nsgvd.detector import _preprocess

        x = _preprocess([np.full((48, 64, 3), 255, dtype=np.uint8)] * 2)
        assert x.shape == (2, 3, 224, 224) and x.max() <= 1.0 and x.min() >= 0.0

    def test_predict_requires_reference(self):
        require_modality("video")
        with pytest.raises(RuntimeError, match="set_reference"):
            load_detector("nsgvd", frame_backend="opencv").predict(
                np.zeros((8, 8, 8, 3), dtype=np.uint8)
            )

    def test_array_input_needs_eight_frames(self):
        require_modality("video")
        from detectzoo.detectors.video.nsgvd.detector import NSGVDDetector

        with pytest.raises(ValueError, match="exactly 8"):
            NSGVDDetector(frame_backend="opencv")._load_frames(
                np.zeros((5, 8, 8, 3), dtype=np.uint8)
            )

    def _stubbed(self, monkeypatch):
        """Detector with tiny stand-ins for the diffusion model and discriminator."""
        from detectzoo.detectors.video.nsgvd.detector import NSGVDDetector

        class _Net(torch.nn.Module):
            def forward(self, x, out_feature=False):
                feat = x.flatten(1)[:, :16]
                return None, feat

        det = NSGVDDetector(frame_backend="opencv")
        det._model = _Net()
        det._kernel = {
            "sigma": torch.tensor(1e3),
            "sigma0_u": torch.tensor(1.0),
            "ep": torch.tensor(1e-3),
        }
        monkeypatch.setattr(det, "nsg_features", lambda x: torch.as_tensor(x, dtype=torch.float32))
        return det

    def test_reference_scoring_and_roundtrip(self, monkeypatch, tmp_path):
        require_modality("video")
        det = self._stubbed(monkeypatch)
        g = torch.Generator().manual_seed(0)
        reals = [torch.randn(8, 3, 4, 4, generator=g) * 0.1 for _ in range(6)]
        assert det.set_reference(reals) == 6
        typical = det.predict(torch.randn(8, 3, 4, 4, generator=g) * 0.1).score
        outlier = det.predict(torch.randn(8, 3, 4, 4, generator=g) * 0.1 + 5.0).score
        assert np.isfinite(typical) and outlier > typical

        path = tmp_path / "ref.pt"
        det.save_reference(path)
        det2 = self._stubbed(monkeypatch)
        det2.load_reference(path)
        probe = torch.randn(8, 3, 4, 4, generator=g) * 0.1 + 5.0
        again = det2.predict(probe)
        assert again.metadata["num_reference"] == 6
        assert again.score == pytest.approx(det.predict(probe).score)

    def test_set_reference_needs_two_readable_videos(self, monkeypatch):
        require_modality("video")
        det = self._stubbed(monkeypatch)

        def _fail(x):
            raise ValueError("unreadable")

        monkeypatch.setattr(det, "nsg_features", _fail)
        with pytest.raises(ValueError, match="at least 2"):
            det.set_reference(["a.mp4", "b.mp4"])


@pytest.mark.slow
class TestNSGVDDetector:
    def test_predict_with_reference(self, tmp_path):
        require_modality("video")
        if not torch.cuda.is_available():
            pytest.skip("NSG-VD's diffusion model needs a GPU")
        cv2 = pytest.importorskip("cv2")

        def _video(name, seed):
            path = tmp_path / name
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (96, 96))
            rng = np.random.default_rng(seed)
            base = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
            for i in range(16):
                writer.write(np.roll(base, i, axis=1))
            writer.release()
            return path

        det = load_detector("nsgvd", device="cuda")
        assert det.set_reference([_video(f"r{i}.avi", i) for i in range(3)]) == 3
        result = det.predict(_video("test.avi", 9))
        assert isinstance(result, DetectionResult)
        assert np.isfinite(result.score)
        assert result.metadata["num_reference"] == 3
