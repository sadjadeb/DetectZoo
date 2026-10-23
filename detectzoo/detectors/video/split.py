"""SPLIT — training-free AI-generated and partially edited video detection (ECCV 2026).

Reference:
    Hyun & Kim, "SPLIT: Training-Free AI-Generated and Partially Edited Video
    Detection via Spatial Patch-Level Incoherence and Temporal Roughness",
    ECCV 2026.  https://arxiv.org/abs/2607.02886

A frozen vision encoder turns each frame into patch tokens.  Two signals are
computed on them:

* **Two-step Temporal Roughness (TTR)** — per patch, log2 of the ratio between
  the summed 1-step feature change and the (length-normalised) summed 2-step
  change; real motion is smooth, generated motion jitters.
* **Local Spatial Motion Incoherence (LSMI)** — the mean spatial gradient of the
  feature-space motion field between consecutive frames.

The score is ``mean(TTR) ** 8 * LSMI`` (higher = more likely generated).  It is
unbounded, so the paper thresholds it with *real-only calibration*: pick the
threshold that gives a target false-positive rate on real videos
(:meth:`SPLITDetector.calibrate`).

Preprocessing follows the upstream code: frames at 8 fps from a 3 s window
starting at a random whole second, the first 16 frames (8 if fewer than 16),
10% cropped from each side of the longer dimension, resized to 224×224 and
ImageNet-normalised.  Upstream reads frames with ``cv2.imread``, so the encoder
sees BGR channel order; this is kept so scores match.  Upstream extracts frames
to JPEG with ffmpeg first; here frames are decoded directly.

Upstream: https://github.com/mldljyh/SPLIT
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

# encoder name -> (HF model id, pinned revision)
SPLIT_ENCODERS = {
    "XCLIP-16": ("microsoft/xclip-base-patch16", "d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6"),
    "XCLIP-32": ("microsoft/xclip-base-patch32", None),
    "CLIP-16": ("openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3"),
    "CLIP-32": ("openai/clip-vit-base-patch32", None),
    "DINO-base": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
    "DINO-large": ("facebook/dinov2-large", None),
}

_SAMPLE_FPS = 8
_WINDOW_SECONDS = 3
_MIN_FRAMES = 8
_MAX_FRAMES = 16
_SIZE = 224
_CROP = 0.1
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_GAMMA = 8.0


# ---------------------------------------------------------------------------
# Frame sampling and preprocessing (mirrors utils/video2frame.py and
# data/datasets.py upstream)
# ---------------------------------------------------------------------------


def _window_frame_indices(total_frames: int, fps: float, start: float) -> List[int]:
    """Source frames for an 8 fps sampling of ``[start, start + 3 s)``."""
    duration = total_frames / fps
    end = min(start + _WINDOW_SECONDS, duration)
    ts = np.arange(start, end, 1.0 / _SAMPLE_FPS)
    idx = np.clip(np.round(ts * fps).astype(np.int64), 0, total_frames - 1)
    return idx.tolist()


def _keep_count(n: int) -> int:
    """Upstream keeps 16 frames, or 8 when fewer than 16 are available."""
    if n < _MIN_FRAMES:
        raise ValueError(
            f"SPLIT needs at least {_MIN_FRAMES} frames at {_SAMPLE_FPS} fps (~1 s); got {n}."
        )
    return _MAX_FRAMES if n >= _MAX_FRAMES else _MIN_FRAMES


def _crop_center(image: np.ndarray, fraction: float = _CROP) -> np.ndarray:
    """Upstream ``crop_center_by_percentage``: trim the longer side on both ends."""
    h, w = image.shape[:2]
    if w > h:
        px = int(w * fraction)
        return image[:, px : w - px]
    px = int(h * fraction)
    return image[px : h - px, :]


def _preprocess(frames_bgr: Sequence[np.ndarray]) -> torch.Tensor:
    """BGR ``uint8`` frames → ``float [T, 3, 224, 224]`` (albumentations equivalent)."""
    import cv2

    out = []
    for frame in frames_bgr:
        img = cv2.resize(_crop_center(frame), (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        out.append(img.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))


# ---------------------------------------------------------------------------
# Score (mirrors models/SPLIT_model.py upstream)
# ---------------------------------------------------------------------------


def _ttr(tokens: torch.Tensor) -> torch.Tensor:
    """Two-step temporal roughness for ``[N, T, D]`` patch trajectories → ``[N]``."""
    t = tokens.shape[1]
    l1 = torch.norm(tokens[:, 1:] - tokens[:, :-1], p=2, dim=-1).sum(dim=1)
    l2 = torch.norm(tokens[:, 2:] - tokens[:, :-2], p=2, dim=-1).sum(dim=1)
    l2 = (l2 / 2) * (t - 1) / (t - 2)
    return (torch.log(l1 + 1e-8) - torch.log(l2 + 1e-8)) / 0.693147


def split_components(patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[T, P, D]`` patch tokens of one video → ``(mean TTR, LSMI)`` scalars."""
    t, p, d = patch_tokens.shape
    mean_ttr = _ttr(patch_tokens.permute(1, 0, 2)).mean()
    g = int(p**0.5)
    grid = patch_tokens.reshape(t, g, g, d)
    motion = grid[1:] - grid[:-1]
    grad_x = torch.norm(motion[:, :, :-1] - motion[:, :, 1:], p=2, dim=-1)
    grad_y = torch.norm(motion[:, :-1] - motion[:, 1:], p=2, dim=-1)
    lsmi = (grad_x.mean() + grad_y.mean()) / 2.0
    return mean_ttr, lsmi


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("split", aliases=["split_eccv"])
class SPLITDetector(BaseVideoDetector):
    """SPLIT training-free video detector (Hyun & Kim, ECCV 2026).

    Parameters
    ----------
    encoder : str
        Frozen encoder from :data:`SPLIT_ENCODERS` (default ``"XCLIP-16"``,
        the upstream evaluation default).
    threshold : float
        Decision boundary on the raw SPLIT score.  The score is unbounded and
        dataset-dependent, so set this with :meth:`calibrate` on real videos
        for meaningful labels; ranking metrics (ROC-AUC, AP) do not depend on
        it.
    device : str
        Torch device string.
    seed : int
        Seed for the per-video random window start (default 42, as upstream).
    cache_dir : str or Path, optional
        Hugging Face cache directory for the encoder weights.

    Non-path inputs (arrays, tensors, frame lists) are taken as RGB frames
    already sampled at 8 fps.
    """

    def __init__(
        self,
        *,
        encoder: str = "XCLIP-16",
        threshold: float = 0.5,
        device: str = "cpu",
        seed: int = 42,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, num_frames=None, **kwargs)
        if encoder not in SPLIT_ENCODERS:
            raise ValueError(f"Unknown SPLIT encoder '{encoder}'. Valid: {list(SPLIT_ENCODERS)}")
        self.encoder = encoder
        self.seed = seed
        self.cache_dir = str(cache_dir) if cache_dir is not None else None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModel, CLIPVisionModel, XCLIPVisionModel

        model_id, revision = SPLIT_ENCODERS[self.encoder]
        cls = {"XCLIP": XCLIPVisionModel, "CLIP-": CLIPVisionModel}.get(self.encoder[:5], AutoModel)
        logger.info("Loading SPLIT encoder %s (%s) …", self.encoder, model_id)
        model = cls.from_pretrained(model_id, revision=revision, cache_dir=self.cache_dir)
        self._model = model.to(self._device).eval()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _read_window(self, path: Path) -> List[np.ndarray]:
        """Decode the BGR frames of the sampled 3 s window of *path*."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total <= 0:
                raise ValueError(f"Could not read the frame rate / length of {path}")
            length = total / fps
            rng = random.Random(f"{self.seed}:{path.name}")
            start = (
                0.0
                if length <= _WINDOW_SECONDS
                else float(int(rng.uniform(0, length - _WINDOW_SECONDS)))
            )
            wanted = _window_frame_indices(total, fps, start)
            wanted = wanted[: _keep_count(len(wanted))]
            targets = set(wanted)
            by_index: dict[int, np.ndarray] = {}
            # Decode sequentially: frame-accurate seeking is not reliable for all codecs.
            i = 0
            while i <= max(targets):
                ok, frame = cap.read()
                if not ok:
                    break
                if i in targets:
                    by_index[i] = frame
                i += 1
        finally:
            cap.release()
        missing = [j for j in wanted if j not in by_index]
        if missing:
            raise ValueError(f"Could not decode {len(missing)} sampled frame(s) of {path}")
        return [by_index[j] for j in wanted]

    def _load_frames(self, input_data: Any) -> List[np.ndarray]:
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            return self._read_window(path)
        rgb = self._normalise_input(input_data).permute(0, 2, 3, 1).numpy()
        frames = [np.ascontiguousarray(f[..., ::-1]) for f in rgb]  # RGB -> BGR, as upstream
        return frames[: _keep_count(len(frames))]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_components(self, input_data: Any) -> dict[str, float]:
        """Return ``{"score", "ttr", "lsmi", "num_frames"}`` for one video."""
        frames = self._load_frames(input_data)
        x = _preprocess(frames).to(self._device)
        out = self.model(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, 1:, :].float()  # drop CLS
        mean_ttr, lsmi = split_components(patch_tokens)
        score = (mean_ttr**_GAMMA) * lsmi
        return {
            "score": float(score),
            "ttr": float(mean_ttr),
            "lsmi": float(lsmi),
            "num_frames": len(frames),
        }

    def predict(self, input_data: Any) -> DetectionResult:
        c = self.score_components(input_data)
        return self._make_result(
            c["score"],
            ttr=c["ttr"],
            lsmi=c["lsmi"],
            num_frames=c["num_frames"],
            encoder=self.encoder,
        )

    def calibrate(self, real_inputs: Sequence[Any], target_fpr: float = 0.05) -> float:
        """Set :attr:`threshold` from real videos only, for *target_fpr*.

        Uses the upstream rule ``quantile(real_scores, 1 - target_fpr,
        method="higher")`` and returns the new threshold.
        """
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        scores = np.array([self.score_components(x)["score"] for x in real_inputs])
        if scores.size == 0:
            raise ValueError("calibrate() needs at least one real video")
        self.threshold = float(np.quantile(scores, 1 - target_fpr, method="higher"))
        return self.threshold
