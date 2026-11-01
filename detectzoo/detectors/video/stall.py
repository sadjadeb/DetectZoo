"""STALL — training-free detection of generated videos via spatial-temporal likelihoods (CVPR 2026).

Reference:
    Ben Hayun et al., "Training-free Detection of Generated Videos via
    Spatial-Temporal Likelihoods", CVPR 2026.  https://arxiv.org/abs/2603.15026

Each frame is embedded with DINOv3 ViT-L/16 (CLS token).  Two Gaussian
log-likelihoods are computed under whitening transforms fitted on real VATEX
videos:

* **spatial** — of the whitened frame embeddings, aggregated with ``max`` over
  frames;
* **temporal** — of the whitened, L2-normalised differences between consecutive
  embeddings, aggregated with ``min`` over frame pairs (identical consecutive
  frames are ignored).

Each aggregate becomes a percentile against the VATEX calibration scores and the
STALL score is their mean (higher = more likely real).  DetectZoo's ``score`` is
``1 - STALL score`` so that higher means generated, as for every other detector;
the original value is in ``metadata["stall_score"]``.

Frame sampling follows ``src/video_index.py`` upstream: downsample to 8 fps,
then a random contiguous 2 s window (16 frames) drawn with
``np.random.RandomState(42)`` after the 1 s window draw.  Upstream seeks each
frame with OpenCV; here frames are decoded sequentially.  (The window indices
stored with the released ``OmerXYZ/comgenvid`` embeddings were drawn differently
from ``video_index.py``; scoring those embeddings with their stored windows
reproduces the published comgenvid table exactly.)

Upstream: https://github.com/OmerBenHayun/STALL
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_PARAMS_URL = (
    "https://raw.githubusercontent.com/OmerBenHayun/STALL/"
    "bfcc603ae83b4e609681277b9b5e80e7a9497e15/precomputed/stall_params_vatex_dino_v3.npz"
)
_PARAMS_NAME = "stall_params_vatex_dino_v3.npz"
_DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"

_TARGET_FPS = 8.0
_WINDOW_SEED = 42
_DURATIONS = (1, 2, 3, 4)
_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Frame sampling (mirrors src/video_index.py upstream)
# ---------------------------------------------------------------------------


def downsample_frames(num_frames: int, current_fps: float, target_fps: float = _TARGET_FPS):
    """Upstream ``downsample_frames``: ``round(ratio * j)`` while inside the video."""
    if target_fps > current_fps:
        raise ValueError(
            f"STALL needs a frame rate of at least {target_fps} fps; got {current_fps:.2f}."
        )
    ratio = current_fps / target_fps
    indices, j = [], 0
    while True:
        idx = round(ratio * j)
        if idx >= num_frames:
            break
        indices.append(idx)
        j += 1
    return indices


def compute_windows(downsample_idxs: List[int], target_fps: float = _TARGET_FPS, seed=_WINDOW_SEED):
    """Upstream ``compute_windows``: one seeded RNG draws the 1/2/3/4 s windows in order."""
    rng = np.random.RandomState(seed)
    n_total = len(downsample_idxs)
    result: Dict[int, Optional[List[int]]] = {}
    for dur in _DURATIONS:
        n = int(round(target_fps * dur))
        if n_total < n:
            result[dur] = None
        else:
            start = rng.randint(0, n_total - n + 1)
            result[dur] = list(downsample_idxs[start : start + n])
    return result


# ---------------------------------------------------------------------------
# Likelihoods (mirrors src/stall.py upstream)
# ---------------------------------------------------------------------------


def _log_likelihood(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    return -0.5 * (d * np.log(2.0 * np.pi) + (x**2).sum(axis=-1))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def stall_scores(embs: np.ndarray, params: Dict[str, np.ndarray]) -> Dict[str, float]:
    """STALL scores from ``[T, D]`` frame embeddings and the calibration *params*."""
    embs = embs[np.newaxis]  # [1, T, D]
    spat_ll = _log_likelihood(np.matmul(embs - params["mu_spat"], params["W_spat"]))
    diffs = embs[:, 1:] - embs[:, :-1]
    temp_ll = _log_likelihood(np.matmul(_l2_normalize(diffs) - params["mu_temp"], params["W_temp"]))
    # Identical consecutive frames carry no temporal information: never the min.
    temp_ll[np.linalg.norm(diffs, axis=-1) == 0] = np.inf

    spat_agg = spat_ll.max(axis=1)
    temp_agg = temp_ll.min(axis=1)
    spat_pct = np.searchsorted(params["calib_spat_sorted"], spat_agg, side="right")
    temp_pct = np.searchsorted(params["calib_temp_sorted"], temp_agg, side="right")
    spat_pct = spat_pct / len(params["calib_spat_sorted"])
    temp_pct = temp_pct / len(params["calib_temp_sorted"])
    return {
        "spat_ll_agg": float(spat_agg[0]),
        "temp_ll_agg": float(temp_agg[0]),
        "spat_percentile": float(spat_pct[0]),
        "temp_percentile": float(temp_pct[0]),
        "stall_score": float(0.5 * (spat_pct[0] + temp_pct[0])),
    }


def load_params(path: str | Path) -> Dict[str, np.ndarray]:
    """Load the upstream ``.npz`` and pre-sort the aggregated calibration scores."""
    data = np.load(path)
    params = {k: data[k] for k in ("mu_spat", "W_spat", "mu_temp", "W_temp")}
    params["calib_spat_sorted"] = np.sort(data["calib_ll_spat"].max(axis=1))
    params["calib_temp_sorted"] = np.sort(data["calib_ll_temp"].min(axis=1))
    return params


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("stall", aliases=["stall_cvpr2026"])
class STALLDetector(BaseVideoDetector):
    """STALL training-free video detector (Ben Hayun et al., CVPR 2026).

    Parameters
    ----------
    duration : int
        Window length in seconds at 8 fps: 1, 2 (default, as upstream), 3 or 4.
    threshold : float
        Decision boundary on ``1 - STALL score`` (default 0.5).
    device : str
        Torch device string.
    batch_size : int
        Frames per DINOv3 forward pass.
    params_path : str or Path, optional
        Local copy of ``stall_params_vatex_dino_v3.npz``; downloaded from the
        upstream repository when omitted.
    cache_dir : str or Path, optional
        Cache directory for the calibration parameters (``.detectzoo_data``)
        and, for the DINOv3 weights, the Hugging Face cache.

    DINOv3 (``facebook/dinov3-vitl16-pretrain-lvd1689m``) is gated on Hugging
    Face: request access on the model page and log in with ``hf auth login``.

    Non-path inputs (arrays, tensors, frame lists) are taken as RGB frames
    already sampled at 8 fps and used as the window directly.
    """

    def __init__(
        self,
        *,
        duration: int = 2,
        threshold: float = 0.5,
        device: str = "cpu",
        batch_size: int = 32,
        params_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, num_frames=None, **kwargs)
        if duration not in _DURATIONS:
            raise ValueError(f"duration must be one of {_DURATIONS}; got {duration}")
        self.duration = int(duration)
        self.batch_size = max(int(batch_size), 1)
        self.params_path = Path(params_path) if params_path is not None else None
        self.cache_dir = cache_dir
        self._params: Optional[Dict[str, np.ndarray]] = None
        self._transform = None

    # ------------------------------------------------------------------
    # Model and calibration parameters
    # ------------------------------------------------------------------

    @property
    def params(self) -> Dict[str, np.ndarray]:
        if self._params is None:
            path = self.params_path
            if path is None:
                path = get_cache_dir("stall", self.cache_dir) / _PARAMS_NAME
                if not path.is_file():
                    download_file(_PARAMS_URL, path)
            self._params = load_params(path)
        return self._params

    def _load_model(self) -> None:
        from transformers import AutoModel

        logger.info("Loading DINOv3 ViT-L/16 (%s) …", _DINO_ID)
        hf_cache = str(Path(self.cache_dir) / "hf") if self.cache_dir is not None else None
        model = AutoModel.from_pretrained(_DINO_ID, revision=_DINO_REVISION, cache_dir=hf_cache)
        self._model = model.to(self._device).eval()

    def _get_transform(self):
        if self._transform is None:
            from torchvision import transforms

            # Upstream ``create_dinov3_transform``.
            self._transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((_SIZE, _SIZE), antialias=True),
                    transforms.Normalize(mean=_MEAN, std=_STD),
                ]
            )
        return self._transform

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def window_indices(self, path: str | Path) -> List[int]:
        """Source frame indices of the scored window for the video at *path*."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if fps <= 0 or total <= 0:
            raise ValueError(f"Could not read the frame rate / length of {path}")
        if total / fps < 1.0:
            raise ValueError(f"STALL needs videos of at least 1 s; {path} is {total / fps:.2f} s")
        window = compute_windows(downsample_frames(total, fps))[self.duration]
        if window is None:
            raise ValueError(f"{path} is too short for a {self.duration} s window at 8 fps")
        return window

    def _load_frames(self, input_data: Any) -> List[np.ndarray]:
        if isinstance(input_data, (str, Path)):
            import cv2

            path = Path(input_data)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            wanted = self.window_indices(path)
            targets, frames, i = set(wanted), {}, 0
            cap = cv2.VideoCapture(str(path))
            try:
                while i <= wanted[-1]:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if i in targets:
                        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1
            finally:
                cap.release()
            if len(frames) < len(targets):
                raise ValueError(f"Could not decode the sampled frames of {path}")
            return [frames[j] for j in wanted]
        rgb = self._normalise_input(input_data).permute(0, 2, 3, 1).numpy()
        if len(rgb) < 2:
            raise ValueError("STALL needs at least 2 frames")
        return list(rgb)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, frames: List[np.ndarray]) -> np.ndarray:
        """DINOv3 CLS embeddings ``[T, 1024]`` of RGB frames."""
        from PIL import Image

        tf = self._get_transform()
        out = []
        for start in range(0, len(frames), self.batch_size):
            batch = torch.stack(
                [tf(Image.fromarray(f)) for f in frames[start : start + self.batch_size]]
            )
            out.append(self.model(pixel_values=batch.to(self._device)).pooler_output.float().cpu())
        return torch.cat(out).numpy()

    def predict(self, input_data: Any) -> DetectionResult:
        frames = self._load_frames(input_data)
        scores = stall_scores(self.embed(frames), self.params)
        return self._make_result(1.0 - scores["stall_score"], num_frames=len(frames), **scores)
