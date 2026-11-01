"""NSG-VD — physics-driven AI-generated video detection (NeurIPS 2025).

Reference:
    Zhang et al., "Physics-Driven Spatiotemporal Modeling for AI-Generated Video
    Detection", NeurIPS 2025.  https://openreview.net/forum?id=HiBoJLCyEo

The *Normalized Spatiotemporal Gradient* (NSG) of a video is the ratio of the
spatial score (∇ log p, estimated by a pre-trained diffusion model) to the
temporal change of the frames, motivated by probability-flow conservation.
NSG-VD embeds the NSG features with a trained Swin/TALL network and scores a
test video by the Maximum Mean Discrepancy (MMD) between it and a reference set
of **real** videos under that learned deep kernel; ``score > 1`` means
generated.

Pipeline (following the upstream code):

1. 8 frames, every ``total_frames // 8``-th frame from the start, resized to
   224×224 (Lanczos) in ``[0, 1]``.
2. Score of OpenAI's ``256x256_diffusion_uncond`` guided-diffusion UNet (fp16) at
   ``t = (5 + 1) / 1000`` under the VP-SDE (``β ∈ [0.1, 20]``):
   ``score = -ε(2x - 1, 999 t) / σ(t)``, first 3 channels.
3. NSG = score / Σ(score · Δx) per frame (central pixel differences), with the
   first and last frame scaled by 100.
4. Swin/TALL features of the NSG tensor, then the smoothed deep-kernel MMD
   (``MMD_batch2``) against the reference features.

Released checkpoints (``checkpoint=``): ``{standard,unbalance}-{SEINE,Pika}-{mp,d}``
(trained on GenVideo with one generator; see upstream README for availability).
Upstream extracts frames to JPEG with ffmpeg first; here frames are decoded
directly with OpenCV.

Upstream: https://github.com/ZSHsh98/NSG-VD
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_REPO_RAW = (
    "https://raw.githubusercontent.com/ZSHsh98/NSG-VD/9a60f57fcb464b418899110a4717c8c8fd7687bc"
)
NSGVD_CHECKPOINTS = (
    "standard-SEINE-mp",
    "standard-Pika-mp",
    "unbalance-SEINE-mp",
    "standard-SEINE-d",
    "standard-Pika-d",
    "unbalance-SEINE-d",
)
_DIFFUSION_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
)
_DIFFUSION_NAME = "256x256_diffusion_uncond.pt"

_NUM_FRAMES = 8
_SIZE = 224
_DIFFUSE_STEPS = 5
_BETA_MIN, _BETA_MAX = 0.1, 20.0
_EDGE_SCALE = 100.0


# ---------------------------------------------------------------------------
# Frame sampling and NSG features
# ---------------------------------------------------------------------------


def _frame_indices(total_frames: int) -> List[int]:
    """Upstream ffmpeg ``select='not(mod(n, k))'`` with ``k = total // 8``, first 8."""
    if total_frames < _NUM_FRAMES:
        raise ValueError(f"NSG-VD needs at least {_NUM_FRAMES} frames; got {total_frames}.")
    interval = max(1, total_frames // _NUM_FRAMES)
    return [i * interval for i in range(_NUM_FRAMES)]


def _frame_count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total <= 0:
        raise ValueError(f"Could not read the frame rate / length of {path}")
    return total


def _frames_ffmpeg(path: Path, total_frames: int) -> List[np.ndarray]:
    """Upstream ``process_video``: ffmpeg ``select`` to JPEG files, read back in order."""
    from PIL import Image

    _frame_indices(total_frames)  # validates the minimum length
    interval = max(1, total_frames // _NUM_FRAMES)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"select=not(mod(n\\,{interval})),setpts=N/FRAME_RATE/TB",
                "-vframes",
                str(_NUM_FRAMES),
                f"{tmp}/frame%d.jpg",
            ],
            check=True,
        )
        files = sorted(Path(tmp).glob("frame*.jpg"))  # frame1..frame8 (upstream sorts too)
        frames = [np.array(Image.open(f).convert("RGB")) for f in files]
    if len(frames) < _NUM_FRAMES:
        raise ValueError(f"ffmpeg returned {len(frames)} frame(s) for {path}; need {_NUM_FRAMES}")
    return frames


def _frames_opencv(path: Path, total_frames: int) -> List[np.ndarray]:
    import cv2

    wanted = _frame_indices(total_frames)
    frames, i = {}, 0
    cap = cv2.VideoCapture(str(path))
    try:
        while i <= wanted[-1]:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
    if len(frames) < _NUM_FRAMES:
        raise ValueError(f"Could not decode the sampled frames of {path}")
    return [frames[j] for j in wanted]


def _preprocess(frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
    """RGB ``uint8`` frames → ``float [T, 3, 224, 224]`` in ``[0, 1]`` (PIL Lanczos)."""
    from PIL import Image

    out = []
    for frame in frames_rgb:
        img = Image.fromarray(frame).resize((_SIZE, _SIZE), Image.Resampling.LANCZOS)
        out.append(torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1))
    return torch.stack(out)


def _vp_std(t: torch.Tensor) -> torch.Tensor:
    """σ(t) of the VP-SDE marginal (``score_sde.sde_lib.VPSDE.marginal_prob``)."""
    log_mean_coeff = -0.25 * t**2 * (_BETA_MAX - _BETA_MIN) - 0.5 * t * _BETA_MIN
    return torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))


def nsg_from_score(score: torch.Tensor, frames: torch.Tensor, epsilon: float = 1e-10):
    """NSG features ``[T, C, H, W]`` from the score and frames of one video.

    Mirrors ``ScoreFeaturesDataset._get_score_velocity`` plus the edge-frame
    scaling applied when the features are loaded (``get_data``).
    """
    t = frames.shape[0]
    pixel_diff = torch.zeros_like(frames)
    for i in range(1, t - 1):
        pixel_diff[i] = (frames[i + 1] - frames[i - 1]) / 2
    pixel_diff[0] = frames[1] - frames[0]
    pixel_diff[-1] = frames[-1] - frames[-2]
    denominator = (score * pixel_diff).sum(dim=(1, 2, 3)).view(t, 1, 1, 1) + epsilon
    nsg = score / denominator
    nsg[0] = nsg[0] * _EDGE_SCALE
    nsg[-1] = nsg[-1] * _EDGE_SCALE
    return nsg


def _build_unet():
    """``create_model`` with the upstream ``libs/eps_ad/imagenet.yml`` settings."""
    from detectzoo.detectors.video.nsgvd._upstream.guided_diffusion.unet import UNetModel

    image_size = 256
    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=256,
        out_channels=6,  # learn_sigma
        num_res_blocks=2,
        attention_resolutions=tuple(image_size // r for r in (32, 16, 8)),
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("nsgvd", aliases=["nsg_vd", "nsg-vd"])
class NSGVDDetector(BaseVideoDetector):
    """NSG-VD deep-kernel MMD detector (Zhang et al., NeurIPS 2025).

    NSG-VD scores a video *relative to real reference videos*: call
    :meth:`set_reference` (or :meth:`load_reference`) with real videos before
    :meth:`predict`.  Upstream uses 100 real MSR-VTT clips.

    Parameters
    ----------
    checkpoint : str
        One of :data:`NSGVD_CHECKPOINTS` (default ``"standard-SEINE-mp"``) or a
        path to a local ``.pth`` file.
    threshold : float
        Decision boundary on the MMD score (default 1.0, as upstream).
    device : str
        Torch device string.
    frame_backend : str
        ``"ffmpeg"`` (default) runs the upstream ffmpeg command, which writes the
        sampled frames as JPEG before they are read back; the released
        checkpoints were trained on such frames and NSG is sensitive to the JPEG
        step, so this is needed to reproduce upstream scores (requires the
        ``ffmpeg`` program on ``PATH``).  ``"opencv"`` decodes frames directly
        (no ffmpeg needed) but scores then deviate from upstream.
    cache_dir : str or Path, optional
        Override the default cache directory (``.detectzoo_data``) for the
        diffusion model (~2.2 GB) and the NSG-VD checkpoint.

    Non-path inputs (arrays, tensors, frame lists) must hold exactly the 8
    frames to use.
    """

    def __init__(
        self,
        *,
        checkpoint: str = "standard-SEINE-mp",
        threshold: float = 1.0,
        device: str = "cpu",
        frame_backend: str = "ffmpeg",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, num_frames=None, **kwargs)
        if checkpoint not in NSGVD_CHECKPOINTS and not Path(checkpoint).is_file():
            raise ValueError(
                f"Unknown NSG-VD checkpoint '{checkpoint}'. "
                f"Valid: {list(NSGVD_CHECKPOINTS)} or a local .pth path."
            )
        self.checkpoint = checkpoint
        if frame_backend not in ("ffmpeg", "opencv"):
            raise ValueError("frame_backend must be 'ffmpeg' or 'opencv'")
        if frame_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "NSG-VD's default frame_backend='ffmpeg' needs the ffmpeg program on PATH "
                "(the checkpoints were trained on ffmpeg-extracted JPEG frames). Install "
                "ffmpeg, or pass frame_backend='opencv' to decode frames directly (scores "
                "will then not match upstream)."
            )
        self.frame_backend = frame_backend
        self.cache_dir = cache_dir
        self._unet: Optional[torch.nn.Module] = None
        self._kernel: Optional[dict] = None
        self._ref_data: Optional[torch.Tensor] = None
        self._ref_features: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        if Path(self.checkpoint).is_file():
            return Path(self.checkpoint)
        path = get_cache_dir("nsgvd", self.cache_dir) / f"{self.checkpoint}.pth"
        if not path.is_file():
            download_file(f"{_REPO_RAW}/ckpts/{self.checkpoint}.pth", path)
        return path

    def _load_model(self) -> None:
        from detectzoo.detectors.video.nsgvd._upstream.tall import SingleSwinBlockDiscriminator

        state = torch.load(self._checkpoint_path(), map_location="cpu", weights_only=True)
        net = SingleSwinBlockDiscriminator(num_features=300)
        net.load_state_dict({k[len("net.") :]: v for k, v in state.items() if k.startswith("net.")})
        # Test-time kernel parameters stored by upstream ``deep_MMD.set_info``.
        self._kernel = {k: state[k].float().to(self._device) for k in ("sigma", "sigma0_u", "ep")}
        self._model = net.to(self._device).eval()

    @property
    def unet(self) -> torch.nn.Module:
        if self._unet is None:
            path = get_cache_dir("nsgvd", self.cache_dir) / _DIFFUSION_NAME
            if not path.is_file():
                download_file(_DIFFUSION_URL, path)
            logger.info("Loading guided-diffusion UNet from %s …", path)
            unet = _build_unet()
            unet.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            unet.convert_to_fp16()
            self._unet = unet.to(self._device).eval()
        return self._unet

    def unload(self) -> None:
        self._kernel = None
        super().unload()

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def _load_frames(self, input_data: Any) -> torch.Tensor:
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            total = _frame_count(path)
            if self.frame_backend == "ffmpeg":
                return _preprocess(_frames_ffmpeg(path, total))
            return _preprocess(_frames_opencv(path, total))
        rgb = self._normalise_input(input_data).permute(0, 2, 3, 1).numpy()
        if len(rgb) != _NUM_FRAMES:
            raise ValueError(f"NSG-VD expects exactly {_NUM_FRAMES} frames; got {len(rgb)}.")
        return _preprocess(list(rgb))

    @torch.no_grad()
    def nsg_features(self, input_data: Any) -> torch.Tensor:
        """NSG features ``[8, 3, 224, 224]`` of one video (on the detector's device)."""
        frames = self._load_frames(input_data).to(self._device)
        t = torch.full((frames.shape[0],), (_DIFFUSE_STEPS + 1) / 1000, device=self._device)
        out = self.unet(2 * frames - 1, t * 999)
        score = -out / _vp_std(t)[:, None, None, None]
        score = score[:, :3].float()
        return nsg_from_score(score, frames)

    # ------------------------------------------------------------------
    # Reference set
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_reference(self, real_inputs: Sequence[Any]) -> int:
        """Compute NSG features of real reference videos; returns how many were used.

        Videos that cannot be read (too short, unreadable) are skipped with a
        warning.
        """
        feats = []
        for x in real_inputs:
            try:
                feats.append(self.nsg_features(x))
            except (ValueError, FileNotFoundError) as exc:
                logger.warning("NSG-VD: skipping reference %s (%s)", x, exc)
        if len(feats) < 2:
            raise ValueError("set_reference() needs at least 2 readable real videos")
        self._set_reference_data(torch.stack(feats))
        return len(feats)

    def _set_reference_data(self, ref_data: torch.Tensor) -> None:
        self._ref_data = ref_data.float().to(self._device)
        _, feats = self.model(self._ref_data, out_feature=True)
        self._ref_features = feats

    def save_reference(self, path: str | Path) -> None:
        """Save the reference NSG features so :meth:`load_reference` can skip the diffusion pass."""
        if self._ref_data is None:
            raise RuntimeError("No reference set; call set_reference() first.")
        torch.save({"nsg": self._ref_data.cpu(), "checkpoint": self.checkpoint}, path)

    @torch.no_grad()
    def load_reference(self, path: str | Path) -> None:
        """Load reference NSG features written by :meth:`save_reference`."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        self._set_reference_data(blob["nsg"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        from detectzoo.detectors.video.nsgvd._upstream.mmd import MMD_batch2

        if self._ref_features is None:
            raise RuntimeError(
                "NSG-VD scores videos against real reference videos: call "
                "set_reference([...real videos...]) or load_reference(path) first."
            )
        x = self.nsg_features(input_data)[None]
        _, feat = self.model(x, out_feature=True)
        n_ref = self._ref_features.shape[0]
        k = self._kernel
        mmd = MMD_batch2(
            torch.cat([self._ref_features, feat], dim=0),
            n_ref,
            torch.cat([self._ref_data, x], dim=0).view(n_ref + 1, -1),
            k["sigma"],
            k["sigma0_u"],
            k["ep"],
            is_smooth=True,
        )
        score = float(mmd.reshape(-1)[0])
        if not math.isfinite(score):
            raise RuntimeError("NSG-VD produced a non-finite MMD score.")
        return self._make_result(score, checkpoint=self.checkpoint, num_reference=n_ref)
