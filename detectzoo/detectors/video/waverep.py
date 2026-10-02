"""WaveRep — Generalizable AI-generated Video Detection (NeurIPS 2025).

Reference:
    Corvi et al., "Seeing What Matters: Generalizable AI-generated Video
    Detection with Forensic-Oriented Augmentation", NeurIPS 2025.
    https://arxiv.org/abs/2506.16802

The key idea: a DINOv2 ViT (with registers) is fine-tuned frame-by-frame with a
wavelet-based augmentation that swaps frequency bands between real and fake
frames, pushing the model towards forensic cues that generalise across
generators.  At test time each frame gets a logit (LLR > 0 means synthetic) and
the video-level decision averages the logits of the first 64 frames.

Two checkpoints are released:

* ``"G1"`` — trained on a single generator (Pyramid Flow).
* ``"G4"`` — trained on Pyramid Flow, CogVideoX 1.5, Allegro and OpenSora Plan.

Upstream: https://github.com/grip-unina/WaveRep-SyntheticVideoDetection
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.v2.functional as TF

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.image.resnet50_binary import load_pytorch_checkpoint
from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet"
# variant -> (checkpoint filename, md5)
_CHECKPOINTS = {
    "G1": ("weights_dinov2_G1.ckpt", "bc9852a7c1d3bfdb60a20b4497e575bb"),
    "G4": ("weights_dinov2_G4.ckpt", "8bf19e6f68a92bed600dd97fbed3f2cd"),
}
_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"
_CROP = 504
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@register_detector("waverep", aliases=["wave_rep"])
class WaveRepDetector(BaseVideoDetector):
    """WaveRep frame-level DINOv2 detector (Corvi et al., NeurIPS 2025).

    ``score`` is the sigmoid of the aggregated frame logit, so the default
    ``threshold=0.5`` matches the paper's ``LLR > 0`` decision rule.

    Parameters
    ----------
    variant : str
        ``"G4"`` (four training generators, default) or ``"G1"`` (Pyramid Flow
        only).
    checkpoint_path : str or Path, optional
        Local ``.ckpt`` file.  Downloaded from the official GRIP server when
        omitted.
    threshold : float
        Decision boundary on the sigmoid score (default 0.5).
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    num_frames : int or None
        Frames scored per video (default 64, as in the paper).  ``None`` scores
        every frame, like the upstream demo script.
    sampling : str
        ``"first"`` (default, as in the paper) or ``"uniform"``.
    aggregation : str
        How frame logits are combined: ``"mean"`` (default), ``"max"`` or
        ``"median"``.
    batch_size : int
        Frames per forward pass.
    cache_dir : str or Path, optional
        Override the default cache directory (``.detectzoo_data``).
    """

    def __init__(
        self,
        *,
        variant: str = "G4",
        checkpoint_path: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        num_frames: int | None = 64,
        sampling: str = "first",
        aggregation: str = "mean",
        batch_size: int = 16,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold,
            device=device,
            num_frames=num_frames,
            sampling=sampling,
            aggregation=aggregation,
            **kwargs,
        )
        variant = variant.upper()
        if variant not in _CHECKPOINTS:
            raise ValueError(f"Unknown WaveRep variant '{variant}'; use 'G1' or 'G4'.")
        self.variant = variant
        self.batch_size = max(int(batch_size), 1)

        # ---- resolve checkpoint path ----
        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            filename, md5 = _CHECKPOINTS[variant]
            self._ckpt = get_cache_dir("waverep", cache_dir) / filename
            if not self._ckpt.is_file():
                download_file(f"{_BASE_URL}/{filename}", self._ckpt)
                if _md5(self._ckpt) != md5:
                    self._ckpt.unlink()
                    raise RuntimeError(
                        f"Checksum mismatch for {filename}; the download was removed, please retry."
                    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import timm

        logger.info("Loading WaveRep (%s) from %s …", self.variant, self._ckpt)
        # The checkpoint holds every weight, so skip downloading DINOv2 itself.
        model = timm.create_model(_ARCH, num_classes=1, pretrained=False, img_size=_CROP)
        state = load_pytorch_checkpoint(self._ckpt, torch.device("cpu"))
        if "state_dict" in state:
            state = {
                k[len("model.") :]: v
                for k, v in state["state_dict"].items()
                if k.startswith("model.")
            }
        model.load_state_dict(state)
        self._model = model.to(self._device).eval()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(frames: torch.Tensor) -> torch.Tensor:
        """``uint8 [T, C, H, W]`` → normalised ``float [T, 3, 504, 504]``."""
        if frames.shape[1] == 1:
            frames = frames.expand(-1, 3, -1, -1)
        # Center crop (zero-padding smaller frames), as upstream CenterCrop does.
        frames = TF.center_crop(frames, [_CROP, _CROP])
        frames = TF.to_dtype(frames, torch.float32, scale=True)
        return TF.normalize(frames, mean=_MEAN, std=_STD)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _frame_logits(self, frames: torch.Tensor) -> torch.Tensor:
        """Return one logit per frame of a ``uint8 [T, C, H, W]`` tensor."""
        logits = []
        for start in range(0, frames.shape[0], self.batch_size):
            batch = self._preprocess(frames[start : start + self.batch_size])
            logits.append(self.model(batch.to(self._device))[:, -1].float().cpu())
        return torch.cat(logits)

    def predict(self, input_data: Any) -> DetectionResult:
        max_frames = self.num_frames if self.sampling == "first" else None
        frames = self._sample_frames(self._normalise_input(input_data, max_frames=max_frames))
        frame_logits = self._frame_logits(frames)
        logit = self._aggregate_scores(frame_logits)
        score = float(torch.sigmoid(torch.tensor(logit)))
        return self._make_result(
            score,
            logit=logit,
            frame_logits=frame_logits.tolist(),
            num_frames=int(frames.shape[0]),
            variant=self.variant,
            checkpoint=str(self._ckpt),
        )
