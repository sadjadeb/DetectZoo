"""CNNSpot — CNN-Generated Image Detection (CVPR 2020).

Reference:
    Wang et al., "CNN-Generated Images Are Surprisingly Easy to Spot...For Now",
    CVPR 2020.
    https://arxiv.org/abs/1912.11035

The key idea: CNN-based generators leave systematic frequency-domain artifacts in
synthesized images. A ResNet-50 trained for binary (real vs fake) classification
learns to detect those traces, generalizing across generator architectures.

Upstream: https://github.com/PeterWang512/CNNDetection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.image.resnet50_binary import build_resnet50_binary, load_pytorch_checkpoint
from detectzoo.utils.io import load_image

_CKPT_URL = "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"
_CKPT_NAME = "blur_jpg_prob0.5.pth"


@register_detector("cnnspot", aliases=["cnn_spot"])
class CNNSpotDetector(BaseDetector):
    """CNNSpot binary classifier (Wang et al., CVPR 2020).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to the ResNet-50 ``.pth`` checkpoint.  Downloaded automatically
        from the official Dropbox link when omitted.
    threshold : float
        Decision boundary (default 0.5).
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    cache_dir : str or Path, optional
        Override the default cache directory (``.detectzoo_data``).
    """

    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        # ---- resolve checkpoint path ----
        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = get_cache_dir("cnnspot", cache_dir) / _CKPT_NAME
            download_file(_CKPT_URL, self._ckpt)

        # ---- preprocessing ----
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # ---- build model ----
        self._model = build_resnet50_binary()
        self._model.load_state_dict(load_pytorch_checkpoint(self._ckpt, self._device)["model"])
        self._model.to(self._device).eval()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _normalize_input(self, input_data: Any) -> Image.Image:
        if hasattr(input_data, "mode") and hasattr(input_data, "convert"):
            return input_data.convert("RGB")
        path = Path(str(input_data))
        if path.is_file():
            return load_image(path)
        raise TypeError(
            f"Expected a PIL Image or a path to an image file; got {type(input_data).__name__}."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        img = self._normalize_input(input_data)
        x = self._transform(img).unsqueeze(0).to(self._device)
        score = self._model(x).sigmoid().item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
        )
