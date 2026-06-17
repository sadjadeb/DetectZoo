"""DRCT — Diffusion Reconstruction Contrastive Training (ICML 2024 Spotlight).

Reference:
    Chen et al., "DRCT: Diffusion Reconstruction Contrastive Training towards
    Universal Detection of Diffusion Generated Images", ICML 2024.
    https://proceedings.mlr.press/v235/chen24ay.html

The key idea: train a classifier with contrastive learning using original images,
generated images, and their diffusion-reconstructed counterparts. A margin-based
contrastive loss forces the detector to learn subtle diffusion artifacts that
transfer across generators.

Upstream: https://github.com/beibuwandeluori/DRCT
Weights: https://modelscope.cn/datasets/BokingChen/DRCT-2M/files  (pretrained.zip)
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.utils.io import load_image

_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_DEFAULT_CKPT_NAME = "14_acc0.9996.pth"
_DEFAULT_CKPT_IN_ZIP = (
    "pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"
)
_PRETRAINED_ZIP_URL = (
    "https://modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/pretrained.zip"
)
_PRETRAINED_ZIP_NAME = "pretrained.zip"
_UPSTREAM_README = "https://github.com/beibuwandeluori/DRCT"


class _DRCTContrastiveModel(nn.Module):
    """ConvNeXt-Base (ImageNet-22K) with contrastive embedding head + 2-class classifier."""

    def __init__(self, embedding_size: int = 1024) -> None:
        super().__init__()
        import timm

        backbone = timm.create_model("convnext_base_in22k", pretrained=False)

        in_features = backbone.head.fc.in_features
        backbone.head.fc = nn.Linear(in_features, embedding_size)
        self.model = backbone
        self.fc = nn.Linear(embedding_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.model(x))


@register_detector("drct", aliases=["drct_icml2024", "drct_convnext"])
class DRCTDetector(BaseDetector):
    """DRCT binary detector (Chen et al., ICML 2024).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to a ``.pth`` checkpoint. When omitted the default checkpoint
        (``14_acc0.9996.pth``, trained on DRCT-2M with SDv1 diffusion
        reconstruction) is downloaded automatically from ModelScope.
    embedding_size : int
        Embedding dimension of the contrastive head (default 1024).
    threshold : float
        Decision boundary on the fake-class probability (default 0.5).
    device : str
        Torch device string.
    cache_dir : str or Path, optional
        Override the default cache directory (``.detectzoo_data``).
    """

    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        embedding_size: int = 1024,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        cache = get_cache_dir("drct", cache_dir)
        self._ckpt = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else cache / _DEFAULT_CKPT_NAME
        )

        if not self._ckpt.is_file():
            self._ensure_download(cache)

        self._model = _DRCTContrastiveModel(embedding_size=embedding_size)
        raw = torch.load(self._ckpt, map_location=self._device, weights_only=False)
        if isinstance(raw, dict):
            state = raw.get("model", raw)
            state = {k.replace("module.", ""): v for k, v in state.items()}
        else:
            state = raw
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()

        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(**_IMAGENET),
            ]
        )

    def _ensure_download(self, cache: Path) -> None:
        zip_path = cache / _PRETRAINED_ZIP_NAME
        if not zip_path.is_file():
            download_file(_PRETRAINED_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(_DEFAULT_CKPT_IN_ZIP, cache)

        extracted = cache / _DEFAULT_CKPT_IN_ZIP
        extracted.rename(self._ckpt)
        zip_path.unlink(missing_ok=True)

    # ---------------------------------------------------------------------------
    # Input handling
    # ---------------------------------------------------------------------------

    def _normalize_input(self, input_data: Any) -> Image.Image:
        if hasattr(input_data, "mode"):
            return input_data.convert("RGB")
        return load_image(Path(str(input_data)))

    # ---------------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        x = self._transform(self._normalize_input(input_data)).unsqueeze(0).to(self._device)
        score = self._model(x).softmax(dim=-1)[:, 1].item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
        )
