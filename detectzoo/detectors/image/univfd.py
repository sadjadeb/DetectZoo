"""Univ-FD — Universal Fake Image Detector (CVPR 2023).

Reference:
    Ojha et al., "Towards Universal Fake Image Detectors that Generalize Across
    Generative Models", CVPR 2023.
    https://arxiv.org/abs/2302.10174

The key idea: Univ-FD uses pretrained CLIP image embeddings to represent images in a rich,
semantic feature space. A simple linear classifier is trained on top to distinguish real vs fake.

Upstream: https://github.com/WisconsinAIVision/UniversalFakeDetect
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.image.resnet50_binary import load_pytorch_checkpoint
from detectzoo.utils.io import load_image

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_CLIP_VIT_L14_DIM = 768

_CKPT_URL = (
    "https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/"
    "pretrained_weights/fc_weights.pth"
)
_CKPT_NAME = "fc_weights.pth"


class _CLIPLinearModel(nn.Module):
    """CLIP visual encoder + single-logit linear head (mirrors official ``CLIPModel``)."""

    def __init__(self, clip_model: nn.Module, feature_dim: int = _CLIP_VIT_L14_DIM) -> None:
        super().__init__()
        self.backbone = clip_model
        self.fc = nn.Linear(feature_dim, 1)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_image(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_image(x)
        # open_clip may return float16 on some builds; cast to match fc dtype.
        features = features.to(self.fc.weight.dtype)
        return self.fc(features)


def _build_clip_linear(device: torch.device) -> _CLIPLinearModel:
    """Build the CLIP ViT-L/14 backbone + linear head."""
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "Univ-FD requires `open-clip-torch`. "
            "Install it with: pip install 'open-clip-torch>=2.20'"
        ) from e

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    model = _CLIPLinearModel(clip_model, feature_dim=_CLIP_VIT_L14_DIM)
    return model


@register_detector("univfd", aliases=["univ_fd", "universal_fake_detect"])
class UnivFDDetector(BaseDetector):
    """Universal Fake Image Detector (Ojha et al., CVPR 2023).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to ``fc_weights.pth``.  Downloaded automatically from the
        official GitHub repository when omitted.
    threshold : float
        Decision boundary (default 0.5, matching the paper).
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
            self._ckpt = get_cache_dir("univfd", cache_dir) / _CKPT_NAME
            download_file(_CKPT_URL, self._ckpt)

        # ---- build model ----
        self._model = _build_clip_linear(self._device)

        fc_state = load_pytorch_checkpoint(self._ckpt, self._device)
        self._model.fc.load_state_dict(fc_state)

        self._model.to(self._device).eval()

        # ---- preprocessing (matches official validate.py) ----
        self._transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )

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
