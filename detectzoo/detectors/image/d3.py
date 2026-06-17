"""D³ — Discrepancy Deepfake Detector (CVPR 2025).

Reference:
    Yang et al., "D³: Scaling Up Deepfake Detection by Learning from Discrepancy",
    CVPR 2025.
    https://arxiv.org/abs/2404.04584

The key idea: detects AI-generated images by modeling discrepancies at three levels:
pixel (data), feature distribution, and generation dynamics.It utilizes a dual-branch
approach where CLIP ViT-L/14 processes both the original image and a patch-shuffled
version.  A learned transformer attention head aggregates penultimate-layer features
from both views, using the discrepancy between intact and distorted representations.

Upstream: https://github.com/BigAandSmallq/D3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.utils.io import load_image

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_CKPT_URL = "https://github.com/BigAandSmallq/D3/raw/main/ckpt/classifier.pth"
_CKPT_NAME = "d3_classifier.pth"

_PENULTIMATE_DIM = 1024  # ViT-L/14 pre-projection width


class _TransformerAttention(nn.Module):
    """Transformer attention head for aggregating penultimate-layer features."""

    def __init__(self, input_dim: int, output_dim: int, last_dim: int = 1) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim, last_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = self.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5))
        out = torch.matmul(attn, v)
        return self.fc(out.reshape(out.shape[0], -1))


class _D3Model(nn.Module):
    """CLIP ViT-L/14 (frozen) + patch shuffle + TransformerAttention head."""

    def __init__(
        self,
        shuffle_times: int = 1,
        original_times: int = 1,
        patch_size: int | list[int] = 14,
    ) -> None:
        super().__init__()
        import open_clip

        self.shuffle_times = shuffle_times
        self.original_times = original_times
        self.patch_size = [patch_size] if isinstance(patch_size, int) else list(patch_size)

        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.clip = clip_model.visual
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self._features: torch.Tensor | None = None
        self._register_hook()

        self.attention_head = _TransformerAttention(
            _PENULTIMATE_DIM, shuffle_times + original_times, last_dim=1
        )

    def _register_hook(self) -> None:
        def hook(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            self._features = output.clone()

        for name, mod in self.clip.named_children():
            if name == "ln_post":
                mod.register_forward_hook(hook)
                return

    # ------------------------------------------------------------------
    # Distorting images
    # ------------------------------------------------------------------

    @staticmethod
    def _shuffle_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        shuffled = patches[:, :, torch.randperm(patches.size(-1))]
        return F.fold(
            shuffled,
            output_size=(x.shape[2], x.shape[3]),
            kernel_size=patch_size,
            stride=patch_size,
        )

    def _encode_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        self.clip(x)
        assert self._features is not None
        feat = self._features.clone()
        if feat.dim() == 3:
            feat = feat[:, 0, :]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.shuffle_times):
                features.append(
                    self._encode_penultimate(self._shuffle_patches(x, self.patch_size[0]))
                )
            for _ in range(self.original_times):
                features.append(self._encode_penultimate(x))
        stacked = torch.stack(features, dim=-2)
        return self.attention_head(stacked)


@register_detector("d3", aliases=["d3_cvpr2025", "discrepancy_deepfake_detector"])
class D3Detector(BaseDetector):
    """D³ image detector (Yang et al., CVPR 2025).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to ``classifier.pth`` (attention-head weights only). Downloaded
        automatically from the authors' GitHub repository when omitted.
    shuffle_times : int
        Number of shuffled views (default 1).
    patch_size : int
        Patch size for shuffling (default 14, matching ViT-L/14 patch grid).
    threshold : float
        Decision boundary (default 0.5).
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
        shuffle_times: int = 1,
        patch_size: int = 14,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = get_cache_dir("d3", cache_dir) / _CKPT_NAME
            download_file(_CKPT_URL, self._ckpt)

        self._model = _D3Model(
            shuffle_times=shuffle_times,
            original_times=1,
            patch_size=patch_size,
        )

        head_state = torch.load(self._ckpt, map_location=self._device, weights_only=False)
        if isinstance(head_state, dict) and "model" in head_state:
            head_state = head_state["model"]
        self._model.attention_head.load_state_dict(head_state, strict=True)
        self._model.to(self._device).eval()

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
