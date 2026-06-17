"""SAFE — Simple Preserved and Augmented FEatures (KDD 2025).

Reference:
    Li et al., "Improving Synthetic Image Detection Towards Generalization:
    An Image Transformation Perspective", KDD 2025.
    https://arxiv.org/abs/2408.06741

The key idea: applies a Discrete Wavelet Transform (DWT) high-pass filter as
preprocessing to expose high-frequency artifacts left by image generators, then
classifies with a lightweight truncated ResNet-50 (layers 1–2 only).

Upstream: https://github.com/Ouxiang-Li/SAFE
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
from detectzoo.utils.io import load_image

_CKPT_URL = "https://github.com/Ouxiang-Li/SAFE/raw/main/checkpoint/checkpoint-best.pth"
_CKPT_NAME = "checkpoint-best.pth"


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _SAFEBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _SAFEResNet(nn.Module):
    """Truncated ResNet-50 (layers 1–2) with DWT high-pass preprocessing."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(_SAFEBottleneck, 64, 3)
        self.layer2 = self._make_layer(_SAFEBottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[_SAFEBottleneck], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers: list[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _preprocess_dwt(
        x: torch.Tensor, mode: str = "symmetric", wave: str = "bior1.3"
    ) -> torch.Tensor:
        from pytorch_wavelets import DWTForward

        dwt = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        _, yh = dwt(x)
        hp = yh[0][:, :, 2, :, :]
        return transforms.functional.resize(hp, [x.shape[-2], x.shape[-1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_dwt(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))


def _load_safe_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    raw: Any = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint contents in {path}: {type(state)!r}")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


@register_detector("safe", aliases=["safe_kdd2025", "safe_detector"])
class SAFEDetector(BaseDetector):
    """SAFE image detector.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to ``checkpoint-best.pth``.  Downloaded automatically from the
        official GitHub repository when omitted.
    input_size : int
        Spatial input size (default 200, matching the paper's crop setting).
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
        input_size: int = 200,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = get_cache_dir("safe", cache_dir) / _CKPT_NAME
            download_file(_CKPT_URL, self._ckpt)

        self._model = _SAFEResNet(num_classes=2)
        _load_safe_checkpoint(self._model, self._ckpt, self._device)
        self._model.to(self._device).eval()

        self._transform = transforms.Compose(
            [
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
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
        logits = self._model(x)
        score = logits.softmax(dim=-1)[:, 1].item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
        )
