"""NPR — Neighboring Pixel Relationships (CVPR 2024).

Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative Network
for Generalizable Deepfake Detection".

**Key idea:** CNN upsampling couples neighbors in a generator-specific way. NPR builds a
residual ``x - up(down(x))`` (nearest, half scale then back); a truncated ResNet classifies
from that map instead of raw RGB.

Upstream: https://github.com/chuangchuangtan/NPR-DeepfakeDetection

Architecture matches ``networks/resnet.py``. Default weights ``model_epoch_last_3090.pth``
"""

from __future__ import annotations

import math
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

_DEFAULT_CKPT_URL = (
    "https://raw.githubusercontent.com/chuangchuangtan/NPR-DeepfakeDetection/main/"
    "model_epoch_last_3090.pth"
)
_DEFAULT_CKPT_NAME = "model_epoch_last_3090.pth"
_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _NPRBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
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


class _NPRResNet(nn.Module):
    """Truncated ResNet-50 (layer1–2 only) with NPR in ``forward`` (official NPR repo)."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(_NPRBottleneck, 64, 3)
        self.layer2 = self._make_layer(_NPRBottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type[_NPRBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers_list: list[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers_list.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers_list)

    @staticmethod
    def _interpolate_residual(img: torch.Tensor, factor: float) -> torch.Tensor:
        _, _, h, w = img.shape
        hh, ww = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
        half = F.interpolate(img, size=(hh, ww), mode="nearest")
        return F.interpolate(half, size=(h, w), mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        npr = x - self._interpolate_residual(x, 0.5)
        x = self.conv1(npr * (2.0 / 3.0))
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))


def _load_checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        raw: Any = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected checkpoint contents in {path}: {type(raw)!r}")
    return {k.replace("module.", ""): v for k, v in raw.items()}


def _translate_duplicate(img: Image.Image, crop_size: int) -> Image.Image:
    if min(img.size) >= crop_size:
        return img
    w, h = img.size
    nw, nh = w * math.ceil(crop_size / w), h * math.ceil(crop_size / h)
    canvas = Image.new("RGB", (nw, nh))
    for i in range(0, nw, w):
        for j in range(0, nh, h):
            canvas.paste(img, (i, j))
    return canvas


@register_detector("npr_deepfake", aliases=["npr_cvpr2024", "npr_image"])
class NPRDeepfakeDetector(BaseDetector):
    """NPR image detector (official NPR-DeepfakeDetection)."""

    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        load_size: int | None = None,
        crop_size: int | None = None,
        genimage_protocol: bool = False,
        trim_odd_dims: bool = True,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.genimage_protocol = genimage_protocol
        self.trim_odd_dims = trim_odd_dims

        steps: list[Any] = []
        if genimage_protocol:
            cs = crop_size if crop_size is not None else 224
            self._crop_size = cs
            steps.append(transforms.Lambda(lambda im: _translate_duplicate(im, cs)))
            steps.append(transforms.CenterCrop(cs))
        else:
            self._crop_size = crop_size
            if load_size is not None:
                steps.append(transforms.Resize((load_size, load_size)))
            if crop_size is not None:
                steps.append(transforms.CenterCrop(crop_size))
        steps.extend([transforms.ToTensor(), transforms.Normalize(**_IMAGENET)])
        self._transform = transforms.Compose(steps)

        cache = get_cache_dir("npr_deepfake", cache_dir)
        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = cache / _DEFAULT_CKPT_NAME
            if not self._ckpt.is_file():
                download_file(_DEFAULT_CKPT_URL, self._ckpt)

        if not self._ckpt.is_file():
            raise FileNotFoundError(
                f"NPR checkpoint not found at `{self._ckpt}`. Place the authors' `.pth` there "
                "or pass `checkpoint_path=`."
            )

        self._model = _NPRResNet(num_classes=1)
        self._model.load_state_dict(_load_checkpoint_state(self._ckpt, self._device), strict=True)
        self._model.to(self._device).eval()

    def _normalize_input(self, input_data: Any) -> Image.Image:
        if hasattr(input_data, "mode") and hasattr(input_data, "convert"):
            return input_data.convert("RGB")
        path = Path(str(input_data))
        if path.is_file():
            return load_image(path)
        raise TypeError(
            f"Expected a PIL Image or a path to an image file; got {type(input_data).__name__}."
        )

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        t = self._transform(self._normalize_input(input_data)).unsqueeze(0).to(self._device)
        if self.trim_odd_dims:
            _, _, w, h = t.shape
            if w % 2 == 1:
                t = t[:, :, :-1, :]
            if h % 2 == 1:
                t = t[:, :, :, :-1]
        score = self._model(t).sigmoid().item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
            genimage_protocol=self.genimage_protocol,
        )
