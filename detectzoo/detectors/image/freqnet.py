"""FreqNet — Frequency-Aware Deepfake Detection (AAAI 2024).

Reference:
    Tan et al., "Frequency-Aware Deepfake Detection: Improving Generalizability
    through Frequency Space Domain Learning", AAAI 2024.
    https://arxiv.org/abs/2403.07240

Upstream:   https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
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

_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_CKPT_NAME = "4-classes-freqnet-v2.pth"
_CKPT_URL = (
    "https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection/raw/main/4-classes-freqnet-v2.pth"
)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _Bottleneck(nn.Module):
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
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _FreqNet(nn.Module):
    """Matches upstream ``networks/freqnet.py``."""

    def __init__(
        self,
        block: type[_Bottleneck] = _Bottleneck,
        layers: tuple[int, int] = (3, 4),
        num_classes: int = 1,
        *,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()

        self.weight1 = nn.Parameter(torch.empty(64, 3, 1, 1))
        self.bias1 = nn.Parameter(torch.empty(64))
        self.realconv1 = _conv1x1(64, 64, stride=1)
        self.imagconv1 = _conv1x1(64, 64, stride=1)

        self.weight2 = nn.Parameter(torch.empty(64, 64, 1, 1))
        self.bias2 = nn.Parameter(torch.empty(64))
        self.realconv2 = _conv1x1(64, 64, stride=1)
        self.imagconv2 = _conv1x1(64, 64, stride=1)

        self.weight3 = nn.Parameter(torch.empty(256, 256, 1, 1))
        self.bias3 = nn.Parameter(torch.empty(256))
        self.realconv3 = _conv1x1(256, 256, stride=1)
        self.imagconv3 = _conv1x1(256, 256, stride=1)

        self.weight4 = nn.Parameter(torch.empty(256, 256, 1, 1))
        self.bias4 = nn.Parameter(torch.empty(256))
        self.realconv4 = _conv1x1(256, 256, stride=1)
        self.imagconv4 = _conv1x1(256, 256, stride=1)

        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, _Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block: type[_Bottleneck],
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
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _hfreq_wh(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        assert scale > 2
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        b, c, h, w = x.shape
        x[
            :,
            :,
            h // 2 - h // scale : h // 2 + h // scale,
            w // 2 - w // scale : w // 2 + w // scale,
        ] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        return F.relu(x, inplace=True)

    def _hfreq_c(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        assert scale > 2
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1)
        b, c, h, w = x.shape
        x[:, c // 2 - c // scale : c // 2 + c // scale, :, :] = 0.0
        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        return F.relu(x, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HFRI
        x = self._hfreq_wh(x, 4)
        x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0)
        x = F.relu(x, inplace=True)

        # HFRFC
        x = self._hfreq_c(x, 4)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # HFRFS
        x = self._hfreq_wh(x, 4)
        x = F.conv2d(x, self.weight2, self.bias2, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        # HFRFC
        x = self._hfreq_c(x, 4)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.maxpool(x)
        x = self.layer1(x)

        # HFRFS
        x = self._hfreq_wh(x, 4)
        x = F.conv2d(x, self.weight3, self.bias3, stride=1, padding=0)
        x = F.relu(x, inplace=True)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # HFRFS
        x = self._hfreq_wh(x, 4)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc1(x)


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state.items()}


def _load_freqnet_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location=device, weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type at {path}")
    return _strip_module_prefix(state)


@register_detector("freqnet", aliases=["freq_net", "freqnet_aaai2024"])
class FreqNetDetector(BaseDetector):
    """FreqNet image detector (Tan et al., AAAI 2024).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to the FreqNet ``.pth`` checkpoint. Downloaded automatically
        from the authors' GitHub repository when omitted.
    threshold : float
        Decision boundary (default 0.5).
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...).
    cache_dir : str or Path, optional
        Override the default cache directory (``.detectzoo_data``).
    load_size : int
        Square resize size applied before center cropping (default 256).
    crop_size : int
        Center crop size passed to the model (default 224).
    **kwargs : Any
        Additional keyword arguments forwarded to ``BaseDetector``.
    """

    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        load_size: int = 256,
        crop_size: int = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        cache = get_cache_dir("freqnet", cache_dir)
        self._ckpt = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else cache / _CKPT_NAME
        )
        if not self._ckpt.is_file():
            download_file(_CKPT_URL, self._ckpt)

        self._transform = transforms.Compose(
            [
                transforms.Resize((load_size, load_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(**_IMAGENET),
            ]
        )

        self._model = _FreqNet(num_classes=1)
        state = _load_freqnet_checkpoint(self._ckpt, torch.device("cpu"))
        self._model.load_state_dict(state, strict=True)
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
        img = self._normalize_input(input_data)
        x = self._transform(img).unsqueeze(0).to(self._device)
        score = self._model(x).sigmoid().item()
        return self._make_result(float(score), checkpoint=str(self._ckpt))
