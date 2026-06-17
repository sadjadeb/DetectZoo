"""LaDeDa — Locally Aware Deepfake Detection Algorithm.

Reference:
    Cavia et al., "Real-Time Deepfake Detection in the Real-World", arXiv 2024.
    https://arxiv.org/abs/2406.09398

The key idea: split each image into 9×9 patches, score each patch independently
with a compact ResNet-50 variant, then pool patch scores into the image-level
detection score. Using only local patch information (9×9 receptive field) forces
the model to focus on local generation artifacts rather than global semantics,
which significantly improves cross-generator generalization.

Upstream:   https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld
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
from detectzoo.datasets._download import get_cache_dir
from detectzoo.detectors.image.resnet50_binary import load_pytorch_checkpoint
from detectzoo.utils.io import load_image

_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_CKPT_NAME = "ForenSynth_LaDeDa.pth"
_GDRIVE_ID = "1KxNdnPRJJTuqxmzBPiGsg43tXzO8AN2d"
_PATCH_SIZE = 9
_LOAD_SIZE = 256


# ---------------------------------------------------------------------------
# ResNet-50 variant with 9×9 receptive field
# ---------------------------------------------------------------------------


def _conv3x3(inp: int, out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, out, 3, stride=stride, padding=0, bias=False)


def _conv1x1(inp: int, out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, out, 1, stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_1x1: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = (
            _conv1x1(planes, planes, stride) if use_1x1 else _conv3x3(planes, planes, stride)
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        if identity.shape[-2:] != out.shape[-2:]:
            diff_h = identity.size(-2) - out.size(-2)
            diff_w = identity.size(-1) - out.size(-1)
            identity = identity[:, :, : identity.size(-2) - diff_h, : identity.size(-1) - diff_w]
        return self.relu(out + identity)


class _LaDeDaNet(nn.Module):
    """ResNet-50 variant with restricted receptive field for patch-level scoring."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = _conv1x1(3, 64)
        self.conv2 = _conv3x3(64, 64)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            64, 3, stride=2, first_block_1x1=False, later_blocks_1x1=True
        )
        self.layer2 = self._make_layer(
            128, 4, stride=2, first_block_1x1=False, later_blocks_1x1=True
        )
        self.layer3 = self._make_layer(
            256, 6, stride=2, first_block_1x1=True, later_blocks_1x1=True
        )
        self.layer4 = self._make_layer(
            512, 3, stride=1, first_block_1x1=True, later_blocks_1x1=True
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * _Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        first_block_1x1: bool = False,
        later_blocks_1x1: bool = True,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )
        layers = [_Bottleneck(self.inplanes, planes, stride, downsample, use_1x1=first_block_1x1)]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes, use_1x1=later_blocks_1x1))
        return nn.Sequential(*layers)

    @staticmethod
    def _interpolate_residual(img: torch.Tensor, factor: float) -> torch.Tensor:
        return F.interpolate(
            F.interpolate(
                img,
                scale_factor=factor,
                mode="nearest",
                recompute_scale_factor=True,
            ),
            scale_factor=1.0 / factor,
            mode="nearest",
            recompute_scale_factor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self._interpolate_residual(x, 0.5)
        x = self.relu(self.bn1(self.conv2(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


@register_detector("ladeda", aliases=["la_de_da", "ladeda_realtime"])
class LaDeDaDetector(BaseDetector):
    """LaDeDa patch-level deepfake detector (Cavia et al., 2024).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to ``ForenSynths.pth``. Downloaded automatically from Google Drive
        when omitted.
    patch_size : int
        Local receptive-field size in pixels (default 9, matching the paper).
    load_size : int or None
        Resize each image to ``load_size × load_size`` before inference.
        Pass ``None`` to keep the original image size.
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
        patch_size: int = _PATCH_SIZE,
        load_size: int | None = _LOAD_SIZE,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.patch_size = int(patch_size)
        self.load_size = None if load_size is None else int(load_size)

        cache = get_cache_dir("ladeda", cache_dir)
        self._ckpt = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else cache / _CKPT_NAME
        )

        if not self._ckpt.is_file():
            self._ensure_download(cache)

        self._model = _LaDeDaNet(num_classes=1)
        raw = load_pytorch_checkpoint(self._ckpt, self._device)
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()

        steps: list[Any] = []
        if self.load_size is not None:
            steps.append(transforms.Resize((self.load_size, self.load_size)))
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(**_IMAGENET),
            ]
        )
        self._transform = transforms.Compose(steps)

    def _ensure_download(self, cache: Path) -> None:
        import gdown

        gdown.download_folder(
            id=_GDRIVE_ID,
            output=str(cache),
            quiet=False,
            use_cookies=False,
        )

        if not self._ckpt.is_file():
            matches = list(cache.rglob(_CKPT_NAME))
            if not matches:
                raise FileNotFoundError(
                    f"Downloaded LaDeDa weights folder but `{_CKPT_NAME}` "
                    f"was not found under {cache}."
                )
            matches[0].rename(self._ckpt)

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
        img = self._normalize_input(input_data)
        img_t = self._transform(img).unsqueeze(0).to(self._device)
        logits = self._model(img_t)
        score = logits.sigmoid().item()

        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
            receptive_field=self.patch_size,
            load_size=self.load_size,
        )
