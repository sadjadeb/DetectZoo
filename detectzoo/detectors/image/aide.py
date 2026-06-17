"""AIDE — AI-generated Image DEtector with Hybrid Features (ICLR 2025).

Reference:
    Yan et al., "A Sanity Check for AI-generated Image Detection", ICLR 2025.
    https://arxiv.org/abs/2406.19435

The key idea: combines two complementary branches: (1) SRM high-pass filtered +
DCT-selected frequency patches processed by dual ResNet-50 trunks to capture artifact
cues, and (2) a frozen OpenCLIP ConvNeXt-XXLarge trunk for robust semantic features.
These signals are fused so the model jointly reasons over low-level artifacts and
high-level semantics for detection.

Upstream: https://github.com/shilinyan99/AIDE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.utils.io import load_image

_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_DEFAULT_CKPT_NAME = "progan_train.pth"
_GDRIVE_FOLDER = "1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4"


def _dct_mat(size: int) -> list[list[float]]:
    return [
        [
            (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
            * np.cos((j + 0.5) * np.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]


def _gen_filter(start: float, end: float, size: int) -> list[list[float]]:
    return [
        [0.0 if i + j > end or i + j < start else 1.0 for j in range(size)] for i in range(size)
    ]


class _Filter(nn.Module):
    def __init__(self, size: int, band_start: float, band_end: float, norm: bool = False) -> None:
        super().__init__()
        self.register_buffer("base", torch.tensor(_gen_filter(band_start, band_end, size)))
        self.norm = norm
        if norm:
            self.register_buffer(
                "ft_num", torch.sum(torch.tensor(_gen_filter(band_start, band_end, size)))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.base / self.ft_num) if self.norm else (x * self.base)


class _DCTRecModule(nn.Module):
    """DCT-based frequency patch selector"""

    def __init__(
        self, window_size: int = 32, stride: int = 16, output: int = 256, grade_N: int = 6
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.grade_N = grade_N
        self.N = (output // window_size) ** 2

        self.register_buffer("_D", torch.tensor(_dct_mat(window_size), dtype=torch.float32))
        self.register_buffer("_DT", torch.tensor(_dct_mat(window_size), dtype=torch.float32).T)

        self.unfold = nn.Unfold(kernel_size=window_size, stride=stride)
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), kernel_size=window_size, stride=window_size
        )
        self.level_filters = nn.ModuleList([_Filter(window_size, 0, window_size * 2)])
        self.grade_filters = nn.ModuleList(
            [
                _Filter(
                    window_size,
                    window_size * 2.0 / grade_N * i,
                    window_size * 2.0 / grade_N * (i + 1),
                    norm=True,
                )
                for i in range(grade_N)
            ]
        )

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        C, W, H = x.shape
        ws = self.window_size
        xu = self.unfold(x.unsqueeze(0)).squeeze(0)
        _, L = xu.shape
        xu = xu.T.reshape(L, C, ws, ws)
        xdct = self._D @ xu @ self._DT

        xp = self.level_filters[0](xdct)
        y = self._DT @ xp @ self._D
        level_xu = y

        grade = torch.zeros(L, device=x.device)
        w, k = 1.0, 2.0
        for gf in self.grade_filters:
            grade += w * gf(xdct.abs().log1p()).sum(dim=[1, 2, 3])
            w *= k

        _, idx = torch.sort(grade)

        def pick(i: torch.Tensor) -> torch.Tensor:
            return self.fold0(level_xu[i : i + 1].reshape(1, -1, 1))

        return (
            pick(idx[0]),
            pick(torch.flip(idx, [0])[0]),
            pick(idx[min(1, len(idx) - 1)]),
            pick(torch.flip(idx, [0])[min(1, len(idx) - 1)]),
        )


# ---------------------------------------------------------------------------
# SRM high-pass filters
# ---------------------------------------------------------------------------


class _HPF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from detectzoo.detectors.image.srm_filter_kernel import all_normalized_hpf_list

        hpf_5x5 = []
        for h in all_normalized_hpf_list:
            if h.shape[0] == 3:
                h = np.pad(h, ((1, 1), (1, 1)), mode="constant")
            hpf_5x5.append(h)
        weight = torch.tensor(np.array(hpf_5x5), dtype=torch.float32).view(30, 1, 5, 5)
        weight = nn.Parameter(weight.repeat(1, 3, 1, 1), requires_grad=False)
        self.hpf = nn.Conv2d(3, 30, 5, padding=2, bias=False)
        self.hpf.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hpf(x)


# ---------------------------------------------------------------------------
# ResNet-50 for noise features (input 30-ch from SRM)
# ---------------------------------------------------------------------------


def _conv3x3(inp: int, out: int, s: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, out, 3, stride=s, padding=1, bias=False)


def _conv1x1(inp: int, out: int, s: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inp, out, 1, stride=s, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp: int, planes: int, stride: int = 1, ds: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1, self.bn1 = _conv1x1(inp, planes), nn.BatchNorm2d(planes)
        self.conv2, self.bn2 = _conv3x3(planes, planes, stride), nn.BatchNorm2d(planes)
        self.conv3, self.bn3 = _conv1x1(planes, planes * 4), nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + identity)


class _AIDEResNet(nn.Module):
    """ResNet-50 with 30-channel input (SRM output)."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        ds = None
        if stride != 1 or self.inplanes != planes * 4:
            ds = nn.Sequential(
                _conv1x1(self.inplanes, planes * 4, stride), nn.BatchNorm2d(planes * 4)
            )
        layers = [_Bottleneck(self.inplanes, planes, stride, ds)]
        self.inplanes = planes * 4
        layers.extend(_Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1)


class _Mlp(nn.Module):
    def __init__(self, in_f: int, hid: int, out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_f, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _AIDEModel(nn.Module):
    """Full AIDE model"""

    def __init__(self) -> None:
        super().__init__()
        self.hpf = _HPF()
        self.model_min = _AIDEResNet()
        self.model_max = _AIDEResNet()
        self.fc = _Mlp(2048 + 256, 1024, 2)

        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup"
        )
        trunk = model.visual.trunk
        trunk.head.global_pool = nn.Identity()
        trunk.head.flatten = nn.Identity()
        self.openclip_convnext_xxl = trunk
        self.openclip_convnext_xxl.eval()
        for p in self.openclip_convnext_xxl.parameters():
            p.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(nn.Linear(3072, 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_minmin, x_maxmax = x[:, 0], x[:, 1]
        x_minmin1, x_maxmax1 = x[:, 2], x[:, 3]
        tokens = x[:, 4]

        hp_mm = self.hpf(x_minmin)
        hp_MM = self.hpf(x_maxmax)
        hp_mm1 = self.hpf(x_minmin1)
        hp_MM1 = self.hpf(x_maxmax1)

        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tokens.device).view(
            3, 1, 1
        )
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tokens.device).view(
            3, 1, 1
        )
        dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=tokens.device).view(3, 1, 1)
        dinov2_std = torch.tensor([0.229, 0.224, 0.225], device=tokens.device).view(3, 1, 1)

        with torch.no_grad():
            convnext_in = tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            feat = self.openclip_convnext_xxl(convnext_in)
            x_0 = self.convnext_proj(self.avgpool(feat).flatten(1))

        x_1 = (
            self.model_min(hp_mm)
            + self.model_max(hp_MM)
            + self.model_min(hp_mm1)
            + self.model_max(hp_MM1)
        ) / 4.0
        return self.fc(torch.cat([x_0, x_1], dim=1))


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

_RESIZE = transforms.Compose([transforms.Resize([256, 256]), transforms.Normalize(**_IMAGENET)])


@register_detector("aide", aliases=["aide_iclr2025", "aide_detector"])
class AIDEDetector(BaseDetector):
    """AIDE image detector (Yan et al., ICLR 2025).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to the trained AIDE ``.pth`` checkpoint.  If omitted, attempts
        download from Google Drive.
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
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        cache = get_cache_dir("aide", cache_dir)
        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = cache / _DEFAULT_CKPT_NAME

        if not self._ckpt.is_file():
            import gdown

            cache.mkdir(parents=True, exist_ok=True)
            gdown.download_folder(
                id=_GDRIVE_FOLDER,
                output=str(cache),
                quiet=False,
                use_cookies=False,
            )

        self._model = _AIDEModel()
        raw = torch.load(self._ckpt, map_location="cpu", weights_only=False)
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)
        self._model.to(self._device).eval()

        self._dct = _DCTRecModule()
        self._dct.to(self._device)

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
        tensor = transforms.ToTensor()(img).to(self._device)

        x_mm, x_MM, x_mm1, x_MM1 = self._dct(tensor)

        x_0 = _RESIZE(tensor.unsqueeze(0) if tensor.dim() == 3 else tensor)
        x_mm = _RESIZE(x_mm)
        x_MM = _RESIZE(x_MM)
        x_mm1 = _RESIZE(x_mm1)
        x_MM1 = _RESIZE(x_MM1)

        batch = torch.stack([x_mm, x_MM, x_mm1, x_MM1, x_0], dim=1)
        logits = self._model(batch)
        score = logits.softmax(dim=-1)[:, 1].item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
        )
