"""PatchCraft (RPTC) — rich vs poor texture contrast (arXiv 2311.12397).

Zhong et al., "PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection".

The key idea: Build two versions of the same image: one dominated by *poor* texture patches and
one dominated by *rich* texture patches; detect AI images from the contrast between the two.

This implementation matches the authors' benchmark code path named *RPTC* and uses their released
checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
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

_DEFAULT_CKPT_NAME = "RPTC.pth"
_HF_REPO_ID = "slxhere/PatchCraft"
_UPSTREAM_WEIGHTS = "https://fdmas.github.io/AIGCDetect/"


# ---------------------------------------------------------------------------
# SRM high-pass filters
# ---------------------------------------------------------------------------


def _srm_hpf_weights() -> torch.Tensor:
    from detectzoo.detectors.image.srm_filter_kernel import all_normalized_hpf_list

    hpf_5x5 = []
    for h in all_normalized_hpf_list:
        if h.shape[0] == 3:
            h = F.pad(torch.from_numpy(h).float(), (1, 1, 1, 1)).numpy()
        hpf_5x5.append(h)
    return torch.tensor(hpf_5x5, dtype=torch.float32).view(30, 1, 5, 5)


class _HPF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = nn.Parameter(_srm_hpf_weights(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hpf(x)


# ---------------------------------------------------------------------------
# RPTC network
# ---------------------------------------------------------------------------


def _conv_bn_relu(ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU())


class RPTCNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.group1 = _HPF()
        self.group1_b = nn.Sequential(
            nn.Conv2d(90, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardtanh(min_val=-5, max_val=5),
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )
        self.group4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )
        self.group5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_poor = x[:, 0]
        img_rich = x[:, 1]
        batch, ch, h, w = img_poor.shape

        poor = self.group1(img_poor.reshape(-1, 1, h, w)).reshape(batch, -1, h, w)
        poor = self.group1_b(poor)
        rich = self.group1(img_rich.reshape(-1, 1, h, w)).reshape(batch, -1, h, w)
        rich = self.group1_b(rich)

        out = self.group2(poor - rich)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = self.advpool(out).view(out.size(0), -1)
        return self.fc2(out)


def _load_weights(model: nn.Module, ckpt: Path, device: torch.device) -> None:
    raw = load_pytorch_checkpoint(ckpt, device)
    state = raw.get("model") or raw.get("netC") or raw if isinstance(raw, dict) else raw
    state = {k.replace("module.", "").removeprefix("model."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# RPTC patch preprocessing
# ---------------------------------------------------------------------------


def _edge_density(img: torch.Tensor) -> float:
    return float(
        torch.abs(img[:, :-1] - img[:, 1:]).sum()
        + torch.abs(img[:, :, :-1] - img[:, :, 1:]).sum()
        + torch.abs(img[:, :-1, :-1] - img[:, 1:, 1:]).sum()
        + torch.abs(img[:, :-1, 1:] - img[:, 1:, :-1]).sum()
    )


def _processing_rptc(
    img: Image.Image, *, load_size: int, patch_num: int, seed: int
) -> torch.Tensor:
    num_block = 2**patch_num
    patch_size = load_size // num_block

    if min(img.size) < patch_size:
        img = transforms.Resize((patch_size, patch_size))(img)

    x = transforms.ToTensor()(img)
    _, h, w = x.shape
    rng = Random(seed)

    crops = sorted(
        [
            (
                x[:, cy : cy + patch_size, cx : cx + patch_size],
                _edge_density(x[:, cy : cy + patch_size, cx : cx + patch_size]),
            )
            for _ in range(num_block * num_block * 3)
            for cx, cy in [
                (
                    rng.randrange(0, max(1, w - patch_size + 1)),
                    rng.randrange(0, max(1, h - patch_size + 1)),
                )
            ]
        ],
        key=lambda t: t[1],
    )

    def _fill(indices) -> torch.Tensor:
        t = torch.zeros(3, load_size, load_size, dtype=x.dtype)
        for k, (ii, jj) in enumerate((i, j) for i in range(num_block) for j in range(num_block)):
            t[
                :, ii * patch_size : (ii + 1) * patch_size, jj * patch_size : (jj + 1) * patch_size
            ] = crops[indices[k]][0]
        return t

    n = num_block * num_block
    return torch.stack((_fill(list(range(n))), _fill(list(range(-1, -n - 1, -1)))))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("patchcraft", aliases=["patch_craft", "patchcraft_detector"])
class PatchCraftDetector(BaseDetector):
    """PatchCraft RPTC detector (Zhong et al., arXiv 2311.12397).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to ``RPTC.pth``. Auto-downloaded from HuggingFace when omitted.
    load_size : int
        Resize shorter edge before patch extraction (default 256).
    patch_num : int
        Grid size: ``2**patch_num × 2**patch_num`` patches (default 2).
    seed : int
        RNG seed for reproducible patch sampling (default 42).
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
        load_size: int = 256,
        patch_num: int = 2,
        seed: int = 42,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.load_size = int(load_size)
        self.patch_num = int(patch_num)
        self.seed = int(seed)

        cache = get_cache_dir("patchcraft", cache_dir)
        self._ckpt = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else cache / _DEFAULT_CKPT_NAME
        )

        if not self._ckpt.is_file():
            self._ensure_download(cache)

        self._model = RPTCNet()
        _load_weights(self._model, self._ckpt, self._device)
        self._model.to(self._device).eval()

    def _ensure_download(self, cache: Path) -> None:
        from huggingface_hub import hf_hub_download

        self._ckpt = Path(
            hf_hub_download(repo_id=_HF_REPO_ID, filename=_DEFAULT_CKPT_NAME, cache_dir=str(cache))
        )

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _normalize_input(self, input_data: Any) -> Image.Image:
        if hasattr(input_data, "mode"):
            return input_data.convert("RGB")
        return load_image(Path(str(input_data)))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        x = (
            _processing_rptc(
                self._normalize_input(input_data),
                load_size=self.load_size,
                patch_num=self.patch_num,
                seed=self.seed,
            )
            .unsqueeze(0)
            .to(self._device)
        )
        score = self._model(x).sigmoid().item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
            load_size=self.load_size,
            patch_num=self.patch_num,
            seed=self.seed,
        )
