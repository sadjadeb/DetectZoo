"""CO-SPY - Combining Semantic and Pixel Features (CVPR 2025).

Reference:
    Cheng et al., "CO-SPY: Combining Semantic and Pixel Features to Detect
    Synthetic Images by AI", CVPR 2025.
    https://arxiv.org/abs/2503.18286

Upstream: https://github.com/Megum1/CO-SPY
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.utils.io import load_image

_HF_REPO = "ruojiruoli/Co-Spy-Pretrained-Weights"
_CHECKPOINT_ZIPS = {
    "progan": "progan.zip",
    "sd-v1_4": "sd-v1_4.zip",
}

CoSpyVariant = Literal["progan", "sd-v1_4"]


def _conv3x3(in_p: int, out_p: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_p, out_p, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_p: int, out_p: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_p, out_p, 1, stride=stride, bias=False)


class _ArtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp: int, planes: int, stride: int = 1, ds: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1, self.bn1 = _conv1x1(inp, planes), nn.BatchNorm2d(planes)
        self.conv2, self.bn2 = _conv3x3(planes, planes, stride), nn.BatchNorm2d(planes)
        self.conv3, self.bn3 = _conv1x1(planes, planes * 4), nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _NPRArtifactEncoder(nn.Module):
    """ProGAN artifact branch from the official CO-SPY CNNDet detector."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(_ArtBottleneck, 64, 3)
        self.layer2 = self._make_layer(_ArtBottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        ds = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, ds)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _interpolate(x: torch.Tensor, factor: float) -> torch.Tensor:
        down = F.interpolate(x, scale_factor=factor, mode="nearest", recompute_scale_factor=True)
        return F.interpolate(
            down,
            scale_factor=1 / factor,
            mode="nearest",
            recompute_scale_factor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        artifact = x - self._interpolate(x, 0.5)
        out = self.maxpool(self.relu(self.bn1(self.conv1(artifact * 2.0 / 3.0))))
        out = self.layer2(self.layer1(out))
        return self.avgpool(out).flatten(1)


class _VAEReconEncoder(nn.Module):
    """SD-v1.4 artifact branch: VAE reconstruction residual to truncated ResNet."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(_ArtBottleneck, 64, 3)
        self.layer2 = self._make_layer(_ArtBottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        ds = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, ds)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(x).latent_dist.mean
        return self.vae.decode(latent).sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = (x - self._reconstruct(x)) / 7.0 * 100.0
        out = self.maxpool(self.relu(self.bn1(self.conv1(residual))))
        out = self.layer2(self.layer1(out))
        return self.avgpool(out).flatten(1)


class _ProGANCoSpyFusion(nn.Module):
    """Official CO-SPY fusion detector trained on CNNDet / ProGAN."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import CLIPModel

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.requires_grad_(False)
        self.sem_fc = nn.Linear(768, 1)

        self.artifact_encoder = _NPRArtifactEncoder()
        self.art_fc = nn.Linear(512, 1)
        self.fusion_fc = nn.Linear(2, 1)

        self.sem_norm = transforms.Normalize(
            [0.48145466, 0.45782750, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        )
        self.art_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sem = self.sem_norm(x)
        x_art = self.art_norm(x)

        sem_pred = self.sem_fc(self.clip.get_image_features(x_sem))
        art_pred = self.art_fc(self.artifact_encoder(x_art))
        return self.fusion_fc(torch.cat([sem_pred, art_pred], dim=1))


class _SDV14CoSpyFusion(nn.Module):
    """Official CO-SPY fusion detector trained on DRCT-2M / SD-v1.4."""

    def __init__(self) -> None:
        super().__init__()
        import open_clip
        from diffusers import StableDiffusionPipeline

        sig, _, _ = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384",
            pretrained="webli",
        )
        sig.requires_grad_(False)
        self.clip = sig
        self.sem_fc = nn.Linear(1152, 1)

        vae = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").vae
        vae.requires_grad_(False)
        self.artifact_encoder = _VAEReconEncoder(vae)
        self.art_fc = nn.Linear(512, 1)
        self.fusion_fc = nn.Linear(2, 1)

        self.sem_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.art_resize = transforms.Resize(224, antialias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sem = self.sem_norm(x)
        x_art = self.art_resize(x)

        sem_pred = self.sem_fc(self.clip.encode_image(x_sem))
        art_pred = self.art_fc(self.artifact_encoder(x_art))
        return self.fusion_fc(torch.cat([sem_pred, art_pred], dim=1))


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def _load_weight_files(model: nn.Module, checkpoint_dir: Path) -> None:
    sem = torch.load(
        checkpoint_dir / "semantic_weights.pth", map_location="cpu", weights_only=False
    )
    model.sem_fc.weight.data = sem["fc.weight"]
    model.sem_fc.bias.data = sem["fc.bias"]

    art = torch.load(
        checkpoint_dir / "artifact_weights.pth", map_location="cpu", weights_only=False
    )
    art_enc_state = _strip_prefix(art, "artifact_encoder.")
    art_fc_state = _strip_prefix(art, "fc.")
    if art_enc_state:
        model.artifact_encoder.load_state_dict(art_enc_state, strict=False)
    if art_fc_state:
        model.art_fc.load_state_dict(art_fc_state, strict=True)

    fus = torch.load(checkpoint_dir / "fusion_weights.pth", map_location="cpu", weights_only=False)
    fus_fc_state = _strip_prefix(fus, "fc.")
    if fus_fc_state:
        model.fusion_fc.load_state_dict(fus_fc_state, strict=True)


def _load_cospy_weights(model: nn.Module, cache: Path, variant: CoSpyVariant) -> Path:
    """Load the three-part CO-SPY pretrained weights from HuggingFace."""
    import zipfile

    from huggingface_hub import hf_hub_download

    zip_path = Path(
        hf_hub_download(
            repo_id=_HF_REPO,
            filename=_CHECKPOINT_ZIPS[variant],
            repo_type="model",
        )
    )

    extract_dir = cache / variant
    if not (extract_dir / "fusion_weights.pth").is_file():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir.parent)

    _load_weight_files(model, extract_dir)
    return extract_dir


@register_detector("cospy", aliases=["co_spy", "cospy_cvpr2025", "cospy_progan"])
class CoSpyDetector(BaseDetector):
    """CO-SPY fusion detector (Cheng et al., CVPR 2025).

    Parameters
    ----------
    variant : {"progan", "sd-v1_4"}
        Official pretrained detector variant. ``"progan"`` matches the
        AIGCDetectionBenchmark/CNNDet setup. ``"sd-v1_4"`` matches the
        DRCT-2M/CO-SPY-Bench diffusion setup.
    checkpoint_dir : str or Path, optional
        Directory containing ``semantic_weights.pth``, ``artifact_weights.pth``,
        and ``fusion_weights.pth``. Auto-downloaded from HuggingFace when omitted.
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
        variant: CoSpyVariant = "progan",
        checkpoint_dir: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        if variant not in _CHECKPOINT_ZIPS:
            valid = ", ".join(sorted(_CHECKPOINT_ZIPS))
            raise ValueError(f"Unknown CO-SPY variant {variant!r}; expected one of: {valid}")

        self.variant: CoSpyVariant = variant
        self._model = _ProGANCoSpyFusion() if variant == "progan" else _SDV14CoSpyFusion()

        cache = get_cache_dir("cospy", cache_dir)
        if checkpoint_dir is not None:
            cdir = Path(checkpoint_dir).expanduser().resolve()
            _load_weight_files(self._model, cdir)
            self._checkpoint_dir = cdir
        else:
            self._checkpoint_dir = _load_cospy_weights(self._model, cache, variant)

        self._model.to(self._device).eval()

        resize = 256 if variant == "progan" else 384
        crop = 224 if variant == "progan" else 384
        self._transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
            ]
        )

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
        return self._make_result(
            float(score),
            variant=self.variant,
            checkpoint_dir=str(self._checkpoint_dir),
        )


@register_detector("cospy_sd_v1_4", aliases=["cospy_sd", "cospy_drct"])
class CoSpySDV14Detector(CoSpyDetector):
    """CO-SPY detector with the official DRCT-2M / SD-v1.4 checkpoint."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("variant", "sd-v1_4")
        super().__init__(**kwargs)
