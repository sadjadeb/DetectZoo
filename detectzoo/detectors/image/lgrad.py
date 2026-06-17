"""LGrad — Learning on Gradients (CVPR 2023).

Tan et al., "Learning on Gradients: Generalized Artifacts Representation for GAN-Generated
Images Detection".

The key idea: train a CNN on gradient-domain images (from the authors' img2grad step)
instead of RGB, so detector features capture generator-agnostic high-frequency traces.

DetectZoo pipeline: By default ``input_mode="rgb"`` runs the official **PyTorch img2grad**
path from the LGrad repo, then applies the ResNet-50 head. Use ``input_mode="gradient"``
only if ``data`` are already saved gradient PNGs.

Upstream: https://github.com/chuangchuangtan/LGrad
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import download_file, get_cache_dir
from detectzoo.detectors.image._lgrad_stylegan_discriminator import StyleGANDiscriminator
from detectzoo.detectors.image.resnet50_binary import build_resnet50_binary, load_pytorch_checkpoint
from detectzoo.utils.io import load_image

_DEFAULT_CKPT_NAME = "LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth"
_DEFAULT_GDRIVE_FILE_ID = "1OVUTjlvkiGOggcvLhB0xEkaOf8sxT8Zr"
_DEFAULT_DISC_NAME = "karras2019stylegan-bedrooms-256x256_discriminator.pth"
_DISC_URL = (
    "https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/"
    "karras2019stylegan-bedrooms-256x256_discriminator.pth"
)
_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

EvalProtocol = Literal["test_8gan", "val"]
InputMode = Literal["rgb", "gradient"]


def _eval_steps(name: EvalProtocol | str) -> list[Any]:
    key = str(name).strip().lower().replace("-", "_")
    if key in ("test_8gan", "8gan", "paper"):
        return [transforms.ToTensor(), transforms.Normalize(**_IMAGENET)]
    if key == "val":
        return [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**_IMAGENET),
        ]
    raise ValueError(f"Unknown eval_protocol {name!r}. Use 'test_8gan' or 'val'.")


def _raw_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint contents: {type(ckpt)!r}")
    return {k.replace("module.", ""): v for k, v in ckpt.items()}


def _download_from_gdrive(file_id: str, dest: Path) -> None:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=True)


def _grad_chw_to_pil(grad: torch.Tensor) -> Image.Image:
    x = grad.detach().float().cpu()
    x = x - x.min()
    m = x.max()
    if m > 0:
        x = x / m
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
    hwc = x.permute(1, 2, 0).numpy()
    return Image.fromarray(hwc, mode="RGB")


# ------------------------------------------------------------------
# LGradDetector
# ------------------------------------------------------------------


@register_detector("lgrad", aliases=["lgrad_cvpr2023", "learning_on_gradients"])
class LGradDetector(BaseDetector):
    """LGrad ResNet-50 head with optional built-in RGB→gradient (official PyTorch img2grad)."""

    modality = "image"

    def __init__(
        self,
        *,
        input_mode: InputMode | str = "rgb",
        eval_protocol: EvalProtocol | str = "test_8gan",
        checkpoint_path: str | Path | None = None,
        gdrive_file_id: str | None = None,
        auto_download: bool = True,
        discriminator_path: str | Path | None = None,
        auto_download_discriminator: bool = True,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        mode = str(input_mode).strip().lower()
        if mode not in ("rgb", "gradient"):
            raise ValueError(f"input_mode must be 'rgb' or 'gradient', got {input_mode!r}")
        self._input_mode = "rgb" if mode == "rgb" else "gradient"
        self._eval_protocol = str(eval_protocol)
        self._transform = transforms.Compose(_eval_steps(self._eval_protocol))

        cache = get_cache_dir("lgrad", cache_dir)
        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = cache / _DEFAULT_CKPT_NAME
            if not self._ckpt.is_file() and auto_download:
                fid = gdrive_file_id or _DEFAULT_GDRIVE_FILE_ID
                _download_from_gdrive(fid, self._ckpt)

        raw: Any = load_pytorch_checkpoint(self._ckpt, self._device)
        state = _raw_state_dict(raw)

        self._model = build_resnet50_binary(num_classes=1)
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()

        self._disc: StyleGANDiscriminator | None = None
        self._disc_preprocess: transforms.Compose | None = None
        self._disc_ckpt: Path | None = None

        if self._input_mode == "rgb":
            if discriminator_path is not None:
                disc_path = Path(discriminator_path).expanduser().resolve()
            else:
                disc_path = cache / _DEFAULT_DISC_NAME
                if not disc_path.is_file() and auto_download_discriminator:
                    download_file(_DISC_URL, disc_path)

            self._disc_ckpt = disc_path
            self._disc = StyleGANDiscriminator(resolution=256, image_channels=3, label_size=0)
            d_raw: Any = load_pytorch_checkpoint(disc_path, self._device)

            self._disc.load_state_dict(d_raw, strict=True)
            self._disc.to(self._device).eval()
            self._disc_preprocess = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

    def _img2grad_pil(self, pil: Image.Image) -> Image.Image:
        assert self._disc is not None and self._disc_preprocess is not None
        self._disc.eval()
        x = self._disc_preprocess(pil).to(self._device, dtype=torch.float32).unsqueeze(0)
        x = x.detach().requires_grad_(True)
        pre = self._disc(x)
        grad = torch.autograd.grad(
            pre.sum(),
            x,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        self._disc.zero_grad(set_to_none=True)
        return _grad_chw_to_pil(grad[0])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_data: Any) -> DetectionResult:
        pil = self._normalize_input(input_data)
        if self._input_mode == "rgb":
            pil = self._img2grad_pil(pil)
        with torch.no_grad():
            t = self._transform(pil).unsqueeze(0).to(self._device)
            score = self._model(t).sigmoid().item()
        meta: dict[str, Any] = {
            "checkpoint": str(self._ckpt),
            "eval_protocol": self._eval_protocol,
            "input_mode": self._input_mode,
        }
        if self._disc_ckpt is not None:
            meta["discriminator"] = str(self._disc_ckpt)
        return self._make_result(float(score), **meta)
