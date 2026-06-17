"""AEROBLADE — Training-Free Diffusion Image Detection (CVPR 2024).

Reference:
    Ricker et al., "AEROBLADE: Training-Free Detection of Latent Diffusion Images
    Using Autoencoder Reconstruction Error", CVPR 2024.

The key idea: latent diffusion models synthesize images in the same VAE latent space
they were trained on, so a round-trip through encode → decode yields lower perceptual
error (LPIPS) for AI-generated images than for real photographs. The detector scores
images by that reconstruction gap (negated LPIPS: higher = more likely AI).

Upstream: https://github.com/jonasricker/aeroblade
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import lpips
import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import AutoencoderKL
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.utils.io import load_image


def _resize_to_multiple_of_8(image: Image.Image) -> Image.Image:
    """Resize so both dimensions are multiples of 8 (VAE requirement)."""
    w, h = image.size
    w2, h2 = w - (w % 8), h - (h % 8)
    if w2 < 8 or h2 < 8:
        raise ValueError("Image is too small after aligning to multiples of 8.")
    if (w2, h2) != (w, h):
        image = image.resize((w2, h2), Image.Resampling.LANCZOS)
    return image


def _pil_to_tensor01(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to [1, 3, H, W] tensor in [0, 1]."""
    t = T.ToTensor()(image)
    return t.unsqueeze(0)


@register_detector("aeroblade", aliases=["aeroblade_vae"])
class AerobladeDetector(BaseDetector):
    """AEROBLADE image detector (Ricker et al., CVPR 2024).

    Parameters
    ----------
    repo_ids : sequence of str, optional
        Hugging Face model ids that expose a ``subfolder="vae"``
        ``AutoencoderKL`` (e.g. Stable Diffusion checkpoints). Defaults to
        ``["CompVis/stable-diffusion-v1-1"]``.
    lpips_vgg_index : int
        Which LPIPS term to use. ``0`` = full LPIPS sum; ``1``–``5`` = VGG
        stage (default ``2`` = ``lpips_vgg_2`` in the official code).
    threshold : float
        Decision boundary on the score (``-LPIPS``). Tune on validation
        data; the default is a placeholder.
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    use_fp16 : bool
        Run the VAE in float16 on CUDA.
    seed : int, optional
        Seed for latent sampling reproducibility.
    """

    modality = "image"

    def __init__(
        self,
        repo_ids: Optional[Sequence[str]] = None,
        *,
        lpips_vgg_index: int = 2,
        threshold: float = -0.15,
        device: str = "cpu",
        use_fp16: bool = False,
        seed: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.repo_ids: List[str] = (
            list(repo_ids)
            if repo_ids is not None
            else [
                "CompVis/stable-diffusion-v1-1",
            ]
        )

        self.lpips_vgg_index = lpips_vgg_index
        self.use_fp16 = bool(use_fp16) and str(device).startswith("cuda")
        self.seed = seed

        self._vaes = nn.ModuleDict()
        self._lpips: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.use_fp16 else torch.float32

    def _vae_key(self, repo_id: str) -> str:
        return repo_id.replace("/", "--")

    def _get_vae(self, repo_id: str) -> nn.Module:
        """Lazily load and cache a VAE by repo id."""
        key = self._vae_key(repo_id)
        if key in self._vaes:
            return self._vaes[key]

        vae = AutoencoderKL.from_pretrained(
            repo_id,
            subfolder="vae",
            torch_dtype=self._dtype(),
        )
        vae = vae.to(device=self._device, dtype=self._dtype())
        vae.eval()
        self._vaes[key] = vae
        return vae

    def _get_lpips(self) -> nn.Module:
        """Lazily load and cache the LPIPS network."""
        if self._lpips is not None:
            return self._lpips

        self._lpips = lpips.LPIPS(net="vgg", verbose=False).to(self._device)
        self._lpips.eval()
        return self._lpips

    def to(self, device: str) -> "AerobladeDetector":
        self._device = torch.device(device)
        self.use_fp16 = bool(self.use_fp16) and str(device).startswith("cuda")
        for mod in self._vaes.values():
            mod.to(self._device, dtype=self._dtype())
        if self._lpips is not None:
            self._lpips.to(self._device)
        return self

    @torch.no_grad()
    def _lpips_distance(self, orig_01: torch.Tensor, recon_01: torch.Tensor) -> float:
        """Mean LPIPS between two [0,1] BCHW tensors."""
        lp = self._get_lpips()
        total, per_layer = lp(orig_01, recon_01, retPerLayer=True, normalize=True)
        if self.lpips_vgg_index == 0:
            d = total.mean()
        else:
            d = per_layer[self.lpips_vgg_index - 1].mean()
        return float(d)

    @torch.no_grad()
    def _reconstruct(self, vae: nn.Module, x_m11: torch.Tensor) -> torch.Tensor:
        """VAE round-trip; returns [B,3,H,W] in [0,1]."""
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(self.seed))

        posterior = vae.encode(x_m11).latent_dist
        latents = posterior.sample(generator)
        decoded = vae.decode(latents).sample
        return (decoded / 2 + 0.5).clamp(0, 1)

    def _distance_one_repo(self, repo_id: str, x01: torch.Tensor) -> float:
        """Compute LPIPS reconstruction distance for a single VAE."""
        vae = self._get_vae(repo_id)
        x_m11 = x01 * 2.0 - 1.0
        x_m11 = x_m11.to(device=self._device, dtype=self._dtype())
        recon = self._reconstruct(vae, x_m11).to(dtype=torch.float32)
        x01_f = x01.to(device=self._device, dtype=torch.float32)
        return self._lpips_distance(x01_f, recon)

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

    def predict(self, input_data: Any) -> DetectionResult:
        img = self._normalize_input(input_data)
        img = _resize_to_multiple_of_8(img)
        x01 = _pil_to_tensor01(img)

        distances: list[float] = []
        for rid in self.repo_ids:
            distances.append(self._distance_one_repo(rid, x01))

        # Smallest reconstruction error (most "LDM-like") → strongest AI signal.
        raw_lpips = min(distances)
        score = -raw_lpips

        return self._make_result(
            score,
            raw_lpips=raw_lpips,
            repo_ids=list(self.repo_ids),
            lpips_vgg_index=self.lpips_vgg_index,
            best_repo=self.repo_ids[int(distances.index(raw_lpips))],
        )
