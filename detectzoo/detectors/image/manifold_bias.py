"""Manifold Induced Biases — Zero-shot AI-Generated Image Detection (ICLR 2025).

Reference:
    Brokman et al., "Manifold Induced Biases for Zero-shot and Few-shot Detection
    of Generated Images", ICLR 2025.
    https://arxiv.org/abs/2504.15470

The key idea: real and AI-generated images lie on slightly different data manifolds, and
generative models introduce subtle geometric biases. By measuring how well an image
aligns with the natural image manifold, the method detects fakes in a zero-shot,
generator-agnostic way.

Threshold calibration (required):
    Use ~1000 real images from the target domain. Pass only real images.
    Using fake images during calibration will bias the threshold.

Upstream: https://github.com/JonathanBrok/Manifold-Induced-Biases-for-Zero-shot-and-Few-shot-Detection-of-Generated-Images
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.utils.io import load_image

_SD_REPO = "CompVis/stable-diffusion-v1-4"
_CLIP_REPO = "openai/clip-vit-large-patch14"
_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

_CLIP_DIM = 512
_DEFAULT_SIZ = 512
_DEFAULT_NUM_NOISE = 2
_DEFAULT_TIME_FRAC = 0.01
_DEFAULT_EPSILON = 1e-8


def _resize_and_crop(img_t: torch.Tensor, siz: int) -> torch.Tensor:
    img_t = T.Resize(siz + 3)(img_t)
    start_x = (img_t.size(-1) - siz) // 2
    start_y = (img_t.size(-2) - siz) // 2
    if img_t.dim() == 3:
        return img_t[:, start_y : start_y + siz, start_x : start_x + siz]
    return img_t[:, :, start_y : start_y + siz, start_x : start_x + siz]


def _normalize_batch(batch: torch.Tensor, epsilon: float = _DEFAULT_EPSILON) -> torch.Tensor:
    dims = tuple(range(1, batch.dim()))
    norms = torch.norm(batch, p=2, dim=dims, keepdim=True)
    return batch / (norms + epsilon)


def _pil_to_raw_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.array(img.convert("RGB")))


def _preprocess_for_sd(
    img_hwc_uint8: torch.Tensor, siz: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    x = img_hwc_uint8.permute(2, 0, 1).to(device=device, dtype=dtype)
    x = _resize_and_crop(x, siz)
    return 2.0 * (x / 255.0) - 1.0


def _postprocess_decoded(img_t: torch.Tensor, siz: int) -> torch.Tensor:
    img_t = _resize_and_crop(img_t, siz)
    img_t = (img_t / 2 + 0.5).clamp(0, 1) * 255.0
    return img_t.detach().float()


def _decode_in_subbatches(
    vae: nn.Module,
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    sub_batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = latents_a.size(0)
    if n <= sub_batch_size:
        with torch.no_grad():
            return (
                vae.decode(latents_a, return_dict=False)[0],
                vae.decode(latents_b, return_dict=False)[0],
            )
    chunks = (n + sub_batch_size - 1) // sub_batch_size
    dec_a_list, dec_b_list = [], []
    with torch.no_grad():
        for i in range(chunks):
            s, e = i * sub_batch_size, min((i + 1) * sub_batch_size, n)
            torch.cuda.empty_cache()
            dec_a_list.append(vae.decode(latents_a[s:e], return_dict=False)[0])
            dec_b_list.append(vae.decode(latents_b[s:e], return_dict=False)[0])
    return torch.cat(dec_a_list), torch.cat(dec_b_list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("manifold_bias", aliases=["mib", "manifold_induced_bias", "brokman2025"])
class ManifoldBiasDetector(BaseDetector):
    """Zero-shot AI-generated image detector based on diffusion manifold biases
    (Brokman et al., ICLR 2025).

    Parameters
    ----------
    sd_repo : str
        HuggingFace repo ID for the Stable Diffusion pipeline.
    clip_repo : str
        HuggingFace repo ID for the CLIP model.
    caption_model : str
        HuggingFace repo ID for the image-captioning model. Pass ``prompt=``
        at predict time to skip captioning entirely.
    num_noise : int
        Number of spherical noise perturbations K (default 2).
    time_frac : float
        Fraction of total diffusion timesteps at which to add noise (default 0.01).
    image_size : int
        Spatial size before VAE encoding (default 512).
    epsilon_reg : float
        Regularisation for sphere normalization (default 1e-8).
    threshold : float or None
        Decision boundary. ``None`` (default) means uncalibrated — ``predict``
        will raise until ``calibrate()`` is called.
    use_fp16 : bool
        Run SD UNet and VAE in float16 (default True on CUDA).
    device : str
        Torch device string.
    """

    modality = "image"

    def __init__(
        self,
        *,
        sd_repo: str = _SD_REPO,
        clip_repo: str = _CLIP_REPO,
        caption_model: str = _CAPTION_MODEL,
        num_noise: int = _DEFAULT_NUM_NOISE,
        time_frac: float = _DEFAULT_TIME_FRAC,
        image_size: int = _DEFAULT_SIZ,
        epsilon_reg: float = _DEFAULT_EPSILON,
        threshold: Optional[float] = None,
        use_fp16: bool = True,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        # Pass threshold=0.0 to BaseDetector as a placeholder, we override it below.
        super().__init__(
            threshold=threshold if threshold is not None else 0.0, device=device, **kwargs
        )
        self._calibrated = threshold is not None
        if threshold is not None:
            self.threshold = threshold

        self.sd_repo = sd_repo
        self.clip_repo = clip_repo
        self.caption_model_name = caption_model
        self.num_noise = int(num_noise)
        self.time_frac = float(time_frac)
        self.image_size = int(image_size)
        self.epsilon_reg = float(epsilon_reg)

        is_cuda = str(device).startswith("cuda")
        self._dtype = torch.float16 if (use_fp16 and is_cuda) else torch.float32

        self._unet: Optional[nn.Module] = None
        self._vae: Optional[nn.Module] = None
        self._tokenizer: Any = None
        self._text_encoder: Optional[nn.Module] = None
        self._scheduler: Any = None
        self._clip: Optional[nn.Module] = None
        self._clip_processor: Any = None
        self._captioner: Any = None
        self._cos = nn.CosineSimilarity(dim=1)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        real_images: Sequence[Union[Image.Image, str, Path]],
        k: float = 1.0,
        *,
        prompt: Optional[str] = None,
        verbose: bool = True,
    ) -> float:
        """Set the detection threshold from real images only."""
        scores: List[float] = []
        n = len(real_images)

        for i, img in enumerate(real_images):
            if verbose and (i % 50 == 0 or i == n - 1):
                print(f"  Calibrating {i + 1}/{n} ...", flush=True)
            pil = self._normalize_input(img)
            result = self._compute_criterion(pil, self._get_prompt(pil, prompt))
            scores.append(result["criterion"])

        mean = float(np.mean(scores))
        std = float(np.std(scores))
        self.threshold = mean + k * std
        self._calibrated = True

        if verbose:
            print(
                f"  Calibration done — mean={mean:.4f}, std={std:.4f}, "
                f"threshold (k={k})={self.threshold:.4f}"
            )
        return self.threshold

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load_sd(self) -> None:
        from diffusers import DDPMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(self.sd_repo, torch_dtype=self._dtype).to(
            self._device
        )
        self._unet = pipe.unet.eval()
        self._vae = pipe.vae.eval()
        self._tokenizer = pipe.tokenizer
        self._text_encoder = pipe.text_encoder.eval()
        self._scheduler = DDPMScheduler.from_pretrained(self.sd_repo, subfolder="scheduler")
        del pipe

    def _load_clip(self) -> None:
        from transformers import AutoImageProcessor, CLIPModel

        self._clip = CLIPModel.from_pretrained(self.clip_repo).to(self._device).eval()
        self._clip_processor = AutoImageProcessor.from_pretrained(self.clip_repo)

    def _load_captioner(self) -> None:
        from transformers import pipeline as hf_pipeline

        self._captioner = hf_pipeline(
            "image-to-text",
            model=self.caption_model_name,
            device=self._device,
        )

    @property
    def unet(self) -> nn.Module:
        if self._unet is None:
            self._load_sd()
        return self._unet

    @property
    def vae(self) -> nn.Module:
        if self._vae is None:
            self._load_sd()
        return self._vae

    @property
    def clip(self) -> nn.Module:
        if self._clip is None:
            self._load_clip()
        return self._clip

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            self._load_clip()
        return self._clip_processor

    @property
    def captioner(self):
        if self._captioner is None:
            self._load_captioner()
        return self._captioner

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _normalize_input(self, input_data: Any) -> Image.Image:
        if hasattr(input_data, "mode"):
            return input_data.convert("RGB")
        path = Path(str(input_data))
        if path.is_file():
            return load_image(path)
        raise TypeError(f"Expected a PIL Image or image path; got {type(input_data).__name__}.")

    def _get_prompt(self, pil_img: Image.Image, prompt: Optional[str]) -> str:
        if prompt is not None:
            return prompt
        result = self.captioner(pil_img, max_new_tokens=64)
        text = (
            result[0].get("generated_text", "")
            if isinstance(result, list) and result
            else str(result)
        )
        return text.strip() or "a photograph"

    # ------------------------------------------------------------------
    # Core criterion computation
    # ------------------------------------------------------------------

    def _clip_features(self, images_uint8_float: torch.Tensor) -> torch.Tensor:
        imgs_pil = [
            TF.to_pil_image(images_uint8_float[i].to(torch.uint8))
            for i in range(images_uint8_float.size(0))
        ]
        inputs = self.clip_processor(images=imgs_pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)
        return feats.detach().cpu()

    def _compute_criterion(self, pil_img: Image.Image, prompt: str) -> dict[str, float]:
        _ = self.unet  # ensure SD is loaded

        K = self.num_noise
        siz = self.image_size

        # 1. Preprocess → latent
        raw_hwc = _pil_to_raw_tensor(pil_img)
        img_sd = _preprocess_for_sd(raw_hwc, siz, self._device, self._dtype).unsqueeze(0)
        with torch.no_grad():
            latent = self._vae.encode(img_sd).latent_dist.sample()
            latent = latent * self._vae.config.scaling_factor
        latents = latent.repeat(K, 1, 1, 1).to(self._dtype)

        # 2. Spherical noise
        sphere = _normalize_batch(torch.randn_like(latents), self.epsilon_reg).to(self._dtype)
        sphere = sphere * float(torch.prod(torch.tensor(latents.shape[1:])).float().sqrt())

        # 3. Add noise at chosen timestep
        t_abs = int(self.time_frac * self._scheduler.config.num_train_timesteps)
        timesteps = torch.full((K,), t_abs, device=self._device, dtype=torch.long)
        noisy_latents = self._scheduler.add_noise(
            original_samples=latents, noise=sphere, timesteps=timesteps
        ).to(self._dtype)

        # 4. Text conditioning
        tokens = self._tokenizer(
            [prompt] * K,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_emb = self._text_encoder(tokens.input_ids.to(self._device)).last_hidden_state

        # 5. UNet forward
        with torch.no_grad():
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_emb)[0]
        noise_pred_scaled = (1.0 / self._vae.config.scaling_factor) * noise_pred

        # 6. Decode to pixel space
        dec_noise, dec_sphere = _decode_in_subbatches(
            self._vae, noise_pred_scaled, sphere / self._vae.config.scaling_factor
        )
        dec_noise_pp = _postprocess_decoded(dec_noise, siz)
        dec_sphere_pp = _postprocess_decoded(dec_sphere, siz)

        # 7. CLIP embeddings
        raw_chw_float = raw_hwc.permute(2, 0, 1).float()
        orig_resized_k = _resize_and_crop(raw_chw_float, siz).unsqueeze(0).expand(K, -1, -1, -1)
        clip_orig = self._clip_features(orig_resized_k)
        clip_dnoise = self._clip_features(dec_noise_pp)
        clip_sphere = self._clip_features(dec_sphere_pp)

        # 8. 3 criterion terms
        bias_vec = self._cos(clip_orig, clip_dnoise).numpy()
        kappa_vec = self._cos(clip_dnoise, clip_sphere).numpy()
        D_vec = torch.norm(clip_dnoise, p=2, dim=1).numpy()

        bias_mean = float(bias_vec.mean())
        kappa_mean = float(kappa_vec.mean())
        D_mean = float(D_vec.mean())

        # 9. Final criterion
        sqrt_d = float(_CLIP_DIM**0.5)
        criterion = 1.0 + (sqrt_d * bias_mean - D_mean + kappa_mean) / (sqrt_d + 2.0)

        return {
            "criterion": criterion,
            "bias_mean": bias_mean,
            "kappa_mean": kappa_mean,
            "D_mean": D_mean,
            "timestep": t_abs,
            "num_noise": K,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, input_data: Any, *, prompt: Optional[str] = None) -> DetectionResult:

        if not self._calibrated:
            raise RuntimeError(
                "ManifoldBiasDetector requires threshold calibration before prediction.\n"
            )

        pil_img = self._normalize_input(input_data)
        prompt_str = self._get_prompt(pil_img, prompt)
        result = self._compute_criterion(pil_img, prompt_str)
        score = result["criterion"]

        return self._make_result(
            float(score),
            prompt=prompt_str,
            bias_mean=result["bias_mean"],
            kappa_mean=result["kappa_mean"],
            D_mean=result["D_mean"],
            timestep=result["timestep"],
            num_noise=result["num_noise"],
            sd_repo=self.sd_repo,
            threshold_calibrated=self._calibrated,
        )
