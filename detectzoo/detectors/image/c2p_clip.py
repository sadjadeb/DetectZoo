"""C2P-CLIP — Category Common Prompt in CLIP for Deepfake Detection (AAAI 2025).

Reference:
    Tan et al., "C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance
    Generalization in Deepfake Detection", AAAI 2025.
    https://arxiv.org/abs/2408.09647

The key idea: leverages pretrained CLIP ViT-L/14 visual features with a lightweight
linear classifier.  During training the method injects category-common prompts via
LoRA fine-tuning of the vision encoder; at inference the frozen CLIP visual encoder
plus a single trained linear head suffice for binary detection.

Upstream: https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection
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
from detectzoo.utils.io import load_image

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_CKPT_URL = "https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip"


class _C2PCLIP(nn.Module):
    """CLIP ViT-L/14 visual encoder + single-logit linear head."""

    def __init__(
        self, pretrained_name: str = "openai/clip-vit-large-patch14", num_classes: int = 1
    ) -> None:
        super().__init__()
        from transformers import CLIPModel as HFCLIPModel

        clip = HFCLIPModel.from_pretrained(pretrained_name)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.config = clip.config

        del clip.text_model, clip.text_projection, clip.logit_scale

        self.vision_model.requires_grad_(False)
        self.visual_projection.requires_grad_(False)
        self.fc = nn.Linear(768, num_classes)
        nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model(
            pixel_values=img,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.use_return_dict,
        )
        pooled = vision_outputs[1]
        return self.visual_projection(pooled)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image(img)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return self.fc(feats)


@register_detector("c2p_clip", aliases=["c2pclip", "c2p_clip_deepfake"])
class C2PCLIPDetector(BaseDetector):
    """C2P-CLIP binary detector (Tan et al., AAAI 2025).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to the ``.pth`` state dict.  Downloaded automatically from the
        authors' release URL when omitted.
    threshold : float
        Decision boundary (default 0.5).
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    """

    modality = "image"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        self._model = _C2PCLIP(num_classes=1)

        if checkpoint_path is not None:
            raw = torch.load(
                Path(checkpoint_path).expanduser().resolve(),
                map_location=self._device,
                weights_only=False,
            )
        else:
            raw = torch.hub.load_state_dict_from_url(
                _CKPT_URL, map_location=self._device, progress=True
            )

        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        elif isinstance(raw, dict):
            state = raw

        state = {k.removeprefix("model."): v for k, v in state.items()}

        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()

        self._transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
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
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        img = self._normalize_input(input_data)
        x = self._transform(img).unsqueeze(0).to(self._device)
        score = self._model(x).sigmoid().item()
        return self._make_result(float(score))
