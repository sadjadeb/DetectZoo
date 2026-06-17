"""FatFormer — Forgery-aware Adaptive Transformer (CVPR 2024).

Reference:
    Liu et al., "Forgery-aware Adaptive Transformer for Generalizable Synthetic
    Image Detection", CVPR 2024.
    https://arxiv.org/abs/2312.16649

The key idea: adapts CLIP ViT-L/14 with forgery-aware adapter layers that
operate in both spatial and frequency (DWT) domains.  A *language-guided
alignment* module builds instance-conditioned text prompts ("real" / "fake") and
contrasts adapted image features with prompt embeddings.

Upstream: https://github.com/Michel-liu/FatFormer
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.datasets._download import get_cache_dir
from detectzoo.utils.io import load_image

_IMAGENET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_DEFAULT_CKPT_NAME = "fatformer.pth"
_GDRIVE_FILE_ID = "1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj"

# ---------------------------------------------------------------------------
# Inlined model architecture (inference-only subset of the FatFormer codebase)
# ---------------------------------------------------------------------------


class _LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class _ForgeryAwareAdapter(nn.Module):
    """Spatial conv-bottleneck + DWT frequency-aware attention."""

    def __init__(
        self, d_model: int = 1024, bottleneck: int = 64, dropout: float = 0.1, head: int = 8
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = 0.1

        self.first_conv_layer = nn.Conv1d(d_model, bottleneck, 1)
        self.non_linear_func = nn.ReLU()
        self.second_conv_layer = nn.Conv1d(bottleneck, d_model, 1)
        self.dropout = dropout

        self.freq_scale = nn.Parameter(torch.zeros(1))
        from pytorch_wavelets import DWTForward, DWTInverse

        self.dwt_transform = DWTForward(J=1, wave="haar")
        self.idwt_transform = DWTInverse(wave="haar")

        self.dwt_norm = nn.GroupNorm(128, d_model)
        self.intra_band = nn.MultiheadAttention(d_model, head)
        self.dropout_intra = nn.Dropout(dropout)
        self.norm_intra = nn.LayerNorm(d_model)
        self.inter_band = nn.MultiheadAttention(d_model, head)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def _ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.norm3(
            tgt + self.dropout4(self.linear2(self.dropout3(self.activation(self.linear1(tgt)))))
        )

    def _freq(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        nq = x.shape[2]
        q = k = v = x.transpose(0, 1).flatten(1, 2)
        x = x + self.dropout_intra(
            self.intra_band(q, k, v)[0].reshape(C, B, nq, self.d_model).transpose(0, 1)
        )
        x = self.norm_intra(x)
        q = k = v = x.flatten(0, 1).transpose(0, 1)
        x = x + self.dropout_inter(
            self.inter_band(q, k, v)[0].transpose(0, 1).reshape(B, C, nq, self.d_model)
        )
        x = self.norm_inter(x)
        return self._ffn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nq, bs, md = x.shape
        side = int(math.sqrt(nq))
        patches = x[1:].transpose(0, 1).reshape(bs, side, side, md).permute(0, 3, 1, 2)
        dl, dh = self.dwt_transform(patches)
        dwt = torch.cat([dl[:, :, None], dh[0]], dim=2)
        hh, ww = dwt.shape[-2:]
        dwt = self.dwt_norm(dwt.flatten(-2)).permute(0, 2, 3, 1)
        dwt = self._freq(dwt).reshape(bs, 4, hh, ww, md).permute(0, 4, 1, 2, 3)
        freq_out = self.idwt_transform((dwt[:, :, 0], [dwt[:, :, 1:]])).flatten(-2).permute(2, 0, 1)
        freq_out = torch.cat([torch.zeros_like(x[:1]), freq_out], dim=0)

        down = self.non_linear_func(self.first_conv_layer(x.permute(1, 2, 0)))
        up = (
            F.dropout(self.second_conv_layer(down), p=self.dropout, training=self.training).permute(
                2, 0, 1
            )
            * self.scale
        )
        return up + freq_out * self.freq_scale


class _ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor | None = None,
        add_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = _LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", _QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = _LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.forgery_aware_adapter = _ForgeryAwareAdapter(d_model) if add_adapter else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        x = (
            x
            + self.attn(
                self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False, attn_mask=mask
            )[0]
        )
        adapt = self.forgery_aware_adapter(x) if self.forgery_aware_adapter is not None else 0
        x = x + self.mlp(self.ln_2(x)) + adapt
        return x


class _Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor | None = None,
        add_adapter: list[bool] | None = None,
    ) -> None:
        super().__init__()
        if add_adapter is None:
            add_adapter = [False] * layers
        self.resblocks = nn.Sequential(
            *[
                _ResidualAttentionBlock(width, heads, attn_mask, add_adapter[i])
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.resblocks):
            x = layer(x)
            out[f"layer{idx}"] = x
        return out, x


class _VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        num_adapter: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = _LayerNorm(width)

        adapt_flags = [False] * layers
        if num_adapter > 0:
            n_split = layers // num_adapter
            for i in range(num_adapter):
                adapt_flags[(i + 1) * n_split - 1] = True
        self.transformer = _Transformer(width, layers, heads, add_adapter=adapt_flags)
        self.ln_post = _LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_full: bool = False) -> torch.Tensor:
        x = self.conv1(x).flatten(2).permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = self.ln_pre(x + self.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        _, x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x) if return_full else self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class _CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        num_adapter: int = 3,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = _VisionTransformer(
            image_resolution,
            vision_patch_size,
            vision_width,
            vision_layers,
            vision_heads,
            embed_dim,
            num_adapter,
        )
        mask = torch.empty(context_length, context_length).fill_(float("-inf")).triu_(1)
        self.transformer = _Transformer(
            transformer_width, transformer_layers, transformer_heads, attn_mask=mask
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = _LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.conv1.weight.dtype


class _TextEncoder(nn.Module):
    def __init__(self, clip: _CLIP) -> None:
        super().__init__()
        self.transformer = clip.transformer
        self.positional_embedding = clip.positional_embedding
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

    @property
    def dtype(self) -> torch.dtype:
        return self.ln_final.weight.dtype

    def forward(self, prompts: torch.Tensor, tokenized: torch.Tensor) -> torch.Tensor:
        x = (prompts + self.positional_embedding.type(self.dtype)).permute(1, 0, 2)
        _, x = self.transformer(x)
        x = self.ln_final(x.permute(1, 0, 2)).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.text_projection


class _LanguageGuidedAlignment(nn.Module):
    def __init__(self, clip: _CLIP, classnames: list[str], n_ctx: int = 8) -> None:
        super().__init__()
        dtype = clip.dtype
        ctx_dim = clip.ln_final.weight.shape[0]
        d_model = ctx_dim

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        import open_clip as _oc

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([_oc.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        self.patch_basaed_enhancer = nn.MultiheadAttention(d_model, 12)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _ffn(self, t: torch.Tensor) -> torch.Tensor:
        return self.norm2(t + self.linear2(self.activation(self.linear1(t))))

    def forward(self, im_features: torch.Tensor) -> torch.Tensor:
        tgt = self.ctx[:, None].repeat_interleave(im_features.shape[0], dim=1)
        tgt = self.norm1(
            tgt
            + self.patch_basaed_enhancer(
                tgt, im_features.transpose(0, 1), im_features.transpose(0, 1)
            )[0]
        )
        tgt = self._ffn(tgt).transpose(0, 1)
        prompts = []
        for ctx_i in tgt:
            ctx_exp = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            prompts.append(torch.cat([self.token_prefix, ctx_exp, self.token_suffix], dim=1))
        return torch.stack(prompts)


class _FatFormerModel(nn.Module):
    """Full FatFormer CLIPModel"""

    def __init__(self, num_adapter: int = 3, n_ctx: int = 8) -> None:
        super().__init__()
        # ViT-L/14 configuration
        self.clip_model = _CLIP(
            embed_dim=768,
            image_resolution=224,
            vision_layers=24,
            vision_width=1024,
            vision_patch_size=14,
            context_length=77,
            vocab_size=49408,
            transformer_width=768,
            transformer_heads=12,
            transformer_layers=12,
            num_adapter=num_adapter,
        )
        self.language_guided_alignment = _LanguageGuidedAlignment(
            self.clip_model, ["real", "fake"], n_ctx
        )
        self.tokenized_prompts = self.language_guided_alignment.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = _TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale

        d_model = self.clip_model.ln_final.weight.shape[0]
        self.text_guided_interactor = nn.MultiheadAttention(d_model, 12)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    @property
    def dtype(self) -> torch.dtype:
        return self.clip_model.dtype

    def _ffn(self, t: torch.Tensor) -> torch.Tensor:
        return self.norm2(t + self.linear2(self.activation(self.linear1(t))))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokenized = self.tokenized_prompts.to(image.device)
        logit_scale = self.logit_scale.exp()

        im_feats = self.image_encoder(image.type(self.dtype), return_full=True)
        im_norm = im_feats / im_feats.norm(dim=-1, keepdim=True)
        prompts = self.language_guided_alignment(im_feats)

        logits_list: list[torch.Tensor] = []
        text_feat_list: list[torch.Tensor] = []
        for pts_i, imf_i in zip(prompts, im_norm):
            tf = self.text_encoder(pts_i, tokenized)
            text_feat_list.append(tf)
            tf_n = tf / tf.norm(dim=-1, keepdim=True)
            logits_list.append(logit_scale * imf_i[0] @ tf_n.t())
        logits = torch.stack(logits_list)

        text_feats = torch.stack(text_feat_list, dim=1)
        tgt = im_feats[:, 1:].transpose(0, 1)
        tgt = self.norm1(tgt + self.text_guided_interactor(tgt, text_feats, text_feats)[0])
        tgt = self._ffn(tgt)

        aug_feats = tgt.transpose(0, 1).mean(dim=1)
        aug_feats = aug_feats / aug_feats.norm(dim=-1, keepdim=True)
        tf_n = text_feats / text_feats.norm(dim=-1, keepdim=True)
        aug_logits = torch.stack(
            [logit_scale * af @ tfn.t() for af, tfn in zip(aug_feats, tf_n.transpose(0, 1))]
        )
        return logits + aug_logits


@register_detector("fatformer", aliases=["fat_former", "fatformer_cvpr2024"])
class FatFormerDetector(BaseDetector):
    """FatFormer image detector (Liu et al., CVPR 2024).

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to the FatFormer ``.pth`` checkpoint.  If omitted, attempts
        auto-download from Google Drive.
    num_vit_adapter : int
        Number of forgery-aware adapter layers (default 3).
    num_context_embedding : int
        Number of learnable context tokens (default 8).
    threshold : float
        Decision boundary on the score (default 0.5).
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
        num_vit_adapter: int = 3,
        num_context_embedding: int = 8,
        threshold: float = 0.5,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)

        cache = get_cache_dir("fatformer", cache_dir)
        if checkpoint_path is not None:
            self._ckpt = Path(checkpoint_path).expanduser().resolve()
        else:
            self._ckpt = cache / _DEFAULT_CKPT_NAME
            if not self._ckpt.is_file():
                import gdown

                gdown.download(id=_GDRIVE_FILE_ID, output=str(self._ckpt), quiet=False)

        self._model = _FatFormerModel(num_adapter=num_vit_adapter, n_ctx=num_context_embedding)
        raw = torch.load(self._ckpt, map_location="cpu", weights_only=False)
        state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        self._model.load_state_dict(state, strict=False)
        self._model.to(self._device).eval()

        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(**_IMAGENET),
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
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        img = self._normalize_input(input_data)
        x = self._transform(img).unsqueeze(0).to(self._device)
        logits = self._model(x)
        score = logits.softmax(dim=-1)[:, 1].item()
        return self._make_result(
            float(score),
            checkpoint=str(self._ckpt),
        )
