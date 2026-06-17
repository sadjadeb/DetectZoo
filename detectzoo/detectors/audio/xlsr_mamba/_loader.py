"""Checkpoint loader for XLSR-Mamba (HF Hub weights + fairseq->HF SSL translation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from detectzoo.detectors.audio._anti_deepfake_common import _translate_fairseq_ssl_state_dict
from detectzoo.detectors.audio.xlsr_mamba._model import (
    DEFAULT_EMB_SIZE,
    DEFAULT_NUM_ENCODERS,
    XLSRMambaModel,
)
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_SAFETENSORS_FILENAME = "model.safetensors"
_SSL_CKPT_PREFIX = "ssl_model.model."
_TRANSLATOR_PREFIX = "m_ssl.model."
_XLSR_BACKBONE = "facebook/wav2vec2-large-xlsr-53"

# Keys we deliberately leave at random init (SpecAugment disabled at inference).
_BENIGN_MISSING = frozenset({"masked_spec_embed"})


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _build_hf_ssl_model(checkpoint: Dict[str, torch.Tensor]) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "XLSR-Mamba requires `transformers`. Install with:\n"
            "  pip install 'transformers>=4.30' huggingface_hub safetensors"
        ) from exc

    config = AutoConfig.from_pretrained(_XLSR_BACKBONE)
    has_conv_bias = any(
        k.startswith(_SSL_CKPT_PREFIX)
        and ".feature_extractor.conv_layers." in k
        and k.endswith(".0.bias")
        for k in checkpoint
    )
    if hasattr(config, "conv_bias") and bool(config.conv_bias) != has_conv_bias:
        _LOGGER.debug(
            "XLSR-Mamba: overriding config.conv_bias %s -> %s from checkpoint.",
            config.conv_bias,
            has_conv_bias,
        )
        config.conv_bias = has_conv_bias
    if hasattr(config, "apply_spec_augment"):
        config.apply_spec_augment = False
    if hasattr(config, "mask_time_prob"):
        config.mask_time_prob = 0.0
    if hasattr(config, "mask_feature_prob"):
        config.mask_feature_prob = 0.0

    ssl_model = AutoModel.from_config(config)
    ssl_model.eval()
    return ssl_model


def _translate_ssl_checkpoint(
    checkpoint: Dict[str, torch.Tensor],
    ssl_model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map fairseq-style ``ssl_model.model.*`` tensors to HF Wav2Vec2 naming."""
    expected_keys = set(ssl_model.state_dict().keys())
    remapped: Dict[str, torch.Tensor] = {}
    for key, val in checkpoint.items():
        if not key.startswith(_SSL_CKPT_PREFIX):
            continue
        remapped[_TRANSLATOR_PREFIX + key[len(_SSL_CKPT_PREFIX) :]] = val

    translated, _ = _translate_fairseq_ssl_state_dict(remapped, expected_keys)
    return {_SSL_CKPT_PREFIX + k: v for k, v in translated.items()}


def build_xlsr_mamba_model(
    repo_id: str,
    cache_dir: Path,
    *,
    emb_size: int = DEFAULT_EMB_SIZE,
    num_encoders: int = DEFAULT_NUM_ENCODERS,
) -> Tuple[XLSRMambaModel, Dict[str, Any]]:
    """Download HF weights, wire the model, and return a load report."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "XLSR-Mamba requires `huggingface_hub` and `safetensors`. Install with:\n"
            "  pip install huggingface_hub safetensors"
        ) from exc

    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "XLSR-Mamba requires `mamba-ssm` (CUDA build matching your PyTorch). "
            "Install after PyTorch, for example:\n"
            "  pip install torch  # CUDA build first\n"
            "  pip install mamba-ssm\n"
            "or: pip install detectzoo[audio,xlsr_mamba]"
        ) from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("XLSR-Mamba: downloading %s (cache=%s)", repo_id, cache_dir)
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=_SAFETENSORS_FILENAME,
        cache_dir=str(cache_dir),
    )
    raw_ckpt = _strip_module_prefix(load_file(weights_path))

    # Triton fused norms are unavailable on CPU-only hosts; use the equivalent
    # eager path (same weights, mathematically identical forward).
    fused_add_norm = torch.cuda.is_available()
    if not fused_add_norm:
        _LOGGER.info(
            "XLSR-Mamba: CUDA unavailable -- using fused_add_norm=False for CPU inference."
        )

    ssl_model = _build_hf_ssl_model(raw_ckpt)
    model = XLSRMambaModel(
        ssl_model,
        emb_size=emb_size,
        num_encoders=num_encoders,
        fused_add_norm=fused_add_norm,
    )

    ssl_sd = _translate_ssl_checkpoint(raw_ckpt, ssl_model)
    head_sd = {k: v for k, v in raw_ckpt.items() if not k.startswith(_SSL_CKPT_PREFIX)}
    merged_sd = {**ssl_sd, **head_sd}

    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    silently_random = [k for k in missing if k not in _BENIGN_MISSING]

    matched = len(model.state_dict()) - len(missing)

    report: Dict[str, Any] = {
        "repo_id": repo_id,
        "matched": matched,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "silently_random": silently_random,
        "fused_add_norm": fused_add_norm,
    }

    _LOGGER.info(
        "XLSR-Mamba/%s: load report -- matched=%d, missing=%d, unexpected=%d, "
        "silently_random=%d.",
        repo_id,
        matched,
        len(missing),
        len(unexpected),
        len(silently_random),
    )

    if len(silently_random) > 10:
        raise RuntimeError(
            f"XLSR-Mamba/{repo_id}: too many parameters left at random init after "
            f"loading ({len(silently_random)} > 10). Missing keys (first 20): "
            f"{silently_random[:20]}"
        )
    if silently_random:
        raise RuntimeError(
            f"XLSR-Mamba/{repo_id}: SSL/head translation incomplete -- "
            f"{len(silently_random)} parameter(s) would stay at random init. "
            f"First 10: {silently_random[:10]}"
        )

    model.eval()
    return model, report
