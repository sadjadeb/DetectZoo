"""Checkpoint loader for TCM (fairseq->HF SSL translation + .pth weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from detectzoo.detectors.audio._anti_deepfake_common import _translate_fairseq_ssl_state_dict
from detectzoo.detectors.audio.tcm._model import (
    DEFAULT_EMB_SIZE,
    DEFAULT_HEADS,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_NUM_ENCODERS,
    TCMModel,
)
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_SSL_CKPT_PREFIX = "ssl_model.model."
_TRANSLATOR_PREFIX = "m_ssl.model."
_XLSR_BACKBONE = "facebook/wav2vec2-large-xlsr-53"

# Official pretrained checkpoints are OneDrive-only (see upstream README). Do NOT
# hardcode the SharePoint URL. After the lab mirrors weights to HuggingFace, set
# the repo ids below and auto-download will work with zero manual steps.
_HF_REPO_IDS: Dict[str, Optional[str]] = {
    "LA": None,
    "DF": None,
}
_HF_FILENAMES: Dict[str, str] = {
    "LA": "tcm_la.pth",
    "DF": "tcm_df.pth",
}

_ONEDRIVE_NOTE = (
    "TCM official weights are hosted on OneDrive only "
    "(ductuantruong/tcm_add README). Mirror them to the lab HuggingFace org, "
    "then set _HF_REPO_IDS in detectzoo/detectors/audio/tcm/_loader.py. "
    "Until then, pass checkpoint_path= pointing at a local .pth file."
)

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
            "TCM requires `transformers`. Install with:\n"
            "  pip install 'transformers>=4.30' torchaudio soundfile"
        ) from exc

    config = AutoConfig.from_pretrained(_XLSR_BACKBONE)
    has_conv_bias = any(
        k.startswith(_SSL_CKPT_PREFIX)
        and ".feature_extractor.conv_layers." in k
        and k.endswith(".0.bias")
        for k in checkpoint
    )
    if hasattr(config, "conv_bias") and bool(config.conv_bias) != has_conv_bias:
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
    expected_keys = set(ssl_model.state_dict().keys())
    remapped: Dict[str, torch.Tensor] = {}
    for key, val in checkpoint.items():
        if not key.startswith(_SSL_CKPT_PREFIX):
            continue
        remapped[_TRANSLATOR_PREFIX + key[len(_SSL_CKPT_PREFIX) :]] = val

    translated, _ = _translate_fairseq_ssl_state_dict(remapped, expected_keys)
    return {_SSL_CKPT_PREFIX + k: v for k, v in translated.items()}


def resolve_checkpoint_path(
    variant: str,
    cache_dir: Path,
    checkpoint_path: Optional[Path] = None,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
        return path

    variant = variant.upper()
    repo_id = _HF_REPO_IDS.get(variant)
    if not repo_id:
        raise RuntimeError(
            f"TCM/{variant}: no HuggingFace mirror configured. {_ONEDRIVE_NOTE}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "TCM auto-download requires `huggingface_hub`. Install with:\n"
            "  pip install huggingface_hub"
        ) from exc

    filename = _HF_FILENAMES[variant]
    cache_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("TCM/%s: downloading %s/%s (cache=%s)", variant, repo_id, filename, cache_dir)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return Path(downloaded)


def build_tcm_model(
    weight_path: Path,
    *,
    device: torch.device,
    emb_size: int = DEFAULT_EMB_SIZE,
    heads: int = DEFAULT_HEADS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    num_encoders: int = DEFAULT_NUM_ENCODERS,
) -> Tuple[TCMModel, Dict[str, Any]]:
    """Load a TCM checkpoint and return the wired model plus a load report."""
    raw = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected a state-dict-like checkpoint at {weight_path}")

    ckpt = _strip_module_prefix(raw)
    ssl_model = _build_hf_ssl_model(ckpt)
    model = TCMModel(
        ssl_model,
        emb_size=emb_size,
        heads=heads,
        kernel_size=kernel_size,
        num_encoders=num_encoders,
        device=device,
    )

    ssl_sd = _translate_ssl_checkpoint(ckpt, ssl_model)
    head_sd = {k: v for k, v in ckpt.items() if not k.startswith(_SSL_CKPT_PREFIX)}
    merged_sd = {**ssl_sd, **head_sd}

    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    silently_random = [k for k in missing if k not in _BENIGN_MISSING]
    matched = len(model.state_dict()) - len(missing)

    report: Dict[str, Any] = {
        "weight_path": str(weight_path),
        "matched": matched,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "silently_random": silently_random,
    }

    _LOGGER.info(
        "TCM: load report -- matched=%d, missing=%d, unexpected=%d, silently_random=%d.",
        matched,
        len(missing),
        len(unexpected),
        len(silently_random),
    )

    if len(silently_random) > 10:
        raise RuntimeError(
            f"TCM: too many parameters left at random init after loading "
            f"({len(silently_random)} > 10). Missing keys (first 20): "
            f"{silently_random[:20]}"
        )
    if silently_random:
        raise RuntimeError(
            f"TCM: checkpoint translation incomplete -- {len(silently_random)} "
            f"parameter(s) would stay at random init. First 10: {silently_random[:10]}"
        )

    model.eval()
    return model, report
