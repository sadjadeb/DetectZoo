"""Checkpoint loader for SafeEar (SpeechTokenizer + SafeEar1s)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from detectzoo.detectors.audio.safeear._models.decouple import SpeechTokenizer
from detectzoo.detectors.audio.safeear._models.safeear import SafeEar1s
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Community HuggingFace mirror (LetterLiGo weights are not on official HF).
# Repoint to the lab org once mirrored.
_DEFAULT_HF_REPO_ID = "TEC2004/SafeEar-ASV19-spoof-detection"
_DETECT_CKPT = "model.ckpt"
_CODEC_CKPT = "SpeechTokenizer.pt"

# Default architecture from config/train19.yaml.
_DEFAULT_CODEC_KWARGS: Dict[str, Any] = {
    "n_filters": 64,
    "strides": [8, 5, 4, 2],
    "dimension": 1024,
    "semantic_dimension": 768,
    "bidirectional": True,
    "dilation_base": 2,
    "residual_kernel_size": 3,
    "n_residual_layers": 1,
    "lstm_layers": 2,
    "activation": "ELU",
    "codebook_size": 1024,
    "n_q": 8,
    "sample_rate": 16000,
}
_DEFAULT_DETECT_KWARGS: Dict[str, Any] = {
    "embedding_dim": 1024,
    "dropout_rate": 0.1,
    "attention_dropout": 0.1,
    "stochastic_depth": 0.1,
    "num_layers": 2,
    "num_heads": 8,
    "num_classes": 2,
    "positional_embedding": "sine",
    "mlp_ratio": 1.0,
}
_RVQ_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7]


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _download_hf_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "SafeEar auto-download requires `huggingface_hub`. Install with:\n"
            "  pip install huggingface_hub"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("SafeEar: downloading %s/%s (cache=%s)", repo_id, filename, cache_dir)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return Path(downloaded)


def resolve_weight_paths(
    cache_dir: Path,
    *,
    repo_id: str = _DEFAULT_HF_REPO_ID,
    detect_checkpoint_path: Optional[Path] = None,
    codec_checkpoint_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    if detect_checkpoint_path is not None:
        detect_path = Path(detect_checkpoint_path).expanduser().resolve()
        if not detect_path.is_file():
            raise FileNotFoundError(f"detect_checkpoint_path does not exist: {detect_path}")
    else:
        detect_path = _download_hf_file(repo_id, _DETECT_CKPT, cache_dir)

    if codec_checkpoint_path is not None:
        codec_path = Path(codec_checkpoint_path).expanduser().resolve()
        if not codec_path.is_file():
            raise FileNotFoundError(f"codec_checkpoint_path does not exist: {codec_path}")
    else:
        codec_path = _download_hf_file(repo_id, _CODEC_CKPT, cache_dir)

    return detect_path, codec_path


def build_safeear_models(
    detect_checkpoint: Path,
    codec_checkpoint: Path,
    *,
    device: torch.device,
    codec_kwargs: Optional[Dict[str, Any]] = None,
    detect_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[SpeechTokenizer, SafeEar1s, Dict[str, Any]]:
    """Build and load SafeEar inference models (codec + detector)."""
    codec_cfg = {**_DEFAULT_CODEC_KWARGS, **(codec_kwargs or {})}
    detect_cfg = {**_DEFAULT_DETECT_KWARGS, **(detect_kwargs or {})}

    codec = SpeechTokenizer(**codec_cfg)
    codec_state = torch.load(codec_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(codec_state, dict) and "state_dict" in codec_state:
        codec_state = codec_state["state_dict"]
    codec_missing, codec_unexpected = codec.load_state_dict(codec_state, strict=False)

    detect = SafeEar1s(front=None, **detect_cfg)
    lightning_ckpt = torch.load(detect_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(lightning_ckpt, dict) or "state_dict" not in lightning_ckpt:
        raise RuntimeError(
            f"Expected a PyTorch Lightning checkpoint at {detect_checkpoint}, "
            "with a top-level 'state_dict' entry."
        )
    detect_state = _strip_prefix(lightning_ckpt["state_dict"], "detect_model.")
    detect_missing, detect_unexpected = detect.load_state_dict(detect_state, strict=False)

    codec_matched = len(codec.state_dict()) - len(codec_missing)
    detect_matched = len(detect.state_dict()) - len(detect_missing)

    report: Dict[str, Any] = {
        "detect_checkpoint": str(detect_checkpoint),
        "codec_checkpoint": str(codec_checkpoint),
        "codec_matched": codec_matched,
        "codec_missing": len(codec_missing),
        "codec_unexpected": len(codec_unexpected),
        "detect_matched": detect_matched,
        "detect_missing": len(detect_missing),
        "detect_unexpected": len(detect_unexpected),
        "codec_missing_keys": codec_missing,
        "detect_missing_keys": detect_missing,
    }

    _LOGGER.info(
        "SafeEar codec load: matched=%d missing=%d unexpected=%d",
        codec_matched,
        len(codec_missing),
        len(codec_unexpected),
    )
    _LOGGER.info(
        "SafeEar detect load: matched=%d missing=%d unexpected=%d",
        detect_matched,
        len(detect_missing),
        len(detect_unexpected),
    )

    if len(codec_missing) > 10 or len(detect_missing) > 10:
        raise RuntimeError(
            "SafeEar: too many missing keys after loading "
            f"(codec={len(codec_missing)}, detect={len(detect_missing)})."
        )

    codec.to(device).eval()
    detect.to(device).eval()
    return codec, detect, report


@torch.no_grad()
def safeear_forward(
    codec: SpeechTokenizer,
    detect: SafeEar1s,
    wav: torch.Tensor,
    *,
    rvq_layers: Optional[list[int]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Run upstream SafeEar inference (SpeechTokenizer RVQ tokens -> SafeEar1s)."""
    if seed is not None:
        torch.manual_seed(seed)
    x = wav.to(memory_format=torch.contiguous_format).float()
    if x.dim() == 2:
        x = x.unsqueeze(1)
    layers = rvq_layers or _RVQ_LAYERS
    _, _, _, acoustic_tokens = codec(x, layers=layers)
    logits, _ = detect(acoustic_tokens)
    return torch.softmax(logits, dim=-1)
