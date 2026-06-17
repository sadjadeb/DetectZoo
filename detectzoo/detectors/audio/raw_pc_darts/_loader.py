"""Checkpoint loader for Raw-PC-DARTS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import torch

from detectzoo.datasets._download import download_file
from detectzoo.detectors.audio.raw_pc_darts._genotype import DEFAULT_GENOTYPE, Genotype
from detectzoo.detectors.audio.raw_pc_darts._model import (
    DEFAULT_GRU_HSIZE,
    DEFAULT_GRU_LAYERS,
    DEFAULT_INIT_CHANNELS,
    DEFAULT_LAYERS,
    DEFAULT_SINC_KERNEL,
    DEFAULT_SINC_SCALE,
    RawPCDartsNetwork,
    build_default_args,
    build_raw_pc_darts_network,
)
from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Official pretrained weights (public Nextcloud share linked from upstream README).
_NEXTCLOUD_SHARE_TOKEN = "4DeWffZH6YG8enq"
_DEFAULT_CKPT_NAME = "fix_mel.pth"
_CKPT_URL = (
    f"https://nextcloud.eurecom.fr/s/{_NEXTCLOUD_SHARE_TOKEN}/download"
    f"?path=/&files={quote(_DEFAULT_CKPT_NAME)}"
)

# Optional lab HuggingFace mirror. When set, HF is tried before Nextcloud.
_HF_REPO_ID: Optional[str] = None
_HF_FILENAME = _DEFAULT_CKPT_NAME


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _download_hf_checkpoint(repo_id: str, filename: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Raw-PC-DARTS HuggingFace download requires `huggingface_hub`. Install with:\n"
            "  pip install huggingface_hub"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Raw-PC-DARTS: downloading %s/%s (cache=%s)", repo_id, filename, cache_dir)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return Path(downloaded)


def resolve_checkpoint_path(
    cache_dir: Path,
    checkpoint_path: Optional[Path] = None,
    *,
    repo_id: Optional[str] = None,
) -> Path:
    """Return a local checkpoint path, downloading official weights when needed."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / _DEFAULT_CKPT_NAME

    hf_repo = repo_id or _HF_REPO_ID
    if hf_repo:
        return _download_hf_checkpoint(hf_repo, _HF_FILENAME, cache_dir)

    if cached.is_file():
        _LOGGER.info("Raw-PC-DARTS: using cached checkpoint %s", cached)
        return cached

    _LOGGER.info(
        "Raw-PC-DARTS: downloading official weights (%s) to %s",
        _DEFAULT_CKPT_NAME,
        cached,
    )
    download_file(_CKPT_URL, cached)
    return cached


def build_raw_pc_darts_model(
    weight_path: Path,
    *,
    device: torch.device,
    init_channels: int = DEFAULT_INIT_CHANNELS,
    layers: int = DEFAULT_LAYERS,
    gru_hsize: int = DEFAULT_GRU_HSIZE,
    gru_layers: int = DEFAULT_GRU_LAYERS,
    sinc_scale: str = DEFAULT_SINC_SCALE,
    sinc_kernel: int = DEFAULT_SINC_KERNEL,
    is_mask: bool = False,
    is_trainable: bool = False,
    genotype: Genotype = DEFAULT_GENOTYPE,
) -> Tuple[RawPCDartsNetwork, Dict[str, Any]]:
    """Load a Raw-PC-DARTS checkpoint and return the wired model plus a load report."""
    raw = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected a state-dict-like checkpoint at {weight_path}")

    ckpt = _strip_module_prefix(raw)
    args = build_default_args(
        sinc_scale=sinc_scale,
        sinc_kernel=sinc_kernel,
        gru_hsize=gru_hsize,
        gru_layers=gru_layers,
        is_mask=is_mask,
        is_trainable=is_trainable,
    )
    model = build_raw_pc_darts_network(
        init_channels=init_channels,
        layers=layers,
        args=args,
        genotype=genotype,
    )
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    matched = len(model.state_dict()) - len(missing)

    report: Dict[str, Any] = {
        "weight_path": str(weight_path),
        "matched": matched,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }

    _LOGGER.info(
        "Raw-PC-DARTS: load report -- matched=%d, missing=%d, unexpected=%d.",
        matched,
        len(missing),
        len(unexpected),
    )

    if len(missing) > 10:
        raise RuntimeError(
            f"Raw-PC-DARTS: too many missing keys after loading ({len(missing)} > 10). "
            f"First 20: {missing[:20]}"
        )

    model.to(device)
    model.eval()
    model.drop_path_prob = 0.0
    return model, report
