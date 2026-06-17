"""Hugging Face stack: reduce warnings and log noise when using DetectZoo."""

from __future__ import annotations

import logging
import os
import sys
import warnings

_CONFIGURED = False

_HF_LOGGERS = (
    "transformers",
    "huggingface_hub",
    "tokenizers",
)


def _sync_huggingface_hub_verbosity() -> None:
    """Apply log level if ``huggingface_hub`` is already imported.

    That library configures its root logger at import time from ``HF_HUB_VERBOSITY``
    (default WARNING), which overrides any earlier ``logging.getLogger`` tweaks.
    """
    mod = sys.modules.get("huggingface_hub.logging")
    if mod is not None:
        mod.set_verbosity_error()


def configure_hf_quiet() -> None:
    """Set env vars, warning filters, and log levels for a quieter HF stack.

    Idempotent. Call before the first ``transformers`` import for full effect.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # Must be set before huggingface_hub import — it configures logging from this env.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    for prefix in ("transformers", "huggingface_hub", "tokenizers"):
        warnings.filterwarnings("ignore", module=prefix)

    for name in _HF_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    _sync_huggingface_hub_verbosity()


configure_hf_quiet()
