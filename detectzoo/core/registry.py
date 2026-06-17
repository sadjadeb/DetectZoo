"""Global detector and dataset registries with factory functions."""

from __future__ import annotations

from typing import Any, Type

from detectzoo.core.base import BaseDetector

_REGISTRY: dict[str, Type[BaseDetector]] = {}
_ALIASES: dict[str, str] = {}


def register_detector(name: str, aliases: list[str] | None = None):
    """Class decorator that registers a detector under *name*.

    Parameters:
        name: Primary registration name.
        aliases: Optional list of alternative names that resolve to the
            same detector class.

    Usage::

        @register_detector("my_detector", aliases=["my_det"])
        class MyDetector(BaseDetector):
            ...
    """

    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if name in _REGISTRY:
            raise ValueError(
                f"Detector '{name}' is already registered ({_REGISTRY[name].__name__}). "
                "Use a different name."
            )
        cls.name = name
        _REGISTRY[name] = cls
        for alias in aliases or []:
            _ALIASES[alias] = name
        return cls

    return decorator


def load_detector(name: str, **kwargs: Any) -> BaseDetector:
    """Instantiate a registered detector by name.

    Parameters:
        name: Registered detector identifier (e.g. ``"detectgpt"``).
        **kwargs: Forwarded to the detector constructor.

    Returns:
        An initialised detector instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    resolved = _ALIASES.get(name, name)
    if resolved not in _REGISTRY:
        available = ", ".join(sorted(set(_REGISTRY) | set(_ALIASES))) or "(none)"
        raise ValueError(f"Unknown detector '{name}'. Available detectors: {available}")
    return _REGISTRY[resolved](**kwargs)


def list_detectors(modality: str | None = None) -> list[str]:
    """Return names of all registered detectors, optionally filtered by modality."""
    if modality is None:
        return sorted(_REGISTRY)
    return sorted(name for name, cls in _REGISTRY.items() if cls.modality == modality)


# ======================================================================
# Dataset registry
# ======================================================================

from detectzoo.datasets.base import BaseDataset  # noqa: E402

_DATASET_REGISTRY: dict[str, Type[BaseDataset]] = {}
_DATASET_ALIASES: dict[str, str] = {}


def register_dataset(name: str, aliases: list[str] | None = None):
    """Class decorator that registers a dataset under *name*.

    Parameters:
        name: Primary registration name.
        aliases: Optional list of alternative names that resolve to the
            same dataset class.

    Usage::

        @register_dataset("my_dataset", aliases=["my_ds"])
        class MyDataset(BaseDataset):
            ...
    """

    def decorator(cls: Type[BaseDataset]) -> Type[BaseDataset]:
        if name in _DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' is already registered "
                f"({_DATASET_REGISTRY[name].__name__}). Use a different name."
            )
        cls.name = name
        _DATASET_REGISTRY[name] = cls
        for alias in aliases or []:
            _DATASET_ALIASES[alias] = name
        return cls

    return decorator


def load_dataset(name: str, **kwargs: Any) -> BaseDataset:
    """Instantiate a registered dataset by name.

    Parameters:
        name: Registered dataset identifier (e.g. ``"hc3"``).
        **kwargs: Forwarded to the dataset constructor.

    Returns:
        An initialised dataset instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    resolved = _DATASET_ALIASES.get(name, name)
    if resolved not in _DATASET_REGISTRY:
        available = ", ".join(sorted(set(_DATASET_REGISTRY) | set(_DATASET_ALIASES))) or "(none)"
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return _DATASET_REGISTRY[resolved](**kwargs)


def list_datasets(modality: str | None = None) -> list[str]:
    """Return names of all registered datasets, optionally filtered by modality."""
    if modality is None:
        return sorted(_DATASET_REGISTRY)
    return sorted(name for name, cls in _DATASET_REGISTRY.items() if cls.modality == modality)
