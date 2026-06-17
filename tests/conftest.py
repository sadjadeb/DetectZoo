"""Shared pytest fixtures and helpers for the DetectZoo test-suite."""

from __future__ import annotations

import importlib

import pytest

import detectzoo  # noqa: F401  (ensures registries are populated)
from detectzoo.core.base import BaseDetector, DetectionResult


def require_modality(modality: str) -> None:
    """Skip the current test if a modality's detector package is unavailable.

    DetectZoo loads modality subpackages on a best-effort basis (see
    ``detectzoo/__init__.py``): if an optional heavy dependency such as
    ``diffusers`` or ``timm`` is missing, the whole subpackage is skipped
    with a warning rather than failing import.  Tests that assert on a
    modality's detectors must therefore skip gracefully when that package
    could not be imported, so the suite stays green on partial installs.
    """
    try:
        importlib.import_module(f"detectzoo.detectors.{modality}")
    except ImportError as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"{modality} detectors unavailable ({exc})")


class DummyDetector(BaseDetector):
    """Lightweight detector that scores text by its length (no models)."""

    name = "dummy"
    modality = "text"

    def predict(self, input_data) -> DetectionResult:
        return self._make_result(min(len(str(input_data)) / 100.0, 1.0))


@pytest.fixture
def dummy_detector() -> DummyDetector:
    return DummyDetector(threshold=0.5)
