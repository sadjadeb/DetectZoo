"""Shared base class and helpers for video detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from detectzoo.core.base import BaseDetector
from detectzoo.utils.io import load_video
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVideoDetector(BaseDetector):
    """Base for video detectors.

    Handles lazy model loading, video input normalisation, frame sampling
    and aggregation of per-frame scores into a video-level score.

    Subclasses implement :meth:`_load_model` (which must set ``self._model``)
    and :meth:`predict`.
    """

    modality = "video"

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "cpu",
        num_frames: int | None = 16,
        sampling: str = "uniform",
        aggregation: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, **kwargs)
        self.num_frames = num_frames
        self.sampling = sampling
        self.aggregation = aggregation
        self._model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore[return-value]

    def _load_model(self) -> None:
        """Load the detector's model into ``self._model``."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _load_model().")

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _sample_frame_indices(self, total_frames: int) -> np.ndarray:
        """Pick ``self.num_frames`` indices out of *total_frames*.

        ``sampling`` is ``"uniform"`` (evenly spaced over the whole clip) or
        ``"first"`` (the leading consecutive frames).  When ``num_frames`` is
        ``None`` or not smaller than *total_frames*, every frame is kept.
        """
        if self.sampling not in ("uniform", "first"):
            raise ValueError(
                f"Unknown sampling strategy '{self.sampling}'; use 'uniform' or 'first'."
            )
        if total_frames <= 0:
            raise ValueError("Video contains no frames.")
        if self.num_frames is None or self.num_frames >= total_frames:
            return np.arange(total_frames)
        if self.sampling == "uniform":
            return np.linspace(0, total_frames - 1, self.num_frames).round().astype(int)
        return np.arange(self.num_frames)

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Subsample a ``[T, C, H, W]`` frame tensor along time."""
        idx = self._sample_frame_indices(frames.shape[0])
        return frames[torch.from_numpy(idx)]

    # ------------------------------------------------------------------
    # Score aggregation
    # ------------------------------------------------------------------

    def _aggregate_scores(self, frame_scores: Sequence[float] | torch.Tensor) -> float:
        """Reduce per-frame scores to one video-level score.

        ``aggregation`` is ``"mean"``, ``"max"`` or ``"median"``.
        """
        scores = torch.as_tensor(frame_scores, dtype=torch.float32).flatten()
        if scores.numel() == 0:
            raise ValueError("Cannot aggregate an empty list of frame scores.")
        if self.aggregation == "mean":
            return float(scores.mean())
        if self.aggregation == "max":
            return float(scores.max())
        if self.aggregation == "median":
            return float(scores.median())
        raise ValueError(
            f"Unknown aggregation '{self.aggregation}'; use 'mean', 'max' or 'median'."
        )

    # ------------------------------------------------------------------
    # Video normalisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_input(input_data: Any, max_frames: int | None = None) -> torch.Tensor:
        """Accept a video path, array, tensor or list of frames.

        Returns a ``uint8`` tensor of shape ``[T, C, H, W]`` in RGB order.
        *max_frames* stops decoding a video file early (ignored otherwise).

        * ``str`` / ``Path``: decoded with :func:`~detectzoo.utils.io.load_video`.
        * ``np.ndarray`` / ``torch.Tensor``: ``[T, H, W, C]`` or ``[T, C, H, W]``.
        * list of PIL Images or ``[H, W, C]`` arrays.
        """
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            frames, _ = load_video(path, max_frames=max_frames)
            return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

        if isinstance(input_data, (list, tuple)):
            if not input_data:
                raise ValueError("Received an empty list of frames.")
            input_data = np.stack(
                [
                    np.asarray(f.convert("RGB")) if hasattr(f, "convert") else np.asarray(f)
                    for f in input_data
                ]
            )

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)

        if not isinstance(input_data, torch.Tensor):
            raise TypeError(
                "Expected a video path, numpy array, torch tensor or list of frames; "
                f"got {type(input_data).__name__}."
            )
        if input_data.ndim != 4:
            raise ValueError(f"Expected a 4-D video tensor; got shape {tuple(input_data.shape)}.")
        if input_data.shape[-1] in (1, 3) and input_data.shape[1] not in (1, 3):
            input_data = input_data.permute(0, 3, 1, 2)
        return input_data.contiguous()
