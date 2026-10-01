"""I/O helpers for loading text, images, audio, and video."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def load_text(source: Union[str, Path]) -> str:
    """Load text from a file path or return the string directly.

    If *source* is an existing file, its contents are read; otherwise
    *source* is treated as raw text.
    """
    path = Path(source)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return str(source)


def load_image(source: Union[str, Path]):
    """Load an image as a PIL ``Image`` in RGB mode.

    Parameters:
        source: Path to an image file.

    Returns:
        A ``PIL.Image.Image`` in RGB.
    """
    from PIL import Image

    return Image.open(source).convert("RGB")


def load_audio(
    source: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *target_sr*.

    Attempts to use ``torchaudio`` first, falling back to ``librosa``.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a 1-D numpy
        array (mono, float32).
    """
    path = str(source)

    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.mean(dim=0).numpy()
        return waveform, target_sr
    except ImportError:
        pass

    import librosa

    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform, sr


def load_video(
    source: Union[str, Path],
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Load the frames of a video file as RGB.

    Uses OpenCV (``cv2``) for decoding.
    If *max_frames* is given, decoding stops after that many frames.

    Returns:
        ``(frames, fps)`` where *frames* is a ``uint8`` array of shape
        ``[T, H, W, 3]`` in RGB order.
    """
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frames = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames could be decoded from: {source}")
    return np.stack(frames), fps
