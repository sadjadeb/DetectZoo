"""Video-modality detectors for identifying AI-generated videos."""

from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.detectors.video.busterx import BusterXDetector
from detectzoo.detectors.video.split import SPLITDetector
from detectzoo.detectors.video.waverep import WaveRepDetector

__all__ = [
    "BaseVideoDetector",
    "BusterXDetector",
    "SPLITDetector",
    "WaveRepDetector",
]
