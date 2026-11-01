"""Video-modality detectors for identifying AI-generated videos."""

from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.detectors.video.busterx import BusterXDetector
from detectzoo.detectors.video.nsgvd import NSGVDDetector
from detectzoo.detectors.video.split import SPLITDetector
from detectzoo.detectors.video.stall import STALLDetector
from detectzoo.detectors.video.videoveritas import VideoVeritasDetector
from detectzoo.detectors.video.waverep import WaveRepDetector

__all__ = [
    "BaseVideoDetector",
    "BusterXDetector",
    "NSGVDDetector",
    "SPLITDetector",
    "STALLDetector",
    "VideoVeritasDetector",
    "WaveRepDetector",
]
