"""Image-modality datasets for AI-generated image detection."""

from detectzoo.datasets.image.aigcdetect import AIGCDetectDataset
from detectzoo.datasets.image.chameleon import ChameleonDataset
from detectzoo.datasets.image.cnn_detection import CNNDetectionDataset
from detectzoo.datasets.image.drct2m import DRCT2MDataset
from detectzoo.datasets.image.genimage import GenImageDataset
from detectzoo.datasets.image.self_synthesis import SelfSynthesisDataset
from detectzoo.datasets.image.univfd import UnivFDDataset

__all__ = [
    "AIGCDetectDataset",
    "CNNDetectionDataset",
    "DRCT2MDataset",
    "SelfSynthesisDataset",
    "UnivFDDataset",
    "GenImageDataset",
    "ChameleonDataset",
]
