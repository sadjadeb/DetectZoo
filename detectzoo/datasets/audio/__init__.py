"""Audio-modality datasets for AI-generated audio detection."""

from detectzoo.datasets.audio.asvspoof2019 import ASVspoof2019Dataset
from detectzoo.datasets.audio.deepfake_eval_2024 import DeepfakeEval2024Dataset
from detectzoo.datasets.audio.for_dataset import FoRDataset

__all__ = ["ASVspoof2019Dataset", "DeepfakeEval2024Dataset", "FoRDataset"]
