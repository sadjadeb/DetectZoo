"""Dataset abstractions for AI-content detection benchmarks."""

from detectzoo.datasets.audio import ASVspoof2019Dataset, FoRDataset, InTheWildDataset
from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.datasets.image import (
    AIGCDetectDataset,
    CNNDetectionDataset,
    DRCT2MDataset,
    SelfSynthesisDataset,
    UnivFDDataset,
)
from detectzoo.datasets.text import (
    CHEATDataset,
    HC3Dataset,
    HC3PlusDataset,
    L2RDataset,
    M4Dataset,
    MAGEDataset,
    OpenLLMTextDataset,
    RAIDDataset,
    TuringBenchDataset,
    WritingPromptsDataset,
    XSumDataset,
)

__all__ = [
    "BaseDataset",
    "DatasetItem",
    "ASVspoof2019Dataset",
    "FoRDataset",
    "InTheWildDataset",
    "AIGCDetectDataset",
    "CNNDetectionDataset",
    "DRCT2MDataset",
    "SelfSynthesisDataset",
    "UnivFDDataset",
    "CHEATDataset",
    "HC3Dataset",
    "HC3PlusDataset",
    "L2RDataset",
    "M4Dataset",
    "MAGEDataset",
    "OpenLLMTextDataset",
    "RAIDDataset",
    "TuringBenchDataset",
    "WritingPromptsDataset",
    "XSumDataset",
]

