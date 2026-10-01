"""Text-modality detectors for identifying LLM-generated text."""

from detectzoo.detectors.text.ada_detect_gpt import AdaDetectGPTDetector
from detectzoo.detectors.text.binoculars import BinocularsDetector
from detectzoo.detectors.text.biscope import BiScopeDetector
from detectzoo.detectors.text.coco import CoCoDetector
from detectzoo.detectors.text.detect_gpt import DetectGPTDetector
from detectzoo.detectors.text.detective import DeTeCtiveDetector
from detectzoo.detectors.text.dna_detectllm import DNADetectLLMDetector
from detectzoo.detectors.text.dna_gpt import DNAGPTDetector
from detectzoo.detectors.text.entropy import EntropyDetector
from detectzoo.detectors.text.fast_detect_gpt import FastDetectGPTDetector
from detectzoo.detectors.text.gecscore import GECScoreDetector
from detectzoo.detectors.text.ghostbuster import GhostbusterDetector
from detectzoo.detectors.text.glimpse import GlimpseDetector
from detectzoo.detectors.text.ide import MLEDetector, PHDDetector
from detectzoo.detectors.text.imbd import ImBDDetector
from detectzoo.detectors.text.ipad import IPADDetector
from detectzoo.detectors.text.irm import IRMDetector
from detectzoo.detectors.text.lastde import LastdeDetector, LastdePPDetector
from detectzoo.detectors.text.log_likelihood import LogLikelihoodDetector
from detectzoo.detectors.text.log_rank import LogRankDetector
from detectzoo.detectors.text.lrr import LRRDetector
from detectzoo.detectors.text.npr import NPRDetector
from detectzoo.detectors.text.ood_detectors import DSVDDDetector, EnergyDetector, HRNDetector
from detectzoo.detectors.text.radar import RADARDetector
from detectzoo.detectors.text.raidar import RaidarDetector
from detectzoo.detectors.text.rank import RankDetector
from detectzoo.detectors.text.remodetect import ReMoDetectDetector
from detectzoo.detectors.text.revise_detect import ReviseDetector
from detectzoo.detectors.text.roberta import (
    ChatGPTDetector,
    RobertaBaseDetector,
    RobertaLargeDetector,
)
from detectzoo.detectors.text.text_fluoroscopy import TextFluoroscopyDetector
from detectzoo.detectors.text.tocsin import TOCSINDetector

__all__ = [
    "AdaDetectGPTDetector",
    "BinocularsDetector",
    "BiScopeDetector",
    "ChatGPTDetector",
    "CoCoDetector",
    "DetectGPTDetector",
    "DeTeCtiveDetector",
    "DNADetectLLMDetector",
    "DNAGPTDetector",
    "DSVDDDetector",
    "EnergyDetector",
    "EntropyDetector",
    "FastDetectGPTDetector",
    "GECScoreDetector",
    "GhostbusterDetector",
    "GlimpseDetector",
    "HRNDetector",
    "ImBDDetector",
    "IPADDetector",
    "IRMDetector",
    "LastdeDetector",
    "LastdePPDetector",
    "LogLikelihoodDetector",
    "LogRankDetector",
    "LRRDetector",
    "MLEDetector",
    "NPRDetector",
    "PHDDetector",
    "RADARDetector",
    "RaidarDetector",
    "RankDetector",
    "ReMoDetectDetector",
    "ReviseDetector",
    "RobertaBaseDetector",
    "RobertaLargeDetector",
    "TextFluoroscopyDetector",
    "TOCSINDetector",
]
