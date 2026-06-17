"""Audio-modality detectors for identifying synthetic / deepfake speech."""

from detectzoo.detectors.audio.aasist import AASISTDetector
from detectzoo.detectors.audio.anti_deepfake_hubert.detector import (
    AntiDeepfakeHuBERTDetector,
)
from detectzoo.detectors.audio.anti_deepfake_wav2vec.detector import (
    AntiDeepfakeWav2VecDetector,
)
from detectzoo.detectors.audio.anti_deepfake_xlsr2b.detector import (
    AntiDeepfakeXLSR2BDetector,
)
from detectzoo.detectors.audio.ast_asvspoof.detector import ASTASVspoofDetector
from detectzoo.detectors.audio.raw_pc_darts.detector import RawPCDartsDetector
from detectzoo.detectors.audio.rawgat_st import RawGATSTDetector
from detectzoo.detectors.audio.rawnet2 import RawNet2Detector
from detectzoo.detectors.audio.restssdnet import ResTSSDNetDetector
from detectzoo.detectors.audio.safeear.detector import SafeEarDetector
from detectzoo.detectors.audio.samo import SAMODetector
from detectzoo.detectors.audio.tcm.detector import TCMDetector
from detectzoo.detectors.audio.xlsr_mamba.detector import XLSRMambaDetector
from detectzoo.detectors.audio.xlsr_sls.detector import XLSRSLSDetector

__all__ = [
    "AASISTDetector",
    "ASTASVspoofDetector",
    "AntiDeepfakeHuBERTDetector",
    "AntiDeepfakeWav2VecDetector",
    "AntiDeepfakeXLSR2BDetector",
    "RawGATSTDetector",
    "RawNet2Detector",
    "RawPCDartsDetector",
    "ResTSSDNetDetector",
    "SafeEarDetector",
    "SAMODetector",
    "TCMDetector",
    "XLSRMambaDetector",
    "XLSRSLSDetector",
]
