"""VideoVeritas — MLLM-based AI-generated video detection (ICML 2026).

Reference:
    Tan et al., "VideoVeritas: AI-Generated Video Detection via Perception
    Pretext Reinforcement Learning", ICML 2026.
    https://arxiv.org/abs/2602.08828

A Qwen3-VL-8B model tuned with Joint Preference Alignment and Perception
Pretext Reinforcement Learning (spatio-temporal grounding and object counting
as RL pretext tasks) reasons about the video and gives its verdict inside
``<answer> … </answer>``.

This wrapper follows the upstream evaluation (``self_scripts/infer/infer_vllm.py``
with ``FPS=3``, ``DURATION=5.0``): the first 5 s of the video, ~3 fps (an even
number of frames, at least 4) picked with ``linspace``, each frame resized to at
most 768 visual tokens (bicubic, antialiased) by the authors' ``qwen_vl_utils``
fork, their system prompt and question, sampling at temperature 0.7 with up to
4096 new tokens.  Sampling is seeded per call so results are repeatable; pass
``temperature=0`` for greedy decoding.

``score`` is P(fake) / (P(real) + P(fake)) from the model's raw logits at the
first token of the word inside ``<answer>``; when the answer is not a single
real/fake verdict the parsed label (or 0.5) is used instead.  The full response
is kept in ``metadata["response"]``.

Upstream: https://github.com/EricTan7/VideoVeritas
Weights: ``EricTanh/VideoVeritas`` on ModelScope (~17.5 GB).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_ID = "EricTanh/VideoVeritas"

# self_scripts/infer/infer_vllm.py, verbatim (note the typographic apostrophe).
_SYSTEM_PROMPT = (
    "You are an expert video analyst.\n"
    "Please think about the question as if you were a human pondering deeply. "
    "It’s encouraged to include self-reflection or verification in the reasoning "
    "process. Then, give a final verdict within <answer> </answer> tags."
)
_USER_PROMPT = "Is this video real or fake?"

# Authors' qwen_vl_utils fork (swift/qwen_vl_utils/vision_process.py).
_FRAME_FACTOR = 2
_MIN_FRAMES = 4
_MAX_FRAMES = 768
_VIDEO_MIN_TOKENS = 128
_VIDEO_MAX_TOKENS = 768
_MODEL_SEQ_LEN = 128000

_ANSWER_WORDS = ("real", "fake")


# ---------------------------------------------------------------------------
# Frame sampling (mirrors the authors' qwen_vl_utils fork)
# ---------------------------------------------------------------------------


def _floor_by(x: float, f: int) -> int:
    return math.floor(x / f) * f


def _ceil_by(x: float, f: int) -> int:
    return math.ceil(x / f) * f


def _frame_indices(total_frames: int, video_fps: float, fps: float, duration: float) -> List[int]:
    """``calculate_video_frame_range`` + ``smart_nframes`` + ``linspace`` sampling."""
    if video_fps <= 0 or total_frames <= 0:
        raise ValueError("Video has no frames or an invalid frame rate.")
    max_duration = total_frames / video_fps
    end_frame = min(math.floor(max(0.0, min(duration, max_duration)) * video_fps), total_frames - 1)
    if end_frame <= 0:
        raise ValueError("Video is too short to sample frames from.")
    count = end_frame + 1
    nframes = count / video_fps * fps
    max_frames = _floor_by(min(_MAX_FRAMES, count), _FRAME_FACTOR)
    nframes = min(min(max(nframes, _ceil_by(_MIN_FRAMES, _FRAME_FACTOR)), max_frames), count)
    nframes = _floor_by(nframes, _FRAME_FACTOR)
    if not _FRAME_FACTOR <= nframes <= count:
        raise ValueError(f"Cannot sample an even number of >= 2 frames from {count} frame(s).")
    return torch.linspace(0, end_frame, nframes).round().long().tolist()


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """``qwen_vl_utils.smart_resize``."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resized_hw(height: int, width: int, nframes: int, factor: int) -> Tuple[int, int]:
    """Target size used by the fork's ``fetch_video``."""
    min_pixels = _VIDEO_MIN_TOKENS * factor * factor
    total_pixels = _MODEL_SEQ_LEN * factor * factor * 0.9
    max_pixels = max(
        min(_VIDEO_MAX_TOKENS * factor * factor, total_pixels / nframes * _FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    return _smart_resize(height, width, factor, min_pixels, max_pixels)


def _read_frames(path: Path, fps: float, duration: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode the sampled RGB frames of *path* (``[T, H, W, 3]``) and metadata."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = _frame_indices(total, video_fps, fps, duration)
        targets = set(wanted)
        by_index: Dict[int, np.ndarray] = {}
        i = 0
        while i <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i in targets:
                by_index[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
    missing = [j for j in wanted if j not in by_index]
    if missing:
        raise ValueError(f"Could not decode {len(missing)} sampled frame(s) of {path}")
    meta = {"fps": video_fps, "frames_indices": wanted, "total_num_frames": total}
    return np.stack([by_index[j] for j in wanted]), meta


# ---------------------------------------------------------------------------
# Answer parsing (upstream metric code)
# ---------------------------------------------------------------------------


def _parse_answer(response: str) -> Optional[str]:
    """``"real"``/``"fake"`` from ``<answer>…</answer>``, as upstream judges it."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL)
    if match is None:
        return None
    text = match.group(1).lower()
    has_real, has_fake = "real" in text, "fake" in text
    if has_real != has_fake:
        return "real" if has_real else "fake"
    return None


class _WordLogitRecorder:
    """Logits processor that keeps the logits of a few candidate token ids."""

    def __init__(self, ids: List[int]) -> None:
        self.ids = ids
        self.steps: List[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.steps.append(scores[0, self.ids].float().cpu())
        return scores


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("videoveritas", aliases=["video_veritas"])
class VideoVeritasDetector(BaseVideoDetector):
    """VideoVeritas (Qwen3-VL-8B) video forgery detector with explanations.

    Parameters
    ----------
    model_name : str
        ModelScope model id (default ``EricTanh/VideoVeritas``) or a local
        directory holding the weights.
    revision : str, optional
        ModelScope revision (default: latest).
    threshold : float
        Decision boundary on P(fake) (default 0.5).
    device : str
        Torch device string.
    fps : float
        Sampling rate (default 3.0, as upstream ``FPS``).
    duration : float
        Seconds of video used from the start (default 5.0, as upstream
        ``DURATION``).
    temperature : float
        Sampling temperature (default 0.7, as upstream); 0 means greedy.
    seed : int
        Seed set before each generation when sampling (default 0).
    max_new_tokens : int
        Generation budget (default 4096).
    torch_dtype : str
        ``"bfloat16"`` (default), ``"float16"`` or ``"float32"``.
    cache_dir : str or Path, optional
        ModelScope cache directory for the weights (~17.5 GB).

    Non-path inputs (arrays, tensors, frame lists) are taken as RGB frames
    already sampled at ``fps``.
    """

    def __init__(
        self,
        *,
        model_name: str = _MODEL_ID,
        revision: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        fps: float = 3.0,
        duration: float = 5.0,
        temperature: float = 0.7,
        seed: int = 0,
        max_new_tokens: int = 4096,
        torch_dtype: str = "bfloat16",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, num_frames=None, **kwargs)
        if fps <= 0 or duration <= 0:
            raise ValueError("fps and duration must be > 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.model_name = model_name
        self.revision = revision
        self.fps = float(fps)
        self.duration = float(duration)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.torch_dtype = torch_dtype
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._processor: Any = None
        self._model_dir: Optional[str] = None
        self._answer_ids: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _resolve_model_dir(self) -> str:
        if self._model_dir is None:
            if Path(self.model_name).is_dir():
                self._model_dir = self.model_name
            else:
                from modelscope import snapshot_download

                kwargs = {"cache_dir": self.cache_dir}
                if self.revision:
                    kwargs["revision"] = self.revision
                try:  # use the local cache first, so cached weights work offline
                    self._model_dir = snapshot_download(
                        self.model_name, local_files_only=True, **kwargs
                    )
                except Exception:
                    self._model_dir = snapshot_download(self.model_name, **kwargs)
        return self._model_dir

    def _load_model(self) -> None:
        from transformers import AutoModelForImageTextToText

        path = self._resolve_model_dir()
        logger.info("Loading VideoVeritas from %s …", path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=getattr(torch, self.torch_dtype)
        )
        self._model.to(self._device).eval()

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._resolve_model_dir())
        return self._processor

    def _answer_token_ids(self) -> Dict[int, str]:
        """First-token ids of ``real``/``fake`` (with case/space variants) → word."""
        if self._answer_ids is None:
            tok = self.processor.tokenizer
            ids: Dict[int, str] = {}
            for word in _ANSWER_WORDS:
                for variant in (word, word.capitalize(), word.upper()):
                    for prefix in ("", " "):
                        first = tok.encode(prefix + variant, add_special_tokens=False)[0]
                        # Keep only tokens that start the word itself (not bare whitespace).
                        if tok.decode([first]).strip():
                            ids.setdefault(first, word)
            self._answer_ids = ids
        return self._answer_ids

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _load_video(self, input_data: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            return _read_frames(path, self.fps, self.duration)
        frames = self._normalise_input(input_data).permute(0, 2, 3, 1).numpy()
        if len(frames) % _FRAME_FACTOR:  # the model groups frames in pairs
            frames = np.concatenate([frames, frames[-1:]])
        meta = {
            "fps": self.fps,
            "frames_indices": list(range(len(frames))),
            "total_num_frames": len(frames),
        }
        return frames, meta

    def _build_inputs(self, frames: np.ndarray, meta: Dict[str, Any]):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF
        from transformers.video_utils import VideoMetadata

        ip = self.processor.video_processor
        factor = ip.patch_size * ip.merge_size
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # uint8 TCHW
        h, w = _resized_hw(video.shape[2], video.shape[3], video.shape[0], factor)
        video = TF.resize(
            video, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True
        ).float()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": _USER_PROMPT}],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        metadata = VideoMetadata(
            total_num_frames=meta["total_num_frames"],
            fps=meta["fps"],
            frames_indices=meta["frames_indices"],
        )
        return self.processor(
            text=[text],
            videos=[video],
            video_metadata=[metadata],
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _answer_step(self, generated: List[int]) -> Optional[int]:
        """Step of the first token after the last ``<answer>`` tag."""
        tok = self.processor.tokenizer
        text, seen, tag_step = "", 0, None
        for i, t in enumerate(generated):
            text += tok.decode([t])
            if text.count("<answer>") > seen:  # a new tag completed at step i
                seen, tag_step = text.count("<answer>"), i
        if tag_step is None:
            return None
        answer_ids = self._answer_token_ids()
        for j in range(tag_step + 1, len(generated)):
            if tok.decode([generated[j]]).strip():
                return j if generated[j] in answer_ids else None
        return None

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        from transformers import LogitsProcessorList

        frames, meta = self._load_video(input_data)
        inputs = self._build_inputs(frames, meta).to(self._device)
        answer_ids = self._answer_token_ids()
        id_list = list(answer_ids)
        recorder = _WordLogitRecorder(id_list)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            torch.manual_seed(self.seed)
            # vLLM's SamplingParams(temperature=0.7) upstream: no top-k / top-p.
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_k=0, top_p=1.0)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_k=None, top_p=None)
        out = self.model.generate(
            **inputs, logits_processor=LogitsProcessorList([recorder]), **gen_kwargs
        )
        generated = out[0, inputs["input_ids"].shape[1] :].tolist()
        tok = self.processor.tokenizer
        response = tok.decode(generated, skip_special_tokens=True)
        answer = _parse_answer(response)

        step = self._answer_step(generated)
        p_fake: Optional[float] = None
        if step is not None and step < len(recorder.steps):
            logits = recorder.steps[step]
            per_word = {
                w: torch.logsumexp(
                    logits[[k for k, i in enumerate(id_list) if answer_ids[i] == w]], dim=0
                )
                for w in _ANSWER_WORDS
            }
            p_fake = float(torch.softmax(torch.stack([per_word["real"], per_word["fake"]]), 0)[1])
        if p_fake is None:
            p_fake = {"fake": 1.0, "real": 0.0}.get(answer, 0.5)
            if answer is None:
                logger.warning("VideoVeritas: no single real/fake verdict in <answer>.")

        return self._make_result(
            p_fake,
            answer=answer,
            response=response,
            num_frames=int(frames.shape[0]),
            num_generated_tokens=len(generated),
            truncated=len(generated) >= self.max_new_tokens,
            model=self.model_name,
        )
