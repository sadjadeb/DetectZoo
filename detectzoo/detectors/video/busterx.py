"""BusterX++ — MLLM-based AI-generated video detection and explanation.

Reference:
    Wen et al., "BusterX: MLLM-Powered AI-Generated Video Forgery Detection
    and Explanation", arXiv 2025.  https://arxiv.org/abs/2505.12620
    Wen et al., "BusterX++: Towards Unified Cross-Modal AI-Generated Content
    Detection and Explanation with MLLM", arXiv 2025.

A multimodal LLM (Qwen3.5-4B, RL-tuned) looks at frames sampled from the video,
reasons about forgery cues inside ``<think>…</think>`` and answers a
multiple-choice question with ``\\boxed{A}`` (real) or ``\\boxed{B}`` (fake).

This wrapper follows the upstream GenBuster-Bench evaluation
(``scripts/eval_genbuster_bench.sh`` with ``make build_genbuster_bench``):
frames at 2 fps over the whole clip, a JPEG (quality 95) round-trip, each frame
resized to 64 visual tokens (256×256 for square input) with ``qwen_vl_utils``'
``smart_resize``, timestamped ``<t seconds><image>`` prompt, thinking mode,
greedy decoding with up to 4096 new tokens.

``score`` is P(B) / (P(A) + P(B)) from the logits at the step where the model
writes the letter inside ``\\boxed{}``, so ``score >= 0.5`` matches the
model's own answer.  The full response is kept in ``metadata["response"]``.

Upstream: https://github.com/l8cv/BusterX
Weights: ``l8cv/BusterX-plusplus`` (gated, auto-approved: request access on
the model page and ``hf auth login`` first).
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.video.base import BaseVideoDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_ID = "l8cv/BusterX-plusplus"
# Pinned: the 2026/06 revised edition (Qwen3.5-4B base).
_MODEL_REVISION = "4ec963385d7529015cb079afcf7ab2026cf60327"

# Upstream prompts/default_user_prompt.txt, verbatim (the file contains a
# literal double backslash before "boxed").  ``<video>`` is replaced by the
# timestamped frame placeholders.
_PROMPT = (
    "<video>\n"
    "Please analyze whether there are any inconsistencies or obvious signs of forgery in the "
    "video, and finally come to a conclusion: Is this video real or fake?\n"
    "\n"
    "Then, just answer this MCQ with a single letter:\n"
    "Q: Is this video real or fake?\n"
    "Options:\n"
    "A) real\n"
    "B) fake\n"
    "\n"
    "Put the final answer in \\\\boxed{...}\n"
)
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


# ---------------------------------------------------------------------------
# Frame sampling (mirrors busterx/vl_utils/vision_process.py)
# ---------------------------------------------------------------------------


def _frame_indices(total_frames: int, video_fps: float, sample_fps: float) -> List[int]:
    """Indices of frames at ``sample_fps`` over the whole clip (upstream rule)."""
    if total_frames <= 0:
        return []
    duration = total_frames / video_fps
    ts = np.arange(0.0, duration, 1.0 / sample_fps)
    if ts.size == 0:
        ts = np.array([0.0])
    idx = np.clip(np.round(ts * video_fps).astype(np.int64), 0, total_frames - 1)
    return idx.tolist()


def _read_frames(path: Path, sample_fps: float) -> Tuple[List[np.ndarray], List[float]]:
    """Decode the sampled RGB frames of *path* and their timestamps (seconds)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise ValueError(f"Could not read the frame rate of {path}")
        wanted = _frame_indices(total, fps, sample_fps)
        by_index: dict[int, np.ndarray] = {}
        targets = set(wanted)
        i = 0
        while targets and i <= max(targets):
            ok, frame = cap.read()
            if not ok:
                break
            if i in targets:
                by_index[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()

    if len(by_index) < len(set(wanted)):
        # The container over-reported its frame count; resample on what decoded.
        if i == 0:
            raise ValueError(f"No frames could be decoded from: {path}")
        wanted = _frame_indices(i, fps, sample_fps)
        if not set(wanted) <= by_index.keys():
            return _read_frames_exact(path, sample_fps, i, fps)
    return [by_index[j] for j in wanted], [round(j / fps, 1) for j in wanted]


def _read_frames_exact(
    path: Path, sample_fps: float, total: int, fps: float
) -> Tuple[List[np.ndarray], List[float]]:
    import cv2

    wanted = _frame_indices(total, fps, sample_fps)
    targets = set(wanted)
    by_index: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    try:
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if i in targets:
                by_index[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return [by_index[j] for j in wanted], [round(j / fps, 1) for j in wanted]


def _smart_resize(height: int, width: int, factor: int, pixels: int) -> Tuple[int, int]:
    """``qwen_vl_utils.smart_resize`` with ``min_pixels == max_pixels == pixels``."""

    def round_by(x: float) -> int:
        return round(x / factor) * factor

    h_bar, w_bar = max(factor, round_by(height)), max(factor, round_by(width))
    if h_bar * w_bar > pixels:
        beta = math.sqrt((height * width) / pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < pixels:
        beta = math.sqrt(pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _parse_answer(response: str) -> Optional[str]:
    """Return ``"A"``/``"B"`` from the last ``\\boxed{…}`` outside ``<think>``."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    if "</think>" in text:  # generation starts inside an (unclosed) <think>
        text = text.rsplit("</think>", 1)[-1]
    boxes = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if not boxes:
        return None
    inner = re.sub(r"\\text\{([^{}]*)\}", r"\1", boxes[-1]).strip().upper()
    match = re.match(r"\(?([AB])\b", inner)
    if match:
        return match.group(1)
    if inner.startswith("REAL"):
        return "A"
    if inner.startswith("FAKE"):
        return "B"
    return None


class _ABLogitRecorder:
    """Logits processor that keeps only the ``A``/``B`` logits of every step."""

    def __init__(self, a_id: int, b_id: int) -> None:
        self.ids = [a_id, b_id]
        self.steps: List[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.steps.append(scores[:, self.ids].float().cpu())
        return scores


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@register_detector("busterx", aliases=["busterx_plusplus", "busterx++"])
class BusterXDetector(BaseVideoDetector):
    """BusterX++ (Qwen3.5-4B) video forgery detector with explanations.

    Parameters
    ----------
    model_name : str
        HF model id (default ``l8cv/BusterX-plusplus``, gated).
    revision : str, optional
        Model revision; defaults to the pinned 2026/06 release when
        *model_name* is the default.
    threshold : float
        Decision boundary on P(fake) (default 0.5, i.e. the model's answer).
    device : str
        Torch device string.
    sample_fps : float
        Frames per second sampled over the whole clip (default 2.0).
    image_tokens : int
        Visual tokens per frame (default 64 → 256×256 pixels).
    max_new_tokens : int
        Generation budget for reasoning + answer (default 4096).
    jpeg_quality : int
        Quality of the JPEG round-trip applied to frames (default 95), as
        upstream caches frames as JPEG before inference.
    torch_dtype : str
        ``"bfloat16"`` (default) or ``"float16"`` / ``"float32"``.
    cache_dir : str or Path, optional
        Hugging Face cache directory for the weights (~10 GB).

    Non-path inputs (arrays, tensors, frame lists) are taken as frames already
    sampled at ``sample_fps``.
    """

    def __init__(
        self,
        *,
        model_name: str = _MODEL_ID,
        revision: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        sample_fps: float = 2.0,
        image_tokens: int = 64,
        max_new_tokens: int = 4096,
        jpeg_quality: int = 95,
        torch_dtype: str = "bfloat16",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device, num_frames=None, **kwargs)
        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        self.model_name = model_name
        self.revision = revision or (_MODEL_REVISION if model_name == _MODEL_ID else None)
        self.sample_fps = float(sample_fps)
        self.image_tokens = int(image_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.jpeg_quality = int(jpeg_quality)
        self.torch_dtype = torch_dtype
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModelForImageTextToText

        logger.info("Loading %s@%s …", self.model_name, (self.revision or "main")[:8])
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            revision=self.revision,
            cache_dir=self.cache_dir,
            dtype=getattr(torch, self.torch_dtype),
        )
        self._model.to(self._device).eval()

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_name, revision=self.revision, cache_dir=self.cache_dir
            )
        return self._processor

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _load_frames(self, input_data: Any) -> Tuple[List[np.ndarray], List[float]]:
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            return _read_frames(path, self.sample_fps)
        frames = self._normalise_input(input_data).permute(0, 2, 3, 1).numpy()
        return list(frames), [round(i / self.sample_fps, 1) for i in range(len(frames))]

    def _prepare_image(self, frame: np.ndarray):
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=self.jpeg_quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        factor = (
            self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        )
        h, w = _smart_resize(img.height, img.width, factor, self.image_tokens * factor**2)
        # Same call as qwen_vl_utils.fetch_image (PIL default resampling).
        return img.resize((w, h))

    def _build_inputs(self, frames: List[np.ndarray], timestamps: List[float]):
        images = [self._prepare_image(f) for f in frames]
        placeholders = "".join(f"<{ts:.1f} seconds>{_IMAGE_PLACEHOLDER}" for ts in timestamps)
        content = _PROMPT.replace("<video>", placeholders).strip()
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=[text], images=images, do_resize=False, return_tensors="pt")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _answer_step(self, generated: List[int]) -> Optional[int]:
        """Index of the step that emitted the letter in the final ``\\boxed{``."""
        tok = self.processor.tokenizer
        a_id, b_id = tok.convert_tokens_to_ids(["A", "B"])
        start = 0
        think_end = tok.convert_tokens_to_ids("</think>")
        if think_end in generated:
            start = len(generated) - generated[::-1].index(think_end)
        step = None
        for i in range(start, len(generated)):
            if generated[i] not in (a_id, b_id):
                continue
            prefix = tok.decode(generated[max(start, i - 4) : i])
            if prefix.rstrip().endswith("boxed{") or re.search(r"boxed\{\s*\($", prefix):
                step = i
        return step

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        from transformers import LogitsProcessorList

        frames, timestamps = self._load_frames(input_data)
        inputs = self._build_inputs(frames, timestamps).to(self._device)
        tok = self.processor.tokenizer
        recorder = _ABLogitRecorder(*tok.convert_tokens_to_ids(["A", "B"]))
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([recorder]),
        )
        generated = out[0, inputs["input_ids"].shape[1] :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        while generated and generated[-1] in eos:
            generated.pop()
        # Keep </think> in the text (it may be a special token).
        response = "<think>\n" + tok.decode(generated, skip_special_tokens=False)
        answer = _parse_answer(response)

        step = self._answer_step(generated)
        if step is not None and step < len(recorder.steps):
            p_fake = float(torch.softmax(recorder.steps[step][0], dim=-1)[1])
        elif answer is not None:
            p_fake = 1.0 if answer == "B" else 0.0
        else:
            p_fake = 0.5
            logger.warning("BusterX: no \\boxed{A|B} answer found in the response.")

        return self._make_result(
            p_fake,
            answer=answer,
            response=response,
            num_frames=len(frames),
            num_generated_tokens=len(generated),
            truncated=len(generated) >= self.max_new_tokens,
            model=f"{self.model_name}@{(self.revision or 'main')[:8]}",
        )
