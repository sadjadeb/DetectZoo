"""GECScore — Grammar Error Correction based zero-shot detector.

Reference:
    Wu et al., "Who Wrote This? The Key to Zero-Shot LLM-Generated
    Text Detection Is GECScore", COLING 2025.

The key insight: LLM-generated text contains fewer grammatical errors
than human-written text.  After correcting grammar with a GEC model,
the similarity between the original and corrected text is measured.
High similarity → few corrections → likely AI-generated.

Two GEC backends have been covered following what the authors have done:

* ``backend="openai"`` — GPT-4o-mini, the setting behind the headline
  average AUROC of 98.62%, at temperature 0.01.  Requires
  ``OPENAI_API_KEY``.
* ``backend="coedit"`` (default) — ``grammarly/coedit-large``
  (COEDIT-L).  Table 1 reports about 94.78% AUROC on XSum and 98.05%
  on Writing Prompts for this model.  Flan-T5-Large only accepts 512
  tokens, and the released evaluation texts are often longer than that,
  so each document is corrected in sentence-sized pieces that fit the
  encoder and the pieces are joined before scoring.  Scoring a
  512-token prefix against the full document depresses ROUGE-2 for
  both classes and does not match the paper.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


# Paper §4.1 and official repository.
_GEC_PROMPT = "Correct the grammar errors in the following text: {text}\nCorrected text:"


def _format_gec_prompt(text: str) -> str:
    return _GEC_PROMPT.format(text=text)

# Tokens that look like a sentence boundary but are abbreviations.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "eg",
    "ie",
    "us",
    "uk",
    "no",
    "fig",
    "al",
    "gen",
    "col",
    "sgt",
    "ltd",
    "inc",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

_SENT_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, keeping the closing punctuation.

    Abbreviations such as ``Ms.`` are not treated as boundaries, so a
    news sentence is not fed to CoEdit as a fragment.  The paper notes
    that fragmented sentences confuse the GEC model.
    """
    text = text.strip()
    if not text:
        return []

    parts: list[str] = []
    start = 0
    for match in _SENT_BOUNDARY.finditer(text):
        chunk = text[start : match.start()]
        words = chunk.split()
        if not words:
            continue
        token = re.sub(r"[^A-Za-z.]", "", words[-1]).lower().strip(".")
        # ``U.S.`` → ``us`` after periods are removed; ``Ms`` stays ``ms``.
        compact = token.replace(".", "")
        if token in _ABBREVIATIONS or compact in _ABBREVIATIONS:
            continue
        sentence = text[start : match.end()].strip()
        if sentence:
            parts.append(sentence)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _pieces_for_budget(
    text: str,
    token_length: Callable[[str], int],
    budget: int,
) -> list[str]:
    """Split *text* into pieces whose token length is at most *budget*.

    Sentences are kept intact when they fit.  A sentence longer than
    the encoder budget is split on word boundaries so the tail of the
    document is still corrected.
    """
    if budget < 1:
        raise ValueError(f"token budget must be positive, got {budget}")

    pieces: list[str] = []
    for sentence in _split_sentences(text) or [text.strip()]:
        if not sentence:
            continue
        if token_length(sentence) <= budget:
            pieces.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            trial = " ".join([*current, word])
            if current and token_length(trial) > budget:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
    return [p for p in pieces if p.strip()]


@register_detector("gecscore")
class GECScoreDetector(BaseTextDetector):
    """GECScore detector — grammar correction similarity.

    Parameters:
        gec_model: HuggingFace seq2seq model used when ``backend`` is
            ``"coedit"`` (default ``"grammarly/coedit-large"``).
        backend: ``"coedit"`` for local COEDIT-L, or ``"openai"`` for
            the GPT-4o-mini setting that produces Table 1's main row.
        openai_model: Chat model used when ``backend`` is ``"openai"``
            (default ``"gpt-4o-mini"``).
        threshold: Decision boundary on the ROUGE-2 F-score.  The
            default is the authors' GPT-4o-mini cutoff.
        num_beams: Beam width for CoEdit generation.  The model card
            decodes greedily (``1``).
        batch_size: Sentences per CoEdit generation batch.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        gec_model: str = "grammarly/coedit-large",
        # Released detector default (GPT-4o-mini ROUGE-2 operating point).
        threshold: float = 0.9243697428995128,
        device: str = "cpu",
        max_length: int = 512,
        backend: str = "coedit",
        openai_model: str = "gpt-4o-mini",
        num_beams: int = 1,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=gec_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        backend = backend.lower().strip()
        if backend not in {"coedit", "openai"}:
            raise ValueError(f"backend must be 'coedit' or 'openai', got {backend!r}")
        self.backend = backend
        self.gec_model_name = gec_model
        self.openai_model = openai_model
        self.num_beams = num_beams
        self.batch_size = batch_size
        self._gec_model: torch.nn.Module | None = None
        self._gec_tokenizer: Any = None
        self._text_budget: int | None = None
        self._openai_client: Any = None
        self._rouge: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading GEC model '%s' …", self.gec_model_name)
        self._gec_tokenizer = AutoTokenizer.from_pretrained(self.gec_model_name)
        self._gec_model = AutoModelForSeq2SeqLM.from_pretrained(self.gec_model_name).to(
            self._device
        )
        self._gec_model.eval()
        # Pieces are measured with the full paper prompt, so the budget is
        # the encoder limit itself.
        self._text_budget = self.max_length

    @property
    def gec_model(self) -> torch.nn.Module:
        if self._gec_model is None:
            self._load_model()
        return self._gec_model  # type: ignore[return-value]

    @property
    def gec_tokenizer(self):
        if self._gec_tokenizer is None:
            self._load_model()
        return self._gec_tokenizer

    def _token_length(self, text: str) -> int:
        """Token length of *text* wrapped in the paper GEC prompt."""
        return len(
            self.gec_tokenizer(_format_gec_prompt(text), add_special_tokens=True)["input_ids"]
        )

    @torch.no_grad()
    def _correct_pieces(self, pieces: list[str]) -> list[str]:
        """Grammar-correct each piece with CoEdit.  Pieces already fit."""
        corrected: list[str] = []
        tokenizer = self.gec_tokenizer
        model = self.gec_model
        step = max(1, self.batch_size)
        for start in range(0, len(pieces), step):
            batch = pieces[start : start + step]
            prompts = [_format_gec_prompt(piece) for piece in batch]
            text_lengths = [self._token_length(piece) for piece in batch]
            max_new = min(self.max_length, max(text_lengths) + 32)
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new,
                "num_beams": self.num_beams,
                "do_sample": False,
            }
            if self.num_beams > 1:
                gen_kwargs["length_penalty"] = 1.0
            out = model.generate(**enc, **gen_kwargs)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            corrected.extend(text.strip() for text in decoded)
        return corrected

    def _correct_with_coedit(self, text: str) -> str:
        if self._text_budget is None:
            self._load_model()
        budget = self._text_budget or 8
        pieces = _pieces_for_budget(text, self._token_length, budget)
        if not pieces:
            return ""
        return " ".join(self._correct_pieces(pieces)).strip()

    def _correct_with_openai(self, text: str) -> str:
        """Match ``chat_with_gpt4o`` in the official ``GECScore.py``."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required for backend='openai'. "
                    "Install it with: pip install openai"
                ) from exc
            self._openai_client = OpenAI()
        prompt = _format_gec_prompt(text)
        completion = self._openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01,
        )
        content = completion.choices[0].message.content
        return (content or "").strip()

    def _correct_grammar(self, text: str) -> str:
        if self.backend == "openai":
            return self._correct_with_openai(text)
        return self._correct_with_coedit(text)

    def _rouge2_f(self, hypothesis: str, reference: str) -> float:
        """ROUGE-2 F between original text (hyp) and correction (ref)."""
        if self._rouge is None:
            from rouge import Rouge

            self._rouge = Rouge()
        try:
            scores = self._rouge.get_scores(hypothesis, reference, avg=True)
            return float(scores["rouge-2"]["f"])
        except ValueError:
            return 1.0 if hypothesis.strip() == reference.strip() else 0.0

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        try:
            corrected = self._correct_grammar(text)
        except Exception as exc:
            logger.warning("GEC failed: %s", exc)
            return self._make_result(float("nan"), reason=f"gec failed: {exc}")

        if not corrected.strip():
            return self._make_result(0.0, reason="empty correction")

        score = self._rouge2_f(text, corrected)
        return self._make_result(
            score,
            corrected_text=corrected[:200],
            rouge2_f=score,
            backend=self.backend,
        )
