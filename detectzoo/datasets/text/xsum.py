"""XSum – Extreme Summarization benchmark.

XSum is widely used as a source corpus in LLM-generated text detection
research (e.g. DetectGPT, Fast-DetectGPT).  Human-written BBC article
summaries serve as the ``label=0`` (human) side; researchers pair them
with LLM-generated summaries for detection experiments.

Reference:
    Narayan et al., "Don't Give Me the Details, Just the Summary!
    Topic-Aware Convolutional Neural Networks for Extreme Summarization",
    EMNLP 2018.
    https://arxiv.org/abs/1808.08745

HuggingFace: ``EdinburghNLP/xsum``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem


@register_dataset("xsum")
class XSumDataset(BaseDataset):
    """XSum dataset for text detection benchmarking.

    Loads human-written BBC article summaries (``label=0``) from
    HuggingFace.  Each item's ``data`` field contains the one-sentence
    summary, and the full article is available in ``metadata["document"]``.

    This is a *source* corpus — it only contains human text.  Use it
    together with LLM-generated summaries (``label=1``) to build a
    detection benchmark, or use ``text_field="document"`` to work with
    the full articles instead.

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file.  When *None* the dataset is loaded
        from HuggingFace (``EdinburghNLP/xsum``).
    split : str
        HuggingFace split (default ``"test"``).
    text_field : str
        Which field to use as the primary text: ``"summary"``
        (default) or ``"document"``.
    max_samples : int, optional
        Cap the number of samples loaded (useful for quick experiments).
    """

    name = "xsum"
    modality = "text"

    info = (
        "XSum (Extreme Summarization)\n"
        "============================\n"
        "BBC article summaries widely used as a human-text source corpus\n"
        "in LLM-generated text detection research (e.g. DetectGPT,\n"
        "Fast-DetectGPT, DNA-GPT).\n"
        "\n"
        "Paper  : Narayan et al., 'Don't Give Me the Details, Just the\n"
        "         Summary! Topic-Aware CNNs for Extreme Summarization',\n"
        "         EMNLP 2018.\n"
        "arXiv  : 1808.08745\n"
        "\n"
        "Statistics\n"
        "----------\n"
        "  Total articles : ~227k\n"
        "  train          : 204,045 articles\n"
        "  validation     : 11,332 articles\n"
        "  test           : 11,334 articles\n"
        "\n"
        "Fields\n"
        "------\n"
        "  summary  – one-sentence BBC summary (default text field)\n"
        "  document – full BBC article text\n"
        "\n"
        "Labels: all items are label=0 (human-written).  This is a\n"
        "source corpus — pair with LLM-generated summaries (label=1)\n"
        "to build a detection benchmark.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Common setup: load XSum summaries, generate LLM completions\n"
        "from the same prompts, then evaluate detectors on the combined\n"
        "human + AI data.\n"
        "  XSumDataset(split='test', max_samples=500)\n"
        "Use text_field='document' for full-article experiments.\n"
    )

    def __init__(
        self,
        path: str | Path | None = None,
        split: str = "test",
        text_field: str = "summary",
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)
        self.path = Path(path) if path is not None else None
        self.split = split
        self.text_field = text_field

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        ds = load_dataset("EdinburghNLP/xsum", split=self.split)
        items: List[DatasetItem] = []
        for row in ds:
            items.append(
                DatasetItem(
                    data=row[self.text_field],
                    label=0,
                    metadata={
                        "source": "human",
                        "id": row.get("id", ""),
                        "document": row.get("document", ""),
                        "summary": row.get("summary", ""),
                    },
                )
            )
            if self.max_samples and len(items) >= self.max_samples:
                break
        return items

    def _load_from_local(self) -> List[DatasetItem]:
        import json

        assert self.path is not None
        items: List[DatasetItem] = []
        files = sorted(self.path.glob("*.jsonl")) if self.path.is_dir() else [self.path]
        for fp in files:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    row: dict[str, Any] = json.loads(line)
                    items.append(
                        DatasetItem(
                            data=row.get(self.text_field, row.get("summary", "")),
                            label=int(row.get("label", 0)),
                            metadata={
                                "source": row.get("source", "human"),
                                "id": row.get("id", ""),
                                "document": row.get("document", ""),
                                "summary": row.get("summary", ""),
                            },
                        )
                    )
                    if self.max_samples and len(items) >= self.max_samples:
                        return items
        return items

    def _load_all(self) -> List[DatasetItem]:
        if self.path is not None:
            return self._load_from_local()
        return self._load_from_huggingface()
