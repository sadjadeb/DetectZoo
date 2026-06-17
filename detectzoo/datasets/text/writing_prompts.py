"""WritingPrompts – Creative story generation dataset.

WritingPrompts is a large dataset of human-written stories paired with
writing prompts, originally collected from Reddit's r/WritingPrompts.
It is used as a source corpus in LLM-generated text detection research
where model-generated stories are compared against human-written ones.

Reference:
    Fan et al., "Hierarchical Neural Story Generation", ACL 2018.
    https://arxiv.org/abs/1805.04833

HuggingFace: ``euclaise/writingprompts``
GitHub: https://github.com/facebookresearch/fairseq/blob/main/examples/stories/README.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem


@register_dataset("writing_prompts")
class WritingPromptsDataset(BaseDataset):
    """WritingPrompts dataset for text detection benchmarking.

    Contains ~303k human-written short stories from Reddit's
    r/WritingPrompts, each paired with its writing prompt.

    Each item's ``data`` field contains the story text (``label=0``,
    human-written).  The prompt is available in
    ``metadata["prompt"]``.

    This is a *source* corpus — it only contains human text.  Use it
    together with LLM-generated stories (``label=1``) to build a
    detection benchmark.

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file.  When *None* the dataset is loaded
        from HuggingFace (``euclaise/writingprompts``).
    split : str
        HuggingFace split (default ``"test"``).  Also ``"train"``
        (273k rows) and ``"validation"`` (15.6k rows).
    max_samples : int, optional
        Cap the number of samples loaded (useful for quick experiments).
    """

    name = "writing_prompts"
    modality = "text"

    info = (
        "WritingPrompts (Hierarchical Neural Story Generation)\n"
        "=====================================================\n"
        "~303k human-written creative stories collected from Reddit's\n"
        "r/WritingPrompts, each paired with its original writing prompt.\n"
        "Stories are first-person narratives, sci-fi, fantasy, horror,\n"
        "and other genres with typical length of 100–1000 words.\n"
        "\n"
        "Paper  : Fan et al., 'Hierarchical Neural Story Generation',\n"
        "         ACL 2018.\n"
        "arXiv  : 1805.04833\n"
        "License: MIT\n"
        "\n"
        "Statistics\n"
        "----------\n"
        "  train      : 272,600 stories\n"
        "  validation : 15,620 stories\n"
        "  test       : 15,138 stories\n"
        "  Total      : ~303k stories\n"
        "\n"
        "Fields\n"
        "------\n"
        "  story  – the creative story text (used as DatasetItem.data)\n"
        "  prompt – the writing prompt (stored in metadata['prompt'])\n"
        "\n"
        "Labels: all items are label=0 (human-written).  This is a\n"
        "source corpus — pair with LLM-generated stories (label=1)\n"
        "to build a detection benchmark.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Common setup: load human stories, generate LLM rewrites using\n"
        "the same prompts (e.g. via OpenAI API), then evaluate detectors\n"
        "on the combined human + AI data.  See examples/writing_prompts_\n"
        "benchmark.py for a complete pipeline.\n"
        "  WritingPromptsDataset(split='test', max_samples=200)\n"
    )

    def __init__(
        self,
        path: str | Path | None = None,
        split: str = "test",
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)
        self.path = Path(path) if path is not None else None
        self.split = split

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        ds = load_dataset("euclaise/writingprompts", split=self.split)
        items: List[DatasetItem] = []
        for row in ds:
            items.append(
                DatasetItem(
                    data=row.get("story", ""),
                    label=0,
                    metadata={
                        "source": "human",
                        "prompt": row.get("prompt", ""),
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
                            data=row.get("story", row.get("text", "")),
                            label=int(row.get("label", 0)),
                            metadata={
                                "source": row.get("source", "human"),
                                "prompt": row.get("prompt", ""),
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
