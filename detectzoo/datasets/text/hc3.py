"""HC3 – Human ChatGPT Comparison Corpus.

Reference:
    Guo et al., "How Close is ChatGPT to Human Experts? Comparison Corpus,
    Evaluation, and Detection", 2023.
    https://arxiv.org/abs/2301.07597

HuggingFace: ``Hello-SimpleAI/HC3``
GitHub: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem


@register_dataset("hc3")
class HC3Dataset(BaseDataset):
    """HC3 dataset for human vs. ChatGPT text detection.

    The corpus contains paired human-expert and ChatGPT answers across
    multiple domains (open-domain QA, finance, medicine, law, psychology).

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file path. When *None* the dataset is streamed
        from HuggingFace (``Hello-SimpleAI/HC3``).
    subsets : sequence of str, optional
        Domain subsets to load (e.g. ``["finance", "medicine"]``).
        *None* loads all available subsets.
    split : str
        HuggingFace split to use (default ``"train"``).
    """

    name = "hc3"
    modality = "text"

    info = (
        "HC3 (Human ChatGPT Comparison Corpus)\n"
        "======================================\n"
        "The first large-scale human–ChatGPT comparison corpus, containing\n"
        "~48k paired question-answer entries where each question has both a\n"
        "human-expert answer and a ChatGPT answer.\n"
        "\n"
        "Paper  : Guo et al., 'How Close is ChatGPT to Human Experts?\n"
        "         Comparison Corpus, Evaluation, and Detection', 2023.\n"
        "arXiv  : 2301.07597\n"
        "License: CC-BY-SA 4.0 (varies by source domain)\n"
        "\n"
        "Subsets (domains)\n"
        "-----------------\n"
        "  all        – combined file with all domains\n"
        "  reddit_eli5 – open-domain QA from ELI5 (Reddit)\n"
        "  open_qa    – open-domain QA from WikiQA\n"
        "  wiki_csai  – CS & AI from Wikipedia\n"
        "  finance    – financial QA from FiQA\n"
        "  medicine   – medical QA from Medical Dialog\n"
        "\n"
        "Splits: single 'train' split on HuggingFace (~48,644 rows total).\n"
        "\n"
        "Labels: 0 = human-expert answer, 1 = ChatGPT answer.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Load with HC3Dataset(subsets=['finance']) to restrict to a domain.\n"
        "Each row produces multiple DatasetItems (one per answer). Pair\n"
        "human (label=0) and ChatGPT (label=1) items for binary detection\n"
        "evaluation.\n"
    )

    SUBSETS = ("all", "finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")

    def __init__(
        self,
        path: str | Path | None = None,
        subsets: Sequence[str] | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path) if path is not None else None
        self.subsets = list(subsets) if subsets else ["all"]
        self.split = split

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        items: List[DatasetItem] = []
        for subset in self.subsets:
            ds = load_dataset(
                "json", data_files=f"hf://datasets/Hello-SimpleAI/HC3/{subset}.jsonl"
            )[self.split]
            for row in ds:
                question = row.get("question", "")
                for answer in row.get("human_answers", []):
                    items.append(
                        DatasetItem(
                            data=answer,
                            label=0,
                            metadata={"source": "human", "question": question, "subset": subset},
                        )
                    )
                for answer in row.get("chatgpt_answers", []):
                    items.append(
                        DatasetItem(
                            data=answer,
                            label=1,
                            metadata={"source": "chatgpt", "question": question, "subset": subset},
                        )
                    )
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
                    question = row.get("question", "")
                    for answer in row.get("human_answers", []):
                        items.append(
                            DatasetItem(
                                data=answer,
                                label=0,
                                metadata={"source": "human", "question": question},
                            )
                        )
                    for answer in row.get("chatgpt_answers", []):
                        items.append(
                            DatasetItem(
                                data=answer,
                                label=1,
                                metadata={"source": "chatgpt", "question": question},
                            )
                        )
        return items

    def _load_all(self) -> List[DatasetItem]:
        if self.path is not None:
            return self._load_from_local()
        return self._load_from_huggingface()
