"""MAGE – Machine-generated Text Detection in the Wild.

Reference:
    Li et al., "MAGE: Machine-generated Text Detection in the Wild", ACL 2024.
    https://arxiv.org/abs/2305.13242

HuggingFace: ``yaful/MAGE``
GitHub: https://github.com/yafuly/MAGE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem


@register_dataset("mage")
class MAGEDataset(BaseDataset):
    """MAGE dataset for machine-generated text detection in the wild.

    A comprehensive testbed gathering texts from diverse human writings
    and multiple LLMs for in-distribution and out-of-distribution
    evaluation of text detectors.

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file path.  When *None* the dataset is loaded
        from HuggingFace (``yaful/MAGE``).
    sources : sequence of str, optional
        Filter to specific sources / generators.  *None* loads all.
    split : str
        HuggingFace split to use (default ``"train"``).
    text_column : str
        Column name for the text content (default ``"text"``).
    label_column : str
        Column name for the binary label (default ``"label"``).
    """

    name = "mage"
    modality = "text"

    info = (
        "MAGE (Machine-generated Text Detection in the Wild)\n"
        "====================================================\n"
        "A large-scale benchmark with ~437k texts from diverse human\n"
        "writings and 27 mainstream LLMs (7 families), designed for\n"
        "in-distribution and out-of-distribution detection evaluation.\n"
        "\n"
        "Paper  : Li et al., 'MAGE: Machine-generated Text Detection\n"
        "         in the Wild', ACL 2024.\n"
        "arXiv  : 2305.13242\n"
        "\n"
        "Sources\n"
        "-------\n"
        "Domains: 10 human-text datasets covering news articles, stories,\n"
        "  opinion statements, long-form answers, scientific writing, etc.\n"
        "LLMs   : 27 models from 7 families (OpenAI, LLaMA, EleutherAI,\n"
        "  GLM, Dolly, and others).\n"
        "\n"
        "Splits\n"
        "------\n"
        "  train      : 319,071 rows\n"
        "  validation : 56,792 rows\n"
        "  test       : 60,743 rows\n"
        "  Total      : 436,606 rows\n"
        "\n"
        "Columns: text, label, src.  The 'src' column encodes both\n"
        "  domain and model, e.g. 'cmv_human' (human-written CMV text)\n"
        "  or 'roct_machine_continuation_flan_t5_large' (flan-t5-large\n"
        "  generated text in the ROCStories domain).\n"
        "\n"
        "Labels: DetectZoo normalises to 0 = human, 1 = AI.\n"
        "  NOTE: raw MAGE labels are inverted (0=machine, 1=human);\n"
        "  this loader flips them automatically.\n"
        "\n"
        "Testbeds\n"
        "--------\n"
        "The paper defines 8 testbeds with increasing difficulty:\n"
        "  - Cross-domain & cross-model detection\n"
        "  - Unseen domains & unseen models (GPT-4)\n"
        "  - Paraphrased text (human & machine)\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Load the test split for evaluation:\n"
        "  MAGEDataset(split='test')\n"
        "Optionally filter by src values using the sources parameter:\n"
        "  MAGEDataset(sources=['cmv_human', 'wp_machine_...'], split='test')\n"
        "The dataset is ready for binary detection out of the box.\n"
    )

    def __init__(
        self,
        path: str | Path | None = None,
        sources: Sequence[str] | None = None,
        split: str = "train",
        text_column: str = "text",
        label_column: str = "label",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path) if path is not None else None
        self.sources = {s.lower() for s in sources} if sources else None
        self.split = split
        self.text_column = text_column
        self.label_column = label_column

    @staticmethod
    def _flip_label(raw_label: int) -> int:
        """MAGE uses 0=machine, 1=human; DetectZoo uses 0=human, 1=AI."""
        return 1 - raw_label

    def _load_from_huggingface(self) -> List[DatasetItem]:
        from datasets import load_dataset

        ds = load_dataset("yaful/MAGE", split=self.split)
        items: List[DatasetItem] = []
        for row in ds:
            source = row.get("src", "unknown")
            if self.sources and str(source).lower() not in self.sources:
                continue
            items.append(
                DatasetItem(
                    data=row[self.text_column],
                    label=self._flip_label(int(row[self.label_column])),
                    metadata={"source": source},
                )
            )
        return items

    def _load_from_local(self) -> List[DatasetItem]:
        import csv
        import json

        items: List[DatasetItem] = []
        files = sorted(self.path.iterdir()) if self.path.is_dir() else [self.path]  # type: ignore[union-attr]
        for fp in files:
            if fp.suffix == ".csv":
                with open(fp, newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        source = row.get("source", row.get("model", "unknown"))
                        if self.sources and source.lower() not in self.sources:
                            continue
                        items.append(
                            DatasetItem(
                                data=row[self.text_column],
                                label=self._flip_label(int(row[self.label_column])),
                                metadata={"source": source},
                            )
                        )
            elif fp.suffix in (".json", ".jsonl"):
                with open(fp, encoding="utf-8") as fh:
                    if fp.suffix == ".json":
                        data = json.load(fh)
                    else:
                        data = [json.loads(line) for line in fh]
                for row in data:
                    source = row.get("source", row.get("model", "unknown"))
                    if self.sources and str(source).lower() not in self.sources:
                        continue
                    items.append(
                        DatasetItem(
                            data=row.get(self.text_column, ""),
                            label=self._flip_label(int(row.get(self.label_column, 0))),
                            metadata={"source": source},
                        )
                    )
        return items

    def _load_all(self) -> List[DatasetItem]:
        if self.path is not None:
            return self._load_from_local()
        return self._load_from_huggingface()
