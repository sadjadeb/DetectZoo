"""HC3 Plus – Semantic-Invariant Human ChatGPT Comparison Corpus.

Reference:
    Su et al., "HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus",
    2023.
    https://arxiv.org/abs/2309.02731

GitHub: https://github.com/suu990901/chatgpt-comparison-detection-HC3-Plus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/suu990901/chatgpt-comparison-detection-HC3-Plus/main/data/en"
)

_FILES = (
    "train.jsonl",
    "val_hc3_QA.jsonl",
    "val_hc3_si.jsonl",
    "test_hc3_QA.jsonl",
    "test_hc3_si.jsonl",
)


@register_dataset("hc3_plus")
class HC3PlusDataset(BaseDataset):
    """HC3 Plus dataset for human vs. ChatGPT text detection (English).

    Extends the original HC3 corpus with *semantic-invariant* tasks
    (summarisation, translation, paraphrasing) that are harder to detect
    than plain question-answering.

    Each sample has ``{text, label}`` where label is ``0`` (human) or
    ``1`` (ChatGPT).

    When *path* is omitted the English JSONL files are downloaded
    automatically from `GitHub <https://github.com/suu990901/
    chatgpt-comparison-detection-HC3-Plus>`_ and cached under
    ``.detectzoo_data/hc3_plus/``.

    Parameters
    ----------
    path : str or Path, optional
        Local directory containing the JSONL files.  When *None* the
        data is downloaded from GitHub.
    splits : sequence of str, optional
        Filter to specific splits.  Valid values: ``"train"``,
        ``"val_qa"``, ``"val_si"``, ``"test_qa"``, ``"test_si"``.
        *None* loads all splits.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "hc3_plus"
    modality = "text"

    info = (
        "HC3 Plus (Semantic-Invariant Human ChatGPT Comparison Corpus)\n"
        "=============================================================\n"
        "Extends the original HC3 corpus with semantic-invariant tasks\n"
        "(summarisation, translation, paraphrasing) where detecting\n"
        "ChatGPT-generated text is harder than in plain QA.\n"
        "\n"
        "Paper  : Su et al., 'HC3 Plus: A Semantic-Invariant Human ChatGPT\n"
        "         Comparison Corpus', 2023.\n"
        "arXiv  : 2309.02731\n"
        "\n"
        "Splits\n"
        "------\n"
        "  train    – training set\n"
        "  val_qa   – QA-style validation set (val_hc3_QA.jsonl)\n"
        "  val_si   – semantic-invariant validation set (val_hc3_si.jsonl)\n"
        "  test_qa  – QA-style test set (test_hc3_QA.jsonl)\n"
        "  test_si  – semantic-invariant test set (test_hc3_si.jsonl)\n"
        "\n"
        "Labels: 0 = human, 1 = ChatGPT.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "The semantic-invariant splits (val_si, test_si) are specifically\n"
        "designed for harder evaluation.  Compare detector performance\n"
        "across QA vs. SI splits to measure robustness on paraphrased or\n"
        "summarised text.  Load specific splits:\n"
        "  HC3PlusDataset(splits=['test_si'])\n"
    )

    _SPLIT_MAP: dict[str, str] = {
        "train": "train.jsonl",
        "val_qa": "val_hc3_QA.jsonl",
        "val_si": "val_hc3_si.jsonl",
        "test_qa": "test_hc3_QA.jsonl",
        "test_si": "test_hc3_si.jsonl",
    }

    def __init__(
        self,
        path: str | Path | None = None,
        splits: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path) if path is not None else None
        self.splits = list(splits) if splits else None
        self.cache_dir = cache_dir

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import download_file, get_cache_dir

        data_dir = get_cache_dir("hc3_plus", self.cache_dir)
        for filename in _FILES:
            download_file(f"{_GITHUB_RAW}/{filename}", data_dir / filename)
        return data_dir

    def _files_to_load(self, data_dir: Path) -> list[tuple[Path, str]]:
        """Return ``(path, split_name)`` pairs to load."""
        if self.splits:
            return [(data_dir / self._SPLIT_MAP[s], s) for s in self.splits if s in self._SPLIT_MAP]
        return [(data_dir / fname, sname) for sname, fname in self._SPLIT_MAP.items()]

    def _load_jsonl(self, fp: Path, split_name: str) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                row: dict[str, Any] = json.loads(line)
                items.append(
                    DatasetItem(
                        data=row["text"],
                        label=int(row["label"]),
                        metadata={"split": split_name},
                    )
                )
        return items

    def _load_all(self) -> List[DatasetItem]:
        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        items: List[DatasetItem] = []
        for fp, split_name in self._files_to_load(data_dir):
            if fp.exists():
                items.extend(self._load_jsonl(fp, split_name))
        return items
