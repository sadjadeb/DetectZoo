"""CHEAT – Detecting CHatGPT-writtEn AbsTracts.

Reference:
    Yu et al., "CHEAT: A Large-scale Dataset for Detecting CHatGPT-writtEn
    AbsTracts", IEEE Transactions on Big Data 2025.
    https://arxiv.org/abs/2304.12008

GitHub: https://github.com/botianzhe/CHEAT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_GITHUB_RAW = "https://raw.githubusercontent.com/botianzhe/CHEAT/main/data"

_FILES: dict[str, tuple[str, int]] = {
    "ieee-init.jsonl": ("human", 0),
    "ieee-chatgpt-generation.jsonl": ("generation", 1),
    "ieee-chatgpt-polish.jsonl": ("polish", 1),
    "ieee-chatgpt-fusion.jsonl": ("fusion", 1),
}


@register_dataset("cheat")
class CHEATDataset(BaseDataset):
    """CHEAT dataset for detecting ChatGPT-written academic abstracts.

    Contains 35,304 synthetic abstracts in three categories:

    * **generation** – first-pass ChatGPT output
    * **polish** – ChatGPT-refined versions of human abstracts
    * **fusion** – human / ChatGPT hybrid abstracts

    When *path* is omitted the JSONL files are downloaded automatically
    from `GitHub <https://github.com/botianzhe/CHEAT>`_ and cached
    under ``.detectzoo_data/cheat/``.

    Parameters
    ----------
    path : str or Path, optional
        Local directory containing the JSONL files.  When *None* the
        data is downloaded from GitHub.
    categories : sequence of str, optional
        Filter to specific categories (``"generation"``, ``"polish"``,
        ``"fusion"``).  *None* loads all including the human baseline.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "cheat"
    modality = "text"

    info = (
        "CHEAT (Detecting CHatGPT-writtEn AbsTracts)\n"
        "============================================\n"
        "A large-scale dataset of 35,304 ChatGPT-written academic abstracts\n"
        "collected from IEEE papers, along with their original human-written\n"
        "counterparts.\n"
        "\n"
        "Paper  : Yu et al., 'CHEAT: A Large-scale Dataset for Detecting\n"
        "         CHatGPT-writtEn AbsTracts', IEEE Trans. Big Data, 2025.\n"
        "arXiv  : 2304.12008\n"
        "\n"
        "Categories\n"
        "----------\n"
        "  human      – original human-written IEEE abstracts (ieee-init.jsonl)\n"
        "  generation – first-pass ChatGPT output (ieee-chatgpt-generation.jsonl)\n"
        "  polish     – ChatGPT-refined human abstracts (ieee-chatgpt-polish.jsonl)\n"
        "  fusion     – human/ChatGPT hybrid abstracts (ieee-chatgpt-fusion.jsonl)\n"
        "\n"
        "Splits: no predefined train/test split; use categories for filtering.\n"
        "\n"
        "Labels: 0 = human (ieee-init), 1 = ChatGPT (generation/polish/fusion).\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Detection difficulty increases with human involvement: pure\n"
        "generation is easiest, fusion is hardest.  Filter by category\n"
        "to measure difficulty gradients:\n"
        "  CHEATDataset(categories=['generation'])  # easiest\n"
        "  CHEATDataset(categories=['fusion'])       # hardest\n"
    )

    CATEGORIES = ("generation", "polish", "fusion")

    def __init__(
        self,
        path: str | Path | None = None,
        categories: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path) if path is not None else None
        self.categories = set(categories) if categories else None
        self.cache_dir = cache_dir

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import download_file, get_cache_dir

        data_dir = get_cache_dir("cheat", self.cache_dir)
        for filename in _FILES:
            download_file(f"{_GITHUB_RAW}/{filename}", data_dir / filename)
        return data_dir

    def _load_jsonl(self, fp: Path, category: str, label: int) -> List[DatasetItem]:
        if self.categories and category not in self.categories and label != 0:
            return []
        items: List[DatasetItem] = []
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                row: dict[str, Any] = json.loads(line)
                items.append(
                    DatasetItem(
                        data=row["abstract"],
                        label=label,
                        metadata={
                            "category": category,
                            "title": row.get("title", ""),
                        },
                    )
                )
        return items

    def _load_all(self) -> List[DatasetItem]:
        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        items: List[DatasetItem] = []
        for filename, (category, label) in _FILES.items():
            fp = data_dir / filename
            if fp.exists():
                items.extend(self._load_jsonl(fp, category, label))
        return items
