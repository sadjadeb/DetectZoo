"""TuringBench – Benchmark environment for the Turing Test in the age of
neural text generation.

Reference:
    Uchendu et al., "TURINGBENCH: A Benchmark Environment for Turing Test
    in the Age of Neural Text Generation", EMNLP 2021 Findings.
    https://aclanthology.org/2021.findings-emnlp.172.pdf

HuggingFace: ``turingbench/TuringBench``
GitHub     : https://github.com/TuringBench/TuringBench
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_HF_ZIP_URL = "https://huggingface.co/datasets/turingbench/TuringBench/resolve/main/TuringBench.zip"

# Binary Turing-Test (human vs. one generator) configurations.
_TT_MODELS: tuple[str, ...] = (
    "gpt1",
    "gpt2_small",
    "gpt2_medium",
    "gpt2_large",
    "gpt2_xl",
    "gpt2_pytorch",
    "gpt3",
    "grover_base",
    "grover_large",
    "grover_mega",
    "ctrl",
    "xlm",
    "xlnet_base",
    "xlnet_large",
    "fair_wmt19",
    "fair_wmt20",
    "transfo_xl",
    "pplm_distil",
    "pplm_gpt2",
)

_CONFIGS: tuple[str, ...] = ("AA",) + tuple(f"TT_{m}" for m in _TT_MODELS)

_SPLIT_FILES: dict[str, str] = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test": "test.csv",
}


@register_dataset("turingbench")
class TuringBenchDataset(BaseDataset):
    """TuringBench benchmark for machine-generated text detection.

    TuringBench provides two task settings on the same underlying corpus
    (human news articles + neural generations from 19 models):

    * **AA – Authorship Attribution** (20-way classification:
      ``human`` + 19 generator names).
    * **TT_<model> – Turing Test** (binary classification: ``human`` vs.
      a specific generator).

    Each row is ``(Generation, label)``.  In the official release the
    ``test`` split's labels are hidden (empty strings) — use ``train`` or
    ``valid`` when running offline evaluation.

    When *path* is omitted the zipped corpus is downloaded automatically
    from HuggingFace and cached under ``.detectzoo_data/turingbench/``.

    Parameters
    ----------
    path : str or Path, optional
        Local directory containing the ``TuringBench/`` extracted tree.
        When *None* the archive is downloaded from HuggingFace.
    config : str
        Which benchmark configuration to load.  Default ``"TT_gpt3"``.
        Pass ``"AA"`` for the authorship attribution task, or
        ``"TT_<model>"`` for a binary setting (see
        :data:`TuringBenchDataset.CONFIGS`).
    split : str
        One of ``"train"`` (default), ``"valid"``, ``"test"``.  The
        ``test`` split has no labels in the public release — items are
        still returned with ``label=-1`` as a placeholder and the
        original empty label in ``metadata['raw_label']``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "turingbench"
    modality = "text"

    info = (
        "TuringBench (Benchmark for Turing Test in Neural Text Generation)\n"
        "=================================================================\n"
        "Large-scale corpus and benchmark released with EMNLP 2021 Findings.\n"
        "Pairs human news articles with generations from 19 neural models\n"
        "to support both authorship attribution and binary detection.\n"
        "\n"
        "Paper  : Uchendu et al., 'TURINGBENCH: A Benchmark Environment\n"
        "         for Turing Test in the Age of Neural Text Generation',\n"
        "         EMNLP 2021 Findings.\n"
        "\n"
        "Configurations\n"
        "--------------\n"
        "  AA          – Authorship Attribution (20-way)\n"
        "  TT_<model>  – Turing Test (human vs. one generator, binary)\n"
        "Generators: gpt1, gpt2_{small,medium,large,xl,pytorch}, gpt3,\n"
        "  grover_{base,large,mega}, ctrl, xlm, xlnet_{base,large},\n"
        "  fair_wmt19, fair_wmt20, transfo_xl, pplm_distil, pplm_gpt2.\n"
        "\n"
        "Splits\n"
        "------\n"
        "  train, valid, test — same split across every configuration.\n"
        "  The `test` split has labels hidden in the public release.\n"
        "\n"
        "Labels: DetectZoo normalises to 0 = human, 1 = AI.\n"
        "  For AA the original label (model name) is preserved in\n"
        "  metadata['raw_label'].  For the hidden `test` split items\n"
        "  carry label=-1 as a placeholder.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Binary detection (e.g. GPT-3):\n"
        "  TuringBenchDataset(config='TT_gpt3', split='valid')\n"
        "Attribution across all 19 generators:\n"
        "  TuringBenchDataset(config='AA', split='valid')\n"
    )

    CONFIGS = _CONFIGS
    SPLITS = tuple(_SPLIT_FILES.keys())

    def __init__(
        self,
        path: str | Path | None = None,
        config: str = "TT_gpt3",
        split: str = "train",
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config not in _CONFIGS:
            raise ValueError(f"Unknown TuringBench config '{config}'. Valid: {list(_CONFIGS)}")
        if split not in _SPLIT_FILES:
            raise ValueError(f"Unknown TuringBench split '{split}'. Valid: {list(_SPLIT_FILES)}")
        self.path = Path(path) if path is not None else None
        self.config = config
        self.split = split
        self.cache_dir = cache_dir

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import (
            download_and_extract_zip,
            get_cache_dir,
        )

        data_dir = get_cache_dir("turingbench", self.cache_dir)
        download_and_extract_zip(_HF_ZIP_URL, data_dir)
        return data_dir

    def _resolve_csv(self, data_dir: Path) -> Path:
        filename = _SPLIT_FILES[self.split]
        # Archive extracts to TuringBench/<config>/<split>.csv
        candidates = [
            data_dir / "TuringBench" / self.config / filename,
            data_dir / self.config / filename,
            data_dir / filename,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Could not find {self.config}/{filename} under {data_dir}.")

    @staticmethod
    def _map_label(raw_label: str) -> int:
        """Normalise raw string labels to 0 (human) / 1 (AI) / -1 (hidden)."""
        if raw_label is None:
            return -1
        label = raw_label.strip().lower()
        if not label:
            return -1
        if label in {"human", "0"}:
            return 0
        if label in {"machine", "ai", "1"}:
            return 1
        # AA labels are model names; anything not "human" is machine-generated.
        return 1

    def _load_all(self) -> List[DatasetItem]:
        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        csv_path = self._resolve_csv(data_dir)

        items: List[DatasetItem] = []
        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = row.get("Generation") or row.get("generation") or ""
                raw_label = row.get("label", "")
                items.append(
                    DatasetItem(
                        data=text,
                        label=self._map_label(raw_label),
                        metadata={
                            "config": self.config,
                            "split": self.split,
                            "raw_label": raw_label,
                        },
                    )
                )
        return items
