"""RAID – Robust AI-generated Text Detection Benchmark.

Reference:
    Dugan et al., "RAID: A Shared Benchmark for Robust Evaluation of
    Machine-Generated Text Detectors", ACL 2024.
    https://aclanthology.org/2024.acl-long.674.pdf

Labeled splits sourced from ``Shengkun/Raid_split`` (re-split by the
authors of "Human Texts Are Outliers: Detecting LLM-generated Texts
via Out-of-distribution Detection").

Original data: ``liamdugan/raid``
GitHub       : https://github.com/liamdugan/raid
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem


@register_dataset("raid")
class RAIDDataset(BaseDataset):
    """RAID dataset for robust machine-generated text detection.

    The largest & most comprehensive detection benchmark: over 10 million
    documents spanning 11 LLMs, 11 genres, 4 decoding strategies, and 12
    adversarial attacks.  Each row contains one *generation* (either a
    human reference or an LLM output under a specific configuration).

    By default data is streamed from the **labeled** re-split published
    at ``Shengkun/Raid_split`` on HuggingFace, which provides fully
    labeled train/test/test_new/test_attack splits with the complete
    column set (id, model, domain, attack, decoding, repetition_penalty,
    generation, …).  To use the original ``liamdugan/raid`` leaderboard
    files instead, pass ``hf_repo="liamdugan/raid"`` and ``split="train"``
    or ``split="extra"``.

    For offline use, download the Parquet/CSV files and point *path* to
    the containing directory.

    Parameters
    ----------
    path : str or Path, optional
        Local directory or file.  When *None* the dataset is loaded from
        HuggingFace.
    split : str
        Which split to load.  Default ``"train"``.

        With the default ``Shengkun/Raid_split`` repo the available
        splits are:

        * ``"train"`` – 337 k fully-labeled rows
        * ``"test"`` – 112 k fully-labeled rows
        * ``"test_new"`` – 28.7 k rows
        * ``"test_attack"`` – 103 k rows

        When using ``hf_repo="liamdugan/raid"`` the available splits are
        ``"train"``, ``"test"`` (unlabeled leaderboard split), and
        ``"extra"``.
    hf_repo : str
        HuggingFace dataset identifier.  Default
        ``"Shengkun/Raid_split"`` (fully-labeled re-split).  Set to
        ``"liamdugan/raid"`` for the original leaderboard files.
    models : sequence of str, optional
        Filter to specific generators (e.g. ``["chatgpt", "gpt4"]``).
        Use ``"human"`` to keep the human baseline.  *None* loads all.
    domains : sequence of str, optional
        Filter to specific genres (e.g. ``["news", "books"]``).
    attacks : sequence of str, optional
        Filter to specific adversarial attacks.  Pass ``["none"]`` to
        get the non-adversarial subset (recommended for initial
        experiments).  *None* loads all.
    decoding : sequence of str, optional
        Filter to specific decoding strategies (``"greedy"`` or
        ``"sampling"``).
    repetition_penalty : sequence of str, optional
        Filter to specific repetition-penalty flags (``"yes"`` / ``"no"``).
    include_human : bool
        When ``models`` is provided, still keep the human reference
        samples (label=0).  Default ``True``.
    max_samples : int, optional
        Cap the number of samples loaded (useful for quick experiments).
    """

    name = "raid"
    modality = "text"

    info = (
        "RAID (Robust AI-generated text Detection benchmark)\n"
        "===================================================\n"
        "The largest shared benchmark for machine-generated text\n"
        "detectors: >10M documents spanning 11 LLMs, 11 genres,\n"
        "4 decoding strategies, and 12 adversarial attacks.\n"
        "\n"
        "Paper  : Dugan et al., 'RAID: A Shared Benchmark for Robust\n"
        "         Evaluation of Machine-Generated Text Detectors',\n"
        "         ACL 2024.\n"
        "\n"
        "Models\n"
        "------\n"
        "  chatgpt, gpt4, gpt3, gpt2, llama-chat, mistral, mistral-chat,\n"
        "  mpt, mpt-chat, cohere, cohere-chat, human\n"
        "\n"
        "Domains\n"
        "-------\n"
        "  abstracts, books, code, czech, german, news, poetry, recipes,\n"
        "  reddit, reviews, wiki\n"
        "\n"
        "Attacks\n"
        "-------\n"
        "  none, homoglyph, number, article_deletion, insert_paragraphs,\n"
        "  perplexity_misspelling, upper_lower, whitespace,\n"
        "  zero_width_space, synonym, paraphrase, alternative_spelling\n"
        "\n"
        "Splits  (default: Shengkun/Raid_split, fully labeled)\n"
        "------\n"
        "  train       – 337 k rows\n"
        "  test        – 112 k rows\n"
        "  test_new    – 28.7 k rows\n"
        "  test_attack – 103 k rows\n"
        "  extra       – labeled, 3 extra domains (code, czech, german)\n"
        "         (available only when using `liamdugan/raid` repo)\n"
        "\n"
        "Labels: 0 = human, 1 = AI (any model).\n"
        "        The full model name is preserved in metadata['model'].\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Non-adversarial training data for a single model:\n"
        "  RAIDDataset(split='train', models=['chatgpt'],\n"
        "              attacks=['none'], decoding=['greedy'])\n"
        "Evaluate on the labeled test split:\n"
        "  RAIDDataset(split='test', max_samples=5000)\n"
        "Adversarial robustness evaluation:\n"
        "  RAIDDataset(split='test_attack')\n"
    )

    MODELS = (
        "chatgpt",
        "gpt4",
        "gpt3",
        "gpt2",
        "llama-chat",
        "mistral",
        "mistral-chat",
        "mpt",
        "mpt-chat",
        "cohere",
        "cohere-chat",
        "human",
    )
    DOMAINS = (
        "abstracts",
        "books",
        "code",
        "czech",
        "german",
        "news",
        "poetry",
        "recipes",
        "reddit",
        "reviews",
        "wiki",
    )
    ATTACKS = (
        "none",
        "homoglyph",
        "number",
        "article_deletion",
        "insert_paragraphs",
        "perplexity_misspelling",
        "upper_lower",
        "whitespace",
        "zero_width_space",
        "synonym",
        "paraphrase",
        "alternative_spelling",
    )
    SPLITS = ("train", "test", "test_new", "test_attack")

    _DEFAULT_HF_REPO = "Shengkun/Raid_split"

    def __init__(
        self,
        path: str | Path | None = None,
        split: str = "train",
        hf_repo: str = _DEFAULT_HF_REPO,
        models: Sequence[str] | None = None,
        domains: Sequence[str] | None = None,
        attacks: Sequence[str] | None = None,
        decoding: Sequence[str] | None = None,
        repetition_penalty: Sequence[str] | None = None,
        include_human: bool = True,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)
        self.path = Path(path) if path is not None else None
        self.split = split
        self.hf_repo = hf_repo
        self.models = {m.lower() for m in models} if models else None
        self.domains = {d.lower() for d in domains} if domains else None
        self.attacks = {a.lower() for a in attacks} if attacks else None
        self.decoding = {d.lower() for d in decoding} if decoding else None
        self.repetition_penalty = (
            {r.lower() for r in repetition_penalty} if repetition_penalty else None
        )
        self.include_human = include_human

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _keep(self, row: dict[str, Any]) -> bool:
        model = str(row.get("model") or "").lower()
        domain = str(row.get("domain") or "").lower()
        attack = str(row.get("attack") or "").lower()
        dec = str(row.get("decoding") or "").lower()
        rep = str(row.get("repetition_penalty") or "").lower()

        if self.models is not None:
            if model not in self.models:
                if not (self.include_human and model == "human"):
                    return False
        if self.domains is not None and domain not in self.domains:
            return False
        if self.attacks is not None and attack not in self.attacks:
            return False
        if self.decoding is not None and dec not in self.decoding:
            return False
        if self.repetition_penalty is not None and rep not in self.repetition_penalty:
            return False
        return True

    @staticmethod
    def _row_to_item(row: dict[str, Any]) -> DatasetItem:
        model = str(row.get("model", "")).lower()
        label = 0 if model == "human" else 1
        return DatasetItem(
            data=row.get("generation", ""),
            label=label,
            metadata={
                "id": row.get("id", ""),
                "model": model,
                "domain": row.get("domain", ""),
                "attack": row.get("attack", ""),
                "decoding": row.get("decoding", ""),
                "repetition_penalty": row.get("repetition_penalty", ""),
                "source_id": row.get("source_id", ""),
                "adv_source_id": row.get("adv_source_id", ""),
                "title": row.get("title", ""),
                "prompt": row.get("prompt", ""),
            },
        )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _iter_rows_huggingface(self) -> Iterable[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, split=self.split, streaming=True)
        for row in ds:
            yield row

    def _iter_rows_local(self) -> Iterable[dict[str, Any]]:
        assert self.path is not None
        if self.path.is_file():
            files = [self.path]
        elif self.path.is_dir():
            target = self.path / self._SPLIT_TO_FILE[self.split]
            files = [target] if target.exists() else sorted(self.path.glob("*.csv"))
        else:
            raise FileNotFoundError(f"RAID path does not exist: {self.path}")

        for fp in files:
            with open(fp, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    yield row

    def _load_all(self) -> List[DatasetItem]:
        iterator = (
            self._iter_rows_local() if self.path is not None else self._iter_rows_huggingface()
        )
        items: List[DatasetItem] = []
        for row in iterator:
            if not self._keep(row):
                continue
            items.append(self._row_to_item(row))
            if self.max_samples is not None and len(items) >= self.max_samples:
                break
        return items
