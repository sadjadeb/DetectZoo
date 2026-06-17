"""L2R – Learning to Rewrite Data.

Reference:
    Hao et al., "Learning to Rewrite: Generalized LLM-Generated Text
    Detection", ACL 2025.
    https://aclanthology.org/2025.acl-long.322.pdf

GitHub: https://github.com/ranhli/l2r_data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_GITHUB_RAW = "https://raw.githubusercontent.com/ranhli/l2r_data/main"

_DOMAINS: tuple[str, ...] = (
    "AcademicResearch",
    "ArtCulture",
    "Business",
    "Code",
    "EducationMaterial",
    "Entertainment",
    "Environmental",
    "Finance",
    "FoodCusine",
    "GovernmentPublic",
    "LegalDocument",
    "LiteratureCreativeWriting",
    "MedicalText",
    "NewsArticle",
    "OnlineContent",
    "PersonalCommunication",
    "ProductReview",
    "Religious",
    "Sports",
    "TechnicalWriting",
    "TravelTourism",
)

_MODELS: tuple[str, ...] = (
    "GPT-3-Turbo",
    "GPT-4o",
    "Gemini-1.5-Pro",
    "Llama-3-70B",
)


@register_dataset("l2r")
class L2RDataset(BaseDataset):
    """L2R dataset for generalized LLM-generated text detection.

    The corpus is organised into 21 domain folders; each folder contains
    a ``human.json`` with human-written passages and one JSON file per
    LLM (``GPT-3-Turbo.json``, ``GPT-4o.json``, ``Gemini-1.5-Pro.json``,
    ``Llama-3-70B.json``).  Every JSON file is a flat array of strings.

    When *path* is omitted the requested JSON files are downloaded
    automatically from `GitHub <https://github.com/ranhli/l2r_data>`_
    and cached under ``.detectzoo_data/l2r/``.

    Parameters
    ----------
    path : str or Path, optional
        Local root directory containing per-domain subfolders.  When
        *None* the data is downloaded from GitHub.
    domains : sequence of str, optional
        Filter to specific domains (case-insensitive match against
        :data:`L2RDataset.DOMAINS`).  *None* loads all 21 domains.
    models : sequence of str, optional
        Filter to specific generators (case-insensitive match against
        :data:`L2RDataset.MODELS`).  *None* loads all four LLMs.  The
        human baseline is always loaded unless ``include_human=False``.
    include_human : bool
        Whether to load ``human.json`` files (label=0).  Default ``True``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "l2r"
    modality = "text"

    info = (
        "L2R (Learning to Rewrite Data)\n"
        "==============================\n"
        "Dataset accompanying 'Learning to Rewrite: Generalized\n"
        "LLM-Generated Text Detection'.  Texts are organised by 21\n"
        "content domains to probe detector generalisation.\n"
        "\n"
        "Paper  : Hao et al., ACL 2025.  (Preprint 2024, arXiv:2408.04237)\n"
        "\n"
        "Domains (21)\n"
        "------------\n"
        "  AcademicResearch, ArtCulture, Business, Code,\n"
        "  EducationMaterial, Entertainment, Environmental, Finance,\n"
        "  FoodCusine, GovernmentPublic, LegalDocument,\n"
        "  LiteratureCreativeWriting, MedicalText, NewsArticle,\n"
        "  OnlineContent, PersonalCommunication, ProductReview,\n"
        "  Religious, Sports, TechnicalWriting, TravelTourism\n"
        "\n"
        "Generators (4)\n"
        "--------------\n"
        "  GPT-3-Turbo, GPT-4o, Gemini-1.5-Pro, Llama-3-70B\n"
        "\n"
        "File layout\n"
        "-----------\n"
        "Each domain folder holds:\n"
        "  human.json  – list of human-written passages (label=0)\n"
        "  <MODEL>.json – list of LLM-generated passages (label=1)\n"
        "\n"
        "Labels: 0 = human, 1 = AI.  Model name and domain are\n"
        "preserved in metadata.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Pick a cross-domain or cross-model slice to measure\n"
        "generalisation:\n"
        "  L2RDataset(domains=['MedicalText'], models=['GPT-4o'])\n"
        "  L2RDataset(models=['Gemini-1.5-Pro'])  # across all domains\n"
    )

    DOMAINS = _DOMAINS
    MODELS = _MODELS

    def __init__(
        self,
        path: str | Path | None = None,
        domains: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        include_human: bool = True,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path) if path is not None else None
        self.cache_dir = cache_dir
        self.include_human = include_human
        self.domains = self._resolve(domains, _DOMAINS, "domain")
        self.models = self._resolve(models, _MODELS, "model")

    @staticmethod
    def _resolve(
        requested: Sequence[str] | None,
        valid: Sequence[str],
        kind: str,
    ) -> list[str]:
        if requested is None:
            return list(valid)
        lookup = {v.lower(): v for v in valid}
        resolved: list[str] = []
        for name in requested:
            key = name.lower()
            if key not in lookup:
                raise ValueError(f"Unknown L2R {kind} '{name}'. Valid: {list(valid)}")
            resolved.append(lookup[key])
        return resolved

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import download_file, get_cache_dir

        data_dir = get_cache_dir("l2r", self.cache_dir)
        files_to_get: list[str] = []
        for domain in self.domains:
            if self.include_human:
                files_to_get.append(f"{domain}/human.json")
            for model in self.models:
                files_to_get.append(f"{domain}/{model}.json")
        for rel_path in files_to_get:
            url = f"{_GITHUB_RAW}/{rel_path}"
            dest = data_dir / rel_path
            try:
                download_file(url, dest)
            except Exception:  # a few (domain, model) combinations may not exist
                continue
        return data_dir

    def _load_json_list(
        self,
        fp: Path,
        domain: str,
        model: str,
        label: int,
    ) -> List[DatasetItem]:
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            texts: list[Any] = list(payload.values())
        else:
            texts = list(payload)
        return [
            DatasetItem(
                data=text if isinstance(text, str) else str(text),
                label=label,
                metadata={"domain": domain, "model": model, "source": model},
            )
            for text in texts
        ]

    def _load_all(self) -> List[DatasetItem]:
        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        items: List[DatasetItem] = []
        for domain in self.domains:
            if self.include_human:
                human_fp = data_dir / domain / "human.json"
                if human_fp.exists():
                    items.extend(self._load_json_list(human_fp, domain, "human", label=0))
            for model in self.models:
                fp = data_dir / domain / f"{model}.json"
                if fp.exists():
                    items.extend(self._load_json_list(fp, domain, model, label=1))
        return items
