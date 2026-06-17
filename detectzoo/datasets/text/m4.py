"""M4 – Multi-generator, Multi-domain, Multi-lingual MGT Detection.

Reference:
    Wang et al., "M4: Multi-generator, Multi-domain, and Multi-lingual
    Black-Box Machine-Generated Text Detection", EACL 2024
    https://aclanthology.org/2024.eacl-long.83/

GitHub: https://github.com/mbzuai-nlp/M4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_GITHUB_RAW = "https://raw.githubusercontent.com/mbzuai-nlp/M4/main/data"

_FILES: tuple[tuple[str, str, str], ...] = (
    # (filename, domain, model)
    ("arabic_chatGPT.jsonl", "arabic", "chatGPT"),
    ("arxiv_bloomz.jsonl", "arxiv", "bloomz"),
    ("arxiv_chatGPT.jsonl", "arxiv", "chatGPT"),
    ("arxiv_cohere.jsonl", "arxiv", "cohere"),
    ("arxiv_davinci.jsonl", "arxiv", "davinci"),
    ("arxiv_dolly.jsonl", "arxiv", "dolly"),
    ("arxiv_flant5.jsonl", "arxiv", "flant5"),
    ("bulgarian_true_and_fake_news_chatGPT.jsonl", "bulgarian", "chatGPT"),
    ("bulgarian_true_and_fake_news_davinci.jsonl", "bulgarian", "davinci"),
    ("germanwikipedia_chatgpt.jsonl", "germanwikipedia", "chatGPT"),
    ("id-newspaper_chatGPT.jsonl", "id-newspaper", "chatGPT"),
    ("peerread_bloomz.jsonl", "peerread", "bloomz"),
    ("peerread_chatgpt.jsonl", "peerread", "chatGPT"),
    ("peerread_cohere.jsonl", "peerread", "cohere"),
    ("peerread_davinci.jsonl", "peerread", "davinci"),
    ("peerread_dolly.jsonl", "peerread", "dolly"),
    ("peerread_llama.jsonl", "peerread", "llama"),
    ("qazh_chatgpt.jsonl", "qazh", "chatGPT"),
    ("qazh_davinci.jsonl", "qazh", "davinci"),
    ("reddit_bloomz.jsonl", "reddit", "bloomz"),
    ("reddit_chatGPT.jsonl", "reddit", "chatGPT"),
    ("reddit_cohere.jsonl", "reddit", "cohere"),
    ("reddit_davinci.jsonl", "reddit", "davinci"),
    ("reddit_dolly.jsonl", "reddit", "dolly"),
    ("reddit_flant5.jsonl", "reddit", "flant5"),
    ("russian_chatGPT.jsonl", "russian", "chatGPT"),
    ("russian_davinci.jsonl", "russian", "davinci"),
    ("urdu_chatGPT.jsonl", "urdu", "chatGPT"),
    ("wikihow_bloomz.jsonl", "wikihow", "bloomz"),
    ("wikihow_chatGPT.jsonl", "wikihow", "chatGPT"),
    ("wikihow_cohere.jsonl", "wikihow", "cohere"),
    ("wikihow_davinci.jsonl", "wikihow", "davinci"),
    ("wikihow_dolly2.jsonl", "wikihow", "dolly"),
    ("wikipedia_bloomz.jsonl", "wikipedia", "bloomz"),
    ("wikipedia_chatgpt.jsonl", "wikipedia", "chatGPT"),
    ("wikipedia_cohere.jsonl", "wikipedia", "cohere"),
    ("wikipedia_davinci.jsonl", "wikipedia", "davinci"),
    ("wikipedia_dolly.jsonl", "wikipedia", "dolly"),
)

_ALL_DOMAINS = tuple(sorted({d for _, d, _ in _FILES}))
_ALL_MODELS = tuple(sorted({m for _, _, m in _FILES}))


def _as_text_field(val: Any) -> str:
    """Normalize M4 JSON text fields to str (some rows use list of segments/tokens)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = [str(x).strip() for x in val if x is not None and str(x).strip()]
        return " ".join(parts)
    return str(val).strip()


@register_dataset("m4")
class M4Dataset(BaseDataset):
    """M4 dataset for multi-generator / multi-domain / multi-lingual
    machine-generated text detection.

    Each JSONL record pairs a human-written text with a machine-generated
    continuation produced from the same prompt.  DetectZoo therefore
    yields **two** :class:`DatasetItem` objects per row: one for the
    human text (label=0) and one for the machine text (label=1).

    When *path* is omitted the requested JSONL files are downloaded
    automatically from `GitHub <https://github.com/mbzuai-nlp/M4>`_ and
    cached under ``.detectzoo_data/m4/``.

    Parameters
    ----------
    path : str or Path, optional
        Local directory containing the JSONL files.  When *None* the
        data is downloaded from GitHub.
    domains : sequence of str, optional
        Filter by domain (case-insensitive).  *None* loads all.  Valid
        values: see :data:`M4Dataset.DOMAINS`.
    models : sequence of str, optional
        Filter by generator model (case-insensitive).  *None* loads all.
        Valid values: see :data:`M4Dataset.MODELS`.
    include_human : bool
        Whether to emit the paired human texts (label=0).  Default
        ``True``.
    include_machine : bool
        Whether to emit the machine-generated texts (label=1).  Default
        ``True``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name = "m4"
    modality = "text"

    info = (
        "M4 (Multi-generator, Multi-domain, Multi-lingual)\n"
        "=================================================\n"
        "EACL 2024 Best Resource Paper.  Machine-generated text\n"
        "detection corpus covering 9 domains, 7 generators, and 6\n"
        "languages (English, Arabic, Bulgarian, Chinese, German,\n"
        "Indonesian, Russian, Urdu).\n"
        "\n"
        "Paper  : Wang et al., 'M4: Multi-generator, Multi-domain, and\n"
        "         Multi-lingual Black-Box Machine-Generated Text\n"
        "         Detection', EACL 2024.\n"
        "\n"
        "Domains\n"
        "-------\n"
        "  arxiv, wikipedia, wikihow, reddit, peerread, germanwikipedia,\n"
        "  id-newspaper, bulgarian, russian, arabic, urdu, qazh\n"
        "\n"
        "Generators\n"
        "----------\n"
        "  davinci, chatGPT, cohere, dolly, bloomz, flant5, llama\n"
        "\n"
        "File layout\n"
        "-----------\n"
        "One JSONL file per (domain, model) pair; each line carries\n"
        "  prompt, human_text, machine_text, model, source, source_ID\n"
        "DetectZoo emits two DatasetItems per record: the human text\n"
        "(label=0) and the machine text (label=1).\n"
        "\n"
        "Labels: 0 = human, 1 = AI.\n"
        "\n"
        "Benchmarking\n"
        "------------\n"
        "Same-generator / cross-domain sweep:\n"
        "  M4Dataset(models=['chatGPT'])\n"
        "Same-domain / cross-generator sweep:\n"
        "  M4Dataset(domains=['arxiv'])\n"
        "The M4 paper also evaluates GPTZero and cross-lingual transfer;\n"
        "filter by domain to reproduce multilingual settings.\n"
    )

    DOMAINS = _ALL_DOMAINS
    MODELS = _ALL_MODELS

    def __init__(
        self,
        path: str | Path | None = None,
        domains: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        include_human: bool = True,
        include_machine: bool = True,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path) if path is not None else None
        self.cache_dir = cache_dir
        self.domains = {d.lower() for d in domains} if domains else None
        self.models = {m.lower() for m in models} if models else None
        self.include_human = include_human
        self.include_machine = include_machine

    def _selected_files(self) -> list[tuple[str, str, str]]:
        selected: list[tuple[str, str, str]] = []
        for filename, domain, model in _FILES:
            if self.domains is not None and domain.lower() not in self.domains:
                continue
            if self.models is not None and model.lower() not in self.models:
                continue
            selected.append((filename, domain, model))
        return selected

    def _ensure_downloaded(self) -> Path:
        from detectzoo.datasets._download import download_file, get_cache_dir

        data_dir = get_cache_dir("m4", self.cache_dir)
        for filename, _domain, _model in self._selected_files():
            url = f"{_GITHUB_RAW}/{filename}"
            download_file(url, data_dir / filename)
        return data_dir

    def _load_jsonl(
        self,
        fp: Path,
        domain: str,
        model: str,
    ) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row: dict[str, Any] = json.loads(line)
                prompt = row.get("prompt", "")
                source = row.get("source", domain)
                source_id = row.get("source_ID", "")
                row_model = row.get("model", model)

                if self.include_human:
                    human_text = _as_text_field(row.get("human_text", ""))
                    if human_text:
                        items.append(
                            DatasetItem(
                                data=human_text,
                                label=0,
                                metadata={
                                    "domain": domain,
                                    "model": "human",
                                    "source": source,
                                    "source_id": source_id,
                                    "prompt": prompt,
                                },
                            )
                        )
                if self.include_machine:
                    machine_text = _as_text_field(row.get("machine_text", ""))
                    if machine_text:
                        items.append(
                            DatasetItem(
                                data=machine_text,
                                label=1,
                                metadata={
                                    "domain": domain,
                                    "model": row_model,
                                    "source": source,
                                    "source_id": source_id,
                                    "prompt": prompt,
                                },
                            )
                        )
        return items

    def _load_all(self) -> List[DatasetItem]:
        data_dir = self.path if self.path is not None else self._ensure_downloaded()
        items: List[DatasetItem] = []
        for filename, domain, model in self._selected_files():
            fp = data_dir / filename
            if fp.exists():
                items.extend(self._load_jsonl(fp, domain, model))
        return items
