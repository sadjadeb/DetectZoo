"""Deepfake-Eval-2024 audio (Chandra et al., 2025).

Reference:
    Chandra et al., "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of
    Deepfakes Circulated in 2024", arXiv:2503.02857, 2025.
    https://arxiv.org/abs/2503.02857

HuggingFace (gated — accept terms and ``huggingface-cli login`` first):
    ``nuriachandra/Deepfake-Eval-2024``

Not to be confused with Müller et al. "In-The-Wild" (Interspeech 2022;
``mueller91/In-The-Wild``) — a separate celebrity YouTube corpus.

Layout (under the dataset root)::

    audio-metadata-publish.csv
    audio-data/<Filename>.mp3   # (or .wav / .m4a)

The published metadata uses columns ``Filename``, ``Ground Truth`` (``real`` /
``fake``; ``unknown`` rows are skipped), and ``Finetuning Set`` (``train`` /
``test`` for the official 60/40 evaluation split).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_HF_REPO = "nuriachandra/Deepfake-Eval-2024"
_METADATA_CANDIDATES = (
    "audio-metadata-publish.csv",
    "audio-metadata-publish-with-links.csv",
)
_AUDIO_SUBDIR = "audio-data"
_SPLITS = frozenset({"train", "test", "all"})
_AUDIO_EXT = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"})


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("_", " ")


def _row_get(row: dict[str, str], *candidates: str) -> str:
    """Return the first matching column value (case/space insensitive)."""
    lookup = {_norm_key(k): v for k, v in row.items()}
    for name in candidates:
        val = lookup.get(_norm_key(name))
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _parse_ground_truth(value: str) -> Optional[int]:
    g = value.strip().lower()
    if g == "real":
        return 0
    if g == "fake":
        return 1
    return None


def _parse_finetuning_split(value: str) -> Optional[str]:
    s = value.strip().lower()
    if s == "train":
        return "train"
    if s == "test":
        return "test"
    return None


def _find_metadata_csv(root: Path) -> Path:
    for name in _METADATA_CANDIDATES:
        p = root / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Deepfake-Eval-2024: no metadata CSV under {root}. "
        f"Expected one of: {', '.join(_METADATA_CANDIDATES)}"
    )


def _resolve_dataset_root(user_path: Path) -> Path:
    u = user_path.expanduser().resolve()
    try:
        _find_metadata_csv(u)
        return u
    except FileNotFoundError:
        pass
    for sub in (u / "Deepfake-Eval-2024", u / "deepfake-eval-2024"):
        if sub.is_dir():
            try:
                _find_metadata_csv(sub)
            except FileNotFoundError:
                continue
            return sub
    raise FileNotFoundError(
        f"Deepfake-Eval-2024: could not locate {_METADATA_CANDIDATES[0]!r} under {user_path}"
    )


def _resolve_audio_path(audio_dir: Path, filename: str) -> Optional[Path]:
    name = filename.strip()
    if not name:
        return None
    direct = audio_dir / name
    if direct.is_file():
        return direct
    stem = Path(name).stem
    for ext in _AUDIO_EXT:
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _load_from_metadata(
    root: Path,
    *,
    split: str,
    skip_missing: bool,
) -> List[DatasetItem]:
    root = _resolve_dataset_root(root)
    meta_path = _find_metadata_csv(root)
    audio_dir = root / _AUDIO_SUBDIR
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Deepfake-Eval-2024: expected audio directory {audio_dir}"
        )

    items: List[DatasetItem] = []
    missing: List[str] = []

    with open(meta_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Deepfake-Eval-2024: empty metadata file {meta_path}")

        for row in reader:
            filename = _row_get(row, "Filename", "filename", "file")
            if not filename:
                continue

            label = _parse_ground_truth(_row_get(row, "Ground Truth", "ground_truth", "label"))
            if label is None:
                continue

            finetune = _parse_finetuning_split(
                _row_get(row, "Finetuning Set", "finetuning set", "split", "set")
            )
            if split != "all":
                if finetune is None or finetune != split:
                    continue

            apath = _resolve_audio_path(audio_dir, filename)
            if apath is None:
                missing.append(filename)
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f"Deepfake-Eval-2024: audio missing for {filename!r} "
                    f"(looked under {audio_dir})"
                )

            meta: dict[str, Any] = {
                "modality": "audio",
                "class": "real" if label == 0 else "fake",
                "filename": filename,
                "finetuning_set": finetune or "",
            }
            for key, val in row.items():
                if val is None or not str(val).strip():
                    continue
                nk = _norm_key(key).replace(" ", "_")
                if nk not in meta and nk not in ("filename", "ground_truth", "finetuning_set"):
                    meta[nk] = str(val).strip()

            items.append(DatasetItem(data=str(apath), label=label, metadata=meta))

    if not items:
        raise RuntimeError(
            f"Deepfake-Eval-2024: no labelled audio loaded from {meta_path} "
            f"(split={split!r})."
        )
    if missing and skip_missing:
        from detectzoo.utils.logger import get_logger

        get_logger(__name__).warning(
            "Deepfake-Eval-2024: skipped %d missing files (first: %s)",
            len(missing),
            missing[0],
        )
    return items


@register_dataset(
    "deepfake_eval_2024",
    aliases=["deepfake-eval-2024", "deepfake_eval", "df_eval_2024"],
)
class DeepfakeEval2024Dataset(BaseDataset):
    """Deepfake-Eval-2024 audio — bonafide vs. deepfake speech (2024).

    Multi-modal benchmark (Chandra et al., 2025); this loader exposes the
    **audio** split only (~40k labelled clips from social media and
    TrueMedia.org). When ``path`` is omitted the dataset is fetched from
    HuggingFace (``nuriachandra/Deepfake-Eval-2024``). Access is **gated**:
    accept the dataset terms on HuggingFace and run ``huggingface-cli login``
    before downloading.

    Parameters
    ----------
    path
        Local root containing ``audio-metadata-publish.csv`` and
        ``audio-data/``. When omitted, files are cached under
        ``<cache_dir>/deepfake_eval_2024/``.
    split
        ``\"train\"``, ``\"test\"``, or ``\"all\"`` (default). Uses the
        official ``Finetuning Set`` column from the metadata CSV.
    download
        If ``True`` and ``path`` is ``None``, download from HuggingFace.
    cache_dir
        Root cache directory when downloading (default ``.detectzoo_data``).
    skip_missing
        Skip rows whose audio file is absent instead of raising.
    hf_repo
        HuggingFace dataset id (default ``nuriachandra/Deepfake-Eval-2024``).
    max_samples
        Optional cap (see :class:`~detectzoo.datasets.base.BaseDataset`).
    """

    name = "deepfake_eval_2024"
    modality = "audio"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        split: str = "all",
        download: bool = True,
        cache_dir: str | Path | None = None,
        skip_missing: bool = False,
        hf_repo: str = _HF_REPO,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.path = Path(path) if path is not None else None
        self.split = split.strip().lower()
        self.download = download
        self.cache_dir = cache_dir
        self.skip_missing = skip_missing
        self.hf_repo = hf_repo

        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}, got {split!r}")

    def _ensure_root(self) -> Path:
        if self.path is not None:
            return _resolve_dataset_root(self.path)

        if not self.download:
            raise ValueError(
                "DeepfakeEval2024Dataset: path is None and download=False — "
                "provide a local `path` or enable download."
            )

        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError

        from detectzoo.datasets._download import get_cache_dir

        dest = get_cache_dir("deepfake_eval_2024", self.cache_dir)
        marker = dest / ".download_complete"
        if marker.is_file():
            return _resolve_dataset_root(dest)

        dest.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=self.hf_repo,
                repo_type="dataset",
                local_dir=dest,
                allow_patterns=[
                    "audio-metadata-publish.csv",
                    "audio-data/*",
                ],
            )
        except GatedRepoError as exc:
            raise PermissionError(
                f"Cannot download gated dataset {self.hf_repo!r}. "
                "Accept the dataset terms on HuggingFace, then run "
                "`huggingface-cli login` and retry."
            ) from exc

        marker.touch()
        return _resolve_dataset_root(dest)

    def _load_all(self) -> List[DatasetItem]:
        root = self._ensure_root()
        return _load_from_metadata(
            root,
            split=self.split,
            skip_missing=self.skip_missing,
        )
