"""In-The-Wild audio deepfake dataset (Müller et al., Interspeech 2022).

Reference:
    Müller et al., "Does Audio Deepfake Detection Generalize?",
    Interspeech 2022. arXiv:2203.16263.
    https://arxiv.org/abs/2203.16263

HuggingFace:
    ``mueller91/In-The-Wild`` — ``release_in_the_wild.zip`` (~8 GB).

After extraction the release typically contains ``meta.csv`` or
``modified_meta.csv`` (columns ``file``, ``label`` with ``bona-fide`` / ``real`` /
``fake``) and ``.wav`` clips, often organised under ``real/`` and ``fake/``
subfolders.

Not to be confused with Deepfake-Eval-2024 (Chandra et al., 2025).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_HF_REPO = "mueller91/In-The-Wild"
_ZIP_NAME = "release_in_the_wild.zip"
_ZIP_URL = f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/{_ZIP_NAME}"
_META_CANDIDATES = ("modified_meta.csv", "meta.csv", "metadata.csv")
_AUDIO_EXT = frozenset({".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3"})
_REAL_DIR = frozenset({"real", "bonafide", "genuine", "human", "original"})
_FAKE_DIR = frozenset({"fake", "spoof", "synthetic", "deepfake"})


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def _label_from_str(value: str) -> Optional[int]:
    v = _norm_name(value)
    if v in ("real", "bonafide", "genuine", "human", "original", "0"):
        return 0
    if v in ("fake", "spoof", "synthetic", "deepfake", "1"):
        return 1
    return None


def _has_real_fake_dirs(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    names = {_norm_name(p.name) for p in directory.iterdir() if p.is_dir()}
    return bool(names & {_norm_name(x) for x in _REAL_DIR}) and bool(
        names & {_norm_name(x) for x in _FAKE_DIR}
    )


def _find_release_root(user_path: Path) -> Path:
    u = user_path.expanduser().resolve()
    if u.is_dir():
        if (u / _META_CANDIDATES[0]).is_file() or _has_real_fake_dirs(u):
            return u
        for subname in ("release_in_the_wild", "In-The-Wild", "in_the_wild"):
            sub = u / subname
            if sub.is_dir() and (
                any((sub / m).is_file() for m in _META_CANDIDATES) or _has_real_fake_dirs(sub)
            ):
                return sub
        for meta_name in _META_CANDIDATES:
            hits = list(u.rglob(meta_name))
            if hits:
                return hits[0].parent
        if _has_real_fake_dirs(u):
            return u
    raise FileNotFoundError(
        f"In-The-Wild: could not locate extracted release under {user_path}. "
        f"Expected ``release_in_the_wild/`` with ``modified_meta.csv`` or "
        "``real/`` + ``fake/`` folders."
    )


def _find_metadata_csv(root: Path) -> Optional[Path]:
    for name in _META_CANDIDATES:
        p = root / name
        if p.is_file():
            return p
    return None


def _class_dirs(root: Path) -> List[tuple[Path, int, str]]:
    """Return (directory, label, role) for real/fake class folders under *root*."""
    pairs: List[tuple[Path, int, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        n = _norm_name(child.name)
        if n in {_norm_name(x) for x in _REAL_DIR}:
            pairs.append((child, 0, "real"))
        elif n in {_norm_name(x) for x in _FAKE_DIR}:
            pairs.append((child, 1, "fake"))
    return pairs


def _collect_wavs(
    directory: Path,
    label: int,
    *,
    metadata: dict[str, Any],
) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in _AUDIO_EXT:
            items.append(DatasetItem(data=str(path), label=label, metadata=dict(metadata)))
    return items


def _resolve_audio_path(root: Path, filename: str) -> Optional[Path]:
    name = filename.strip()
    if not name:
        return None
    for base in (root, root / "real", root / "fake"):
        direct = base / name
        if direct.is_file():
            return direct
    stem = Path(name).stem
    for path in root.rglob(f"{stem}.*"):
        if path.is_file() and path.suffix in _AUDIO_EXT:
            return path
    return None


def _load_from_metadata_csv(root: Path, meta_path: Path) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    missing: List[str] = []

    with open(meta_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"In-The-Wild: empty metadata file {meta_path}")
        field_map = {f.strip().lower(): f for f in reader.fieldnames}
        file_col = field_map.get("file") or field_map.get("filename") or field_map.get("path")
        label_col = (
            field_map.get("label") or field_map.get("class") or field_map.get("ground_truth")
        )
        if not file_col or not label_col:
            raise ValueError(
                f"In-The-Wild: {meta_path} must contain file and label columns; "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            filename = (row.get(file_col) or "").strip()
            if not filename:
                continue
            label = _label_from_str(row.get(label_col, ""))
            if label is None:
                continue
            apath = _resolve_audio_path(root, filename)
            if apath is None:
                missing.append(filename)
                continue
            meta: dict[str, Any] = {
                "modality": "audio",
                "class": "real" if label == 0 else "fake",
                "filename": filename,
            }
            items.append(DatasetItem(data=str(apath), label=label, metadata=meta))

    if not items:
        raise RuntimeError(
            f"In-The-Wild: no labelled audio resolved from {meta_path} under {root}."
        )
    if missing:
        from detectzoo.utils.logger import get_logger

        get_logger(__name__).warning(
            "In-The-Wild: %d metadata rows had no matching audio (first: %s)",
            len(missing),
            missing[0],
        )
    return items


def _load_from_class_dirs(root: Path) -> List[DatasetItem]:
    pairs = _class_dirs(root)
    if not pairs:
        raise FileNotFoundError(f"In-The-Wild: no real/fake class folders under {root}")
    items: List[DatasetItem] = []
    for dir_path, label, role in pairs:
        meta = {"modality": "audio", "class": role}
        items.extend(_collect_wavs(dir_path, label, metadata=meta))
    if not items:
        raise RuntimeError(f"In-The-Wild: no audio files under {root}")
    return items


def _load_release(
    root: Path,
    *,
    skip_missing: bool,
) -> List[DatasetItem]:
    root = _find_release_root(root)
    meta_path = _find_metadata_csv(root)
    if meta_path is not None:
        return _load_from_metadata_csv(root, meta_path)
    return _load_from_class_dirs(root)


@register_dataset(
    "in_the_wild",
    aliases=["in-the-wild", "inthewild", "itw", "muller_in_the_wild"],
)
class InTheWildDataset(BaseDataset):
    """Müller et al. In-The-Wild celebrity / politician audio deepfake corpus.

    ~31.8k clips (19,963 real + 11,816 fake) from 58 public figures, scraped
    from online video platforms (Interspeech 2022).

    Parameters
    ----------
    path
        Directory containing an extracted ``release_in_the_wild`` tree (or the
        tree itself). When omitted, ``release_in_the_wild.zip`` is downloaded
        from HuggingFace into ``<cache_dir>/in_the_wild/`` (~8 GB).
    download
        If ``True`` and ``path`` is ``None``, download and extract the ZIP.
    cache_dir
        Root cache directory when downloading (default ``.detectzoo_data``).
    skip_missing
        When loading from ``modified_meta.csv``, skip rows with missing audio
        instead of failing immediately after the CSV pass.
    max_samples
        Optional cap (see :class:`~detectzoo.datasets.base.BaseDataset`).
    """

    name = "in_the_wild"
    modality = "audio"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        download: bool = True,
        cache_dir: str | Path | None = None,
        skip_missing: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.path = Path(path) if path is not None else None
        self.download = download
        self.cache_dir = cache_dir
        self.skip_missing = skip_missing

    def _ensure_root(self) -> Path:
        if self.path is not None:
            return _find_release_root(self.path)

        if not self.download:
            raise ValueError(
                "InTheWildDataset: path is None and download=False — "
                "provide a local extract or enable download."
            )

        from detectzoo.datasets._download import download_and_extract_zip, get_cache_dir

        dest = get_cache_dir("in_the_wild", self.cache_dir)
        marker = dest / ".download_complete"
        if marker.is_file():
            return _find_release_root(dest)

        try:
            from huggingface_hub import hf_hub_download

            zip_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename=_ZIP_NAME,
                repo_type="dataset",
                local_dir=dest,
            )
            import zipfile

            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        except Exception:
            download_and_extract_zip(_ZIP_URL, dest, force=False)

        marker.touch()
        return _find_release_root(dest)

    def _load_all(self) -> List[DatasetItem]:
        root = self._ensure_root()
        return _load_release(root, skip_missing=self.skip_missing)
