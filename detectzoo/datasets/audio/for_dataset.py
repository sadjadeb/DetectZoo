"""Fake-or-Real (FoR) dataset for synthetic speech detection.

Reference:
    Reimao & Tzerpos, "The Fake-or-Real (FoR) Dataset",2020.
    https://bil.eecs.yorku.ca/wp-content/uploads/2020/01/FoR-Dataset_RR_VT_final.pdf

Official downloads (GPLv3):
    https://bil.eecs.yorku.ca/datasets/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_BASE_URL = "https://bil.eecs.yorku.ca/share"

_VARIANT_FILES: dict[str, str] = {
    "original": "for-original.tar.gz",
    "norm": "for-norm.tar.gz",
    "two_sec": "for-2sec.tar.gz",
    "rerec": "for-rerec.tar.gz",
}

_VARIANT_ALIASES: dict[str, str] = {
    "for-original": "original",
    "for_original": "original",
    "for-norm": "norm",
    "for_norm": "norm",
    "for-2sec": "two_sec",
    "for_2sec": "two_sec",
    "for-2seconds": "two_sec",
    "for-rerec": "rerec",
    "for_rerec": "rerec",
    "for-rerecorded": "rerec",
}

_PREPROCESSED_SPLITS = frozenset({"train", "val", "test", "all"})

_AUDIO_EXT = frozenset({".wav", ".mp3", ".flac"})


def _norm_dirname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_train_norm(n: str) -> bool:
    return n == "training" or n == "train" or n.startswith("training")


def _is_val_norm(n: str) -> bool:
    return n == "validation" or n == "val" or n.startswith("validation")


def _is_test_norm(n: str) -> bool:
    if n in ("testing", "test"):
        return True
    return "generalization" in n


def _coerce_variant(variant: str) -> str:
    key = variant.strip().lower().replace(" ", "_")
    key = _VARIANT_ALIASES.get(key, key)
    if key not in _VARIANT_FILES:
        choices = ", ".join(sorted(_VARIANT_FILES))
        raise ValueError(f"Unknown FoR variant {variant!r}. Choose one of: {choices}")
    return key


def _split_pick_fn(split: str) -> Callable[[str], bool]:
    if split == "train":
        return _is_train_norm
    if split == "val":
        return _is_val_norm
    if split == "test":
        return _is_test_norm
    raise ValueError(f"split must be 'train', 'val', or 'test' (got {split!r})")


def _real_fake_pairs(split_dir: Path) -> List[Tuple[Path, int, str]]:
    """Return (directory, label, role) for real/fake class folders under a split."""
    pairs: List[Tuple[Path, int, str]] = []
    for c in sorted(split_dir.iterdir()):
        if not c.is_dir():
            continue
        n = _norm_dirname(c.name)
        if n in ("real", "human", "bonafide", "original"):
            pairs.append((c, 0, "real"))
        elif n in ("fake", "synthetic", "spoof", "deepfake"):
            pairs.append((c, 1, "fake"))
        elif n in ("0real",):
            pairs.append((c, 0, "real"))
        elif n in ("1fake",):
            pairs.append((c, 1, "fake"))
    return pairs


def _collect_audio_files(
    directory: Path,
    label: int,
    *,
    metadata: dict[str, Any],
) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _AUDIO_EXT:
            items.append(DatasetItem(data=str(path), label=label, metadata=dict(metadata)))
    return items


def _bfs_dirs(root: Path, max_depth: int) -> List[Path]:
    frontier: List[Tuple[Path, int]] = [(root, 0)]
    out: List[Path] = []
    while frontier:
        d, depth = frontier.pop(0)
        out.append(d)
        if depth >= max_depth:
            continue
        try:
            children = sorted([p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")])
        except OSError:
            continue
        for ch in children:
            frontier.append((ch, depth + 1))
    return out


def _find_preprocessed_split_parent(start: Path) -> Optional[Path]:
    """Locate the directory that contains Training / Validation / Testing (or similar) siblings."""
    best: Optional[Path] = None
    best_depth = 1 << 30
    for d in _bfs_dirs(start.resolve(), max_depth=8):
        try:
            subs = [c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")]
        except OSError:
            continue
        nn = [_norm_dirname(c.name) for c in subs]
        has_train = any(_is_train_norm(n) for n in nn)
        has_val = any(_is_val_norm(n) for n in nn)
        has_test = any(_is_test_norm(n) for n in nn)
        score = int(has_train) + int(has_val) + int(has_test)
        if score >= 2:
            depth = len(d.relative_to(start).parts) if d != start else 0
            if depth < best_depth:
                best = d
                best_depth = depth
    return best


def _pick_split_child(parent: Path, split: str) -> Path:
    pred = _split_pick_fn(split)
    for c in sorted(parent.iterdir()):
        if c.is_dir() and pred(_norm_dirname(c.name)):
            return c
    names = sorted(p.name for p in parent.iterdir() if p.is_dir())
    raise FileNotFoundError(
        f"FoR: could not find a {split!r} split directory under {parent}. Subdirectories: {names}"
    )


def _layout_flat_real_fake(root: Path) -> bool:
    nn = {_norm_dirname(c.name) for c in root.iterdir() if c.is_dir()}
    has_real = bool(nn & {"real", "human", "bonafide"})
    has_fake = bool(nn & {"fake", "synthetic", "spoof"})
    return has_real and has_fake


def _label_original_source(folder_name: str, extra: dict[str, int]) -> Optional[int]:
    if folder_name in extra:
        return extra[folder_name]
    n = _norm_dirname(folder_name)

    fake_markers = (
        "deepvoice",
        "polly",
        "amazon",
        "baidu",
        "wavenet",
        "googlecloud",
        "cloudtts",
        "azure",
        "microsoft",
        "googletexttospeech",
    )
    if any(m in n for m in fake_markers):
        return 1
    if "google" in n and "traditional" in n:
        return 1
    if "google" in n:
        return 1

    real_markers = ("arctic", "ljspeech", "voxforge", "internet", "youtube", "tedtalk")
    if any(m in n for m in real_markers):
        return 0
    if "recording" in n and "rerec" not in n:
        return 0
    return None


def _load_preprocessed(
    root: Path,
    *,
    split: str,
    variant_key: str,
) -> List[DatasetItem]:
    root = root.resolve()
    items: List[DatasetItem] = []

    if _layout_flat_real_fake(root):
        pairs = _real_fake_pairs(root)
        if not pairs:
            names = sorted(p.name for p in root.iterdir() if p.is_dir())
            raise FileNotFoundError(
                f"FoR: no real/fake class folders under {root}. Subdirectories: {names}"
            )
        for dir_path, label, role in pairs:
            meta = {
                "split": "all",
                "variant": variant_key,
                "modality": "audio",
                "class": role,
            }
            items.extend(_collect_audio_files(dir_path, label, metadata=meta))
        return items

    parent = _find_preprocessed_split_parent(root)
    if parent is None:
        raise FileNotFoundError(
            f"FoR ({variant_key}): expected split folders (e.g. Training/Validation/Testing) "
            f"or a top-level real/fake tree under {root}"
        )

    want_splits: Tuple[str, ...] = ("train", "val", "test") if split == "all" else (split,)
    # Track per-split class counts so we can raise a clear error if a
    # particular split is missing one of the two classes (e.g. the FoR
    # `norm` training split, where the `fake/` folder is shipped empty).
    per_split_counts: dict[str, dict[str, int]] = {}
    for sp in want_splits:
        try:
            split_dir = _pick_split_child(parent, sp)
        except FileNotFoundError:
            continue
        pairs = _real_fake_pairs(split_dir)
        if not pairs:
            names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
            raise FileNotFoundError(
                f"FoR: no real/fake class folders under {split_dir}. Subdirectories: {names}"
            )
        sp_counts = {"real": 0, "fake": 0}
        for dir_path, label, role in pairs:
            meta = {
                "split": sp,
                "variant": variant_key,
                "modality": "audio",
                "class": role,
            }
            collected = _collect_audio_files(dir_path, label, metadata=meta)
            sp_counts[role] += len(collected)
            items.extend(collected)
        per_split_counts[sp] = sp_counts

    if not items:
        raise FileNotFoundError(
            f"FoR ({variant_key}): no audio found for split={split!r} under {parent}. "
            "Check that Training/Validation/Testing (or similar) folders exist."
        )

    # Single-split request must yield both classes; otherwise the user
    # would get a one-class dataset and binary metrics (EER/AUC/F1) would
    # silently degenerate. The FoR `norm` *training* split is the known
    # offender: its `fake/` folder ships empty.
    if split != "all":
        c = per_split_counts.get(split, {"real": 0, "fake": 0})
        empty = [role for role, n in c.items() if n == 0]
        if empty:
            avail = {
                sp: cs for sp, cs in per_split_counts.items() if all(v > 0 for v in cs.values())
            }
            hint = (
                f" Try `split={next(iter(avail))!r}` instead — that split "
                f"has both classes ({avail[next(iter(avail))]})."
                if avail
                else " No other split has both classes either; the local extraction is incomplete."
            )
            raise RuntimeError(
                f"FoR ({variant_key}, split={split!r}): the {empty!r} class "
                f"folder is empty (counts: real={c['real']}, fake={c['fake']}). "
                "This is a known issue with the FoR `norm` *training* split "
                "where the `fake/` directory ships empty in the upstream "
                f"tarball.{hint}"
            )
    return items


def _load_original(
    root: Path,
    *,
    variant_key: str,
    extra_labels: dict[str, int],
) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    for c in sorted(root.iterdir()):
        if not c.is_dir() or c.name.startswith("."):
            continue
        label = _label_original_source(c.name, extra_labels)
        if label is None:
            continue
        meta = {
            "split": "all",
            "variant": variant_key,
            "modality": "audio",
            "source_folder": c.name,
            "class": "fake" if label == 1 else "real",
        }
        items.extend(_collect_audio_files(c, label, metadata=meta))
    if not items:
        raise FileNotFoundError(
            f"FoR (original): no labelled audio under {root}. "
            "Expected one subdirectory per speech source; use "
            "extra_source_labels for unknown folder names."
        )
    return items


def _resolve_user_root(path: Path) -> Path:
    p = path.resolve()
    if _layout_flat_real_fake(p):
        return p
    if _find_preprocessed_split_parent(p) is not None:
        return p
    subdirs = [c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")]
    if len(subdirs) == 1:
        child = subdirs[0]
        if _layout_flat_real_fake(child) or _find_preprocessed_split_parent(child) is not None:
            return child
    return p


@register_dataset("for", aliases=["for_dataset", "fake_or_real"])
class FoRDataset(BaseDataset):
    """Fake-or-Real (FoR) audio benchmark (human vs. synthetic speech).

    The official release provides four archives (see the `variant` argument):
    *original*, *norm* (16 kHz mono WAV, level-normalised), *two_sec* (truncated),
    and *rerec* (re-recorded). Preprocessed variants are split into training,
    validation, and generalisation-testing partitions with **Real** / **Fake**
    (or equivalent) class folders — this loader discovers those layouts
    automatically. The original release is organised by **speech source**; a
    lightweight name-based mapping assigns labels (overridable via
    ``extra_source_labels``).

    When ``path`` is omitted, the selected archive is downloaded from the
    `APTLY dataset page <https://bil.eecs.yorku.ca/datasets/>`_ and extracted
    under ``cache_dir`` (default ``.detectzoo_data/for``). Archives are large
    (multi-gigabyte); ensure sufficient disk space.

    Parameters
    ----------
    variant
        ``"original"``, ``"norm"``, ``"two_sec"``, or ``"rerec"``. Aliases such
        as ``"for-norm"`` are accepted.
    split
        ``"train"``, ``"val"``, ``"test"``, or ``"all"`` (concatenate splits).
        For ``variant="original"`` only ``"all"`` is valid.
    path
        Optional directory containing extracted FoR files (or a single
        top-level folder inside the tarball).
    download
        If ``True`` and ``path`` is ``None``, download and extract the archive.
    cache_dir
        Root cache directory when downloading (default ``.detectzoo_data``).
    extra_source_labels
        For ``variant="original"`` only: map **exact** source folder names to
        labels (``0`` human, ``1`` synthetic). Folders not listed and not
        recognised by built-in heuristics are skipped.
    max_samples
        Optional cap on samples (see :class:`~detectzoo.datasets.base.BaseDataset`).
    """

    name = "for"
    modality = "audio"

    def __init__(
        self,
        variant: str = "norm",
        split: str = "train",
        path: str | Path | None = None,
        *,
        download: bool = True,
        cache_dir: str | Path | None = None,
        extra_source_labels: dict[str, int] | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.variant_key = _coerce_variant(variant)
        self.split = split.lower()
        self.path = Path(path) if path is not None else None
        self.download = download
        self.cache_dir = cache_dir
        self.extra_source_labels = dict(extra_source_labels or {})

        if self.split not in _PREPROCESSED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_PREPROCESSED_SPLITS)}, got {split!r}")
        if self.variant_key == "original" and self.split != "all":
            raise ValueError(
                'For variant "original", split must be "all" (source-organised layout).'
            )

    def _ensure_root(self) -> Path:
        if self.path is not None:
            return _resolve_user_root(self.path)

        if not self.download:
            raise ValueError(
                "FoRDataset: path is None and download=False — provide a local `path`."
            )

        from detectzoo.datasets._download import download_and_extract_tar, get_cache_dir

        dest = get_cache_dir("for", self.cache_dir) / self.variant_key
        fname = _VARIANT_FILES[self.variant_key]
        url = f"{_BASE_URL}/{fname}"
        download_and_extract_tar(url, dest, force=False)
        return _resolve_user_root(dest)

    def _load_all(self) -> List[DatasetItem]:
        root = self._ensure_root()
        if self.variant_key == "original":
            return _load_original(
                root,
                variant_key=self.variant_key,
                extra_labels=self.extra_source_labels,
            )
        return _load_preprocessed(root, split=self.split, variant_key=self.variant_key)
