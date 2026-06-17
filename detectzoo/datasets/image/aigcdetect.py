"""AIGCDetect – Comprehensive benchmark for AI-generated image detection.

Reference:
    Zhong et al., "PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection",
    arXiv 2023.
    https://arxiv.org/abs/2311.12397

Note:
    In the original PatchCraft / AIGCDetectBenchmark setup, the **training split** is based on
    the CNNSpot/CNNDetection training data (i.e., the ForenSynths-style ProGAN-based training
    set), while AIGCDetect is primarily used as a large unified test benchmark across
    many generators.

GitHub: https://github.com/Ekko-zn/AIGCDetectBenchmark
ModelScope: ``aemilia/AIGCDetectionBenchmark``
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_MODELSCOPE_AIGCDETECT_DATASET: str = "aemilia/AIGCDetectionBenchmark"

# (table column name, on-disk folder name under *root*)
AIGCDETECT_PARTITIONS: Tuple[Tuple[str, str], ...] = (
    ("ProGAN", "progan"),
    ("StyleGAN", "stylegan"),
    ("BigGAN", "biggan"),
    ("CycleGAN", "cyclegan"),
    ("StarGAN", "stargan"),
    ("GauGAN", "gaugan"),
    ("StyleGAN2", "stylegan2"),
    ("WFIR", "whichfaceisreal"),
    ("ADM", "ADM"),
    ("Glide", "Glide"),
    ("Midjourney", "Midjourney"),
    ("SD v1.4", "stable_diffusion_v_1_4"),
    ("SD v1.5", "stable_diffusion_v_1_5"),
    ("VQDM", "VQDM"),
    ("Wukong", "wukong"),
    ("DALLE2", "DALLE2"),
)

_COLUMN_TO_FOLDER: dict[str, str] = {c: f for c, f in AIGCDETECT_PARTITIONS}
_FOLDER_TO_COLUMN: dict[str, str] = {f: c for c, f in AIGCDETECT_PARTITIONS}

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

# Official / ModelScope layout: ProGAN, StyleGAN, … use ``<root>/<folder>/<class>/0_real``.
_NESTED_CLASS_PARTITION_FOLDERS: frozenset[str] = frozenset(
    {"progan", "stylegan", "stylegan2", "cyclegan"}
)


def _partition_layout_ok(parent: Path, folder_name: str) -> bool:
    base = parent / folder_name
    if not base.is_dir():
        return False
    if (base / "0_real").is_dir() and (base / "1_fake").is_dir():
        return True
    if folder_name not in _NESTED_CLASS_PARTITION_FOLDERS:
        return False
    try:
        for sub in base.iterdir():
            if sub.is_dir() and (sub / "0_real").is_dir() and (sub / "1_fake").is_dir():
                return True
    except OSError:
        return False
    return False


def find_aigcdetect_root(start: Path) -> Optional[Path]:
    start = start.resolve()

    for _, folder in AIGCDETECT_PARTITIONS:
        if folder in _NESTED_CLASS_PARTITION_FOLDERS:
            continue
        for fp in start.rglob(folder):
            try:
                if not fp.is_dir() or fp.name != folder:
                    continue
            except OSError:
                continue
            if (fp / "0_real").is_dir() and (fp / "1_fake").is_dir():
                return fp.parent

    for folder in _NESTED_CLASS_PARTITION_FOLDERS:
        for pp in start.rglob(folder):
            try:
                if not pp.is_dir() or pp.name != folder:
                    continue
            except OSError:
                continue
            if _partition_layout_ok(pp.parent, folder):
                return pp.parent
    return None


def _real_fake_dirs_for_partition(root: Path, folder_name: str) -> List[Tuple[Path, Path]]:
    """Return ``(0_real, 1_fake)`` directory pairs for one partition (flat or class-nested)."""
    base = root / folder_name
    if not base.is_dir():
        return []
    direct_r, direct_f = base / "0_real", base / "1_fake"
    if direct_r.is_dir() and direct_f.is_dir():
        return [(direct_r, direct_f)]
    pairs: List[Tuple[Path, Path]] = []
    try:
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            r, f = sub / "0_real", sub / "1_fake"
            if r.is_dir() and f.is_dir():
                pairs.append((r, f))
    except OSError:
        return pairs
    return pairs


def _extract_archives_recursive(root: Path, *, max_rounds: int = 32) -> None:
    extracted: set[Path] = set()
    for _ in range(max_rounds):
        progress = False
        for zpath in sorted(root.rglob("*.zip")):
            if zpath in extracted:
                continue
            parent = zpath.parent
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(parent)
                extracted.add(zpath)
                progress = True
            except zipfile.BadZipFile:
                extracted.add(zpath)
                continue
        for pattern in ("*.tar.gz", "*.tgz"):
            for tpath in sorted(root.rglob(pattern)):
                if tpath in extracted:
                    continue
                parent = tpath.parent
                try:
                    with tarfile.open(tpath, mode="r:*") as tf:
                        tf.extractall(path=parent)
                    extracted.add(tpath)
                    progress = True
                except (tarfile.TarError, OSError):
                    extracted.add(tpath)
                    continue
        if not progress:
            break


def _try_download_modelscope_aigcdetect(dest: Path) -> Optional[Path]:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        return None

    try:
        snapshot_download(
            model_id=_MODELSCOPE_AIGCDETECT_DATASET,
            repo_type="dataset",
            local_dir=str(dest),
        )
    except Exception:
        return None

    _extract_archives_recursive(dest)
    return find_aigcdetect_root(dest)


def ensure_aigcdetect_downloaded(
    root: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    from detectzoo.datasets._download import get_cache_dir

    dest = Path(root) if root is not None else get_cache_dir("aigcdetect", cache_dir)
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    found = find_aigcdetect_root(dest)
    if found is not None and not force:
        return found

    found = _try_download_modelscope_aigcdetect(dest)
    if found is not None:
        return found

    raise RuntimeError("Could not locate AIGCDetectBenchmark after ModelScope download.")


def resolve_aigcdetect_partition(partition: str) -> Tuple[str, str]:
    if partition in _COLUMN_TO_FOLDER:
        return partition, _COLUMN_TO_FOLDER[partition]
    if partition in _FOLDER_TO_COLUMN:
        return _FOLDER_TO_COLUMN[partition], partition
    valid = sorted(set(_COLUMN_TO_FOLDER) | set(_FOLDER_TO_COLUMN))
    raise ValueError(f"Unknown partition {partition!r}. Valid column or folder names: {valid}")


@register_dataset("aigcdetect", aliases=["aigc_detect", "aigcdetect_benchmark"])
class AIGCDetectDataset(BaseDataset):
    """AIGCDetect benchmark partitions: ``0_real`` / ``1_fake`` image paths per generator.

    Parameters
    ----------
    root : str or Path, optional
        Directory intended to contain partition folders, or a parent to search.
        When omitted, the default cache directory ``.detectzoo_data/aigcdetect/`` is used.
    partitions : sequence of str, optional
        Partition(s) to load. Each entry may be either a **column** name
        (``\"ProGAN\"``, ``\"SD v1.4\"``, …) or the **folder** name
        (``\"progan\"``, ``\"stable_diffusion_v_1_4\"``, …). When None, loads all
        :data:`AIGCDETECT_PARTITIONS`.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name: str = "aigcdetect"
    modality: str = "image"

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        partitions: Optional[Sequence[str]] = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root = Path(root) if root is not None else None
        self.partitions = list(partitions) if partitions is not None else None
        self.cache_dir = cache_dir
        self._resolved_root: Optional[Path] = None

    def _data_root(self) -> Path:

        if self._resolved_root is not None:
            return self._resolved_root
        if self.root is not None:
            r = self.root.resolve()
            found = find_aigcdetect_root(r)
            self._resolved_root = found if found is not None else r
        else:
            self._resolved_root = ensure_aigcdetect_downloaded(
                None, cache_dir=self.cache_dir, force=False
            )
        return self._resolved_root

    def _load_all(self) -> List[DatasetItem]:
        specs: List[Tuple[str, str]]
        if self.partitions is None:
            specs = list(AIGCDETECT_PARTITIONS)
        else:
            specs = [resolve_aigcdetect_partition(p) for p in self.partitions]

        root = self._data_root()
        items: List[DatasetItem] = []

        for col_name, folder_name in specs:
            base = root / folder_name
            pairs = _real_fake_dirs_for_partition(root, folder_name)

            for real_dir, fake_dir in pairs:
                sub_key = real_dir.parent.name if real_dir.parent != base else None
                base_meta: dict[str, Any] = {
                    "partition": col_name,
                    "folder": folder_name,
                }
                if sub_key is not None:
                    base_meta["subfolder"] = sub_key
                for label, source, d in (
                    (0, "real", real_dir),
                    (1, "fake", fake_dir),
                ):
                    for path in sorted(d.rglob("*")):
                        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                            items.append(
                                DatasetItem(
                                    data=str(path),
                                    label=label,
                                    metadata={**base_meta, "source": source},
                                )
                            )

        return items
