"""CNN Detection – CNN-generated images are surprisingly easy to spot (benchmark).

Reference:
    Wang et al., "CNN-generated Images are Surprisingly Easy to Spot... for Now", CVPR 2020.
    https://arxiv.org/abs/1912.11035

HuggingFace: ``sywang/CNNDetection``
GitHub: https://github.com/peterwang512/CNNDetection
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_SPLITS: Tuple[str, ...] = ("train", "val", "test")
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = _SPLITS

_HF = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main"

_CACHE_KEY: dict[str, str] = {
    SPLIT_TRAIN: "cnn_detection_train",
    SPLIT_VAL: "cnn_detection_progan_val",
    SPLIT_TEST: "cnn_detection_test",
}

_REL_ROOT: dict[str, str] = {
    SPLIT_TRAIN: "progan_train",
    SPLIT_VAL: "progan_val",
    SPLIT_TEST: "CNN_synth_testset",
}

_TEST_ZIP = f"{_HF}/CNN_synth_testset.zip"
_VAL_ZIP = f"{_HF}/progan_val.zip"
_TRAIN_7Z = tuple(f"{_HF}/progan_train.7z.{i:03d}" for i in range(1, 8))
_TRAIN_7Z_NAMES = tuple(f"progan_train.7z.{i:03d}" for i in range(1, 8))

CNN_DETECTION_TEST_PARTITIONS: Tuple[Tuple[str, str], ...] = (
    ("ProGAN", "progan"),
    ("StyleGAN", "stylegan"),
    ("BigGAN", "biggan"),
    ("CycleGAN", "cyclegan"),
    ("StarGAN", "stargan"),
    ("GauGAN", "gaugan"),
    ("CRN", "crn"),
    ("IMLE", "imle"),
    ("SITD", "seeingdark"),
    ("SAN", "san"),
    ("Deepfake", "deepfake"),
    ("StyleGAN2", "stylegan2"),
    ("Whichfaceisreal", "whichfaceisreal"),
)

_COLUMN_TO_FOLDER: dict[str, str] = {c: f for c, f in CNN_DETECTION_TEST_PARTITIONS}
_FOLDER_TO_COLUMN: dict[str, str] = {f: c for c, f in CNN_DETECTION_TEST_PARTITIONS}

_PROGAN_CLASS_FOLDERS: Tuple[str, ...] = (
    "airplane",
    "bird",
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "cow",
    "chair",
    "diningtable",
    "dog",
    "person",
    "pottedplant",
    "motorbike",
    "tvmonitor",
    "train",
    "sheep",
    "sofa",
    "horse",
)

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def _resolved_data_root(dest: Path, split: str) -> Path | None:
    """Where image folders live under *dest* after HF extract (nested or flat)."""
    dest = dest.resolve()
    nested = dest / _REL_ROOT[split]
    if nested.is_dir():
        return nested
    if split == SPLIT_TEST and (dest / "progan").is_dir():
        return dest
    if split == SPLIT_VAL and (dest / "airplane").is_dir():
        return dest
    if split == SPLIT_TRAIN and (dest / "airplane").is_dir():
        return dest
    if split == SPLIT_TRAIN and (dest / "progan_train").is_dir():
        return dest / "progan_train"
    return None


def _extract_7z(first_part: Path, dest_dir: Path) -> None:
    import shutil
    import subprocess

    seven = shutil.which("7z") or shutil.which("7za")
    if not seven:
        raise RuntimeError("need `7z` or `7za` on PATH to unpack ProGAN train")
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([seven, "x", str(first_part), f"-o{dest_dir}", "-y"], check=True)


def resolve_cnn_detection_partition(name: str) -> Tuple[str, str]:
    if name in _COLUMN_TO_FOLDER:
        return name, _COLUMN_TO_FOLDER[name]
    if name in _FOLDER_TO_COLUMN:
        return _FOLDER_TO_COLUMN[name], name
    raise ValueError(f"unknown test partition {name!r}")


def _classes(partitions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if partitions is None:
        return _PROGAN_CLASS_FOLDERS
    valid = {c.lower(): c for c in _PROGAN_CLASS_FOLDERS}
    sel = {p.lower() for p in partitions}
    bad = sel - set(valid)
    if bad:
        raise ValueError(f"unknown class(es): {sorted(bad)}")
    return tuple(valid[k] for k in _PROGAN_CLASS_FOLDERS if k in sel)


def _collect_test(data_root: Path, col: str, folder: str) -> List[DatasetItem]:
    partition_dir = data_root / folder
    meta_base = {"split": SPLIT_TEST, "partition": col, "folder": folder, "generator": folder}
    out: List[DatasetItem] = []

    for real_dir in sorted(partition_dir.rglob("0_real")):
        if not real_dir.is_dir():
            continue
        fake_dir = real_dir.parent / "1_fake"
        if not fake_dir.is_dir():
            continue

        subfolder = real_dir.parent.name if real_dir.parent != partition_dir else None
        meta = {**meta_base, **({"subfolder": subfolder} if subfolder else {})}
        for label, directory, source in ((0, real_dir, "real"), (1, fake_dir, "fake")):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                    out.append(
                        DatasetItem(
                            data=str(path), label=label, metadata={**meta, "source": source}
                        )
                    )

    return out


def _collect_train_val(base: Path, split: str, classes: Sequence[str]) -> List[DatasetItem]:
    out: List[DatasetItem] = []
    for cls in classes:
        real, fake = base / cls / "0_real", base / cls / "1_fake"
        for label, directory, source in ((0, real, "real"), (1, fake, "fake")):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                    out.append(
                        DatasetItem(
                            data=str(path),
                            label=label,
                            metadata={
                                "split": split,
                                "partition": "ProGAN",
                                "class": cls,
                                "source": source,
                            },
                        )
                    )
    return out


@register_dataset("cnn_detection", aliases=["foren_synths"])
class CNNDetectionDataset(BaseDataset):
    """CNN Detection train / val / test (HF sywang/CNNDetection layout).

    Parameters
    ----------
    split
        ``"train"``, ``"val"``, or ``"test"``.
    partitions
        **test:** generator names (column or folder). **val** / **train:** ImageNet class folders.
        Omit for all.
    root, cache_dir
        Root cache directory (default ``.detectzoo_data``).
    max_samples
        Optional cap on samples (see :class:`~detectzoo.datasets.base.BaseDataset`).
    """

    name = "cnn_detection"
    modality = "image"

    def __init__(
        self,
        split: str,
        partitions: Optional[Sequence[str]] = None,
        root: str | Path | None = None,
        cache_dir: str | Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        from detectzoo.datasets._download import get_cache_dir

        if split not in _SPLITS:
            raise ValueError(f"split must be one of {list(_SPLITS)}, got {split!r}")
        super().__init__(max_samples=max_samples)
        self.split = split
        self.partitions = list(partitions) if partitions is not None else None
        self.root = Path(root) if root is not None else get_cache_dir(_CACHE_KEY[split], cache_dir)
        self.cache_dir = cache_dir

    def _ensure_download(self) -> Path:
        from detectzoo.datasets._download import download_and_extract_zip, download_file

        dest = self.root.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        inner = _resolved_data_root(dest, self.split)
        if inner is not None:
            return inner

        if self.split == SPLIT_TRAIN:
            for name, url in zip(_TRAIN_7Z_NAMES, _TRAIN_7Z):
                download_file(url, dest / name, force=False)
            _extract_7z(dest / _TRAIN_7Z_NAMES[0], dest)
        elif self.split == SPLIT_VAL:
            download_and_extract_zip(_VAL_ZIP, dest, force=False)
        else:
            download_and_extract_zip(_TEST_ZIP, dest, force=False)

        inner = _resolved_data_root(dest, self.split)
        if inner is None:
            raise RuntimeError(f"CNN Detection data not found under {dest}")
        return inner

    def _load_all(self) -> List[DatasetItem]:
        base = self._ensure_download()

        if self.split == SPLIT_TEST:
            specs = (
                list(CNN_DETECTION_TEST_PARTITIONS)
                if self.partitions is None
                else [resolve_cnn_detection_partition(p) for p in self.partitions]
            )
            items: List[DatasetItem] = []
            for col, folder in specs:
                items.extend(_collect_test(base, col, folder))
            return items

        cls = _classes(self.partitions)
        items = _collect_train_val(base, self.split, cls)
        return items
