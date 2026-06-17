"""GenImage – A Million-Scale Benchmark for Detecting AI-Generated Image.

Reference:
    Zhu et al., "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image", arXiv 2023.
    https://arxiv.org/abs/2306.08571

Official site:
    https://genimage-dataset.github.io/

Hugging Face mirror:
    Dataset repo: https://huggingface.co/datasets/ENSTA-U2IS/GenImage
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_HF_DATASET_ID = "ENSTA-U2IS/GenImage"

GENIMAGE_PARTITIONS: Tuple[str, ...] = (
    "ADM",
    "BigGAN",
    "Midjourney",
    "VQDM",
    "glide",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "wukong",
)

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def _ensure_7z() -> str:
    seven = shutil.which("7z") or shutil.which("7za")
    if not seven:
        raise RuntimeError(
            "GenImage extraction needs `7z` or `7za` on PATH because the archive is split "
            "across multiple files (.z01, .z02, ...)."
        )
    return seven


def _find_split_dirs(part_dir: Path, split: str) -> Optional[Tuple[Path, Path]]:
    """Return ``(real_dir, fake_dir)`` if the split layout exists under *part_dir*."""
    for candidate in part_dir.rglob(split):
        if not candidate.is_dir():
            continue
        ai = candidate / "ai"
        nature = candidate / "nature"
        if ai.is_dir() and nature.is_dir():
            return nature, ai  # (real=nature, fake=ai)
    return None


def _extract_split_only(part_dir: Path, split: str) -> None:
    """Extract only the requested split subfolder from the split zip archive."""
    first = next(iter(sorted(part_dir.glob("*.z01"))), None) or next(
        iter(sorted(part_dir.glob("*.zip"))), None
    )
    if first is None:
        raise FileNotFoundError(f"No archive found under {part_dir}")

    archive_name = first.stem.split(".")[0]
    seven = _ensure_7z()
    subprocess.run(
        [seven, "x", str(first), f"{archive_name}/{split}/*", f"-o{part_dir}", "-y"],
        check=True,
    )


def _try_snapshot_download_hf(dest: Path, *, partition: str, force: bool) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=_HF_DATASET_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[f"{partition}/*"],
        local_dir_use_symlinks=False,
        resume_download=not force,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@register_dataset("genimage", aliases=["gen_image", "genimage_dataset"])
class GenImageDataset(BaseDataset):
    """GenImage partition with ``<split>/ai/`` (fake) and ``<split>/nature/`` (real).

    Parameters
    ----------
    partitions : sequence of str, optional
        Generator folder names from :data:`GENIMAGE_PARTITIONS`. When None, loads all
        partitions.
    split : str
        Which split to load — ``"val"`` (default) or ``"train"``.
    root : str or Path, optional
        Directory that will contain a ``genimage/`` cache folder (when omitted).
    cache_dir : str or Path, optional
        Root cache directory when *root* is ``None`` (default ``.detectzoo_data``).
    max_samples : int, optional
        Cap on samples.
    """

    name: str = "genimage"
    modality: str = "image"

    def __init__(
        self,
        *,
        partitions: Optional[Sequence[str]] = None,
        split: str = "val",
        root: str | Path | None = None,
        cache_dir: str | Path | None = None,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)

        if partitions is None:
            selected = list(GENIMAGE_PARTITIONS)
        else:
            selected = list(partitions)
            invalid = [p for p in selected if p not in GENIMAGE_PARTITIONS]
            if invalid:
                raise ValueError(
                    f"Unknown GenImage partition(s) {invalid!r}. Valid: {list(GENIMAGE_PARTITIONS)}"
                )
        if split not in ("val", "train"):
            raise ValueError(f"split must be 'val' or 'train', got {split!r}")

        self.partitions = selected
        self.split = split
        self.root = Path(root) if root is not None else None
        self.cache_dir = cache_dir

    def _ensure_download(self, partition: str) -> Tuple[Path, Path]:
        from detectzoo.datasets._download import get_cache_dir

        base = (
            self.root.resolve()
            if self.root is not None
            else get_cache_dir("genimage", self.cache_dir)
        )
        part_dir = base / partition
        part_dir.mkdir(parents=True, exist_ok=True)

        found = _find_split_dirs(part_dir, self.split)
        if found is not None:
            return found

        # Archives present — extract
        if any(part_dir.glob("*.z01")) or any(part_dir.glob("*.zip")):
            _extract_split_only(part_dir, self.split)
            found = _find_split_dirs(part_dir, self.split)
            if found is not None:
                return found

        # Nothing found and no archives — download then extract
        _try_snapshot_download_hf(base, partition=partition, force=False)
        _extract_split_only(part_dir, self.split)

        found = _find_split_dirs(part_dir, self.split)
        if found is None:
            raise FileNotFoundError(
                f"Could not find '{self.split}/ai/' and '{self.split}/nature/' under {part_dir}."
            )
        return found

    def _load_all(self) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        for partition in self.partitions:
            real_dir, fake_dir = self._ensure_download(partition)
            meta = {"partition": partition, "split": self.split, "source_dataset": "genimage"}
            for label, directory, source in ((0, real_dir, "real"), (1, fake_dir, "fake")):
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                        items.append(
                            DatasetItem(
                                data=str(path), label=label, metadata={**meta, "source": source}
                            )
                        )
        return items
