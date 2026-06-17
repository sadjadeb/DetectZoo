"""Univ-FD diffusion evaluation datasets.

Reference:
    Ojha et al., "Towards Universal Fake Image Detectors that Generalize Across
    Generative Models", CVPR 2023.
    https://arxiv.org/abs/2302.10174

Upstream diffusion test data (per-domain archives):
    https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets._download import extract_tar_archive, get_cache_dir
from detectzoo.datasets.base import BaseDataset, DatasetItem

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

UNIVFD_DIFFUSION_PARTITIONS: Tuple[str, ...] = (
    "guided",
    "ldm_200",
    "ldm_200_cfg",
    "ldm_100",
    "glide_100_27",
    "glide_50_27",
    "glide_100_10",
    "dalle",
)

_UNIVFD_TEST_DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-"
)


def _collect_images(directory: Path, label: int, partition: str) -> List[DatasetItem]:
    meta = {
        "source": "real" if label == 0 else "fake",
        "partition": partition,
        "source_dataset": "univfd_diffusion",
    }
    return [
        DatasetItem(data=str(p), label=label, metadata=meta)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]


@register_dataset("univfd_diffusion", aliases=["univfd_dataset", "univfd"])
class UnivFDDataset(BaseDataset):
    """Diffusion-model evaluation dataset for Univ-FD.

    Parameters
    ----------
    partitions : sequence of str, optional
        Diffusion partition keys from :data:`UNIVFD_DIFFUSION_PARTITIONS`. When
        None or ``["all"]``, loads all eight partitions.
    real_dir, fake_dir : str or Path, optional
        Explicit image directories. Both must be provided together or both omitted.
    root : str or Path, optional
        Cache root for downloads. Defaults to ``.detectzoo_data/univfd_diffusion/``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name: str = "univfd_diffusion"
    modality: str = "image"

    def __init__(
        self,
        *,
        partitions: Optional[Sequence[str]] = None,
        real_dir: Optional[str | Path] = None,
        fake_dir: Optional[str | Path] = None,
        root: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if (real_dir is None) ^ (fake_dir is None):
            raise ValueError("real_dir and fake_dir must both be set, or both omitted.")

        self._manual_real: Optional[Path] = (
            Path(real_dir).expanduser().resolve() if real_dir else None
        )
        self._manual_fake: Optional[Path] = (
            Path(fake_dir).expanduser().resolve() if fake_dir else None
        )

        if partitions is None or list(partitions) == ["all"]:
            self._keys = UNIVFD_DIFFUSION_PARTITIONS
        else:
            selected = list(partitions)
            invalid = [p for p in selected if p not in UNIVFD_DIFFUSION_PARTITIONS]
            if invalid:
                raise ValueError(
                    f"partitions must be None, ['all'], or values from "
                    f"{sorted(UNIVFD_DIFFUSION_PARTITIONS)}, got {invalid!r}"
                )
            self._keys = tuple(selected)
        self.partitions = list(self._keys)

        self.root: Path = (
            Path(root).expanduser().resolve()
            if root is not None
            else get_cache_dir("univfd_diffusion", cache_dir)
        )

    def _ensure_download(self, key: str) -> Tuple[Path, Path]:
        """Return ``(real_dir, fake_dir)`` for *key*, downloading and extracting if needed."""
        real_dir = self.root / key / "0_real"
        fake_dir = self.root / key / "1_fake"

        if not real_dir.is_dir():
            import gdown

            archive = self.root / f"{key}.tar.gz"
            if not archive.is_file():
                gdown.download_folder(
                    url=_UNIVFD_TEST_DRIVE_FOLDER,
                    output=str(self.root),
                    quiet=False,
                    use_cookies=False,
                )
            extract_tar_archive(archive, self.root)

        return real_dir, fake_dir

    def _load_all(self) -> List[DatasetItem]:
        if self._manual_real is not None:
            meta_key = self.partitions[0] if len(self.partitions) == 1 else "manual"
            return _collect_images(
                self._manual_real, label=0, partition=meta_key
            ) + _collect_images(self._manual_fake, label=1, partition=meta_key)

        items: List[DatasetItem] = []
        for key in self._keys:
            real_dir, fake_dir = self._ensure_download(key)
            items += _collect_images(real_dir, label=0, partition=key)
            items += _collect_images(fake_dir, label=1, partition=key)
        return items
