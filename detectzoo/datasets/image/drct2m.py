"""DRCT-2M – diffusion-generated image detection dataset (2M scale).

Reference:
    Chen et al., "DRCT: Diffusion Reconstruction Contrastive Training towards Universal
    Detection of Diffusion Generated Images", ICML 2024.

ModelScope dataset: ``BokingChen/DRCT-2M``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_MODELSCOPE_DATASET_ID = "BokingChen/DRCT-2M"

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def _find_split_roots(start: Path) -> List[Path]:
    start = start.resolve()
    out: List[Path] = []

    for real_dir in sorted(start.rglob("0_real")):
        if real_dir.is_dir() and (real_dir.parent / "1_fake").is_dir():
            out.append(real_dir.parent)
    return sorted(set(out))


def _iter_image_paths(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            yield path


@register_dataset("drct2m", aliases=["drct-2m", "drct_2m", "DRCT-2M"])
class DRCT2MDataset(BaseDataset):
    """DRCT-2M dataset with ``0_real`` / ``1_fake`` folders.

    Parameters
    ----------
    split : str, optional
        If provided, look under ``<root>/<split>/`` first (e.g. ``train``, ``val``,
        ``test``).
        If omitted, scan the full dataset root for any ``0_real`` / ``1_fake`` pairs.
    root : str or Path, optional
        Download/extract location. When omitted, uses ``.detectzoo_data/drct2m/``.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name: str = "drct2m"
    modality: str = "image"

    def __init__(
        self,
        *,
        split: str | None = None,
        root: str | Path | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        from detectzoo.datasets._download import get_cache_dir

        super().__init__(**kwargs)
        self.split = split
        self.root = Path(root) if root is not None else get_cache_dir("drct2m", cache_dir)
        self.cache_dir = cache_dir

    def _ensure_download(self) -> Path:
        from modelscope.hub.snapshot_download import snapshot_download

        dest = self.root.resolve()
        dest.mkdir(parents=True, exist_ok=True)

        roots = _find_split_roots(dest)
        if roots:
            return dest

        snapshot_download(
            model_id=_MODELSCOPE_DATASET_ID,
            repo_type="dataset",
            local_dir=str(dest),
        )
        return dest

    def _candidate_search_roots(self, base: Path) -> Sequence[Tuple[str, Path]]:
        if self.split is None:
            return [("all", base)]
        return [(self.split, base / self.split), ("all", base)]

    def _load_all(self) -> List[DatasetItem]:
        base = self._ensure_download()

        items: List[DatasetItem] = []
        for split_name, search_root in self._candidate_search_roots(base):
            if not search_root.exists():
                continue

            for root in _find_split_roots(search_root):
                real_dir, fake_dir = root / "0_real", root / "1_fake"
                rel = str(root.relative_to(base)) if root.is_relative_to(base) else str(root)
                meta_base = {"split": split_name, "root": rel, "source_dataset": "drct2m"}
                for label, source, d in ((0, "real", real_dir), (1, "fake", fake_dir)):
                    for path in _iter_image_paths(d):
                        items.append(
                            DatasetItem(
                                data=str(path),
                                label=label,
                                metadata={**meta_base, "source": source},
                            )
                        )

            if items:
                return items

        return items
