"""Chameleon – AI-generated image detection testset (AIDE paper).

Reference:
    Yan et al., "A Sanity Check for AI-generated Image Detection",
    ICLR 2025.
    https://arxiv.org/abs/2406.19435

Download:
    https://drive.google.com/file/d/1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH/view?usp=sharing
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, List, Optional

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_GDRIVE_FILE_ID = "1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH"
_ZIP_NAME = "chameleon.zip"

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def _find_test_root(start: Path) -> Optional[Path]:
    """Return the directory containing ``0_real/`` and ``1_fake/``."""
    for candidate in sorted(start.rglob("0_real")):
        if candidate.is_dir() and (candidate.parent / "1_fake").is_dir():
            return candidate.parent
    return None


@register_dataset("chameleon", aliases=["chameleon_testset", "aide_chameleon"])
class ChameleonDataset(BaseDataset):
    """Chameleon testset from the AIDE paper (Yan et al., 2024).

    Parameters
    ----------
    root : str or Path, optional
        Directory to search for the dataset. When omitted,
        ``.detectzoo_data/chameleon/`` is used.
    cache_dir : str or Path, optional
        Root cache directory (default ``.detectzoo_data``).
    """

    name: str = "chameleon"
    modality: str = "image"

    def __init__(
        self,
        *,
        root: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root = Path(root) if root is not None else None
        self.cache_dir = cache_dir

    def _ensure_download(self) -> Path:
        from detectzoo.datasets._download import get_cache_dir

        dest = (
            self.root.resolve()
            if self.root is not None
            else get_cache_dir("chameleon", self.cache_dir)
        )
        dest.mkdir(parents=True, exist_ok=True)

        found = _find_test_root(dest)
        if found is not None:
            return found

        import gdown

        zip_path = dest / _ZIP_NAME
        if not zip_path.is_file():
            gdown.download(
                id=_GDRIVE_FILE_ID,
                output=str(zip_path),
                quiet=False,
                use_cookies=False,
            )

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink(missing_ok=True)

        found = _find_test_root(dest)
        if found is None:
            raise RuntimeError(
                f"Chameleon download completed but '0_real/' and '1_fake/' "
                f"were not found under {dest}."
            )
        return found

    def _load_all(self) -> List[DatasetItem]:
        test_root = self._ensure_download()
        real_dir = test_root / "0_real"
        fake_dir = test_root / "1_fake"
        meta = {"source_dataset": "chameleon", "split": "test"}
        items: List[DatasetItem] = []
        for label, directory, source in ((0, real_dir, "real"), (1, fake_dir, "fake")):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                    items.append(
                        DatasetItem(
                            data=str(path),
                            label=label,
                            metadata={**meta, "source": source},
                        )
                    )
        return items
