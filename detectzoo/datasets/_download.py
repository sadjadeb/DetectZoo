"""Download and cache utilities for built-in datasets."""

from __future__ import annotations

import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path(".detectzoo_data")


def get_cache_dir(
    dataset_name: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return (and create) the cache directory for *dataset_name*."""
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path = root / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded / total_size * 100)
    print(f"\r  {pct:5.1f}%  ({downloaded:,} / {total_size:,} bytes)", end="", flush=True)


def download_file(
    url: str,
    dest: Path,
    *,
    force: bool = False,
) -> Path:
    """Download a single file, skipping if already cached."""
    if dest.exists() and not force:
        logger.info("Using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress bar
    return dest


def download_and_extract_zip(
    url: str,
    dest_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download a ZIP archive, extract it, and cache the result.

    A ``.download_complete`` marker file prevents re-downloading on
    subsequent calls.
    """
    marker = dest_dir / ".download_complete"
    if marker.exists() and not force:
        logger.info("Using cached %s", dest_dir)
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)

    with tempfile.NamedTemporaryFile(
        suffix=".zip",
        delete=False,
        dir=dest_dir,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook)
        print()  # newline after progress bar
        logger.info("Extracting to %s", dest_dir)
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest_dir)
    finally:
        tmp_path.unlink(missing_ok=True)
    marker.touch()
    return dest_dir


def extract_tar_archive(archive: Path, dest_dir: Path) -> None:
    """Extract a tar archive (``.tar``, ``.tar.gz``, ``.tgz``, ``.tar.bz2``, …) into *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest_dir)


def download_and_extract_tar(
    url: str,
    dest_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download a tar archive (gzip/bzip2/xz optional), extract it, and cache the result.

    A ``.download_complete`` marker file prevents re-downloading on
    subsequent calls.
    """
    marker = dest_dir / ".download_complete"
    if marker.exists() and not force:
        logger.info("Using cached %s", dest_dir)
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)

    with tempfile.NamedTemporaryFile(
        suffix=".tar",
        delete=False,
        dir=dest_dir,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook)
        print()  # newline after progress bar
        logger.info("Extracting to %s", dest_dir)
        extract_tar_archive(tmp_path, dest_dir)
    finally:
        tmp_path.unlink(missing_ok=True)
    marker.touch()
    return dest_dir
