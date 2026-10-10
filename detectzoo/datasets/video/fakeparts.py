"""FakeParts / FakePartsBench — full and partial AI-generated video forgeries.

Reference:
    Brison et al., "FakeParts: A New Family of AI-Generated Video Forgeries",
    arXiv 2025.  https://arxiv.org/abs/2508.21052

GitHub: https://github.com/hi-paris/FakeParts
HuggingFace: ``hi-paris/FakeParts`` (public, ~126 GB, test split only)

Hub layout::

    metadata.jsonl                                   # one row per video
    <Task>/<Method>/<real_videos|fake_videos>/shard-NNNNNN.parquet

Each parquet shard stores one mp4 per row group (columns ``filename``,
``size_bytes``, ``bytes``), and ``metadata.jsonl``'s ``file_name`` is the
shard directory joined with ``filename``.  This loader reads the row-group
statistics from each shard's footer to locate videos, then streams only the
selected videos with HTTP range requests and writes them to
``<cache>/fakeparts/videos/<file_name>``.  With ``max_samples`` only those
videos are fetched, never whole shards.  Videos already on disk are reused, so
a selection fetched once (e.g. on a login node) loads offline afterwards.
"""

from __future__ import annotations

import json
import os
import posixpath
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_HF_REPO = "hi-paris/FakeParts"
# Pinned so the cached footers and seeded selections stay reproducible.
_HF_REVISION = "8e49f5791d99cf7e1200785942cb3e55f4435acc"
_METADATA_FILE = "metadata.jsonl"

FAKEPARTS_TASKS: Tuple[str, ...] = (
    "T2V",
    "TI2V",
    "Inpainting",
    "Outpainting",
    "Interpolation",
    "Extrapolation",
    "Change_of_style",
    "Faceswap",
    "Real",
)

FAKEPARTS_METHODS: Dict[str, Tuple[str, ...]] = {
    "T2V": (
        "CogVideoX",
        "HunyuanVideo",
        "LTXVideo",
        "Mochi1",
        "Open-Sora",
        "Wan_21",
        "allegroai",
        "sora-1080p",
        "sora-720p",
        "veo2-720p",
    ),
    "TI2V": ("Open-Sora-768px", "WAN"),
    "Inpainting": ("DiffuEraser", "Propainter", "ROVI/Inpainting_MP4", "ROVI/REAL_JPEG_MP4"),
    "Outpainting": ("akira", "akira2"),
    "Interpolation": ("Framer",),
    "Extrapolation": ("Cosmos-Predict2",),
    "Change_of_style": ("AnyV2V", "RAVE"),
    "Faceswap": ("Insightface",),
    "Real": ("TenKReal",),
}


def _fetch_metadata(base: Path) -> Path:
    """Return the local ``metadata.jsonl``, downloading it if missing."""
    path = base / _METADATA_FILE
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            _HF_REPO,
            _METADATA_FILE,
            repo_type="dataset",
            revision=_HF_REVISION,
            local_dir=str(base),
        )
    return path


def _read_metadata(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class _ShardReader:
    """Locate and stream single videos out of FakeParts parquet shards.

    Shards found under the dataset root (a local copy of the Hub repo) are
    read from disk; otherwise they are read from the Hub with range requests.
    Remote footers are cached under ``<base>/footers/``.
    """

    def __init__(self, base: Path, num_workers: int) -> None:
        self.base = base
        self.num_workers = max(int(num_workers), 1)
        self._fs = None
        self._shards: Optional[List[str]] = None

    # ---- file access ----

    @property
    def fs(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem()
        return self._fs

    def _remote(self, shard: str) -> str:
        return f"datasets/{_HF_REPO}@{_HF_REVISION}/{shard}"

    def _open(self, shard: str):
        local = self.base / shard
        if local.is_file():
            return open(local, "rb")
        # Exact range reads: one request per video, no read-ahead.
        return self.fs.open(self._remote(shard), cache_type="none")

    def _list_shards(self) -> List[str]:
        if self._shards is None:
            local = sorted(
                p.relative_to(self.base).as_posix() for p in self.base.rglob("shard-*.parquet")
            )
            if local:
                self._shards = local
            else:
                from huggingface_hub import HfApi

                info = HfApi().dataset_info(_HF_REPO, revision=_HF_REVISION)
                self._shards = sorted(
                    s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")
                )
        return self._shards

    def _footer(self, shard: str):
        import pyarrow.parquet as pq

        local = self.base / shard
        if local.is_file():
            return pq.read_metadata(local)
        cached = self.base / "footers" / f"{shard}.footer"
        if cached.is_file():
            return pq.read_metadata(cached)
        with self.fs.open(self._remote(shard)) as fh:
            md = pq.ParquetFile(fh).metadata
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        md.write_metadata_file(str(tmp))
        os.replace(tmp, cached)
        return md

    # ---- index + fetch ----

    def _index(self, shard_dirs: set[str]) -> Dict[str, Tuple[str, Any, int]]:
        """Map ``file_name`` → ``(shard, footer, row_group)`` for *shard_dirs*."""
        shards = [s for s in self._list_shards() if posixpath.dirname(s) in shard_dirs]
        with ThreadPoolExecutor(self.num_workers) as ex:
            footers = list(ex.map(self._footer, shards))
        index: Dict[str, Tuple[str, Any, int]] = {}
        for shard, md in zip(shards, footers):
            directory = posixpath.dirname(shard)
            for rg in range(md.num_row_groups):
                stats = md.row_group(rg).column(0).statistics
                index[f"{directory}/{stats.min}"] = (shard, md, rg)
        return index

    def _fetch_one(self, file_name: str, dest: Path, loc: Tuple[str, Any, int]) -> None:
        import pyarrow.parquet as pq

        shard, md, rg = loc
        with self._open(shard) as fh:
            table = pq.ParquetFile(fh, metadata=md).read_row_group(rg, columns=["bytes"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(table.column(0)[0].as_py())
        os.replace(tmp, dest)

    def fetch(self, file_names: Sequence[str], video_dir: Path) -> None:
        """Write each of *file_names* to ``video_dir/<file_name>`` unless present."""
        todo = [f for f in file_names if not (video_dir / f).is_file()]
        if not todo:
            return
        logger.info("FakeParts: fetching %d video(s) …", len(todo))
        index = self._index({posixpath.dirname(f) for f in todo})
        missing = [f for f in todo if f not in index]
        if missing:
            raise FileNotFoundError(
                f"FakeParts: {len(missing)} video(s) not found in any shard, e.g. {missing[:3]}"
            )
        with ThreadPoolExecutor(self.num_workers) as ex:
            list(ex.map(lambda f: self._fetch_one(f, video_dir / f, index[f]), todo))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@register_dataset("fakeparts", aliases=["fake_parts", "fakepartsbench"])
class FakePartsDataset(BaseDataset):
    """FakePartsBench real-vs-fake video detection (test split).

    Parameters
    ----------
    tasks : sequence of str, optional
        Tasks from :data:`FAKEPARTS_TASKS` (``"T2V"``, ``"Inpainting"``, …).
        ``None`` loads every task.
    methods : sequence of str, optional
        Restrict to these methods (see :data:`FAKEPARTS_METHODS`), e.g.
        ``["sora-720p", "veo2-720p"]``.  Methods from tasks not in *tasks* are
        an error.  The ``TenKReal`` real videos added by *include_real* are
        not affected.
    include_real : bool
        Add the ``Real`` task (TenKReal, 10k real videos) so fake-only tasks
        still yield a two-class set (default ``True``).  ``Inpainting`` also
        ships its own real source clips (``ROVI/REAL_JPEG_MP4``).
    max_samples : int, optional
        Pick this many videos, balanced across real/fake, and fetch only
        those.
    seed : int
        Seed for the ``max_samples`` selection (default 0), so the same
        videos are picked on every run.
    root : str or Path, optional
        Dataset directory (holds ``metadata.jsonl`` and ``videos/``; may also
        hold a local copy of the parquet shards).  Defaults to
        ``<cache_dir>/fakeparts``.
    cache_dir : str or Path, optional
        Root cache directory when *root* is ``None`` (default
        ``.detectzoo_data``).
    num_workers : int
        Parallel downloads (default 8).
    """

    name: str = "fakeparts"
    modality: str = "video"

    def __init__(
        self,
        *,
        tasks: Optional[Sequence[str]] = None,
        methods: Optional[Sequence[str]] = None,
        include_real: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
        root: str | Path | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)

        selected = list(FAKEPARTS_TASKS) if tasks is None else list(tasks)
        invalid = [t for t in selected if t not in FAKEPARTS_TASKS]
        if invalid:
            raise ValueError(
                f"Unknown FakeParts task(s) {invalid!r}. Valid: {list(FAKEPARTS_TASKS)}"
            )
        if include_real and "Real" not in selected:
            selected.append("Real")

        if methods is not None:
            allowed = {m for t in selected for m in FAKEPARTS_METHODS[t]}
            invalid = [m for m in methods if m not in allowed]
            if invalid:
                raise ValueError(
                    f"Unknown FakeParts method(s) {invalid!r} for tasks {selected}. "
                    f"Valid: {sorted(allowed)}"
                )

        self.tasks = selected
        self.methods = list(methods) if methods is not None else None
        self.seed = seed
        self.root = Path(root) if root is not None else None
        self.cache_dir = cache_dir
        self.num_workers = num_workers

    def _base_dir(self) -> Path:
        if self.root is not None:
            base = self.root.expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            return base
        from detectzoo.datasets._download import get_cache_dir

        return get_cache_dir("fakeparts", self.cache_dir)

    def _keep(self, row: dict) -> bool:
        if row["task"] not in self.tasks:
            return False
        if self.methods is None or row["task"] == "Real":
            return True
        return row["method"] in self.methods

    def _load_all(self) -> List[DatasetItem]:
        base = self._base_dir()
        rows = [r for r in _read_metadata(_fetch_metadata(base)) if self._keep(r)]
        video_dir = base / "videos"

        items = [
            DatasetItem(
                data=str(video_dir / r["file_name"]),
                label=int(r["label"]),
                metadata={
                    "task": r["task"],
                    "method": r["method"],
                    "name": r["name"],
                    "file_name": r["file_name"],
                    "source": r["label_name"],
                    "source_dataset": "fakeparts",
                },
            )
            for r in rows
        ]
        if self.max_samples is not None:
            # Seeded shuffle, then the base class's balanced pick, so only the
            # chosen videos are fetched.  load() re-applies the same pick,
            # which keeps exactly these items.
            random.Random(self.seed).shuffle(items)
            items = self._balance_and_truncate(items, self.max_samples)

        _ShardReader(base, self.num_workers).fetch(
            [it.metadata["file_name"] for it in items], video_dir
        )
        return items
