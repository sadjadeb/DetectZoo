"""GenBuster-Bench — AI-generated video detection benchmark from BusterX.

Reference:
    Wen et al., "BusterX: MLLM-Powered AI-Generated Video Forgery Detection
    and Explanation", arXiv 2025.  https://arxiv.org/abs/2505.12620

GitHub: https://github.com/l8cv/BusterX
HuggingFace: ``l8cv/GenBuster-Bench`` (public, one ~2.2 GB zip, 3,150 videos)

Layout (inside ``GenBuster-Bench.zip``)::

    GenBuster-Bench/id/real/*.mp4              # 1,000 real videos
    GenBuster-Bench/id/fake/<generator>/*.mp4  # in-distribution generators
    GenBuster-Bench/ood/fake/<generator>/*.mp4 # unseen generators
    GenBuster-Bench/wild/fake/<generator>/*.mp4 # recent commercial generators

Only ``id`` ships real videos; they are reused as the real class when
evaluating ``ood`` or ``wild``.  There is no metadata file: labels and
generator names come from the folder layout.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_HF_REPO = "l8cv/GenBuster-Bench"
# Pinned so the benchmark contents stay reproducible.
_HF_REVISION = "68e41ab55726013f1c7455980180114302196c23"
_ZIP_NAME = "GenBuster-Bench.zip"
_TOP_DIR = "GenBuster-Bench"

GENBUSTER_SETTINGS: Tuple[str, ...] = ("id", "ood", "wild")

GENBUSTER_GENERATORS: Dict[str, Tuple[str, ...]] = {
    "id": ("cogvideox", "easyanimate", "hunyuanvideo", "ltxvideo"),
    "ood": ("gen3", "jimeng", "kling", "luma", "pika", "sora", "vidu", "wanx"),
    "wild": (
        "gen4.5",
        "hailuo2.3",
        "kling2.5_turbo",
        "kling2.6",
        "pixverse_v5",
        "seedance1.5_pro",
        "seedance2.0",
        "sora2",
        "veo3.1",
        "wan2.5",
        "wan2.6",
    ),
}


def _find_bench_dir(base: Path) -> Optional[Path]:
    """Return the ``GenBuster-Bench`` directory under (or at) *base*, if present."""
    for candidate in (base, base / _TOP_DIR):
        if (candidate / "id" / "real").is_dir():
            return candidate
    return None


def _download_and_extract(base: Path) -> None:
    """Fetch the benchmark zip from the Hub, extract it into *base*, drop the zip.

    ``hf_hub_download`` retries and resumes interrupted downloads, which matters
    for a single 2.2 GB file.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    logger.info("GenBuster-Bench: downloading %s …", _ZIP_NAME)
    zip_path = Path(
        hf_hub_download(
            _HF_REPO,
            _ZIP_NAME,
            repo_type="dataset",
            revision=_HF_REVISION,
            local_dir=str(base),
        )
    )
    logger.info("GenBuster-Bench: extracting to %s", base)
    # Extract into a scratch folder and move it into place at the end, so an
    # interrupted extraction never looks like a complete dataset.
    staging = base / ".extracting"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)
    os.replace(staging / _TOP_DIR, base / _TOP_DIR)
    shutil.rmtree(staging)
    zip_path.unlink()


def _list_videos(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


@register_dataset("genbuster_bench", aliases=["genbuster", "genbuster-bench"])
class GenBusterBenchDataset(BaseDataset):
    """GenBuster-Bench real-vs-fake video detection.

    Parameters
    ----------
    settings : sequence of str, optional
        Fake-video settings from :data:`GENBUSTER_SETTINGS`: ``"id"``
        (in-distribution), ``"ood"`` (unseen generators) and ``"wild"``
        (recent commercial generators).  ``None`` loads all three.
    generators : sequence of str, optional
        Restrict fakes to these generators (see
        :data:`GENBUSTER_GENERATORS`), e.g. ``["sora", "kling"]``.
    include_real : bool
        Add the 1,000 real videos (``id/real``), the only reals in the
        benchmark (default ``True``).
    max_samples : int, optional
        Cap on samples, balanced across real/fake.
    seed : int
        Seed for the ``max_samples`` selection (default 0).
    root : str or Path, optional
        Directory holding (or receiving) the extracted ``GenBuster-Bench/``
        folder.  Defaults to ``<cache_dir>/genbuster_bench``.
    cache_dir : str or Path, optional
        Root cache directory when *root* is ``None`` (default
        ``.detectzoo_data``).
    """

    name: str = "genbuster_bench"
    modality: str = "video"

    def __init__(
        self,
        *,
        settings: Optional[Sequence[str]] = None,
        generators: Optional[Sequence[str]] = None,
        include_real: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
        root: str | Path | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_samples=max_samples, **kwargs)

        selected = list(GENBUSTER_SETTINGS) if settings is None else list(settings)
        invalid = [s for s in selected if s not in GENBUSTER_SETTINGS]
        if invalid:
            raise ValueError(
                f"Unknown GenBuster-Bench setting(s) {invalid!r}. Valid: {list(GENBUSTER_SETTINGS)}"
            )

        if generators is not None:
            allowed = {g for s in selected for g in GENBUSTER_GENERATORS[s]}
            invalid = [g for g in generators if g not in allowed]
            if invalid:
                raise ValueError(
                    f"Unknown GenBuster-Bench generator(s) {invalid!r} for settings "
                    f"{selected}. Valid: {sorted(allowed)}"
                )

        self.settings = selected
        self.generators = list(generators) if generators is not None else None
        self.include_real = include_real
        self.seed = seed
        self.root = Path(root) if root is not None else None
        self.cache_dir = cache_dir

    def _ensure_download(self) -> Path:
        from detectzoo.datasets._download import get_cache_dir

        if self.root is not None:
            base = self.root.expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
        else:
            base = get_cache_dir("genbuster_bench", self.cache_dir)

        bench = _find_bench_dir(base)
        if bench is None:
            _download_and_extract(base)
            bench = _find_bench_dir(base)
        if bench is None:
            raise FileNotFoundError(
                f"GenBuster-Bench: expected '{_TOP_DIR}/id/real/' under {base}."
            )
        return bench

    def _load_all(self) -> List[DatasetItem]:
        bench = self._ensure_download()
        items: List[DatasetItem] = []

        if self.include_real:
            for path in _list_videos(bench / "id" / "real"):
                items.append(
                    DatasetItem(
                        data=str(path),
                        label=0,
                        metadata={
                            "setting": "id",
                            "generator": "real",
                            "source": "real",
                            "source_dataset": "genbuster_bench",
                        },
                    )
                )

        for setting in self.settings:
            for generator in GENBUSTER_GENERATORS[setting]:
                if self.generators is not None and generator not in self.generators:
                    continue
                for path in _list_videos(bench / setting / "fake" / generator):
                    items.append(
                        DatasetItem(
                            data=str(path),
                            label=1,
                            metadata={
                                "setting": setting,
                                "generator": generator,
                                "source": "fake",
                                "source_dataset": "genbuster_bench",
                            },
                        )
                    )

        if self.max_samples is not None:
            # Seeded shuffle so the balanced pick in load() spans all generators
            # instead of the first ones in folder order.
            random.Random(self.seed).shuffle(items)
        return items
