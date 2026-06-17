"""Replicate GECScore paper baselines on the authors' released *normal* data.

Lists JSON files in ``NLP2CT/GECScore`` under ``data/normal_data/`` (see
`normal_data on GitHub <https://github.com/NLP2CT/GECScore/tree/main/data/normal_data>`_),
downloads each, and runs detectors with
:class:`detectzoo.benchmarks.BenchmarkEvaluator`.  Filenames follow::

    <source>.<generator_model>.normal.test_data.json

e.g. ``xsum.GPT-4o.normal.test_data.json`` (corpus = *source*, e.g. xsum
or writing).  Each row has ``text`` and ``label`` where ``label`` is
``human`` or ``llm``.

Usage::

    python gecscore_replicate.py
    python gecscore_replicate.py --sources xsum
    python gecscore_replicate.py --models GPT-4o gpt3.5
    python gecscore_replicate.py --sources xsum --models Claude-3.5-Sonnet
    python gecscore_replicate.py --max-samples 20
    python gecscore_replicate.py --data-url <url>   # single file (legacy)

Results: one JSON per (source, model) file under ``experiments/``.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

from detectzoo import load_detector
from detectzoo.benchmarks import BenchmarkEvaluator
from detectzoo.datasets.base import BaseDataset, DatasetItem

# ---------------------------------------------------------------------------
# Detectors (registry names)
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_NAMES: List[str] = [
    "log_likelihood",
    "rank",
    "log_rank",
    "lrr",
    "npr",
    "detectgpt",
    "fast_detectgpt",
    "revise_detect",
    "roberta_base",
    "roberta_large",
    "gecscore",
]

# ---------------------------------------------------------------------------
# GECScore GitHub layout (normal_data)
# ---------------------------------------------------------------------------

GECSCORE_OWNER_REPO = "NLP2CT/GECScore"
GECSCORE_API_BASE = f"https://api.github.com/repos/{GECSCORE_OWNER_REPO}/contents/data/normal_data"
GECSCORE_RAW_BASE = (
    f"https://raw.githubusercontent.com/{GECSCORE_OWNER_REPO}/refs/heads/main/data/normal_data"
)

NORMAL_DATA_SUFFIX = ".normal.test_data.json"

# Corpora in published normal_data (for argparse choices; new files on GitHub still work
# if you omit --sources to allow all, or we discover dynamically only).
GECSCORE_SOURCES = ("xsum", "writing")

# Generators that appear in published filenames (optional filter examples).
GECSCORE_MODELS = (
    "Claude-3.5-Sonnet",
    "GPT-4o",
    "Google-PaLM",
    "Llama-3-70B-T",
    "gpt3.5",
)


@dataclass(frozen=True)
class GECScoreFile:
    """One ``<source>.<model>.normal.test_data.json`` on GitHub."""

    source: str
    model: str
    filename: str

    @property
    def url(self) -> str:
        return f"{GECSCORE_RAW_BASE}/{self.filename}"

    @property
    def cache_path(self) -> Path:
        return Path("data") / "gecscore" / "normal_data" / self.filename

    @property
    def slug(self) -> str:
        return f"{self.source}__{self.model}"


# ---------------------------------------------------------------------------
# Discovery + download
# ---------------------------------------------------------------------------


def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "DetectZoo-GECScore-replicate"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "DetectZoo-GECScore-replicate"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def parse_normal_data_filename(name: str) -> Optional[Tuple[str, str]]:
    """Parse ``<source>.<model>.normal.test_data.json`` -> (source, model)."""
    if not name.endswith(NORMAL_DATA_SUFFIX):
        return None
    core = name[: -len(NORMAL_DATA_SUFFIX)]
    if "." not in core:
        return None
    source, model = core.split(".", 1)
    if not source or not model:
        return None
    return source, model


def discover_files(
    sources: Optional[Set[str]] = None,
    models: Optional[Set[str]] = None,
) -> List[GECScoreFile]:
    """List *normal* test JSONs from the GitHub API, optionally filtered."""
    try:
        entries = _http_get_json(GECSCORE_API_BASE)
    except Exception as exc:
        print(f"  [FAIL] could not list {GECSCORE_API_BASE}: {exc}")
        return []

    files: List[GECScoreFile] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = entry.get("name") or ""
        parsed = parse_normal_data_filename(name)
        if not parsed:
            continue
        source, model = parsed
        if sources is not None and source not in sources:
            continue
        if models is not None and model not in models:
            continue
        files.append(GECScoreFile(source=source, model=model, filename=name))

    files.sort(key=lambda f: (f.source.lower(), f.model.lower(), f.filename))
    return files


def download_if_needed(f: GECScoreFile) -> Path:
    path = f.cache_path
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {f.url}")
    path.write_bytes(_http_get_bytes(f.url))
    return path


def download_url_to_path(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    dest.write_bytes(_http_get_bytes(url))
    return dest


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _label_to_binary(label: Any) -> int:
    s = str(label).strip().lower()
    if s in ("human", "0", "real"):
        return 0
    if s in ("llm", "machine", "ai", "fake", "1"):
        return 1
    raise ValueError(f"Unsupported label: {label!r} (expected 'human' or 'llm')")


class GECScoreJsonDataset(BaseDataset):
    """Load GECScore ``data/normal_data/*.json`` rows with ``text`` and ``label``."""

    modality = "text"

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "gecscore",
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.path = Path(path)
        self.name = name

    def _load_all(self) -> List[DatasetItem]:
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise TypeError("Expected a JSON array of {text, label} objects")
        items: List[DatasetItem] = []
        for idx, row in enumerate(raw):
            if not isinstance(row, dict):
                continue
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            label = _label_to_binary(row.get("label"))
            items.append(
                DatasetItem(
                    data=text,
                    label=label,
                    metadata={
                        "index": idx,
                        "file": str(self.path),
                    },
                )
            )
        return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_") or "unnamed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Only these corpora (e.g. xsum writing). Default: all files on GitHub.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Only these generator names as in the filename (e.g. GPT-4o gpt3.5). "
            "Default: all files on GitHub."
        ),
    )
    p.add_argument(
        "--data-url",
        type=str,
        default=None,
        help=(
            "If set, evaluate this single URL only and skip API discovery. "
            "Cache path defaults to data/gecscore/normal_data/ from the URL name."
        ),
    )
    p.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="With --data-url, where to save the file (default: under data/gecscore/normal_data/).",
    )
    p.add_argument(
        "--max-samples", type=int, default=None, help="Cap samples per file for quick debug runs."
    )
    p.add_argument("--device", type=str, default="cuda", help="Device for detectors.")
    p.add_argument(
        "--detectors", nargs="+", default=DEFAULT_DETECTOR_NAMES, help="Detector registry names."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory for per-file benchmark JSONs.",
    )
    p.add_argument(
        "--save-scores",
        action="store_true",
        help="Store per-sample scores in each output JSON (like imbd).",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="List discovered files and exit (no download or evaluation).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- build file list: single-URL or discovery ---
    files: List[GECScoreFile] = []
    if args.data_url:
        from urllib.parse import unquote, urlparse

        path_part = unquote(urlparse(args.data_url).path)
        name = Path(path_part).name
        if not name.endswith(NORMAL_DATA_SUFFIX) and not name.endswith(".json"):
            print("[FAIL] --data-url must end with a .json filename.")
            return
        parsed = parse_normal_data_filename(name)
        if not parsed:
            print(
                "[WARN] filename does not match *.<model>.normal.test_data.json; "
                "using generic slug from filename."
            )
            source, model = "custom", _safe_slug(name.rsplit(".", 1)[0])
        else:
            source, model = parsed
        files = [GECScoreFile(source=source, model=model, filename=name)]
    else:
        src_set: Optional[Set[str]] = set(args.sources) if args.sources else None
        mod_set: Optional[Set[str]] = set(args.models) if args.models else None
        print(
            f"Discovering GECScore normal_data files"
            f"{f', sources={args.sources}' if args.sources else ''}"
            f"{f', models={args.models}' if args.models else ''}"
        )
        files = discover_files(sources=src_set, models=mod_set)
        if not files:
            print("No matching GECScore files found. Aborting.")
            return
        if args.list_only:
            print(f"Found {len(files)} file(s):")
            for f in files:
                print(f"  - {f.filename}  ({f.slug})")
            return
        print(f"Found {len(files)} data file(s).")
        for f in files:
            print(f"  - {f.slug}")

    print(f"\nLoading {len(args.detectors)} detector(s) on {args.device} …")
    detectors: List = []
    for name in args.detectors:
        try:
            detectors.append(load_detector(name, device=args.device))
            print(f"  [OK] {name}")
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    if not detectors:
        print("No detectors loaded; aborting.")
        return

    for f in files:
        print(f"\n{'=' * 72}\nGECScore file: {f.slug} ({f.filename})\n{'=' * 72}")
        if args.data_url:
            dest = args.cache_path or (Path("data") / "gecscore" / "normal_data" / f.filename)
            try:
                local_path = download_url_to_path(args.data_url, dest)
            except Exception:
                print(f"  [SKIP] download failed: {args.data_url!r}")
                traceback.print_exc()
                continue
        else:
            try:
                local_path = download_if_needed(f)
            except Exception:
                print(f"  [SKIP] failed to download {f.url}")
                traceback.print_exc()
                continue

        try:
            dataset = GECScoreJsonDataset(local_path, name=f.slug, max_samples=args.max_samples)
            items = dataset.load()
        except Exception:
            print(f"  [SKIP] failed to build dataset for {local_path}")
            traceback.print_exc()
            continue

        n = len(items)
        n_h = sum(1 for it in items if it.label == 0)
        n_ai = sum(1 for it in items if it.label == 1)
        print(f"  loaded {n} items (human={n_h}, ai={n_ai})")

        meta: dict = {
            "dataset": "gecscore",
            "source": f.source,
            "model": f.model,
            "data_file": f.filename,
            "n_samples": n,
            "max_samples": args.max_samples,
            "device": args.device,
            "detectors_requested": list(args.detectors),
        }

        out_slug = _safe_slug(f"{f.source}_{f.model}")
        out_path = args.output_dir / f"gecscore__{out_slug}__{ts}.json"
        evaluator = BenchmarkEvaluator(dataset)
        try:
            evaluator.run_and_save(
                detectors, out_path, save_scores=args.save_scores, meta=meta, incremental=True
            )
            print(f"  results -> {out_path}")
        except Exception:
            print(f"  [ERROR] evaluation failed for {f.slug}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
