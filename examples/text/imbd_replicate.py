"""Replicate ImBD baselines by running DetectZoo detectors on ImBD's released data.

For every (task, generator_model, source_corpus) combination hosted in the
Jiaqi-Chen-00/ImBD GitHub repo under ``data/<task>/<model>/<source>_<task>_<model>.raw_data.json``,
we treat the ``original`` field as human text (label=0) and the ``rewritten``
field as AI text (label=1), and evaluate every detector listed in
DetectZoo with :class:`detectzoo.benchmarks.BenchmarkEvaluator`.

Usage::

    python imbd_replicate.py                     # run everything
    python imbd_replicate.py --tasks rewrite     # only the rewrite task
    python imbd_replicate.py --models gpt-4o gpt-3.5-turbo
    python imbd_replicate.py --sources xsum
    python imbd_replicate.py --max-samples 20    # quick debug run

Results are written under ``experiments/`` (one JSON per task/model/source).
"""

from __future__ import annotations

import argparse
import json
import traceback
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

from detectzoo import load_detector
from detectzoo.benchmarks import BenchmarkEvaluator
from detectzoo.datasets.base import BaseDataset, DatasetItem

# ---------------------------------------------------------------------------
# Detectors (registry names)
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_NAMES: List[str] = [
    "log_likelihood",
    "entropy",
    "rank",
    "log_rank",
    "lrr",
    "dna_gpt",
    "npr",
    "detectgpt",
    "fast_detectgpt",
    "imbd",
    "remodetect",
]

# ---------------------------------------------------------------------------
# ImBD data layout on GitHub
# ---------------------------------------------------------------------------

IMBD_RAW_BASE = "https://raw.githubusercontent.com/Jiaqi-Chen-00/ImBD/main/data"
IMBD_API_BASE = "https://api.github.com/repos/Jiaqi-Chen-00/ImBD/contents/data"

TASKS = ["polish", "rewrite", "expand", "generation"]
MODELS = ["Deepseek-7b", "Llama-3-8B", "Mistral-7B", "Qwen2-7B", "gpt-3.5-turbo", "gpt-4o"]


@dataclass(frozen=True)
class ImBDFile:
    task: str
    model: str
    source: str  # e.g. "xsum", "writing", "pubmed", "squad"

    @property
    def filename(self) -> str:
        return f"{self.source}_{self.task}_{self.model}.raw_data.json"

    @property
    def url(self) -> str:
        return f"{IMBD_RAW_BASE}/{self.task}/{self.model}/{self.filename}"

    @property
    def cache_path(self) -> Path:
        return Path("data") / "imbd" / self.task / self.model / self.filename

    @property
    def slug(self) -> str:
        return f"{self.task}__{self.model}__{self.source}"


# ---------------------------------------------------------------------------
# Discovery + download
# ---------------------------------------------------------------------------


def _http_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "DetectZoo-ImBD-replicate"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def discover_files(
    tasks: Iterable[str],
    models: Iterable[str],
    sources: Optional[Iterable[str]] = None,
) -> List[ImBDFile]:
    """Ask the GitHub API which data files exist for each (task, model)."""
    wanted_sources = set(sources) if sources else None
    files: List[ImBDFile] = []
    for task in tasks:
        for model in models:
            api_url = f"{IMBD_API_BASE}/{task}/{model}"
            try:
                entries = _http_get_json(api_url)
            except Exception as exc:  # 404 when combo doesn't exist, rate limit, etc.
                print(f"  [SKIP] {task}/{model}: {exc}")
                continue
            for entry in entries:
                if entry.get("type") != "file":
                    continue
                name = entry["name"]
                if not name.endswith(".raw_data.json"):
                    continue
                # "<source>_<task>_<model>.raw_data.json"
                suffix = f"_{task}_{model}.raw_data.json"
                if not name.endswith(suffix):
                    continue
                source = name[: -len(suffix)]
                if wanted_sources and source not in wanted_sources:
                    continue
                files.append(ImBDFile(task=task, model=model, source=source))
    return files


def download_if_needed(file: ImBDFile) -> Path:
    path = file.cache_path
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {file.url}")
    with urllib.request.urlopen(file.url, timeout=120) as resp:
        payload = resp.read()
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ImBDRawDataset(BaseDataset):
    """Turn an ImBD ``*.raw_data.json`` file into a DetectZoo dataset.

    The JSON has two parallel lists: ``original`` (human, label=0) and
    ``rewritten`` (AI, label=1).  Items are interleaved so that truncation
    via ``max_samples`` still yields a roughly balanced subset.
    """

    modality = "text"

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "imbd",
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.path = Path(path)
        self.name = name

    def _load_all(self) -> List[DatasetItem]:
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        originals = data.get("original", []) or []
        rewrittens = data.get("rewritten", []) or data.get("sampled", []) or []

        items: List[DatasetItem] = []
        for idx, (human, ai) in enumerate(zip(originals, rewrittens)):
            if isinstance(human, str) and human.strip():
                items.append(
                    DatasetItem(
                        data=human,
                        label=0,
                        metadata={"source": "human", "index": idx, "file": str(self.path)},
                    )
                )
            if isinstance(ai, str) and ai.strip():
                items.append(
                    DatasetItem(
                        data=ai,
                        label=1,
                        metadata={"source": "ai", "index": idx, "file": str(self.path)},
                    )
                )
        return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="ImBD task folders to evaluate (default: all).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Generator model folders to evaluate (default: all).",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Restrict to these source corpora (e.g. xsum writing pubmed squad).",
    )
    p.add_argument(
        "--detectors",
        nargs="+",
        default=DEFAULT_DETECTOR_NAMES,
        help="Detector registry names to run.",
    )
    p.add_argument(
        "--device", type=str, default="cuda", help="Device for detectors (default: cuda)."
    )
    p.add_argument(
        "--max-samples", type=int, default=None, help="Cap samples per file for quick debug runs."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory for per-file benchmark JSONs.",
    )
    p.add_argument(
        "--save-scores", action="store_true", help="Store per-sample scores in each output JSON."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"Discovering ImBD data files for tasks={args.tasks}, models={args.models}"
        + (f", sources={args.sources}" if args.sources else "")
    )
    files = discover_files(args.tasks, args.models, args.sources)
    if not files:
        print("No matching ImBD data files found. Aborting.")
        return
    print(f"Found {len(files)} data file(s).")
    for f in files:
        print(f"  - {f.slug}")

    print(f"\nLoading {len(args.detectors)} detector(s) on {args.device} \u2026")
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

    for file in files:
        print(f"\n{'=' * 72}\nImBD file: {file.slug}\n{'=' * 72}")
        try:
            local_path = download_if_needed(file)
        except Exception:
            print(f"  [SKIP] failed to download {file.url}")
            traceback.print_exc()
            continue

        try:
            dataset = ImBDRawDataset(local_path, name=file.slug, max_samples=args.max_samples)
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
            "dataset": "imbd",
            "task": file.task,
            "model": file.model,
            "source": file.source,
            "data_file": file.filename,
            "n_samples": n,
            "max_samples": args.max_samples,
            "device": args.device,
            "detectors_requested": list(args.detectors),
        }

        out_path = args.output_dir / f"imbd__{file.slug}__{ts}.json"
        evaluator = BenchmarkEvaluator(dataset)
        try:
            evaluator.run_and_save(
                detectors, out_path, save_scores=args.save_scores, meta=meta, incremental=True
            )
            print(f"  results -> {out_path}")
        except Exception:
            print(f"  [ERROR] evaluation failed for {file.slug}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
