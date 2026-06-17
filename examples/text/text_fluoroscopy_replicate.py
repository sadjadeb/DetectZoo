"""Replicate Text-Fluoroscopy paper baselines on the authors' released data.

Loads JSON arrays from
``Fish-and-Sheep/Text-Fluoroscopy`` under ``dataset/processed_data/``
(each row: ``text``, ``result`` — ``0`` human, ``1`` machine) and runs
detectors with :class:`detectzoo.benchmarks.BenchmarkEvaluator`.

You must pass exact filenames as released in the repo, e.g.
``gpt4-Xsum-gpt3.5.json``, ``HC3_en_test.json``.

Usage::

    python text_fluoroscopy_replicate.py --files gpt4-Xsum-gpt3.5.json
    python text_fluoroscopy_replicate.py --files gpt4-Xsum-gpt3.5.json writing_gpt-3.5-turbo.json
    python text_fluoroscopy_replicate.py --files gpt4-Xsum-gpt3.5.json --max-samples 50

Default data URL is the repo file ``dataset/processed_data/``
(see ``DEFAULT_DATA_URL`` in the script).
Results are written under ``experiments/``.
"""

from __future__ import annotations

import argparse
import json
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, List

from detectzoo import load_detector
from detectzoo.benchmarks import BenchmarkEvaluator
from detectzoo.datasets.base import BaseDataset, DatasetItem

# ---------------------------------------------------------------------------
# Detectors (registry names)
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_NAMES: List[str] = [
    "roberta_base",
    "roberta_large",
    "radar",
    "coco",
    "log_likelihood",
    "entropy",
    "log_rank",
    "lrr",
    "dna_gpt",
    "npr",
    "detectgpt",
    "fast_detectgpt",
    "text_fluoroscopy",
]

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/Fish-and-Sheep/Text-Fluoroscopy/"
    "refs/heads/main/dataset/processed_data"
)

CACHE_DIR = Path("data") / "text_fluoroscopy" / "processed_data"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DetectZoo-Text-Fluoroscopy-replicate"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def raw_url(filename: str) -> str:
    return f"{DEFAULT_DATA_URL}/{filename}"


def download_if_needed(filename: str, *, force: bool = False) -> Path:
    if Path(filename).name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Invalid filename (no paths): {filename!r}")
    if not filename.endswith(".json"):
        raise ValueError(f"Expected a .json filename, got {filename!r}")
    path = CACHE_DIR / filename
    if force and path.exists():
        path.unlink()
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = raw_url(filename)
    print(f"  downloading {url}")
    path.write_bytes(_http_get_bytes(url))
    return path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _result_to_label(result: Any) -> int:
    if isinstance(result, int):
        return 1 if result else 0
    s = str(result).strip()
    if s in ("0", "1"):
        return int(s)
    raise ValueError(f"Unsupported label value: {result!r}")


class TextFluoroscopyJsonDataset(BaseDataset):
    """Load a Text-Fluoroscopy ``processed_data`` JSON array of ``text``/``result`` rows."""

    modality = "text"

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "text_fluoroscopy",
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        self.path = Path(path)
        self.name = name

    def _load_all(self) -> List[DatasetItem]:
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise TypeError("Expected a JSON array of {text, result} objects")
        items: List[DatasetItem] = []
        for idx, row in enumerate(raw):
            if not isinstance(row, dict):
                continue
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            label = _result_to_label(row["result"])
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="NAME",
        help="Exact processed_data/*.json basename(s) from the Text-Fluoroscopy repo.",
    )
    p.add_argument(
        "--force-download", action="store_true", help="Re-download even if cache exists."
    )
    p.add_argument(
        "--max-samples", type=int, default=None, help="Cap items per file (default: all)."
    )
    p.add_argument("--device", type=str, default="cuda", help="Device for detectors.")
    p.add_argument(
        "--detectors", nargs="+", default=DEFAULT_DETECTOR_NAMES, help="Detector registry names."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory for result JSON files.",
    )
    p.add_argument(
        "--save-scores",
        action="store_true",
        help="Store per-sample labels and scores in the output JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    for filename in args.files:
        stem = Path(filename).stem
        print(f"\n{'=' * 72}\nText-Fluoroscopy file: {filename}\n{'=' * 72}")

        try:
            local_path = download_if_needed(filename, force=args.force_download)
        except ValueError as exc:
            print(f"  [SKIP] {exc}")
            continue
        except Exception:
            print(f"  [SKIP] download failed for {raw_url(filename)}")
            traceback.print_exc()
            continue

        try:
            dataset = TextFluoroscopyJsonDataset(
                local_path,
                name=stem,
                max_samples=args.max_samples,
            )
            items = dataset.load()
        except Exception:
            print(f"  [SKIP] could not build dataset for {local_path}")
            traceback.print_exc()
            continue

        n = len(items)
        n_h = sum(1 for it in items if it.label == 0)
        n_ai = sum(1 for it in items if it.label == 1)
        print(f"  loaded {n} items (human={n_h}, ai={n_ai})")

        meta: dict = {
            "dataset": "text_fluoroscopy",
            "data_file": filename,
            "data_url": raw_url(filename),
            "local_path": str(local_path.resolve()),
            "n_samples": n,
            "max_samples": args.max_samples,
            "device": args.device,
            "detectors_requested": list(args.detectors),
        }

        out_path = args.output_dir / f"text_fluoroscopy__{stem}__{ts}.json"
        evaluator = BenchmarkEvaluator(dataset)
        try:
            evaluator.run_and_save(
                detectors, out_path, save_scores=args.save_scores, meta=meta, incremental=True
            )
            print(f"  results -> {out_path}")
        except Exception:
            print(f"  [ERROR] evaluation failed for {filename}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
