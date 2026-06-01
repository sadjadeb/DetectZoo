"""Run DetectZoo audio detectors for reproducibility experiments.

Loads one of the four built-in audio benchmarks with a balanced
``max_samples`` cap (default **1000**, about 500 bonafide + 500 spoof), runs
detectors, and saves metrics with :class:`detectzoo.benchmarks.BenchmarkEvaluator`.

Usage::

    python audio_replicate.py --dataset in_the_wild --detectors rawnet2 aasist
    python audio_replicate.py --dataset for --max-samples 1000 --save-scores
    python audio_replicate.py --dataset asvspoof2019 --path /data/ASVspoof2019/LA
    python audio_replicate.py --dataset deepfake_eval_2024 --split test

Results are written under ``experiments/`` by default.
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectzoo import load_dataset, load_detector
from detectzoo.benchmarks import BenchmarkEvaluator

# ---------------------------------------------------------------------------
# Datasets (registry names)
# ---------------------------------------------------------------------------

DATASETS_DICT: dict[str, dict[str, Any]] = {
    "asvspoof2019": {
        "loader_name": "asvspoof2019",
        "output_prefix": "asvspoof2019_la_eval",
        "dataset_kwargs": {"track": "LA", "partition": "eval"},
        "requires_path": True,
        "supports_download": False,
    },
    "for": {
        "loader_name": "for",
        "output_prefix": "for_norm_val",
        "dataset_kwargs": {"variant": "norm", "split": "val"},
        "supports_download": True,
    },
    "in_the_wild": {
        "loader_name": "in_the_wild",
        "output_prefix": "in_the_wild",
        "dataset_kwargs": {},
        "supports_download": True,
    },
    "deepfake_eval_2024": {
        "loader_name": "deepfake_eval_2024",
        "output_prefix": "deepfake_eval_2024_test",
        "dataset_kwargs": {"split": "test"},
        "supports_download": True,
    },
}

# ---------------------------------------------------------------------------
# Detectors (registry names — paper / DetectZoo audio set)
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_NAMES: List[str] = [
    "rawnet2",
    "aasist",
    "rawgat_st",
    "res_tssdnet",
    "samo",
    "ast_asvspoof",
    "anti_deepfake_wav2vec",
    "anti_deepfake_hubert",
    "anti_deepfake_xlsr2b",
    "xlsr_sls",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS_DICT),
        help="Audio benchmark to evaluate.",
    )
    p.add_argument(
        "--detectors",
        nargs="+",
        default=DEFAULT_DETECTOR_NAMES,
        help="Detector registry names (default: all built-in audio detectors).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Balanced sample cap: ~half bonafide + half spoof (default: 1000).",
    )
    p.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Local dataset root (required for ASVspoof 2019; optional override).",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override split for FoR or Deepfake-Eval-2024 (e.g. val, test).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Disable auto-download; require a local --path where needed.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for detectors (default: cuda).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for balanced subsampling shuffle (default: 42).",
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
        help="Store per-sample scores in the output JSON.",
    )
    return p.parse_args()


def build_output_path(dataset_name: str, max_samples: int, output_dir: Path) -> Path:
    info = DATASETS_DICT[dataset_name]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"audio__{info['output_prefix']}__n{max_samples}__{ts}.json"


def build_dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    info = DATASETS_DICT[args.dataset]
    kwargs = dict(info["dataset_kwargs"])
    kwargs["max_samples"] = args.max_samples

    if args.path is not None:
        kwargs["path"] = args.path
    if args.no_download and info.get("supports_download", False):
        kwargs["download"] = False
    if args.split is not None:
        if args.dataset == "for":
            kwargs["split"] = args.split
        elif args.dataset == "deepfake_eval_2024":
            kwargs["split"] = args.split
        else:
            raise ValueError(
                f"--split is only supported for 'for' and 'deepfake_eval_2024', "
                f"not {args.dataset!r}"
            )

    if info.get("requires_path") and args.path is None:
        raise ValueError(
            f"Dataset {args.dataset!r} requires a local extract — pass --path "
            "(ASVspoof 2019 has no public auto-download)."
        )

    return kwargs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_kwargs = build_dataset_kwargs(args)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return

    info = DATASETS_DICT[args.dataset]
    out_path = build_output_path(args.dataset, args.max_samples, args.output_dir)

    print(
        f"Loading {args.dataset}: loader={info['loader_name']}, "
        f"max_samples={args.max_samples}, kwargs={ds_kwargs}"
    )

    random.seed(args.seed)
    try:
        dataset = load_dataset(info["loader_name"], **ds_kwargs)
    except Exception:
        print("[FAIL] Could not build dataset.")
        traceback.print_exc()
        return

    try:
        items = dataset.load()
    except Exception:
        print("[FAIL] Could not load dataset items.")
        traceback.print_exc()
        return

    n = len(items)
    n_h = sum(1 for it in items if it.label == 0)
    n_ai = sum(1 for it in items if it.label == 1)
    print(f"  loaded {n} items (bonafide={n_h}, spoof={n_ai})")
    if n_h == 0 or n_ai == 0:
        print(
            "  [WARN] Single-class eval set — EER / ROC-AUC will be undefined (NaN)."
        )

    print(f"\nLoading {len(args.detectors)} detector(s) on {args.device} …")
    detectors: List[Any] = []
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

    meta: dict[str, Any] = {
        "modality": "audio",
        "dataset": args.dataset,
        "loader_name": info["loader_name"],
        "n_samples": n,
        "n_bonafide": n_h,
        "n_spoof": n_ai,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "device": args.device,
        "detectors_requested": list(args.detectors),
        "dataset_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in ds_kwargs.items()},
    }

    evaluator = BenchmarkEvaluator(dataset)
    try:
        results = evaluator.run_and_save(
            detectors,
            out_path,
            save_scores=args.save_scores,
            meta=meta,
            incremental=True,
            unload_between=True,
        )
        print(f"  results -> {out_path}")
        print(results)
    except Exception:
        print("  [ERROR] evaluation failed")
        traceback.print_exc()


if __name__ == "__main__":
    main()
