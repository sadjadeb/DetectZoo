"""Replicate the OOD paper baselines on RAID.

Runs DetectZoo detectors on the *labeled* RAID re-split
(``Shengkun/Raid_split``) **test** split, defaulting to 1000 samples, and
saves metrics with :class:`detectzoo.benchmarks.BenchmarkEvaluator`.


Usage::

    python ood_replicate.py
    python ood_replicate.py --max-samples 100 --device cuda
    python ood_replicate.py --detectors lrr fast_detectgpt
    python ood_replicate.py --attacks none
    python ood_replicate.py --save-scores

Results are written under ``experiments/`` by default.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

from detectzoo import load_dataset, load_detector
from detectzoo.benchmarks import BenchmarkEvaluator

# ---------------------------------------------------------------------------
# Detectors (registry names)
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_NAMES: List[str] = [
    "lrr",
    "npr",
    "detectgpt",
    "dna_gpt",
    "fast_detectgpt",
    "binoculars",
    "glimpse",
    "radar",
    "ghostbuster",
    "biscope",
    "detective",
    "dsvdd",
    "hrn",
    "energy_detector",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--max-samples", type=int, default=10000, help="Max RAID rows to load (default: 10000)."
    )
    p.add_argument("--split", type=str, default="test", help="RAID split (default: test).")
    p.add_argument(
        "--device", type=str, default="cuda", help="Device for detectors (default: cuda)."
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default="Shengkun/Raid_split",
        help="HuggingFace dataset id for RAID split.",
    )
    p.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="Pass through to RAIDDataset (e.g. `none` for non-adversarial only).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Pass through to RAIDDataset: restrict to these generators (e.g. `human` `chatgpt`).",
    )
    p.add_argument(
        "--detectors",
        nargs="+",
        default=DEFAULT_DETECTOR_NAMES,
        help="Detector registry names to run (default: OOD-paper set).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory for result JSON files.",
    )
    p.add_argument(
        "--save-scores", action="store_true", help="Store per-sample scores in the output JSON."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"ood__{args.split}__n{args.max_samples}__{ts}.json"

    ds_kwargs: dict = {
        "split": args.split,
        "hf_repo": args.hf_repo,
        "max_samples": args.max_samples,
    }
    if args.attacks is not None:
        ds_kwargs["attacks"] = args.attacks
    if args.models is not None:
        ds_kwargs["models"] = args.models

    print(
        f"Loading RAID: split={args.split}, max_samples={args.max_samples}, hf_repo={args.hf_repo}"
    )
    try:
        dataset = load_dataset("raid", **ds_kwargs)
    except Exception:
        print("[FAIL] Could not build RAID dataset.")
        traceback.print_exc()
        return

    items = dataset.load()
    n = len(items)
    n_h = sum(1 for it in items if it.label == 0)
    n_ai = sum(1 for it in items if it.label == 1)
    print(f"  loaded {n} items (human={n_h}, ai={n_ai})")

    _RAID_CHECKPOINT = "model_raid.pth"
    _DETECTOR_KWARGS: dict = {
        "detective": {"checkpoint": _RAID_CHECKPOINT},
        "dsvdd": {"checkpoint_path": None},
        "hrn": {"detective_checkpoint": _RAID_CHECKPOINT},
        "energy_detector": {"detective_checkpoint": _RAID_CHECKPOINT},
    }

    print(f"\nLoading {len(args.detectors)} detector(s) on {args.device} …")
    detectors: List = []
    for name in args.detectors:
        try:
            extra = _DETECTOR_KWARGS.get(name, {})
            detectors.append(load_detector(name, device=args.device, **extra))
            print(f"  [OK] {name}")
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    if not detectors:
        print("No detectors loaded; aborting.")
        return

    meta: dict = {
        "dataset": "raid",
        "split": args.split,
        "hf_repo": args.hf_repo,
        "n_samples": n,
        "max_samples": args.max_samples,
        "attacks_filter": list(args.attacks) if args.attacks else None,
        "models_filter": list(args.models) if args.models else None,
        "device": args.device,
        "detectors_requested": list(args.detectors),
    }

    evaluator = BenchmarkEvaluator(dataset)
    try:
        evaluator.run_and_save(
            detectors, out_path, save_scores=args.save_scores, meta=meta, incremental=True
        )
        print(f"  results -> {out_path}")
    except Exception:
        print("  [ERROR] evaluation failed")
        traceback.print_exc()


if __name__ == "__main__":
    main()
