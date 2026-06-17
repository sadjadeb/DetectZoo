"""Run DetectZoo image detectors for reproducibility experiments.

Example:
    python image_replicate.py \
        --dataset self_synthesis \
        --partitions AttGAN BEGAN \
        --detectors cnnspot patchcraft univfd
"""

import argparse
from pathlib import Path

import torch

from detectzoo import load_dataset, load_detector
from detectzoo.benchmarks.evaluator import BenchmarkEvaluator

DATASETS_DICT = {
    "self_synthesis": {
        "loader_name": "self_synthesis",
        "output_prefix": "self_synthesis",
        "dataset_kwargs": {},
    },
    "aigcdetect": {
        "loader_name": "aigcdetect",
        "output_prefix": "aigcdetect",
        "dataset_kwargs": {},
    },
    "cnn_detection": {
        "loader_name": "cnn_detection",
        "output_prefix": "cnn_detection_test",
        "dataset_kwargs": {"split": "test"},
    },
    "genimage": {
        "loader_name": "genimage",
        "output_prefix": "genimage",
        "dataset_kwargs": {},
    },
    "univfd_diffusion": {
        "loader_name": "univfd_diffusion",
        "output_prefix": "univfd_diffusion",
        "dataset_kwargs": {},
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS_DICT))
    parser.add_argument("--partitions", nargs="+", required=True)
    parser.add_argument("--detectors", nargs="+", required=True)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser.parse_args()


def build_output_path(dataset_name, partitions, output_dir):
    dataset_info = DATASETS_DICT[dataset_name]
    part_str = "_".join(partitions)
    return output_dir / f"{dataset_info['output_prefix']}_{part_str}.json"


def main():
    args = parse_args()
    dataset_info = DATASETS_DICT[args.dataset]

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_kwargs = dict(dataset_info["dataset_kwargs"])
    dataset_kwargs["partitions"] = args.partitions

    ds = load_dataset(
        dataset_info["loader_name"],
        **dataset_kwargs,
    )

    items = ds.load()

    detectors = []
    for name in args.detectors:
        det = load_detector(name, device=device)

        if name.lower() in {"mib", "manifold_bias"}:
            real_images = [item.data for item in items if item.label == 0][:1000]
            det.calibrate(real_images, k=1.0, prompt="a photograph")

        detectors.append(det)

    output_path = build_output_path(args.dataset, args.partitions, args.output_dir)

    results = BenchmarkEvaluator(ds).run_and_save(
        detectors,
        output_path=output_path,
        save_scores=args.save_scores,
        unload_between=True,
    )

    print(f"Saved to: {output_path}")
    print(results)


if __name__ == "__main__":
    main()
