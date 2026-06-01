"""Smoke test reproducibility/audio/audio_replicate.py (run manually, not part of CI)."""
from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from detectzoo import load_dataset
from detectzoo.core.base import BaseDetector, DetectionResult

SCRIPT = REPO / "reproducibility" / "audio" / "audio_replicate.py"
spec = importlib.util.spec_from_file_location("audio_replicate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def touch_wavs(directory: Path, prefix: str, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"{prefix}{i:04d}.wav").write_bytes(b"RIFF")


def make_in_the_wild(root: Path, n: int = 120) -> Path:
    touch_wavs(root / "real", "r", n)
    touch_wavs(root / "fake", "f", n)
    return root


def make_for(root: Path, n: int = 120) -> Path:
    touch_wavs(root / "Validation" / "real", "r", n)
    touch_wavs(root / "Validation" / "fake", "f", n)
    return root


def make_asvspoof(root: Path, n: int = 120) -> Path:
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    flac_dir = root / "ASVspoof2019_LA_eval" / "flac"
    proto_dir.mkdir(parents=True)
    flac_dir.mkdir(parents=True)
    lines = []
    for i in range(n):
        utt = f"LA_E_{i:07d}"
        key = "bonafide" if i % 2 == 0 else "spoof"
        lines.append(f"LA_{i:04d} {utt} - A07 {key}")
        (flac_dir / f"{utt}.flac").write_bytes(b"")
    (proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return root


def make_deepfake(root: Path, n: int = 120) -> Path:
    audio_dir = root / "audio-data"
    audio_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        fname = f"clip_{i:04d}.wav"
        (audio_dir / fname).write_bytes(b"RIFF")
        gt = "real" if i % 2 == 0 else "fake"
        split = "test"
        rows.append({"Filename": fname, "Ground Truth": gt, "Finetuning Set": split})
    with open(root / "audio-metadata-publish.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Filename", "Ground Truth", "Finetuning Set"])
        writer.writeheader()
        writer.writerows(rows)
    return root


class MockAudioDetector(BaseDetector):
    name = "mock_audio"
    modality = "audio"

    def __init__(self, threshold: float = 0.5, device: str = "cpu", **kwargs):
        super().__init__(threshold=threshold, device=device, **kwargs)

    def predict(self, input_data):
        path = str(input_data).lower()
        score = 0.8 if ("fake" in path or "spoof" in path or "/f0" in path) else 0.2
        return DetectionResult(
            score=score,
            label="ai" if score >= self.threshold else "human",
            confidence=abs(score - 0.5) * 2,
        )

    def unload(self):
        pass


def assert_balanced(items, max_samples: int) -> None:
    n0 = sum(1 for i in items if i.label == 0)
    n1 = sum(1 for i in items if i.label == 1)
    half = max_samples // 2
    assert len(items) == max_samples, (len(items), max_samples)
    assert n0 == half and n1 == half, (n0, n1, max_samples)


def main() -> int:
    errors: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="dz_audio_smoke_"))
    print("fixtures:", tmp)

    fixtures = {
        "in_the_wild": make_in_the_wild(tmp / "in_the_wild"),
        "for": make_for(tmp / "for"),
        "asvspoof2019": make_asvspoof(tmp / "asvspoof2019"),
        "deepfake_eval_2024": make_deepfake(tmp / "deepfake_eval_2024"),
    }

    for name, path in fixtures.items():
        for cap in (4, 10, 100):
            kwargs: dict = {"path": path, "max_samples": cap}
            if name in ("in_the_wild", "for", "deepfake_eval_2024"):
                kwargs["download"] = False
            if name == "for":
                kwargs.update({"variant": "norm", "split": "val"})
            elif name == "deepfake_eval_2024":
                kwargs["split"] = "test"
            elif name == "asvspoof2019":
                kwargs.update({"track": "LA", "partition": "eval"})
            random.seed(99)
            try:
                ds = load_dataset(name, **kwargs)
                items = ds.load()
                assert_balanced(items, cap)
                n0 = sum(i.label == 0 for i in items)
                print(f"[OK] load_dataset {name} max_samples={cap} -> {len(items)} ({n0}+{len(items)-n0})")
            except Exception as exc:
                errors.append(f"load_dataset {name} n={cap}: {exc}")

    class Args:
        pass

    for ds_name, path in fixtures.items():
        a = Args()
        a.dataset = ds_name
        a.path = path
        a.no_download = True
        a.split = None
        a.max_samples = 1000
        try:
            kw = mod.build_dataset_kwargs(a)
            assert kw["max_samples"] == 1000
            if mod.DATASETS_DICT[ds_name].get("supports_download", False):
                assert kw.get("download") is False
            else:
                assert "download" not in kw
            print(f"[OK] build_dataset_kwargs {ds_name}")
        except Exception as exc:
            errors.append(f"build_dataset_kwargs {ds_name}: {exc}")

    a = Args()
    a.dataset = "asvspoof2019"
    a.path = None
    a.no_download = False
    a.split = None
    a.max_samples = 1000
    try:
        mod.build_dataset_kwargs(a)
        errors.append("asvspoof path guard did not raise")
    except ValueError:
        print("[OK] asvspoof requires --path")

    out_dir = tmp / "experiments"
    for ds_name, path in fixtures.items():
        argv = [
            "audio_replicate.py",
            "--dataset",
            ds_name,
            "--path",
            str(path),
            "--no-download",
            "--max-samples",
            "10",
            "--detectors",
            "mock_audio",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--output-dir",
            str(out_dir),
            "--save-scores",
        ]
        if ds_name == "deepfake_eval_2024":
            argv.extend(["--split", "test"])
        with patch.object(mod, "load_detector", side_effect=lambda name, **kw: MockAudioDetector(**kw)):
            with patch.object(sys, "argv", argv):
                mod.main()

        found = False
        for p in sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(p.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            if meta.get("dataset") == ds_name and meta.get("max_samples") == 10:
                res = data["results"]["mock_audio"]
                assert res["n_samples"] == 10
                assert "eer" in res and res["eer"] == res["eer"]
                assert "samples" in res
                print(f"[OK] main() {ds_name} -> {p.name} eer={res['eer']:.4f}")
                found = True
                break
        if not found:
            errors.append(f"main {ds_name}: no output json with expected meta")

    src = inspect.getsource(mod.parse_args)
    for flag in (
        "dataset",
        "detectors",
        "max-samples",
        "path",
        "split",
        "no-download",
        "device",
        "seed",
        "output-dir",
        "save-scores",
    ):
        if f"--{flag}" not in src:
            errors.append(f"missing CLI flag --{flag}")
        else:
            print(f"[OK] CLI has --{flag}")

    print("\n=== SUMMARY ===")
    if errors:
        print("FAILURES:")
        for err in errors:
            print(" -", err)
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
