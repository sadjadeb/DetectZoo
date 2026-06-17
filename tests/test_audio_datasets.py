"""Smoke tests for audio dataset loaders (fixture-based, no large downloads)."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from detectzoo import list_datasets, load_dataset
from detectzoo.datasets.audio.asvspoof2019 import (
    ASVspoof2019Dataset,
    _key_to_label,
    _parse_protocol_line,
)
from detectzoo.datasets.audio.deepfake_eval_2024 import (
    DeepfakeEval2024Dataset,
    _load_from_metadata as _load_dfeval,
)
from detectzoo.datasets.audio.for_dataset import FoRDataset, _coerce_variant
from detectzoo.datasets.audio.in_the_wild import (
    InTheWildDataset,
    _label_from_str,
    _load_release,
)


def test_audio_registry_lists_four_datasets() -> None:
    names = sorted(list_datasets("audio"))
    assert names == ["asvspoof2019", "deepfake_eval_2024", "for", "in_the_wild"]


def test_asvspoof_protocol_parsing() -> None:
    parsed = _parse_protocol_line("LA_0009 LA_E_2834763 - A07 spoof")
    assert parsed is not None
    assert _key_to_label(parsed[3]) == 1
    assert _key_to_label("bonafide") == 0


def test_asvspoof_minimal_tree(tmp_path: Path) -> None:
    track = "LA"
    root = tmp_path / track
    proto = root / f"ASVspoof2019_{track}_cm_protocols"
    flac = root / f"ASVspoof2019_{track}_eval" / "flac"
    proto.mkdir(parents=True)
    flac.mkdir(parents=True)
    proto_file = proto / f"ASVspoof2019.{track}.cm.eval.trl.txt"
    proto_file.write_text(
        "LA_0009 LA_E_0001 - A07 bonafide\nLA_0010 LA_E_0002 - A07 spoof\n",
        encoding="utf-8",
    )
    (flac / "LA_E_0001.flac").write_bytes(b"a")
    (flac / "LA_E_0002.flac").write_bytes(b"b")

    ds = ASVspoof2019Dataset(path=root, track=track, partition="eval")
    items = ds.load()
    assert len(items) == 2
    assert {it.label for it in items} == {0, 1}


def test_for_real_fake_layout(tmp_path: Path) -> None:
    root = tmp_path / "for-norm"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    (root / "real" / "a.wav").write_bytes(b"0")
    (root / "fake" / "b.wav").write_bytes(b"1")

    assert _coerce_variant("for-norm") == "norm"
    ds = FoRDataset(path=root, variant="norm", split="all", download=False)
    items = ds.load()
    assert len(items) == 2
    assert sum(1 for it in items if it.label == 0) == 1


def test_in_the_wild_bona_fide_label() -> None:
    assert _label_from_str("bona-fide") == 0
    assert _label_from_str("Bona-Fide") == 0
    assert _label_from_str("fake") == 1


def test_in_the_wild_metadata_csv(tmp_path: Path) -> None:
    root = tmp_path / "release_in_the_wild"
    root.mkdir()
    with open(root / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "label"])
        writer.writeheader()
        writer.writerow({"file": "r.wav", "label": "bona-fide"})
        writer.writerow({"file": "f.wav", "label": "fake"})
    (root / "r.wav").write_bytes(b"0")
    (root / "f.wav").write_bytes(b"1")

    items = _load_release(root, skip_missing=True)
    assert len(items) == 2
    assert sum(1 for it in items if it.label == 0) == 1
    assert sum(1 for it in items if it.label == 1) == 1
    ds = InTheWildDataset(path=root, download=False)
    assert len(ds.load()) == 2


def test_in_the_wild_class_folders(tmp_path: Path) -> None:
    root = tmp_path / "release_in_the_wild"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    (root / "real" / "a.wav").write_bytes(b"0")
    (root / "fake" / "b.wav").write_bytes(b"1")

    items = _load_release(root, skip_missing=True)
    assert len(items) == 2


def test_deepfake_eval_2024_metadata(tmp_path: Path) -> None:
    root = tmp_path / "dfeval"
    (root / "audio-data").mkdir(parents=True)
    with open(root / "audio-metadata-publish.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["Filename", "Ground Truth", "Finetuning Set"]
        )
        writer.writeheader()
        writer.writerow(
            {"Filename": "r.mp3", "Ground Truth": "real", "Finetuning Set": "train"}
        )
        writer.writerow(
            {"Filename": "f.mp3", "Ground Truth": "fake", "Finetuning Set": "test"}
        )
    (root / "audio-data" / "r.mp3").write_bytes(b"0")
    (root / "audio-data" / "f.mp3").write_bytes(b"1")

    assert len(_load_dfeval(root, split="all", skip_missing=False)) == 2
    ds = DeepfakeEval2024Dataset(path=root, download=False)
    assert len(ds.load()) == 2


def test_load_dataset_aliases(tmp_path: Path) -> None:
    dfeval_root = tmp_path / "dfeval"
    (dfeval_root / "audio-data").mkdir(parents=True)
    (dfeval_root / "audio-metadata-publish.csv").write_text(
        "Filename,Ground Truth,Finetuning Set\n"
        "x.mp3,real,train\n",
        encoding="utf-8",
    )
    (dfeval_root / "audio-data" / "x.mp3").write_bytes(b"1")

    itw_root = tmp_path / "release_in_the_wild"
    itw_root.mkdir()
    (itw_root / "real").mkdir()
    (itw_root / "real" / "a.wav").write_bytes(b"0")

    assert load_dataset("deepfake_eval_2024", path=dfeval_root, download=False).name == (
        "deepfake_eval_2024"
    )
    assert load_dataset("itw", path=itw_root, download=False).name == "in_the_wild"
