"""Smoke tests for video dataset loaders (fixture-based, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detectzoo import list_datasets, load_dataset
from detectzoo.datasets.video.fakeparts import FakePartsDataset
from detectzoo.datasets.video.genbuster import GenBusterBenchDataset

# (file_name, task, method, label) — a miniature FakeParts with two fake tasks,
# the TenKReal reals, and Inpainting's own real clips.
_ROWS = [
    ("T2V/sora-720p/fake_videos/sora_0.mp4", "T2V", "sora-720p", 1),
    ("T2V/sora-720p/fake_videos/sora_1.mp4", "T2V", "sora-720p", 1),
    ("T2V/veo2-720p/fake_videos/veo_0.mp4", "T2V", "veo2-720p", 1),
    ("Inpainting/Propainter/fake_videos/pp_0.mp4", "Inpainting", "Propainter", 1),
    ("Inpainting/ROVI/REAL_JPEG_MP4/real_videos/rovi_0.mp4", "Inpainting", "ROVI/REAL_JPEG_MP4", 0),
    ("Real/TenKReal/real_videos/1.mp4", "Real", "TenKReal", 0),
    ("Real/TenKReal/real_videos/2.mp4", "Real", "TenKReal", 0),
    ("Real/TenKReal/real_videos/3.mp4", "Real", "TenKReal", 0),
]


def _video_bytes(file_name: str) -> bytes:
    return f"fake-mp4:{file_name}".encode()


@pytest.fixture
def fakeparts_root(tmp_path: Path) -> Path:
    """Write ``metadata.jsonl`` plus one-video-per-row-group parquet shards."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    root = tmp_path / "fakeparts"
    root.mkdir()
    with open(root / "metadata.jsonl", "w", encoding="utf-8") as fh:
        for file_name, task, method, label in _ROWS:
            row = {
                "file_name": file_name,
                "task": task,
                "method": method,
                "name": Path(file_name).stem,
                "label": label,
                "label_name": "fake" if label else "real",
                "split": "test",
            }
            fh.write(json.dumps(row) + "\n")

    by_dir: dict[str, list[str]] = {}
    for file_name, *_ in _ROWS:
        by_dir.setdefault(str(Path(file_name).parent), []).append(file_name)
    for directory, files in by_dir.items():
        shard = root / directory / "shard-000000.parquet"
        shard.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "filename": [Path(f).name for f in files],
                "size_bytes": [len(_video_bytes(f)) for f in files],
                "bytes": [_video_bytes(f) for f in files],
            }
        )
        pq.write_table(table, shard, row_group_size=1)
    return root


def test_video_registry_lists_datasets() -> None:
    assert list_datasets("video") == ["fakeparts", "genbuster_bench"]


def test_all_tasks_extracts_videos(fakeparts_root: Path) -> None:
    items = FakePartsDataset(root=fakeparts_root).load()
    assert len(items) == len(_ROWS)
    for it in items:
        assert Path(it.data).read_bytes() == _video_bytes(it.metadata["file_name"])
        assert it.label == (0 if it.metadata["source"] == "real" else 1)
        assert it.metadata["source_dataset"] == "fakeparts"


def test_task_filter_adds_tenkreal(fakeparts_root: Path) -> None:
    items = FakePartsDataset(tasks=["T2V"], root=fakeparts_root).load()
    assert {it.metadata["task"] for it in items} == {"T2V", "Real"}
    assert sum(it.label == 0 for it in items) == 3


def test_include_real_false(fakeparts_root: Path) -> None:
    items = FakePartsDataset(tasks=["T2V"], include_real=False, root=fakeparts_root).load()
    assert {it.label for it in items} == {1}


def test_inpainting_brings_its_own_reals(fakeparts_root: Path) -> None:
    items = FakePartsDataset(tasks=["Inpainting"], include_real=False, root=fakeparts_root).load()
    assert sorted(it.metadata["method"] for it in items) == ["Propainter", "ROVI/REAL_JPEG_MP4"]


def test_method_filter_keeps_reals(fakeparts_root: Path) -> None:
    items = FakePartsDataset(tasks=["T2V"], methods=["veo2-720p"], root=fakeparts_root).load()
    fakes = [it for it in items if it.label == 1]
    assert [it.metadata["method"] for it in fakes] == ["veo2-720p"]
    assert sum(it.label == 0 for it in items) == 3


def test_invalid_task_and_method(fakeparts_root: Path) -> None:
    with pytest.raises(ValueError, match="task"):
        FakePartsDataset(tasks=["Deepfake"], root=fakeparts_root)
    with pytest.raises(ValueError, match="method"):
        FakePartsDataset(tasks=["T2V"], methods=["Framer"], root=fakeparts_root)


def test_max_samples_fetches_only_selection(fakeparts_root: Path) -> None:
    items = FakePartsDataset(max_samples=4, seed=1, root=fakeparts_root).load()
    assert len(items) == 4
    assert sorted(it.label for it in items) == [0, 0, 1, 1]
    fetched = {p for p in (fakeparts_root / "videos").rglob("*.mp4")}
    assert fetched == {Path(it.data) for it in items}


def test_max_samples_is_deterministic(fakeparts_root: Path) -> None:
    def pick(seed: int) -> list[str]:
        ds = FakePartsDataset(max_samples=4, seed=seed, root=fakeparts_root)
        return sorted(it.data for it in ds.load())

    assert pick(3) == pick(3)
    assert len({tuple(pick(s)) for s in range(6)}) > 1


def test_existing_videos_are_reused(fakeparts_root: Path) -> None:
    FakePartsDataset(tasks=["T2V"], root=fakeparts_root).load()
    for shard in fakeparts_root.rglob("shard-*.parquet"):
        shard.unlink()  # no source left: a second load must not need it
    items = FakePartsDataset(tasks=["T2V"], root=fakeparts_root).load()
    assert len(items) == 6


def test_load_dataset_alias(fakeparts_root: Path) -> None:
    ds = load_dataset("fake_parts", tasks=["T2V"], include_real=False, root=fakeparts_root)
    assert isinstance(ds, FakePartsDataset)
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# GenBuster-Bench
# ---------------------------------------------------------------------------

# Paths relative to the ``GenBuster-Bench/`` folder.
_GENBUSTER_FILES = [
    "id/real/r0.mp4",
    "id/real/r1.mp4",
    "id/real/r2.mp4",
    "id/fake/cogvideox/c0.mp4",
    "id/fake/ltxvideo/l0.mp4",
    "ood/fake/sora/s0.mp4",
    "ood/fake/sora/s1.mp4",
    "ood/fake/kling/k0.mp4",
    "wild/fake/veo3.1/v0.mp4",
]


@pytest.fixture
def genbuster_root(tmp_path: Path) -> Path:
    root = tmp_path / "genbuster"
    for rel in _GENBUSTER_FILES:
        path = root / "GenBuster-Bench" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
    return root


def test_genbuster_all_settings(genbuster_root: Path) -> None:
    items = GenBusterBenchDataset(root=genbuster_root).load()
    assert len(items) == len(_GENBUSTER_FILES)
    for it in items:
        rel = Path(it.data).relative_to(genbuster_root / "GenBuster-Bench").as_posix()
        assert it.metadata["setting"] == rel.split("/")[0]
        assert it.label == (0 if "/real/" in rel else 1)
        assert it.metadata["generator"] == ("real" if it.label == 0 else rel.split("/")[2])


def test_genbuster_root_may_point_at_bench_folder(genbuster_root: Path) -> None:
    items = GenBusterBenchDataset(root=genbuster_root / "GenBuster-Bench").load()
    assert len(items) == len(_GENBUSTER_FILES)


def test_genbuster_ood_reuses_id_reals(genbuster_root: Path) -> None:
    items = GenBusterBenchDataset(settings=["ood"], root=genbuster_root).load()
    assert sum(it.label == 0 for it in items) == 3
    assert {it.metadata["setting"] for it in items if it.label == 1} == {"ood"}


def test_genbuster_generator_filter_and_no_reals(genbuster_root: Path) -> None:
    items = GenBusterBenchDataset(
        settings=["ood"], generators=["sora"], include_real=False, root=genbuster_root
    ).load()
    assert [it.metadata["generator"] for it in items] == ["sora", "sora"]


def test_genbuster_invalid_setting_and_generator(genbuster_root: Path) -> None:
    with pytest.raises(ValueError, match="setting"):
        GenBusterBenchDataset(settings=["test"], root=genbuster_root)
    with pytest.raises(ValueError, match="generator"):
        GenBusterBenchDataset(settings=["wild"], generators=["sora"], root=genbuster_root)


def test_genbuster_max_samples_balanced_and_seeded(genbuster_root: Path) -> None:
    def pick(seed: int) -> list[str]:
        ds = GenBusterBenchDataset(max_samples=4, seed=seed, root=genbuster_root)
        items = ds.load()
        assert sorted(it.label for it in items) == [0, 0, 1, 1]
        return sorted(it.data for it in items)

    assert pick(5) == pick(5)
    assert len({tuple(pick(s)) for s in range(8)}) > 1


def test_genbuster_downloads_and_extracts_zip(tmp_path: Path, monkeypatch) -> None:
    import zipfile

    import huggingface_hub

    def fake_hf_hub_download(repo_id, filename, *, local_dir, **kwargs):
        zip_path = Path(local_dir) / filename
        with zipfile.ZipFile(zip_path, "w") as zf:
            for rel in _GENBUSTER_FILES:
                zf.writestr(f"GenBuster-Bench/{rel}", b"mp4")
        return str(zip_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    root = tmp_path / "fresh"
    items = GenBusterBenchDataset(root=root).load()
    assert len(items) == len(_GENBUSTER_FILES)
    assert sorted(p.name for p in root.iterdir()) == ["GenBuster-Bench"]  # zip + staging removed


def test_genbuster_load_dataset_alias(genbuster_root: Path) -> None:
    ds = load_dataset("genbuster", settings=["wild"], include_real=False, root=genbuster_root)
    assert isinstance(ds, GenBusterBenchDataset)
    assert len(ds) == 1
