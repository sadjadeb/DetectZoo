"""Tests for the dataset module (base classes + registry)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from detectzoo.core.registry import (
    _DATASET_ALIASES,
    _DATASET_REGISTRY,
    list_datasets,
    load_dataset,
)
from detectzoo.datasets.base import (
    BaseDataset,
    CSVDataset,
    DatasetItem,
    SimpleDirectoryDataset,
)


class TestDatasetItem:
    def test_fields(self):
        item = DatasetItem(data="hello", label=1, metadata={"src": "gpt"})
        assert item.label == 1
        assert item.metadata["src"] == "gpt"

    def test_default_metadata(self):
        item = DatasetItem(data="x", label=0)
        assert item.metadata == {}


class TestSimpleDirectoryDataset:
    def test_loads_files(self, tmp_path: Path):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        (real / "a.txt").write_text("r1")
        (real / "b.txt").write_text("r2")
        (fake / "c.txt").write_text("f1")

        ds = SimpleDirectoryDataset(real, fake)
        items = ds.load()
        assert len(items) == 3
        labels = {it.label for it in items}
        assert labels == {0, 1}

    def test_labels_match_directory(self, tmp_path: Path):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        (real / "a.txt").write_text("r")
        (fake / "b.txt").write_text("f")

        items = {Path(it.data).name: it.label for it in SimpleDirectoryDataset(real, fake).load()}
        assert items["a.txt"] == 0
        assert items["b.txt"] == 1

    def test_extension_filter(self, tmp_path: Path):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        (real / "a.png").write_text("")
        (real / "b.jpg").write_text("")
        (real / "c.txt").write_text("")
        (fake / "d.png").write_text("")

        ds = SimpleDirectoryDataset(real, fake, extensions=[".png"])
        assert len(ds.load()) == 2

    def test_iter_and_len(self, tmp_path: Path):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        (real / "a.txt").write_text("r")
        (fake / "b.txt").write_text("f")

        ds = SimpleDirectoryDataset(real, fake)
        assert len(ds) == 2
        assert sum(1 for _ in ds) == 2

    def test_caches_items(self, tmp_path: Path):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        (real / "a.txt").write_text("r")
        (fake / "b.txt").write_text("f")

        ds = SimpleDirectoryDataset(real, fake)
        assert ds.load() is ds.load()


class TestMaxSamples:
    def _make(self, tmp_path: Path, n_real: int, n_fake: int, **kw):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        for i in range(n_real):
            (real / f"r{i}.txt").write_text("r")
        for i in range(n_fake):
            (fake / f"f{i}.txt").write_text("f")
        return SimpleDirectoryDataset(real, fake, **kw)

    def test_balanced_truncation(self, tmp_path: Path):
        ds = self._make(tmp_path, 10, 10, max_samples=4)
        items = ds.load()
        assert len(items) == 4
        labels = [it.label for it in items]
        assert labels.count(0) == 2
        assert labels.count(1) == 2

    def test_fills_from_other_class_when_short(self, tmp_path: Path):
        # Only 1 real sample, 10 fake; ask for 6 -> 1 real + 5 fake.
        ds = self._make(tmp_path, 1, 10, max_samples=6)
        items = ds.load()
        assert len(items) == 6
        labels = [it.label for it in items]
        assert labels.count(0) == 1
        assert labels.count(1) == 5


class TestCSVDataset:
    def test_loads_csv(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "hello world", "label": "0"})
            writer.writerow({"text": "ai text here", "label": "1"})

        ds = CSVDataset(csv_path)
        items = ds.load()
        assert len(items) == 2
        assert items[0].label == 0
        assert items[1].data == "ai text here"

    def test_custom_columns(self, tmp_path: Path):
        csv_path = tmp_path / "custom.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["body", "y"])
            writer.writeheader()
            writer.writerow({"body": "content", "y": "1"})

        ds = CSVDataset(csv_path, text_column="body", label_column="y")
        items = ds.load()
        assert items[0].data == "content"
        assert items[0].label == 1


class TestFromFactoryMethods:
    def test_from_directory(self, tmp_path: Path):
        real = tmp_path / "real"
        fake = tmp_path / "fake"
        real.mkdir()
        fake.mkdir()
        (real / "a.txt").write_text("r")
        (fake / "b.txt").write_text("f")

        ds = BaseDataset.from_directory(real, fake)
        assert isinstance(ds, SimpleDirectoryDataset)
        assert len(ds.load()) == 2

    def test_from_csv(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "sample", "label": "0"})

        ds = BaseDataset.from_csv(csv_path)
        assert isinstance(ds, CSVDataset)
        assert len(ds.load()) == 1


class TestDatasetRegistry:
    def test_datasets_registered(self):
        names = set(list_datasets())
        assert names, "No datasets registered"
        # Text datasets have no heavy optional deps and should be present.
        for n in ("hc3", "raid", "m4"):
            assert n in names, f"{n} not registered; got {sorted(names)}"

    def test_registry_invariants(self):
        for name, cls in _DATASET_REGISTRY.items():
            assert cls.name == name, f"{name}: cls.name mismatch ({cls.name!r})"

    def test_alias_targets_exist(self):
        for alias, target in _DATASET_ALIASES.items():
            assert target in _DATASET_REGISTRY, f"Alias {alias!r} -> unknown {target!r}"

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset_xyz")

    def test_list_by_modality(self):
        text_names = list_datasets("text")
        for n in text_names:
            assert _DATASET_REGISTRY[n].modality == "text"
