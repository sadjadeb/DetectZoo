"""Tests for the BenchmarkEvaluator (no model downloads).

A trivial in-memory dataset and a length-based dummy detector are used so
the evaluator's orchestration, metric aggregation, and persistence can be
exercised without any heavy dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from detectzoo.benchmarks.evaluator import BenchmarkEvaluator
from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.datasets.base import BaseDataset, DatasetItem


class _MemoryDataset(BaseDataset):
    name = "memory"
    modality = "text"

    def __init__(self, items: List[DatasetItem], **kw):
        super().__init__(**kw)
        self._mem = items

    def _load_all(self) -> List[DatasetItem]:
        return self._mem


class _KeywordDetector(BaseDetector):
    """Scores 0.9 if 'ai' appears in the text, else 0.1 — perfectly separable."""

    name = "keyword"
    modality = "text"

    def predict(self, input_data) -> DetectionResult:
        return self._make_result(0.9 if "ai" in str(input_data).lower() else 0.1)


@pytest.fixture
def dataset() -> _MemoryDataset:
    items = [
        DatasetItem(data="a human wrote this", label=0),
        DatasetItem(data="another genuine note", label=0),
        DatasetItem(data="this is ai generated", label=1),
        DatasetItem(data="ai produced output", label=1),
    ]
    return _MemoryDataset(items)


class TestBenchmarkEvaluator:
    def test_evaluate_single(self, dataset):
        ev = BenchmarkEvaluator(dataset)
        metrics = ev.evaluate_single(_KeywordDetector())
        assert metrics["detector"] == "keyword"
        assert metrics["n_samples"] == 4
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_save_scores(self, dataset):
        ev = BenchmarkEvaluator(dataset)
        metrics = ev.evaluate_single(_KeywordDetector(), save_scores=True)
        assert "samples" in metrics
        assert len(metrics["samples"]) == 4
        assert {"label", "score"} <= set(metrics["samples"][0])

    def test_run_multiple(self, dataset):
        ev = BenchmarkEvaluator(dataset)
        results = ev.run([_KeywordDetector()])
        assert "keyword" in results
        assert results["keyword"]["accuracy"] == 1.0

    def test_run_and_save(self, dataset, tmp_path: Path):
        ev = BenchmarkEvaluator(dataset)
        out = tmp_path / "nested" / "results.json"
        ev.run_and_save([_KeywordDetector()], out)
        assert out.is_file()
        payload = json.loads(out.read_text())
        assert payload["keyword"]["n_samples"] == 4

    def test_run_and_save_with_meta(self, dataset, tmp_path: Path):
        ev = BenchmarkEvaluator(dataset)
        out = tmp_path / "results.json"
        ev.run_and_save([_KeywordDetector()], out, meta={"run": "test"})
        payload = json.loads(out.read_text())
        assert payload["meta"] == {"run": "test"}
        assert "keyword" in payload["results"]

    def test_modality_inferred_from_dataset(self, dataset):
        assert BenchmarkEvaluator(dataset).modality == "text"

    def test_modality_override(self, dataset):
        assert BenchmarkEvaluator(dataset, modality="audio").modality == "audio"
