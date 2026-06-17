"""Tests for utility modules (io, metrics, logger)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from detectzoo.utils.io import load_text
from detectzoo.utils.logger import get_logger
from detectzoo.utils.metrics import compute_metrics


class TestLoadText:
    def test_raw_string(self):
        assert load_text("hello world") == "hello world"

    def test_file_path(self, tmp_path: Path):
        p = tmp_path / "sample.txt"
        p.write_text("file content", encoding="utf-8")
        assert load_text(str(p)) == "file content"

    def test_pathlib_input(self, tmp_path: Path):
        p = tmp_path / "sample.txt"
        p.write_text("via path object", encoding="utf-8")
        assert load_text(p) == "via path object"

    def test_nonexistent_path_treated_as_text(self):
        assert load_text("/no/such/file/here.txt") == "/no/such/file/here.txt"


class TestLoadImage:
    def test_loads_rgb(self, tmp_path: Path):
        Image = pytest.importorskip("PIL.Image")
        from detectzoo.utils.io import load_image

        src = tmp_path / "img.png"
        Image.new("L", (8, 8), color=128).save(src)
        img = load_image(src)
        assert img.mode == "RGB"
        assert img.size == (8, 8)


class TestMetrics:
    def test_perfect_predictions(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        m = compute_metrics(labels, scores, threshold=0.5)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["pr_auc"] == pytest.approx(1.0)
        assert m["eer"] == pytest.approx(0.0, abs=1e-9)

    def test_all_wrong(self):
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.1, 0.2]
        m = compute_metrics(labels, scores, threshold=0.5)
        assert m["accuracy"] == 0.0
        assert m["roc_auc"] == 0.0
        assert m["eer"] == pytest.approx(1.0)

    def test_threshold_dependent_keys_present(self):
        m = compute_metrics([0, 1], [0.2, 0.8], threshold=0.5)
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tpr",
            "fpr",
            "roc_auc",
            "pr_auc",
            "avg_precision",
            "eer",
        ):
            assert key in m

    def test_tpr_equals_recall(self):
        labels = [0, 0, 1, 1]
        scores = [0.4, 0.6, 0.4, 0.9]  # one FP, one FN
        m = compute_metrics(labels, scores, threshold=0.5)
        assert m["tpr"] == pytest.approx(m["recall"])

    def test_single_class_auc_is_nan(self):
        labels = [1, 1, 1]
        scores = [0.9, 0.8, 0.7]
        m = compute_metrics(labels, scores, threshold=0.5)
        assert np.isnan(m["roc_auc"])
        assert np.isnan(m["pr_auc"])
        assert np.isnan(m["eer"])
        # threshold metrics are still computable for a single class
        assert m["accuracy"] == 1.0

    def test_non_finite_scores_are_dropped(self):
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, float("nan"), float("inf")]
        m = compute_metrics(labels, scores, threshold=0.5)
        # Only the two finite, correctly-classified samples remain.
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_all_non_finite_returns_nan(self):
        m = compute_metrics([0, 1], [float("nan"), float("inf")], threshold=0.5)
        assert np.isnan(m["accuracy"])
        assert np.isnan(m["roc_auc"])
        assert np.isnan(m["eer"])

    def test_threshold_changes_predictions(self):
        labels = [0, 1]
        scores = [0.4, 0.6]
        assert compute_metrics(labels, scores, threshold=0.5)["accuracy"] == 1.0
        # With a threshold above both scores, everything is predicted "human".
        assert compute_metrics(labels, scores, threshold=0.99)["accuracy"] == 0.5


class TestLogger:
    def test_returns_logger(self):
        log = get_logger("test_logger")
        assert log.name == "test_logger"
        assert isinstance(log, logging.Logger)

    def test_same_name_returns_same_instance(self):
        assert get_logger("dz_shared") is get_logger("dz_shared")
