"""Benchmark evaluator for running detectors across datasets."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from tqdm import tqdm

from detectzoo.core.base import BaseDetector, DetectionResult
from detectzoo.datasets.base import BaseDataset, DatasetItem
from detectzoo.utils.logger import get_logger
from detectzoo.utils.metrics import compute_metrics

logger = get_logger(__name__)


# Per-modality print views for `run_and_print`. Audio uses the canonical
# anti-spoofing trio (EER / AUC / F1); image and text keep DetectZoo's
# original column set unchanged.
_DEFAULT_PRINT_COLUMNS = [
    "detector", "accuracy", "precision", "recall", "f1",
    "tpr", "fpr", "roc_auc", "pr_auc",
]
_PRINT_VIEWS = {
    "audio": ["detector", "eer", "roc_auc", "f1"],
    "image": _DEFAULT_PRINT_COLUMNS,
    "text": _DEFAULT_PRINT_COLUMNS,
}


class BenchmarkEvaluator:
    """Run one or more detectors on a dataset and compute metrics.

    Example::

        evaluator = BenchmarkEvaluator(dataset)
        results = evaluator.run([detector_a, detector_b])
        for name, metrics in results.items():
            print(name, metrics)

    Parameters
    ----------
    dataset:
        Dataset to evaluate against.
    modality:
        Controls only :meth:`run_and_print`'s column selection — execution
        and saved JSON are unaffected. When ``None`` (the default), the
        modality is auto-inferred from ``dataset.modality``. Pass
        ``modality="audio"`` to force the audio view (``detector | eer |
        roc_auc | f1``), or any other value to fall back to the default
        column set. Image and text behaviour is unchanged from earlier
        versions.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        *,
        modality: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self.modality = modality if modality is not None else getattr(dataset, "modality", None)

    def evaluate_single(
        self,
        detector: BaseDetector,
        items: Sequence[DatasetItem] | None = None,
        *,
        save_scores: bool = False,
    ) -> Dict[str, Any]:
        """Run *detector* on all items and return a metrics dictionary.

        Parameters
        ----------
        save_scores:
            If ``True``, the returned dict includes a ``"samples"`` key with
            per-sample labels and scores.
        """
        if items is None:
            items = self.dataset.load()

        labels: List[int] = []
        scores: List[float] = []
        predictions: List[DetectionResult] = []

        for item in tqdm(items, desc=detector.name):
            result = detector.predict(item.data)
            labels.append(item.label)
            scores.append(result.score)
            predictions.append(result)

        metrics = compute_metrics(labels, scores, threshold=detector.threshold)
        metrics["detector"] = detector.name
        metrics["n_samples"] = len(labels)

        if save_scores:
            metrics["samples"] = [
                {"label": lbl, "score": scr}
                for lbl, scr in zip(labels, scores)
            ]

        return metrics

    def run(
        self,
        detectors: Sequence[BaseDetector],
        *,
        save_scores: bool = False,
        unload_between: bool = True,
        on_detector_done: Optional[Callable[[Dict[str, Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple detectors and return ``{name: metrics}``.

        Parameters
        ----------
        unload_between:
            If ``True`` (default), each detector's :meth:`BaseDetector.unload`
            is called after it finishes so its GPU weights are released
            before the next detector starts loading.  Disable this only if
            you intentionally want the models to stay resident (e.g. for
            repeated calls on the same evaluator).
        on_detector_done:
            Optional callback invoked with the accumulated results dict
            after each detector finishes. Useful for incremental saving.
        """
        items = self.dataset.load()
        results: Dict[str, Dict[str, Any]] = {}
        for det in detectors:
            logger.info("Evaluating '%s' …", det.name)
            try:
                results[det.name] = self.evaluate_single(det, items=items, save_scores=save_scores)
            finally:
                if unload_between:
                    try:
                        det.unload()
                    except Exception:
                        logger.exception("Failed to unload detector '%s'", det.name)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if on_detector_done is not None:
                on_detector_done(results)
        return results

    def run_and_print(self, detectors: Sequence[BaseDetector]) -> None:
        """Evaluate detectors and print a comparison table.

        For audio datasets/detectors the printed columns are reduced to the
        canonical anti-spoofing trio ``detector | eer | roc_auc | f1``.
        Image and text use the same column set DetectZoo has always printed.
        """
        all_results = self.run(detectors)
        header_keys = _PRINT_VIEWS.get(self.modality, _DEFAULT_PRINT_COLUMNS)
        header = " | ".join(f"{k:>18s}" for k in header_keys)
        print(header)
        print("-" * len(header))
        for metrics in all_results.values():
            row = " | ".join(f"{metrics.get(k, ''):>18}" if isinstance(metrics.get(k), str)
                             else f"{metrics.get(k, 0):>18.4f}" for k in header_keys)
            print(row)

    def _save_payload(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Path,
        meta: Optional[Dict[str, Any]],
    ) -> None:
        payload: Any = results
        if meta is not None:
            payload = {"meta": meta, "results": results}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str))

    def run_and_save(
        self,
        detectors: Sequence[BaseDetector],
        output_path: Union[str, Path],
        *,
        save_scores: bool = False,
        unload_between: bool = True,
        meta: Optional[Dict[str, Any]] = None,
        incremental: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate detectors and save the results to a JSON file.

        Parameters
        ----------
        detectors:
            Detectors to benchmark.
        output_path:
            Destination file path.  Parent directories are created
            automatically if they don't exist.
        save_scores:
            If ``True``, per-sample labels and scores are included in the
            saved output under each detector's ``"samples"`` key.
        meta:
            Optional metadata dict.  When provided the saved JSON becomes
            ``{"meta": <meta>, "results": <detector_metrics>}`` instead of
            the flat ``{detector: metrics}`` form.
        incremental:
            If ``True``, the output file is rewritten after each detector
            finishes so partial results survive if a later detector fails.

        Returns
        -------
        The same ``{name: metrics}`` dictionary that :meth:`run` returns.
        """
        output_path = Path(output_path)
        callback = (
            (lambda results: self._save_payload(results, output_path, meta))
            if incremental
            else None
        )

        all_results = self.run(
            detectors,
            save_scores=save_scores,
            unload_between=unload_between,
            on_detector_done=callback,
        )
        self._save_payload(all_results, output_path, meta)

        logger.info("Results saved to %s", output_path)
        return all_results
