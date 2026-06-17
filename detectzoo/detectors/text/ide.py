"""IDE — Intrinsic Dimension Estimation for AI-text detection.

Reference:
    Tulchinskii et al., "Intrinsic Dimension Estimation for Robust
    Detection of AI-Generated Texts", NeurIPS 2023.

Each text is encoded with a pre-trained language model (RoBERTa by
default), producing a point cloud of token embeddings.  The intrinsic
dimensionality of this manifold is estimated via:

- **PHD** (Persistent Homology Dimension): MST weight scaling in
  random subsamples → log-log linear regression → dimension.
- **MLE** (Maximum Likelihood Estimation, Levina-Bickel): k-NN
  distance ratios → per-point dimension → harmonic mean.

Human texts have higher intrinsic dimension (~9) than AI texts (~7.5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# PHD helpers
# ------------------------------------------------------------------


def _prim_mst_weight(dist_matrix: np.ndarray, alpha: float = 1.0) -> float:
    """Compute MST weight via Prim's algorithm with weight exponent *alpha*."""
    n = dist_matrix.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    min_cost = np.full(n, np.inf)
    min_cost[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = -1
        for v in range(n):
            if not in_tree[v] and (u == -1 or min_cost[v] < min_cost[u]):
                u = v
        in_tree[u] = True
        total += min_cost[u] ** alpha
        for v in range(n):
            if not in_tree[v]:
                d = dist_matrix[u, v]
                if d < min_cost[v]:
                    min_cost[v] = d
    return total


def _phd_estimate(
    points: np.ndarray,
    alpha: float = 1.0,
    min_subsample: int = 40,
    intermediate_points: int = 7,
    n_reruns: int = 3,
    restarts: int = 9,
) -> float:
    """Estimate PHD from a point cloud."""
    from scipy.spatial.distance import cdist

    N = points.shape[0]
    if N < min_subsample + 10:
        return 0.0

    step = max(1, (N - min_subsample) // intermediate_points)
    test_n = list(range(min_subsample, N - step, step))
    if not test_n:
        return 0.0

    all_dims: list[float] = []
    for _ in range(n_reruns):
        lengths = []
        for n_pts in test_n:
            mst_weights = []
            r = max(3, restarts) if n_pts < N // 2 else max(3, restarts // 3)
            for _ in range(r):
                idx = np.random.choice(N, size=n_pts, replace=False)
                sub = points[idx]
                D = cdist(sub, sub, metric="euclidean")
                w = _prim_mst_weight(D, alpha)
                mst_weights.append(w)
            lengths.append(np.median(mst_weights))

        x = np.log(np.array(test_n, dtype=np.float64))
        y = np.log(np.array(lengths, dtype=np.float64) + 1e-30)

        if len(x) < 2:
            continue
        slope = np.polyfit(x, y, 1)[0]
        if abs(1.0 - slope) < 1e-10:
            continue
        dim = alpha / (1.0 - slope)
        all_dims.append(max(dim, 0.0))

    return float(np.mean(all_dims)) if all_dims else 0.0


# ------------------------------------------------------------------
# MLE helpers (Levina-Bickel)
# ------------------------------------------------------------------


def _mle_estimate(points: np.ndarray, k: int = 20) -> float:
    """Estimate intrinsic dimension via Levina-Bickel MLE."""
    from scipy.spatial.distance import cdist

    N = points.shape[0]
    if N < k + 2:
        return 0.0

    D = cdist(points, points, metric="euclidean")
    np.fill_diagonal(D, np.inf)

    dims: list[float] = []
    for i in range(N):
        dists = np.sort(D[i])[:k]
        R_k = dists[-1]
        if R_k < 1e-10:
            continue
        log_ratios = np.log(R_k / np.maximum(dists[:-1], 1e-30))
        total = log_ratios.sum()
        if total < 1e-10:
            continue
        dims.append((k - 1) / total)

    if not dims:
        return 0.0
    inv_dims = [1.0 / d for d in dims if d > 1e-10]
    if not inv_dims:
        return 0.0
    return 1.0 / np.mean(inv_dims)


# ------------------------------------------------------------------
# Shared base
# ------------------------------------------------------------------


class _IDEBase(BaseTextDetector):
    """Shared base for PHD and MLE intrinsic-dimension detectors."""

    def __init__(
        self,
        encoder_model: str = "roberta-base",
        threshold: float = 0.0,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=encoder_model,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.encoder_model_name = encoder_model
        self._encoder: torch.nn.Module | None = None
        self._enc_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading encoder '%s' for IDE …", self.encoder_model_name)
        self._enc_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name)
        self._encoder = AutoModel.from_pretrained(self.encoder_model_name).to(self._device)
        self._encoder.eval()

    @property
    def encoder(self) -> torch.nn.Module:
        if self._encoder is None:
            self._load_model()
        return self._encoder  # type: ignore[return-value]

    @property
    def enc_tokenizer(self):
        if self._enc_tokenizer is None:
            self._load_model()
        return self._enc_tokenizer

    @torch.no_grad()
    def _get_point_cloud(self, text: str) -> np.ndarray:
        """Encode text into a point cloud of token embeddings."""
        import re

        text = re.sub(r"\n", " ", text)
        text = re.sub(r"  +", " ", text)

        enc = self.enc_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)

        out = self.encoder(**enc)
        hidden = out.last_hidden_state.squeeze(0).cpu().numpy()
        # Strip CLS and SEP tokens
        if hidden.shape[0] > 2:
            hidden = hidden[1:-1]
        return hidden


@register_detector("phd")
class PHDDetector(_IDEBase):
    """PHD (Persistent Homology Dimension) detector.

    Higher intrinsic dimension → more likely human.  The score is
    negated so that higher → more likely AI (DetectZoo convention).

    Parameters:
        encoder_model: HuggingFace encoder model for embeddings
            (default ``"roberta-base"``).
        alpha: MST weight exponent (default ``1.0``).
        n_reruns: Number of independent PHD reruns (default ``3``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "roberta-base",
        alpha: float = 1.0,
        n_reruns: int = 3,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold, device=device, **kwargs)
        self.alpha = alpha
        self.n_reruns = n_reruns

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        cloud = self._get_point_cloud(text)

        dim = _phd_estimate(cloud, alpha=self.alpha, n_reruns=self.n_reruns)

        # Human ~9, AI ~7.5 → negate so higher = more likely AI
        score = -dim

        return self._make_result(
            score,
            intrinsic_dim=dim,
            cloud_size=cloud.shape[0],
        )


@register_detector("mle_ide")
class MLEDetector(_IDEBase):
    """MLE (Levina-Bickel) intrinsic-dimension detector.

    Parameters:
        encoder_model: HuggingFace encoder model for embeddings
            (default ``"roberta-base"``).
        n_neighbors: Number of nearest neighbours (default ``20``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "roberta-base",
        n_neighbors: int = 20,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold, device=device, **kwargs)
        self.n_neighbors = n_neighbors

    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        cloud = self._get_point_cloud(text)

        dim = _mle_estimate(cloud, k=self.n_neighbors)

        score = -dim

        return self._make_result(
            score,
            intrinsic_dim=dim,
            cloud_size=cloud.shape[0],
        )
