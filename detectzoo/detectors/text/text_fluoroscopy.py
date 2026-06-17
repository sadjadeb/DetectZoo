"""Text Fluoroscopy — detecting LLM-generated text through intrinsic features.

Reference:
    Yu et al., "Text Fluoroscopy: Detecting LLM-Generated Text through
    Intrinsic Features", EMNLP 2024.

Text Fluoroscopy captures intrinsic text features by identifying the
transformer layer with the largest KL-divergence from both the first
and last layers when projected to the vocabulary space.  The embedding
from that layer, obtained via last-token pooling, is fed to a
lightweight MLP classifier for detection.

The default encoder is ``Alibaba-NLP/gte-Qwen1.5-7B-instruct``.  For
faster inference with minimal accuracy loss (<0.7%), set
``fixed_layer=30`` (~6.5x speed-up).

Without a trained classifier (``classifier_path=None`` and no prior
call to :meth:`fit`), the detector returns the max summed
KL-divergence as a proxy detection score.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# MLP classifier
# ------------------------------------------------------------------


class _BinaryClassifier(nn.Module):
    """Dropout → Linear → Tanh stack, then a 2-class output head."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [1024, 512]
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Dropout(dropout), nn.Linear(prev, h), nn.Tanh()])
            prev = h
        self.dense = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dense(x))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token from *hidden_states*."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, seq_lengths]


def _get_vocab_head(model: nn.Module) -> nn.Module:
    """Return the vocabulary projection head from a CausalLM model."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    head = model.get_output_embeddings()
    if head is not None:
        return head
    raise AttributeError(
        f"Cannot locate vocabulary head on {type(model).__name__}. "
        "Expected 'lm_head' attribute or get_output_embeddings()."
    )


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------


@register_detector("text_fluoroscopy")
class TextFluoroscopyDetector(BaseTextDetector):
    """Text Fluoroscopy — intrinsic-feature detector via max-KL layer selection.

    Parameters:
        model_name: HuggingFace causal LM used as encoder
            (default ``"Alibaba-NLP/gte-Qwen1.5-7B-instruct"``).
        fixed_layer: If set, skip dynamic KL computation and always
            use this transformer layer index (1-indexed into
            ``output_hidden_states``).  The paper recommends ``30``
            for a ~6.5x speed-up with <0.7% accuracy loss.
        classifier_hidden_sizes: Hidden layer sizes for the MLP
            classifier (default ``[1024, 512]``).
        classifier_dropout: Dropout rate for the MLP (default ``0.4``).
        classifier_path: Path to a saved classifier ``state_dict``.
            When ``None`` and :meth:`fit` has not been called, the
            detector falls back to the max KL-divergence value as a
            proxy score.
        threshold: Decision boundary on the output score.
        device: ``"cpu"`` or ``"cuda"``.
        max_length: Maximum token length for the tokenizer.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        fixed_layer: int | None = None,
        classifier_hidden_sizes: list[int] | None = None,
        classifier_dropout: float = 0.4,
        classifier_path: str | None = None,
        threshold: float = 0.5,
        device: str = "cpu",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            threshold=threshold,
            device=device,
            max_length=max_length,
            **kwargs,
        )
        self.fixed_layer = fixed_layer
        self.classifier_hidden_sizes = classifier_hidden_sizes or [1024, 512]
        self.classifier_dropout = classifier_dropout
        self.classifier_path = classifier_path
        self._classifier: _BinaryClassifier | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Text Fluoroscopy encoder '%s' …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch.float16,
        }
        if self._device.type == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        if self._device.type != "cuda":
            self._model.to(self._device)
        self._model.eval()

    def _ensure_classifier(self, input_dim: int) -> _BinaryClassifier:
        """Create (and optionally load) the MLP classifier."""
        if self._classifier is not None:
            return self._classifier
        self._classifier = _BinaryClassifier(
            input_dim,
            hidden_sizes=self.classifier_hidden_sizes,
            dropout=self.classifier_dropout,
        ).to(self._device)
        if self.classifier_path is not None:
            state = torch.load(self.classifier_path, map_location=self._device, weights_only=True)
            self._classifier.load_state_dict(state)
            logger.info("Loaded classifier weights from '%s'", self.classifier_path)
        self._classifier.eval()
        return self._classifier

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_hidden(
        self,
        text: str,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Tokenise *text*, run the encoder, and return (hidden_states, attention_mask).

        The full ``outputs`` object is discarded immediately so only
        the hidden-state tuple and attention mask remain in memory.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self._device)
        outputs = self.model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        del outputs
        return hidden_states, enc["attention_mask"]

    @torch.no_grad()
    def _compute_kl_divergences(
        self,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> list[float]:
        """KL(middle‖first) + KL(middle‖last) for each middle layer.

        Returns a list of length ``N-1`` (one entry per candidate
        middle layer, where index 0 corresponds to ``hidden_states[1]``).

        Projections are done one layer at a time and cast to float32 for
        numerical stability; intermediates are deleted to stay within VRAM.
        """
        vocab_head = _get_vocab_head(self.model)

        first_probs = F.softmax(
            vocab_head(hidden_states[0]).squeeze(0).float(),
            dim=-1,
        )
        last_probs = F.softmax(
            vocab_head(hidden_states[-1]).squeeze(0).float(),
            dim=-1,
        )

        kls: list[float] = []
        for i in range(1, len(hidden_states) - 1):
            mid_logits = vocab_head(hidden_states[i]).squeeze(0).float()
            mid_log_probs = F.log_softmax(mid_logits, dim=-1)
            del mid_logits
            kl = (
                F.kl_div(mid_log_probs, first_probs, reduction="batchmean").item()
                + F.kl_div(mid_log_probs, last_probs, reduction="batchmean").item()
            )
            del mid_log_probs
            kls.append(kl)

        del first_probs, last_probs
        return kls

    def _select_layer(
        self,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> tuple[int, list[float]]:
        """Return ``(layer_index, kl_values)``.

        *layer_index* is an index into *hidden_states*.  When
        ``fixed_layer`` is set, KL computation is skipped.
        """
        if self.fixed_layer is not None:
            return self.fixed_layer, []
        kls = self._compute_kl_divergences(hidden_states)
        best_offset = int(max(range(len(kls)), key=lambda i: kls[i]))
        layer_idx = best_offset + 1  # kls[0] ↔ hidden_states[1]
        return layer_idx, kls

    @torch.no_grad()
    def _extract_features(
        self,
        text: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Extract the intrinsic-layer embedding and metadata for *text*."""
        hidden_states, attention_mask = self._forward_hidden(text)
        layer_idx, kls = self._select_layer(hidden_states)

        embedding = _last_token_pool(hidden_states[layer_idx], attention_mask)
        n_layers = len(hidden_states)
        del hidden_states

        meta: dict[str, Any] = {
            "selected_layer": layer_idx,
            "n_layers": n_layers,
        }
        if kls:
            meta["max_kl"] = max(kls)
            meta["kl_divergences"] = kls
        return embedding, meta

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        embedding, meta = self._extract_features(text)

        if self.classifier_path is not None or self._classifier is not None:
            clf = self._ensure_classifier(embedding.shape[-1])
            logits = clf(embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            score = float(probs[1])
            meta["ai_prob"] = score
            meta["human_prob"] = float(probs[0])
        else:
            # Proxy: max combined KL-divergence (no trained classifier)
            score = meta.get("max_kl", 0.0)

        return self._make_result(score, **meta)

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[int] | None = None,
        *,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 3e-3,
        save_path: str | None = None,
    ) -> _BinaryClassifier:
        """Train the MLP classifier on extracted embeddings.

        Parameters:
            train_texts: Training texts.
            train_labels: Binary labels (``1`` = AI, ``0`` = human).
            val_texts: Optional validation texts.
            val_labels: Optional validation labels.
            epochs: Training epochs (default ``10``).
            batch_size: Mini-batch size (default ``16``).
            lr: Learning rate (default ``3e-3``).
            save_path: If given, save the best classifier state dict.

        Returns:
            The trained :class:`_BinaryClassifier` module.
        """
        logger.info("Extracting training embeddings (%d samples) …", len(train_texts))
        train_embs = torch.stack([self._extract_features(t)[0] for t in train_texts])
        train_y = torch.tensor(list(train_labels), dtype=torch.long, device=self._device)

        val_embs: torch.Tensor | None = None
        val_y: torch.Tensor | None = None
        if val_texts is not None and val_labels is not None:
            logger.info("Extracting validation embeddings (%d samples) …", len(val_texts))
            val_embs = torch.stack([self._extract_features(t)[0] for t in val_texts])
            val_y = torch.tensor(list(val_labels), dtype=torch.long, device=self._device)

        clf = self._ensure_classifier(train_embs.shape[-1])
        optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_state: dict[str, Any] | None = None

        for epoch in range(epochs):
            clf.train()
            perm = torch.randperm(len(train_embs), device=self._device)
            for start in range(0, len(train_embs), batch_size):
                idx = perm[start : start + batch_size]
                logits = clf(train_embs[idx])
                loss = criterion(logits, train_y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            clf.eval()
            if val_embs is not None and val_y is not None:
                with torch.no_grad():
                    preds = clf(val_embs).argmax(dim=-1)
                    acc = float((preds == val_y).float().mean())
                logger.info("Epoch %d/%d — val acc %.4f", epoch + 1, epochs, acc)
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_state = {k: v.clone() for k, v in clf.state_dict().items()}

        if best_state is not None:
            clf.load_state_dict(best_state)
        if save_path is not None:
            torch.save(clf.state_dict(), save_path)
            logger.info("Saved classifier to '%s'", save_path)

        clf.eval()
        self._classifier = clf
        return clf
