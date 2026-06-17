"""DeTeCtive — multi-level contrastive learning detector.

Reference:
    Guo et al., "DeTeCtive: Detecting AI-generated Text via Multi-Level
    Contrastive Learning", NeurIPS 2024.

DeTeCtive learns a text embedding space where texts from the same
source cluster tightly using a three-level contrastive hierarchy
(model -> model-family -> human/machine).  At inference, classification
is done via KNN majority voting over a reference database.

Without a reference database, the detector uses the embedding's
L2 norm concentration as a zero-shot proxy (LLM text tends to
produce more structured, compact embeddings).

The default encoder is ``princeton-nlp/unsup-simcse-roberta-base``
(matching the official implementation).  Official fine-tuned
checkpoints are available on HuggingFace at
``heyongxin233/DeTeCtive`` (e.g. ``Deepfake_best.pth``,
``M4_monolingual_best.pth``) and ``Shengkun/ood-detection``
(``model_raid.pth`` — trained on the RAID split).

When a checkpoint is loaded, the full contrastive embedding model is
used for inference.  Without a checkpoint, the detector uses the
embedding's L2 norm concentration as a zero-shot proxy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)

_HF_CHECKPOINT_REPO = "heyongxin233/DeTeCtive"
_AVAILABLE_CHECKPOINTS = {
    "Deepfake_best.pth": "heyongxin233/DeTeCtive",
    "M4_monolingual_best.pth": "heyongxin233/DeTeCtive",
    "M4_multilingual_best.pth": "heyongxin233/DeTeCtive",
    "OUTFOX_best.pth": "heyongxin233/DeTeCtive",
    "TuringBench_best.pth": "heyongxin233/DeTeCtive",
    "model_raid.pth": "Shengkun/ood-detection",
}


@register_detector("detective")
class DeTeCtiveDetector(BaseTextDetector):
    """DeTeCtive contrastive-learning detector.

    Parameters:
        encoder_model: HuggingFace encoder model (default
            ``"princeton-nlp/unsup-simcse-roberta-base"``).
        checkpoint: Name of an official checkpoint file from
            ``heyongxin233/DeTeCtive`` on HuggingFace (e.g.
            ``"Deepfake_best.pth"``), or a local path to a ``.pth``
            file.  When provided, fine-tuned weights are loaded into
            the encoder.  Default ``None`` (zero-shot proxy).
        threshold: Decision boundary.
        max_length: Max token length.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        checkpoint: str | None = None,
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
        self.checkpoint = checkpoint
        self._enc_model: torch.nn.Module | None = None
        self._enc_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading DeTeCtive encoder '%s' …", self.encoder_model_name)
        self._enc_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name)
        self._enc_model = AutoModel.from_pretrained(self.encoder_model_name).to(self._device)

        if self.checkpoint is not None:
            self._load_checkpoint()

        self._enc_model.eval()

    def _load_checkpoint(self) -> None:
        """Load official .pth checkpoint into the encoder."""
        import os

        ckpt_path = self.checkpoint
        if not os.path.isfile(str(ckpt_path)):
            if ckpt_path in _AVAILABLE_CHECKPOINTS:
                repo_id = _AVAILABLE_CHECKPOINTS[ckpt_path]
                try:
                    from huggingface_hub import hf_hub_download

                    ckpt_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=str(self.checkpoint),
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not download checkpoint '%s' from %s: %s",
                        self.checkpoint,
                        repo_id,
                        exc,
                    )
                    return
            else:
                logger.warning(
                    "Checkpoint '%s' not found locally and is not a known "
                    "official checkpoint. Skipping.",
                    self.checkpoint,
                )
                return

        logger.info("Loading DeTeCtive checkpoint from '%s' …", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=self._device)

        # Official checkpoints store the encoder under "model." prefix
        # (from TextEmbeddingModel wrapping AutoModel)
        enc_state: dict[str, Any] = {}
        for key, val in state_dict.items():
            if key.startswith("model.model."):
                enc_state[key[len("model.model.") :]] = val
            elif key.startswith("model."):
                enc_state[key[len("model.") :]] = val

        if enc_state:
            missing, unexpected = self._enc_model.load_state_dict(  # type: ignore[union-attr]
                enc_state,
                strict=False,
            )
            if missing:
                logger.debug("Missing keys when loading checkpoint: %s", missing[:5])
            if unexpected:
                logger.debug("Unexpected keys when loading checkpoint: %s", unexpected[:5])
            logger.info("Loaded %d parameters from checkpoint.", len(enc_state))
        else:
            logger.warning("No encoder parameters found in checkpoint.")

    @property
    def enc_model(self) -> torch.nn.Module:
        if self._enc_model is None:
            self._load_model()
        return self._enc_model  # type: ignore[return-value]

    @property
    def enc_tokenizer(self):
        if self._enc_tokenizer is None:
            self._load_model()
        return self._enc_tokenizer

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        enc = self.enc_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        ).to(self._device)

        out = self.enc_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        hidden = out.last_hidden_state * mask
        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1).squeeze(0)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        emb = self._embed(text)

        weight = None
        for name, param in self.enc_model.named_parameters():
            if "word_embeddings" in name:
                weight = param
                break

        if weight is not None:
            ref_dir = F.normalize(weight.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            score = float(F.cosine_similarity(emb.unsqueeze(0), ref_dir.unsqueeze(0)).squeeze())
        else:
            score = float(emb.abs().mean())

        return self._make_result(
            score,
            embedding_norm=float(emb.norm()),
        )
