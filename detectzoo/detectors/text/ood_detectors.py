"""OOD-based LLM text detectors — D-SVDD, HRN, and Energy.

Reference:
    Zeng et al., "Human Texts Are Outliers: Detecting LLM-generated
    Texts via Out-of-distribution Detection", NeurIPS 2025.

All three methods reframe AI-text detection as an OOD problem:
  - In-distribution = LLM-generated text
  - Out-of-distribution = Human-written text

They share a common encoder (``unsup-simcse-roberta-base``) that is
fine-tuned with an OOD objective **plus** a supervised contrastive
loss (SimCLR-style).  The three OOD objectives are:

- **D-SVDD**: Softplus margin on distance to a learned hypersphere
  centre.  Trained from scratch.
- **HRN**: Per-model one-class classifiers with sigmoid scoring and
  WGAN-GP gradient penalty (λ=0.1, p=12).  Optionally initialised
  from DeTeCtive pre-trained encoder weights via
  ``detective_checkpoint``.
- **Energy**: Multi-class classifier over LLM generators with
  ``-logsumexp`` energy scoring and margin regularisation
  (m_in=-27, m_out=-5).  Optionally initialised from DeTeCtive
  pre-trained encoder weights via ``detective_checkpoint``.

Without trained weights (``checkpoint_path=None`` and no call to
:meth:`fit`), each detector still runs inference using the frozen
encoder with proxy scoring so that :meth:`predict` never fails.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Classification head (shared by HRN and Energy)
# ------------------------------------------------------------------


class _ClassificationHead(nn.Module):
    """3-layer MLP: in → in/4 → in/16 → out."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        h1, h2 = in_dim // 4, in_dim // 16
        act = nn.Tanh if activation == "tanh" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            act(),
            nn.Linear(h1, h2),
            act(),
            nn.Linear(h2, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Contrastive (SimCLR) loss used by all three methods
# ------------------------------------------------------------------


def _contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (SimCLR-style).

    Positive pairs share the same label; negatives differ.
    """
    sim = (
        F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=-1,
        )
        / temperature
    )
    batch = embeddings.size(0)
    mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask_self = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    mask_pos = mask_pos & ~mask_self

    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0
    for i in range(batch):
        pos_idx = mask_pos[i].nonzero(as_tuple=True)[0]
        neg_idx = (~mask_pos[i] & ~mask_self[i]).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        avg_pos = sim[i, pos_idx].mean()
        logits = torch.cat([avg_pos.unsqueeze(0), sim[i, neg_idx]])
        target = torch.zeros(1, dtype=torch.long, device=embeddings.device)
        loss = loss + F.cross_entropy(logits.unsqueeze(0), target)
        count += 1
    return loss / max(count, 1)


# ------------------------------------------------------------------
# Shared encoder base
# ------------------------------------------------------------------


_DETECTIVE_CHECKPOINTS = {
    "Deepfake_best.pth": "heyongxin233/DeTeCtive",
    "M4_monolingual_best.pth": "heyongxin233/DeTeCtive",
    "M4_multilingual_best.pth": "heyongxin233/DeTeCtive",
    "OUTFOX_best.pth": "heyongxin233/DeTeCtive",
    "TuringBench_best.pth": "heyongxin233/DeTeCtive",
    "model_raid.pth": "Shengkun/ood-detection",
}


class _OODTextBase(BaseTextDetector):
    """Shared base for OOD-based text detectors.

    Uses ``unsup-simcse-roberta-base`` as the text encoder with
    average pooling and L2 normalisation.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
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
        self._enc_model: torch.nn.Module | None = None
        self._enc_tokenizer: Any = None

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading OOD encoder '%s' …", self.encoder_model_name)
        self._enc_tokenizer = AutoTokenizer.from_pretrained(self.encoder_model_name)
        self._enc_model = AutoModel.from_pretrained(self.encoder_model_name).to(self._device)
        self._enc_model.eval()

    def _load_detective_weights(self, checkpoint: str) -> None:
        """Initialise encoder from a DeTeCtive ``.pth`` checkpoint.

        Accepts a local path or the filename of an official checkpoint
        from ``heyongxin233/DeTeCtive`` on HuggingFace (e.g.
        ``"Deepfake_best.pth"``).
        """
        import os

        ckpt_path = checkpoint
        if not os.path.isfile(ckpt_path):
            if checkpoint in _DETECTIVE_CHECKPOINTS:
                repo_id = _DETECTIVE_CHECKPOINTS[checkpoint]
                try:
                    from huggingface_hub import hf_hub_download

                    ckpt_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=checkpoint,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not download DeTeCtive checkpoint '%s' from %s: %s",
                        checkpoint,
                        repo_id,
                        exc,
                    )
                    return
            else:
                logger.warning(
                    "DeTeCtive checkpoint '%s' not found locally and is not "
                    "a known official checkpoint. Skipping.",
                    checkpoint,
                )
                return

        logger.info("Loading DeTeCtive weights from '%s' …", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        enc_state: dict[str, Any] = {}
        for key, val in state_dict.items():
            if key.startswith("model.model."):
                enc_state[key[len("model.model.") :]] = val
            elif key.startswith("model."):
                enc_state[key[len("model.") :]] = val

        if enc_state:
            missing, unexpected = self.enc_model.load_state_dict(enc_state, strict=False)
            if missing:
                logger.debug("Missing keys: %s", missing[:5])
            if unexpected:
                logger.debug("Unexpected keys: %s", unexpected[:5])
            logger.info("Loaded %d DeTeCtive encoder parameters.", len(enc_state))
        else:
            logger.warning("No encoder parameters found in DeTeCtive checkpoint.")

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
        """Encode text into a normalised embedding vector."""
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

    def _embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed a list of texts, returning ``[N, D]``."""
        return torch.stack([self._embed(t) for t in texts])


# ------------------------------------------------------------------
# D-SVDD
# ------------------------------------------------------------------


@register_detector("dsvdd")
class DSVDDDetector(_OODTextBase):
    """D-SVDD (Deep Support Vector Data Description) detector.

    Measures squared L2 distance from the embedding to a learned
    hypersphere centre.  The centre is the L2-normalised mean of
    machine-generated embeddings computed during :meth:`fit`.

    Training loss (from reference code):
        ``softplus(avg_dist_machine - avg_dist_human)``
    combined with SimCLR contrastive loss.

    Parameters:
        encoder_model: HuggingFace encoder model.
        center: Pre-computed centre vector.  If ``None``, must be set
            via :meth:`fit` or :meth:`set_center` before meaningful
            inference.
        checkpoint_path: Path to a saved state dict containing
            ``encoder`` and ``center`` keys.
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        center: list[float] | None = None,
        checkpoint_path: str | None = None,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold, device=device, **kwargs)
        self._center: torch.Tensor | None = None
        if center is not None:
            self._center = torch.tensor(center, dtype=torch.float32)
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=False)
        if "encoder" in state:
            self.enc_model.load_state_dict(state["encoder"])
            logger.info("Loaded encoder weights from checkpoint")
        if "center" in state:
            self._center = state["center"].to(self._device)
            logger.info("Loaded center from checkpoint")

    @property
    def center(self) -> torch.Tensor:
        if self._center is not None:
            return self._center.to(self._device)
        return torch.zeros(768, device=self._device)

    def set_center(self, center: torch.Tensor) -> None:
        """Set the hypersphere centre from a pre-computed tensor."""
        self._center = F.normalize(center.unsqueeze(0), dim=-1).squeeze(0)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        emb = self._embed(text)

        dist_sq = float(((emb - self.center) ** 2).sum())

        # Closer to centre → AI; farther → human
        # Negate distance so higher → more likely AI
        score = -dist_sq

        return self._make_result(score, distance_squared=dist_sq)

    def fit(
        self,
        machine_texts: Sequence[str],
        human_texts: Sequence[str],
        *,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 2e-5,
        alpha: float = 1.0,
        beta: float = 1.0,
        save_path: str | None = None,
    ) -> None:
        """Train the encoder with D-SVDD + contrastive loss.

        Parameters:
            machine_texts: In-distribution (AI-generated) texts.
            human_texts: Out-of-distribution (human-written) texts.
            epochs: Training epochs (default ``20``).
            batch_size: Mini-batch size (default ``32``).
            lr: Learning rate (default ``2e-5``).
            alpha: Weight for contrastive loss.
            beta: Weight for D-SVDD loss.
            save_path: If given, save encoder + center.
        """
        logger.info("Computing initial center from %d machine texts …", len(machine_texts))
        with torch.no_grad():
            machine_embs = self._embed_batch(machine_texts)
            self._center = F.normalize(machine_embs.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

        self.enc_model.train()
        optimizer = torch.optim.Adam(self.enc_model.parameters(), lr=lr, betas=(0.9, 0.98))

        all_texts = list(machine_texts) + list(human_texts)
        # label: 0 = machine (ID), 1 = human (OOD)
        all_labels = [0] * len(machine_texts) + [1] * len(human_texts)

        for epoch in range(epochs):
            perm = torch.randperm(len(all_texts))
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(all_texts), batch_size):
                idx = perm[start : start + batch_size]
                batch_texts = [all_texts[i] for i in idx]
                batch_labels = torch.tensor([all_labels[i] for i in idx], device=self._device)

                embs = torch.stack([self._embed_no_grad_off(t) for t in batch_texts])

                machine_mask = batch_labels == 0
                human_mask = batch_labels == 1

                loss_dsvdd = torch.tensor(0.0, device=self._device)
                if machine_mask.any() and human_mask.any():
                    dist_m = ((embs[machine_mask] - self.center) ** 2).sum(dim=-1).clamp(1e-12, 1e6)
                    dist_h = ((embs[human_mask] - self.center) ** 2).sum(dim=-1).clamp(1e-12, 1e6)
                    diff = (dist_m.mean() - dist_h.mean()).clamp(-100, 100)
                    loss_dsvdd = F.softplus(diff)

                loss_contrastive = _contrastive_loss(embs, batch_labels)
                loss = alpha * loss_contrastive + beta * loss_dsvdd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                n_batches += 1

            logger.info(
                "Epoch %d/%d — loss %.4f", epoch + 1, epochs, total_loss / max(n_batches, 1)
            )

        self.enc_model.eval()
        if save_path is not None:
            torch.save(
                {
                    "encoder": self.enc_model.state_dict(),
                    "center": self._center,
                },
                save_path,
            )
            logger.info("Saved D-SVDD checkpoint to '%s'", save_path)

    def _embed_no_grad_off(self, text: str) -> torch.Tensor:
        """Embed with gradient tracking (for training)."""
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


# ------------------------------------------------------------------
# HRN
# ------------------------------------------------------------------


@register_detector("hrn")
class HRNDetector(_OODTextBase):
    """HRN (Holistic Regularised Network) detector.

    Trains one one-class classifier per LLM model family.  Each
    classifier outputs ``sigmoid(f(φ(x)))`` ∈ (0,1); high values
    indicate in-distribution (AI).  At inference, scores from all
    classifiers are averaged.

    Loss per classifier (from paper Eq. 7, reference code):
        ``-log(sigmoid(f(φ(x)))) + λ · GP``
    where GP is a WGAN-GP gradient penalty with ``p=12``, ``λ=0.1``.

    Parameters:
        encoder_model: HuggingFace encoder.
        detective_checkpoint: DeTeCtive ``.pth`` checkpoint (local
            path or filename from ``heyongxin233/DeTeCtive``) used
            to initialise the encoder before training.
        checkpoint_path: Path to saved state dict with ``encoder``
            and ``classifiers`` keys.
        n_classifiers: Number of per-model classifiers (set
            automatically by :meth:`fit`).
        gp_lambda: Gradient penalty weight (default ``0.1``).
        gp_power: Gradient penalty exponent (default ``12``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        detective_checkpoint: str | None = None,
        checkpoint_path: str | None = None,
        n_classifiers: int = 0,
        gp_lambda: float = 0.1,
        gp_power: int = 12,
        threshold: float = 0.5,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold, device=device, **kwargs)
        self.gp_lambda = gp_lambda
        self.gp_power = gp_power
        self._classifiers: nn.ModuleList | None = None
        if detective_checkpoint is not None:
            self._load_detective_weights(detective_checkpoint)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        elif n_classifiers > 0:
            self._init_classifiers(n_classifiers, self._get_embed_dim())

    def _get_embed_dim(self) -> int:
        """Infer embedding dimension from the encoder config."""
        cfg = self.enc_model.config
        return getattr(cfg, "hidden_size", 768)

    def _init_classifiers(self, n: int, embed_dim: int) -> None:
        self._classifiers = nn.ModuleList(
            [
                _ClassificationHead(embed_dim, 1, activation="relu").to(self._device)
                for _ in range(n)
            ]
        )

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=False)
        if "encoder" in state:
            self.enc_model.load_state_dict(state["encoder"])
        if "classifiers" in state:
            self._classifiers = nn.ModuleList()
            for sd in state["classifiers"]:
                in_dim = sd["net.0.weight"].shape[1]
                head = _ClassificationHead(in_dim, 1, activation="relu").to(self._device)
                head.load_state_dict(sd)
                self._classifiers.append(head)
            logger.info("Loaded %d HRN classifiers from checkpoint", len(self._classifiers))

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        emb = self._embed(text)

        if self._classifiers is not None and len(self._classifiers) > 0:
            scores = []
            for clf in self._classifiers:
                clf.eval()
                out = torch.sigmoid(clf(emb.unsqueeze(0))).squeeze()
                scores.append(float(out))
            avg_score = sum(scores) / len(scores)
            # Higher sigmoid → AI (ID); DetectZoo convention: higher → AI
            score = avg_score
        else:
            # Fallback: no trained classifiers — use embedding norm proxy
            score = float(emb.norm())

        return self._make_result(score, sigmoid_scores=scores if self._classifiers else [])

    def fit(
        self,
        texts_by_model: Dict[str, List[str]],
        human_texts: Sequence[str],
        *,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 2e-5,
        alpha: float = 1.0,
        beta: float = 1.0,
        freeze_encoder: bool = True,
        save_path: str | None = None,
    ) -> None:
        """Train per-model HRN classifiers.

        Parameters:
            texts_by_model: ``{model_name: [texts]}`` — one entry per
                LLM generator family.
            human_texts: Human-written (OOD) texts for contrastive loss.
            epochs: Training epochs per classifier (default ``20``).
            batch_size: Mini-batch size.
            lr: Learning rate (default ``2e-5``).
            alpha: Contrastive loss weight.
            beta: HRN loss weight.
            freeze_encoder: If ``True``, encoder is frozen and only
                classifiers are trained (matches reference default).
            save_path: If given, save encoder + classifier state dicts.
        """
        embed_dim = self._get_embed_dim()
        model_names = sorted(texts_by_model.keys())
        self._init_classifiers(len(model_names), embed_dim)
        assert self._classifiers is not None

        if freeze_encoder:
            for p in self.enc_model.parameters():
                p.requires_grad_(False)
            self.enc_model.eval()

        for i, model_name in enumerate(model_names):
            logger.info(
                "Training HRN classifier %d/%d for '%s' …", i + 1, len(model_names), model_name
            )
            clf = self._classifiers[i]
            optimizer = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.9, 0.98))

            model_texts = texts_by_model[model_name]
            all_texts = list(model_texts) + list(human_texts)
            all_labels = [0] * len(model_texts) + [1] * len(human_texts)

            for epoch in range(epochs):
                clf.train()
                perm = torch.randperm(len(all_texts))
                total_loss = 0.0
                n_batches = 0
                for start in range(0, len(all_texts), batch_size):
                    idx = perm[start : start + batch_size]
                    batch_texts = [all_texts[j] for j in idx]
                    batch_labels = torch.tensor([all_labels[j] for j in idx], device=self._device)
                    with torch.no_grad():
                        embs = self._embed_batch(batch_texts)

                    machine_mask = batch_labels == 0
                    if not machine_mask.any():
                        continue
                    machine_embs = embs[machine_mask]

                    # HRN loss: -log(sigmoid(f(x)))
                    out = clf(machine_embs)
                    loss_main = -torch.log(torch.sigmoid(out) + 1e-8).mean()

                    # WGAN-GP gradient penalty (p=12)
                    loss_gp = self._gradient_penalty(clf, machine_embs)

                    loss_hrn = loss_main + self.gp_lambda * loss_gp

                    # Contrastive loss on full batch
                    loss_contrastive = _contrastive_loss(embs, batch_labels)

                    loss = alpha * loss_contrastive + beta * loss_hrn
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)
                    n_batches += 1

                logger.info(
                    "  [%s] epoch %d/%d — loss %.4f",
                    model_name,
                    epoch + 1,
                    epochs,
                    total_loss / max(n_batches, 1),
                )

        if not freeze_encoder:
            self.enc_model.eval()
        if save_path is not None:
            torch.save(
                {
                    "encoder": self.enc_model.state_dict(),
                    "classifiers": [c.state_dict() for c in self._classifiers],
                },
                save_path,
            )
            logger.info("Saved HRN checkpoint to '%s'", save_path)

    def _gradient_penalty(
        self,
        clf: _ClassificationHead,
        real: torch.Tensor,
    ) -> torch.Tensor:
        """WGAN-GP style gradient penalty with ``p=gp_power``."""
        eps = torch.rand(real.size(0), 1, device=real.device)
        interp = (eps * real + (1 - eps) * real).detach().requires_grad_(True)
        out = clf(interp)
        grad = torch.autograd.grad(
            outputs=out.sum(),
            inputs=interp,
            create_graph=True,
            retain_graph=True,
        )[0]
        penalty = ((grad.norm(2, dim=1) - 1) ** self.gp_power).mean()
        return penalty


# ------------------------------------------------------------------
# Energy
# ------------------------------------------------------------------


@register_detector("energy_detector")
class EnergyDetector(_OODTextBase):
    """Energy-based OOD text detector.

    Uses a multi-class classifier head trained to distinguish among
    LLM generator families.  The energy score is:

        ``E(x) = -log Σ_i exp(f_i(x))``

    where ``f`` is the classifier logits.  Lower energy → in-distribution
    (AI).  Score is ``-E(x)`` so higher → more likely AI.

    Training loss combines:
      1. Cross-entropy on LLM generator classes (machine text only).
      2. Energy margin regularisation (``m_in=-27``, ``m_out=-5``).
      3. SimCLR contrastive loss.

    Parameters:
        encoder_model: HuggingFace encoder.
        detective_checkpoint: DeTeCtive ``.pth`` checkpoint (local
            path or filename from ``heyongxin233/DeTeCtive``) used
            to initialise the encoder before training.
        n_classes: Number of LLM generator classes for the classifier
            head.  Set automatically by :meth:`fit`.
        checkpoint_path: Path to a saved state dict with ``encoder``
            and ``classifier`` keys.
        m_in: Energy margin for in-distribution samples (default ``-27``).
        m_out: Energy margin for out-of-distribution samples
            (default ``-5``).
        threshold: Decision boundary.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        encoder_model: str = "princeton-nlp/unsup-simcse-roberta-base",
        detective_checkpoint: str | None = None,
        n_classes: int = 0,
        checkpoint_path: str | None = None,
        m_in: float = -27.0,
        m_out: float = -5.0,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder_model=encoder_model, threshold=threshold, device=device, **kwargs)
        self.m_in = m_in
        self.m_out = m_out
        self._classifier: _ClassificationHead | None = None
        if detective_checkpoint is not None:
            self._load_detective_weights(detective_checkpoint)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        elif n_classes > 0:
            embed_dim = self._get_embed_dim()
            self._classifier = _ClassificationHead(
                embed_dim,
                n_classes,
                activation="tanh",
            ).to(self._device)

    def _get_embed_dim(self) -> int:
        cfg = self.enc_model.config
        return getattr(cfg, "hidden_size", 768)

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=False)
        if "encoder" in state:
            self.enc_model.load_state_dict(state["encoder"])
        if "classifier" in state:
            sd = state["classifier"]
            in_dim = sd["net.0.weight"].shape[1]
            out_dim = sd["net.4.weight"].shape[0]
            self._classifier = _ClassificationHead(
                in_dim,
                out_dim,
                activation="tanh",
            ).to(self._device)
            self._classifier.load_state_dict(sd)
            logger.info("Loaded Energy classifier (%d classes) from checkpoint", out_dim)

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)
        emb = self._embed(text)

        if self._classifier is not None:
            self._classifier.eval()
            logits = self._classifier(emb.unsqueeze(0))
            energy = -float(torch.logsumexp(logits, dim=-1))
            # Higher -E(x) → more likely AI (ID)
            score = -energy
        else:
            # Fallback: no classifier trained — use embedding norm proxy
            score = float(emb.norm())
            energy = -score

        return self._make_result(score, energy=energy)

    def fit(
        self,
        texts_by_model: Dict[str, List[str]],
        human_texts: Sequence[str],
        *,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 2e-5,
        alpha: float = 1.0,
        beta: float = 1.0,
        energy_weight: float = 0.01,
        save_path: str | None = None,
    ) -> None:
        """Train the encoder + classifier with Energy + contrastive loss.

        Parameters:
            texts_by_model: ``{model_name: [texts]}`` — one key per
                LLM generator family.
            human_texts: Human-written (OOD) texts.
            epochs: Training epochs (default ``20``).
            batch_size: Mini-batch size.
            lr: Learning rate (default ``2e-5``).
            alpha: Contrastive loss weight.
            beta: Weight for classification + energy loss.
            energy_weight: Relative weight of energy margin loss
                vs. classification cross-entropy (default ``0.01``).
            save_path: If given, save encoder + classifier.
        """
        model_names = sorted(texts_by_model.keys())
        n_classes = len(model_names)
        model_to_idx = {m: i for i, m in enumerate(model_names)}
        embed_dim = self._get_embed_dim()

        self._classifier = _ClassificationHead(
            embed_dim,
            n_classes,
            activation="tanh",
        ).to(self._device)

        self.enc_model.train()
        params = list(self.enc_model.parameters()) + list(self._classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98))

        # Build dataset: (text, binary_label, class_idx)
        all_texts: list[str] = []
        all_binary: list[int] = []
        all_class: list[int] = []
        for model_name, texts in texts_by_model.items():
            for t in texts:
                all_texts.append(t)
                all_binary.append(0)  # machine = ID
                all_class.append(model_to_idx[model_name])
        for t in human_texts:
            all_texts.append(t)
            all_binary.append(1)  # human = OOD
            all_class.append(-1)

        for epoch in range(epochs):
            self._classifier.train()
            perm = torch.randperm(len(all_texts))
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(all_texts), batch_size):
                idx = perm[start : start + batch_size]
                batch_texts = [all_texts[i] for i in idx]
                batch_binary = torch.tensor([all_binary[i] for i in idx], device=self._device)
                batch_class = torch.tensor([all_class[i] for i in idx], device=self._device)

                embs = torch.stack([self._embed_no_grad_off(t) for t in batch_texts])

                machine_mask = batch_binary == 0
                human_mask = batch_binary == 1

                # 1) Classification cross-entropy (machine text only)
                loss_classify = torch.tensor(0.0, device=self._device)
                if machine_mask.any():
                    logits_m = self._classifier(embs[machine_mask])
                    targets_m = batch_class[machine_mask]
                    loss_classify = F.cross_entropy(logits_m, targets_m)

                # 2) Energy margin regularisation
                logits_all = self._classifier(embs)
                energy_all = -torch.logsumexp(logits_all, dim=-1)
                loss_energy = torch.tensor(0.0, device=self._device)
                if machine_mask.any():
                    loss_energy = (
                        loss_energy + F.relu(energy_all[machine_mask] - self.m_in).pow(2).mean()
                    )
                if human_mask.any():
                    loss_energy = (
                        loss_energy + F.relu(self.m_out - energy_all[human_mask]).pow(2).mean()
                    )

                # 3) Contrastive loss
                loss_contrastive = _contrastive_loss(embs, batch_binary)

                loss = alpha * loss_contrastive + beta * (
                    loss_classify + energy_weight * loss_energy
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                n_batches += 1

            logger.info(
                "Epoch %d/%d — loss %.4f", epoch + 1, epochs, total_loss / max(n_batches, 1)
            )

        self.enc_model.eval()
        self._classifier.eval()
        if save_path is not None:
            torch.save(
                {
                    "encoder": self.enc_model.state_dict(),
                    "classifier": self._classifier.state_dict(),
                },
                save_path,
            )
            logger.info("Saved Energy checkpoint to '%s'", save_path)

    def _embed_no_grad_off(self, text: str) -> torch.Tensor:
        """Embed with gradient tracking (for training)."""
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
