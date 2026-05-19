"""Shared helpers for the NII Yamagishi-Lab AntiDeepfake detector family.

The three AntiDeepfake checkpoints

* ``nii-yamagishilab/wav2vec-large-anti-deepfake``
* ``nii-yamagishilab/hubert-xlarge-anti-deepfake``
* ``nii-yamagishilab/xls-r-2b-anti-deepfake``

share the same lightweight head architecture on top of an SSL frontend:

    raw waveform (1, T) at 16 kHz
        |
        v
    SSL frontend (Wav2Vec2 / HuBERT)         -> [1, T', D]
        |
        v
    transpose to (1, D, T')
        |
        v
    AdaptiveAvgPool1d(output_size=1)         -> [1, D, 1]
        |
        v
    squeeze last dim                          -> [1, D]
        |
        v
    Linear(D -> 2)                            -> [1, 2]  = (fake_logit, real_logit)

The official inference script published on each model card builds the
SSL frontend with ``fairseq``.  fairseq is heavy and brittle to install
(and explicitly broken on recent PyPI for the HuBERT case), so this
module re-implements the same forward pass on top of HuggingFace
``transformers`` (``Wav2Vec2Model`` / ``HubertModel`` via ``AutoModel``).

The model weights are still pulled from each HuggingFace repo as a
single ``model.safetensors`` file, but the state-dict keys -- which were
saved with fairseq's naming convention through
``huggingface_hub.PyTorchModelHubMixin`` -- are translated on the fly to
the equivalent HF transformers names.  Pretraining-only weights
(``final_proj``, ``project_q``, ``quantizer.*`` for wav2vec2;
``label_embs_concat`` for HuBERT) are silently dropped because they are
not used at inference.

Score convention
----------------
The classifier outputs a 2-element logit vector.  Following the
inference script printed on every model card

    probs = softmax(logits, dim=1)         # [B, 2]
    fake_prob = probs[:, 0]                # index 0 = fake
    real_prob = probs[:, 1]                # index 1 = real

so DetectZoo's ``score_ai`` corresponds to ``probs[0]`` and ``score_human``
to ``probs[1]`` -- higher ``score`` means more likely AI / spoof, which
matches the rest of DetectZoo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectzoo.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# All AntiDeepfake checkpoints expect 16 kHz mono input.
SAMPLE_RATE = 16_000

# Filename of the safetensors blob saved by ``PyTorchModelHubMixin`` on
# every nii-yamagishilab/*-anti-deepfake repo.
_SAFETENSORS_FILENAME = "model.safetensors"


# ---------------------------------------------------------------------------
# Audio I/O helpers (same conventions as the other audio detectors in
# DetectZoo: ast_asvspoof, xlsr_sls, ...).
# ---------------------------------------------------------------------------

def load_audio_to_numpy(
    path: Union[str, Path], target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Load an audio file -> mono float32 numpy array at ``target_sr``."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)
        wav = data.astype(np.float32)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1, keepdims=True)
        wav = wav[:, 0]
        if sr != target_sr:
            import torchaudio

            wav_t = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, target_sr
            )
            wav = wav_t.squeeze(0).numpy().astype(np.float32)
        return wav


def normalize_input(input_data: Any, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Accept path / numpy / tensor -> mono float32 numpy at ``sample_rate``."""
    if isinstance(input_data, np.ndarray):
        wav = input_data.astype(np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=0) if wav.shape[0] < wav.shape[1] else wav.mean(axis=1)
        return wav
    if isinstance(input_data, torch.Tensor):
        wav = input_data.detach().to(torch.float32).cpu()
        if wav.dim() == 2:
            wav = wav.mean(dim=0) if wav.shape[0] < wav.shape[1] else wav.mean(dim=1)
        return wav.numpy().astype(np.float32)
    return load_audio_to_numpy(input_data, sample_rate)


# ---------------------------------------------------------------------------
# Fairseq -> HF transformers state-dict translation
# ---------------------------------------------------------------------------

# Prefix used by the published checkpoints (the SSL frontend lives under
# ``self.m_ssl.model = fairseq.Wav2Vec2Model(...)`` in the original
# DeepfakeDetector class).
_SSL_PREFIX = "m_ssl.model."

# Pretraining-only prefixes that we drop at inference.
_DROP_PREFIXES = (
    "final_proj",
    "project_q",
    "quantizer",
    "label_embs_concat",
)


def _translate_fairseq_ssl_state_dict(
    fairseq_sd: Dict[str, torch.Tensor],
    expected_keys: set[str],
) -> Tuple[Dict[str, torch.Tensor], list[str]]:
    """Translate a fairseq Wav2Vec2/HuBERT state-dict into HF transformers naming.

    Parameters
    ----------
    fairseq_sd
        Raw state-dict loaded from ``model.safetensors`` (keys still
        include the ``m_ssl.model.`` prefix and the trailing
        ``proj_fc.weight`` / ``proj_fc.bias`` head tensors).
    expected_keys
        Set of keys the *target* HF model exposes via ``state_dict()``;
        used to pick between legacy ``weight_g``/``weight_v`` and the
        modern ``parametrizations.weight.original0/1`` naming for the
        positional convolution's weight-norm parameters.

    Returns
    -------
    new_sd
        Dict suitable for ``hf_ssl_model.load_state_dict(new_sd, strict=False)``.
    head_kept
        The two head tensor names actually present in ``fairseq_sd``
        (``["proj_fc.weight", "proj_fc.bias"]`` if both exist).  This is
        returned only as a sanity check for the caller.
    """
    new_sd: Dict[str, torch.Tensor] = {}

    pos_conv_legacy = any(
        "pos_conv_embed.conv.weight_g" in k for k in expected_keys
    )
    pos_conv_param = any(
        "pos_conv_embed.conv.parametrizations.weight.original" in k
        for k in expected_keys
    )

    def _put(target: str, value: torch.Tensor) -> None:
        if target in expected_keys:
            new_sd[target] = value

    head_kept: list[str] = []

    for key, val in fairseq_sd.items():
        if key in ("proj_fc.weight", "proj_fc.bias"):
            head_kept.append(key)
            continue
        if not key.startswith(_SSL_PREFIX):
            continue

        k = key[len(_SSL_PREFIX):]

        if any(k.startswith(p) for p in _DROP_PREFIXES) or k == "label_embs_concat":
            continue

        if k == "mask_emb":
            _put("masked_spec_embed", val)
            continue

        if k.startswith("feature_extractor.conv_layers."):
            parts = k.split(".")
            i = parts[2]
            inner = ".".join(parts[3:])
            if inner.startswith("0."):
                tail = inner[len("0."):]
                _put(f"feature_extractor.conv_layers.{i}.conv.{tail}", val)
            elif inner.startswith("2.1."):
                tail = inner[len("2.1."):]
                _put(f"feature_extractor.conv_layers.{i}.layer_norm.{tail}", val)
            continue

        if k.startswith("post_extract_proj."):
            tail = k[len("post_extract_proj."):]
            _put(f"feature_projection.projection.{tail}", val)
            continue

        if k.startswith("layer_norm."):
            tail = k[len("layer_norm."):]
            _put(f"feature_projection.layer_norm.{tail}", val)
            continue

        if k.startswith("encoder.pos_conv.0."):
            tail = k[len("encoder.pos_conv.0."):]
            if tail == "weight_g":
                if pos_conv_legacy:
                    _put("encoder.pos_conv_embed.conv.weight_g", val)
                if pos_conv_param:
                    _put(
                        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
                        val,
                    )
            elif tail == "weight_v":
                if pos_conv_legacy:
                    _put("encoder.pos_conv_embed.conv.weight_v", val)
                if pos_conv_param:
                    _put(
                        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
                        val,
                    )
            else:
                _put(f"encoder.pos_conv_embed.conv.{tail}", val)
            continue

        if k.startswith("encoder.layer_norm."):
            _put(k, val)
            continue

        if k.startswith("encoder.layers."):
            parts = k.split(".")
            i = parts[2]
            inner = ".".join(parts[3:])
            if inner.startswith("self_attn_layer_norm."):
                tail = inner[len("self_attn_layer_norm."):]
                _put(f"encoder.layers.{i}.layer_norm.{tail}", val)
            elif inner.startswith("self_attn."):
                tail = inner[len("self_attn."):]
                _put(f"encoder.layers.{i}.attention.{tail}", val)
            elif inner.startswith("fc1."):
                tail = inner[len("fc1."):]
                _put(f"encoder.layers.{i}.feed_forward.intermediate_dense.{tail}", val)
            elif inner.startswith("fc2."):
                tail = inner[len("fc2."):]
                _put(f"encoder.layers.{i}.feed_forward.output_dense.{tail}", val)
            elif inner.startswith("final_layer_norm."):
                tail = inner[len("final_layer_norm."):]
                _put(f"encoder.layers.{i}.final_layer_norm.{tail}", val)
            continue

    return new_sd, head_kept


# ---------------------------------------------------------------------------
# Detector module: SSL frontend + global-avg-pool + 2-class linear classifier
# ---------------------------------------------------------------------------

class AntiDeepfakeDetectorModule(nn.Module):
    """Pure-HuggingFace re-implementation of the AntiDeepfake DeepfakeDetector.

    Mirrors the architecture printed on every nii-yamagishilab/*-anti-deepfake
    model card -- raw waveform -> SSL frontend -> AdaptiveAvgPool1d ->
    Linear(D -> 2) -- but uses ``transformers.AutoModel`` for the SSL
    frontend instead of fairseq's ``Wav2Vec2Model`` / ``HubertModel``.

    The forward pass follows the model card script exactly:

    1. apply ``F.layer_norm(wav, wav.shape)`` to the (1-D) input waveform
       (the ``do_normalize=True`` behaviour of the corresponding HF
       feature extractor) and add a batch dimension,
    2. extract SSL hidden states via ``AutoModel(input_values).last_hidden_state``,
    3. transpose to ``(B, D, T')`` and adaptive-average-pool over time,
    4. project the pooled embedding to 2 logits via ``proj_fc``.

    The output ``logits[..., 0]`` is the **fake** logit and
    ``logits[..., 1]`` is the **real** logit, matching the score
    convention of the original inference script.
    """

    def __init__(self, ssl_model: nn.Module, hidden_size: int) -> None:
        super().__init__()
        # Wrap the SSL model under ``self.m_ssl.model`` so the structure
        # matches the keys printed in the README (helps debugging).
        self.m_ssl = nn.Module()
        self.m_ssl.model = ssl_model
        self.adap_pool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = nn.Linear(in_features=hidden_size, out_features=2)
        self._hidden_size = hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Run the SSL frontend; return ``[B, T', D]`` hidden states."""
        if wav.ndim == 3:
            wav = wav[:, :, 0]
        out = self.m_ssl.model(wav)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        # Some HF models return a tuple (last_hidden_state, ...).
        return out[0]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(wav)            # [B, T', D]
        emb = emb.transpose(1, 2)                   # [B, D, T']
        pooled = self.adap_pool1d(emb).squeeze(-1)  # [B, D]
        logits = self.proj_fc(pooled)               # [B, 2]
        return logits


# ---------------------------------------------------------------------------
# End-to-end loader: download safetensors, build HF model, load weights
# ---------------------------------------------------------------------------

def build_anti_deepfake_detector(
    model_name: str,
    cache_dir: Path,
    *,
    expected_hidden_size: Optional[int] = None,
) -> AntiDeepfakeDetectorModule:
    """Construct an :class:`AntiDeepfakeDetectorModule` for *model_name*.

    Steps:

    1. Pull ``model.safetensors`` (the only weight file) and ``config.json``
       from the HuggingFace Hub repo into ``cache_dir``.
    2. Build a randomly-initialised HF SSL model from the published
       config (``Wav2Vec2Model`` / ``HubertModel`` depending on
       ``model_type``).
    3. Translate the fairseq-style state dict keys saved in
       ``model.safetensors`` to HF transformers naming and load them
       (skipping pretraining-only tensors).  Also load the two-tensor
       ``proj_fc`` head onto the wrapper.
    4. Return the wired-up :class:`AntiDeepfakeDetectorModule`, which can
       be moved to a device and put in ``eval`` mode by the caller.

    Parameters
    ----------
    model_name
        Full HuggingFace Hub repo id (e.g.
        ``"nii-yamagishilab/wav2vec-large-anti-deepfake"``).
    cache_dir
        Directory the HF Hub downloads should land in.  Both the
        config and the safetensors blob are cached here.
    expected_hidden_size
        Optional sanity check.  If provided, the resolved
        ``config.hidden_size`` must match this value.
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "AntiDeepfake detectors require `transformers`, "
            "`huggingface_hub`, and `safetensors`. Install with:\n"
            "  pip install 'transformers>=4.30' 'huggingface_hub>=0.20' "
            "'safetensors>=0.4' torchaudio soundfile"
        ) from exc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info(
        "AntiDeepfake: downloading checkpoint %s (cache=%s)",
        model_name,
        cache_dir,
    )
    weights_path = hf_hub_download(
        repo_id=model_name,
        filename=_SAFETENSORS_FILENAME,
        cache_dir=str(cache_dir),
    )

    config = AutoConfig.from_pretrained(model_name, cache_dir=str(cache_dir))
    if expected_hidden_size is not None and int(config.hidden_size) != int(
        expected_hidden_size
    ):
        _LOGGER.warning(
            "AntiDeepfake: %s has hidden_size=%d but the wrapper expected "
            "%d -- proceeding with the value from the HF config.",
            model_name,
            config.hidden_size,
            expected_hidden_size,
        )

    fairseq_sd = load_file(weights_path)

    # The HF config bundled with these repos sometimes lists
    # ``conv_bias=True`` even though the actual fairseq-trained feature
    # extractor was bias-free (HuBERT-XLarge case).  Detect from the
    # state dict and override the config so the architecture we build
    # matches the weights we are about to load.
    has_conv_bias = any(
        ".feature_extractor.conv_layers." in k
        and k.endswith(".0.bias")
        for k in fairseq_sd.keys()
    )
    if hasattr(config, "conv_bias") and bool(config.conv_bias) != has_conv_bias:
        _LOGGER.debug(
            "AntiDeepfake: overriding config.conv_bias %s -> %s based on "
            "presence of bias tensors in the safetensors checkpoint.",
            config.conv_bias,
            has_conv_bias,
        )
        config.conv_bias = has_conv_bias

    # Disable masking at construction time -- we always run in eval()
    # but a few of these configs come with non-zero ``apply_spec_augment``
    # or ``mask_time_prob`` values that produce harmless warnings.
    if hasattr(config, "apply_spec_augment"):
        config.apply_spec_augment = False
    if hasattr(config, "mask_time_prob"):
        config.mask_time_prob = 0.0
    if hasattr(config, "mask_feature_prob"):
        config.mask_feature_prob = 0.0

    ssl_model = AutoModel.from_config(config)
    ssl_model.eval()

    expected_keys = set(ssl_model.state_dict().keys())
    new_sd, head_kept = _translate_fairseq_ssl_state_dict(
        fairseq_sd, expected_keys
    )

    # Pre-load visibility: how many fairseq SSL tensors did we actually
    # find a destination for, vs how many we know we're dropping on
    # purpose, vs how many fell through every rule (these last ones are
    # the silent-failure mode -- if non-zero they get logged below).
    n_fairseq_ssl = sum(1 for k in fairseq_sd if k.startswith(_SSL_PREFIX))
    n_intentionally_dropped = sum(
        1 for k in fairseq_sd
        if k.startswith(_SSL_PREFIX)
        and (
            any(k[len(_SSL_PREFIX):].startswith(p) for p in _DROP_PREFIXES)
            or k[len(_SSL_PREFIX):] == "label_embs_concat"
        )
    )
    n_translated = len(new_sd)
    n_unmapped_fairseq = n_fairseq_ssl - n_intentionally_dropped - n_translated

    sample_fairseq = sorted(
        k for k in fairseq_sd if k.startswith(_SSL_PREFIX)
    )[:5]
    sample_hf = sorted(expected_keys)[:5]
    _LOGGER.info(
        "AntiDeepfake/%s: state-dict translation -- "
        "fairseq SSL=%d, HF expected=%d, translated=%d, "
        "intentionally_dropped=%d, unmapped_fairseq=%d.",
        model_name,
        n_fairseq_ssl,
        len(expected_keys),
        n_translated,
        n_intentionally_dropped,
        n_unmapped_fairseq,
    )
    _LOGGER.debug(
        "AntiDeepfake/%s: fairseq sample (5): %s",
        model_name, sample_fairseq,
    )
    _LOGGER.debug(
        "AntiDeepfake/%s: HF expected sample (5): %s",
        model_name, sample_hf,
    )

    missing, unexpected = ssl_model.load_state_dict(new_sd, strict=False)

    # ``masked_spec_embed`` is the only HF SSL parameter we deliberately
    # leave at random init (we explicitly disable SpecAugment above, so
    # its value never participates in the forward pass at inference).
    # Anything else still missing is a real silent-failure: a fairseq key
    # we should have mapped but didn't, or an HF model parameter we
    # forgot to feed. Log loudly and refuse to proceed.
    benign_missing = {"masked_spec_embed"}
    silently_random = [k for k in missing if k not in benign_missing]
    if silently_random:
        _LOGGER.error(
            "AntiDeepfake/%s: %d HF SSL tensor(s) left at RANDOM INIT after "
            "translation -- the model will produce garbage scores. "
            "First 10: %s",
            model_name,
            len(silently_random),
            silently_random[:10],
        )
        raise RuntimeError(
            f"AntiDeepfake/{model_name}: state-dict translation is incomplete: "
            f"{len(silently_random)} expected HF SSL parameter(s) were not "
            f"populated by _translate_fairseq_ssl_state_dict and would have "
            f"silently stayed at random init. First 10: {silently_random[:10]}. "
            f"Fix the rules in _translate_fairseq_ssl_state_dict to cover "
            f"these keys."
        )
    if missing:
        _LOGGER.debug(
            "AntiDeepfake/%s: %d SSL tensor(s) left at random init "
            "(all benign HF-only buffers): %s",
            model_name, len(missing), missing,
        )
    if unexpected:
        _LOGGER.warning(
            "AntiDeepfake/%s: %d unexpected SSL tensor(s) ignored by HF "
            "model: %s%s",
            model_name,
            len(unexpected),
            unexpected[:8],
            " ..." if len(unexpected) > 8 else "",
        )

    detector = AntiDeepfakeDetectorModule(
        ssl_model=ssl_model, hidden_size=int(config.hidden_size)
    )

    head_sd = {
        "proj_fc.weight": fairseq_sd["proj_fc.weight"],
        "proj_fc.bias": fairseq_sd["proj_fc.bias"],
    }
    head_missing, head_unexpected = detector.load_state_dict(head_sd, strict=False)
    head_missing = [k for k in head_missing if k.startswith("proj_fc.")]
    if head_missing or head_unexpected:
        raise RuntimeError(
            f"AntiDeepfake/{model_name}: failed to load the proj_fc head "
            f"(missing={head_missing}, unexpected={head_unexpected}). "
            f"Got head_kept={head_kept!r}."
        )

    detector.eval()
    return detector


# ---------------------------------------------------------------------------
# Single-utterance forward (matches the inference script on every model card)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    detector: AntiDeepfakeDetectorModule,
    wav: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor]:
    """Run the AntiDeepfake forward pass on a single mono 16 kHz utterance.

    Returns ``(score_ai, score_human, logits)`` where ``score_ai`` and
    ``score_human`` are softmax probabilities (so ``score_ai + score_human == 1``)
    and ``logits`` is the raw 2-element logit tensor (still on ``device``).
    """
    wav_tensor = torch.from_numpy(np.ascontiguousarray(wav)).to(
        device=device, dtype=torch.float32
    )
    wav_tensor = F.layer_norm(wav_tensor, wav_tensor.shape)
    wav_tensor = wav_tensor.unsqueeze(0)  # [1, T]

    logits = detector(wav_tensor).view(-1)
    probs = torch.softmax(logits, dim=-1)
    score_ai = float(probs[0].item())     # index 0 = fake
    score_human = float(probs[1].item())  # index 1 = real
    return score_ai, score_human, logits
