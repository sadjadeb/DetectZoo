"""XLSR-Mamba architecture (HF SSL frontend + dual-column Mamba backend).

Vendored from ``swagshaw/XLSR-Mamba/model.py`` with the fairseq XLSR frontend
replaced by HuggingFace ``Wav2Vec2Model`` (same translation strategy as
:mod:`detectzoo.detectors.audio._anti_deepfake_common`). The Mamba forward
pass and preprocessing are otherwise line-for-line identical to upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

# Default hyper-parameters from ``swagshaw/XLSR-Mamba/main.py``.
DEFAULT_EMB_SIZE = 144
DEFAULT_NUM_ENCODERS = 12


@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8


class SSLModel(nn.Module):
    """XLSR-300M frontend (pure HuggingFace Wav2Vec2Model)."""

    def __init__(self, ssl_model: nn.Module) -> None:
        super().__init__()
        self.model = ssl_model
        self.out_dim = 1024

    def extract_feat(self, input_data: torch.Tensor) -> tuple[torch.Tensor, Any]:
        # put the model to GPU if it not there
        if (
            next(self.model.parameters()).device != input_data.device
            or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # [batch, length, dim] -- HF ``last_hidden_state`` matches fairseq ``['x']``.
        out = self.model(input_tmp)
        if hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state
        else:
            emb = out[0]
        layerresult = None
        return emb, layerresult


class XLSRMambaModel(nn.Module):
    """Full XLSR-Mamba spoofing detector (upstream ``Model`` class)."""

    def __init__(
        self,
        ssl_model: nn.Module,
        *,
        emb_size: int = DEFAULT_EMB_SIZE,
        num_encoders: int = DEFAULT_NUM_ENCODERS,
        fused_add_norm: bool = True,
    ) -> None:
        super().__init__()
        from detectzoo.detectors.audio.xlsr_mamba._mamba_blocks import MixerModel

        args = SimpleNamespace(emb_size=emb_size, num_encoders=num_encoders)
        self.ssl_model = SSLModel(ssl_model)
        self.LL = nn.Linear(1024, args.emb_size)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.config = MambaConfig(
            d_model=args.emb_size,
            n_layer=args.num_encoders // 2,
            fused_add_norm=fused_add_norm,
        )
        self.conformer = MixerModel(
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            ssm_cfg=self.config.ssm_cfg,
            rms_norm=self.config.rms_norm,
            residual_in_fp32=self.config.residual_in_fp32,
            fused_add_norm=self.config.fused_add_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat, _ = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1)  # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out = self.conformer(x)
        return out
