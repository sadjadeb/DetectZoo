"""TCM model architecture (HF SSL frontend + Conformer-W2V backend).

Vendored from ``ductuantruong/tcm_add/model.py`` with the fairseq XLSR frontend
replaced by HuggingFace ``Wav2Vec2Model`` (same translation strategy as
:mod:`detectzoo.detectors.audio._anti_deepfake_common`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones

DEFAULT_EMB_SIZE = 144
DEFAULT_HEADS = 4
DEFAULT_KERNEL_SIZE = 31
DEFAULT_NUM_ENCODERS = 4


def sinusoidal_embedding(n_channels: int, dim: int) -> torch.Tensor:
    pe = torch.FloatTensor(
        [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
    )
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


class MyConformer(nn.Module):
    def __init__(
        self,
        emb_size: int = 128,
        heads: int = 4,
        ffmult: int = 4,
        exp_fac: int = 2,
        kernel_size: int = 16,
        n_encoders: int = 1,
    ):
        super().__init__()
        from detectzoo.detectors.audio.tcm._conformer import ConformerBlock

        self.dim_head = int(emb_size / heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders
        self.positional_emb = nn.Parameter(
            sinusoidal_embedding(10000, emb_size), requires_grad=False
        )
        self.encoder_blocks = _get_clones(
            ConformerBlock(
                dim=emb_size,
                dim_head=self.dim_head,
                heads=heads,
                ff_mult=ffmult,
                conv_expansion_factor=exp_fac,
                conv_kernel_size=kernel_size,
            ),
            n_encoders,
        )
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x, device):  # x shape [bs, tiempo, frecuencia]
        del device
        x = x + self.positional_emb[:, : x.size(1), :]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])
        list_attn_weight = []
        for layer in self.encoder_blocks:
            x, attn_weight = layer(x)
            list_attn_weight.append(attn_weight)
        embedding = x[:, 0, :]
        out = self.fc5(embedding)
        return out, list_attn_weight


class SSLModel(nn.Module):
    """XLSR-300M frontend (pure HuggingFace Wav2Vec2Model)."""

    def __init__(self, ssl_model: nn.Module) -> None:
        super().__init__()
        self.model = ssl_model
        self.out_dim = 1024

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
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

        # [batch, length, dim]
        out = self.model(input_tmp)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]


class TCMModel(nn.Module):
    """Full TCM spoofing detector (upstream ``Model`` class)."""

    def __init__(
        self,
        ssl_model: nn.Module,
        *,
        emb_size: int = DEFAULT_EMB_SIZE,
        heads: int = DEFAULT_HEADS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        num_encoders: int = DEFAULT_NUM_ENCODERS,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(ssl_model)
        self.LL = nn.Linear(1024, emb_size)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MyConformer(
            emb_size=emb_size,
            n_encoders=num_encoders,
            heads=heads,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[Any]]:
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out, attn_score = self.conformer(x, self.device)
        return out, attn_score

    @classmethod
    def from_args(cls, args: SimpleNamespace, ssl_model: nn.Module, device: torch.device) -> "TCMModel":
        return cls(
            ssl_model,
            emb_size=int(args.emb_size),
            heads=int(args.heads),
            kernel_size=int(args.kernel_size),
            num_encoders=int(args.num_encoders),
            device=device,
        )
