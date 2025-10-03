"""Time-series transformer encoder aligned with Fabel et al. and tsai."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
        return tensor


class TimeSeriesTransformerEncoder(nn.Module):
    """Transformer encoder for 30×5 irradiance sequences with learnable positions."""

    def __init__(
        self,
        input_features: int = 5,
        seq_len: int = 30,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_features, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Encode batch of time-series windows.

        Args:
            x: Tensor shaped (batch, seq_len, input_features)
        Returns:
            Tensor shaped (batch, d_model) representing `z_ts` embeddings.
        """
        if x.ndim != 3:
            raise ValueError("Expected input of shape (batch, seq_len, features)")
        batch_size, seq_len, _ = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        x = self.input_proj(x)
        x = self.input_dropout(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)

        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return cls_out


__all__ = ["TimeSeriesTransformerEncoder"]
