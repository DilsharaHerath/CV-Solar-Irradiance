"""End-to-end solar nowcasting model combining time-series and image encoders."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .time_series_transformer import TimeSeriesTransformerEncoder
from .vision_transformer import TimeSformerEncoder


class FusionMLP(nn.Module):
    """Two-layer MLP mapping fused embeddings to 20 forecast steps."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 2048, output_dim: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class SolarNowcastingModel(nn.Module):
    """Multimodal transformer for 1–20 min GHI forecasts."""

    def __init__(
        self,
        ts_encoder: TimeSeriesTransformerEncoder | None = None,
        asi_encoder: TimeSformerEncoder | None = None,
        fusion_head: FusionMLP | None = None,
        output_dim: int = 20,
    ) -> None:
        super().__init__()
        self.ts_encoder = ts_encoder or TimeSeriesTransformerEncoder()
        self.asi_encoder = asi_encoder or TimeSformerEncoder()
        default_fusion = FusionMLP(output_dim=output_dim)
        self.fusion_head = fusion_head or default_fusion

    def forward(self, ts_inputs: Tensor, asi_clip: Tensor) -> Tensor:
        """Compute 20-step GHI forecast."""
        z_ts = self.ts_encoder(ts_inputs)
        z_asi = self.asi_encoder(asi_clip)
        fused = torch.cat([z_ts, z_asi], dim=1)
        return self.fusion_head(fused)


__all__ = ["SolarNowcastingModel", "FusionMLP"]
