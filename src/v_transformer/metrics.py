"""Metric utilities for solar nowcasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


def mae(pred: Tensor, target: Tensor, dim: int = 0) -> Tensor:
    return (pred - target).abs().mean(dim=dim)


def rmse(pred: Tensor, target: Tensor, dim: int = 0) -> Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=dim))


def mbe(pred: Tensor, target: Tensor, dim: int = 0) -> Tensor:
    return (pred - target).mean(dim=dim)


def skill_score(model_metric: Tensor, baseline_metric: Tensor) -> Tensor:
    return 1 - (model_metric / baseline_metric.clamp_min(1e-6))


@dataclass
class MetricResult:
    rmse: Tensor
    mae: Tensor
    mbe: Tensor
    skill_rmse: Optional[Tensor] = None

    def to_dict(self) -> Dict[str, Tensor]:
        data = {
            "rmse": self.rmse,
            "mae": self.mae,
            "mbe": self.mbe,
        }
        if self.skill_rmse is not None:
            data["skill_rmse"] = self.skill_rmse
        return data


class MetricAggregator:
    """Accumulates per-horizon metrics for nowcasting forecasts."""

    def __init__(self, horizon: int = 20) -> None:
        self.horizon = horizon
        self.reset()

    def reset(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []
        self._baseline: list[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor, baseline: Optional[Tensor] = None) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must share shape")
        if preds.shape[-1] != self.horizon:
            raise ValueError(f"expected horizon {self.horizon}, got {preds.shape[-1]}")
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())
        if baseline is not None:
            if baseline.shape != targets.shape:
                raise ValueError("baseline must match target shape")
            self._baseline.append(baseline.detach().cpu())

    def has_data(self) -> bool:
        return bool(self._preds)

    def compute(self) -> MetricResult:
        preds = torch.cat(self._preds, dim=0)
        targets = torch.cat(self._targets, dim=0)
        rmse_val = rmse(preds, targets, dim=0)
        mae_val = mae(preds, targets, dim=0)
        mbe_val = mbe(preds, targets, dim=0)

        skill = None
        if self._baseline:
            baseline = torch.cat(self._baseline, dim=0)
            baseline_rmse = rmse(baseline, targets, dim=0)
            skill = skill_score(rmse_val, baseline_rmse)

        return MetricResult(rmse=rmse_val, mae=mae_val, mbe=mbe_val, skill_rmse=skill)

    @staticmethod
    def summarize_per_horizon(metrics: MetricResult) -> Dict[str, float]:
        summary = {}
        for name, tensor in metrics.to_dict().items():
            summary[f"{name}_mean"] = tensor.mean().item()
        return summary


__all__ = [
    "mae",
    "rmse",
    "mbe",
    "skill_score",
    "MetricResult",
    "MetricAggregator",
]
