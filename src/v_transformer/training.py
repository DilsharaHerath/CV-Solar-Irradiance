"""Training and inference utilities for the solar nowcasting model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .metrics import MetricAggregator
from .model import SolarNowcastingModel

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    mean: Tensor
    std: Tensor

    def normalize(self, tensor: Tensor) -> Tensor:
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device).clamp_min(1e-6)


@dataclass
class NormalizationBundle:
    ts: Optional[NormalizationStats] = None
    images: Optional[NormalizationStats] = None


class SolarNowcastingSystem(nn.Module):
    """Wraps model with normalization and provides convenience methods."""

    def __init__(self, model: Optional[SolarNowcastingModel] = None, normalization: Optional[NormalizationBundle] = None) -> None:
        super().__init__()
        self.model = model or SolarNowcastingModel()
        self.normalization = normalization or NormalizationBundle()

    def _normalize_ts(self, ts_inputs: Tensor) -> Tensor:
        stats = self.normalization.ts
        if stats is None:
            return ts_inputs
        return stats.normalize(ts_inputs)

    def _normalize_images(self, images: Tensor) -> Tensor:
        stats = self.normalization.images
        if stats is None:
            return images
        return stats.normalize(images)

    def forward(self, ts_inputs: Tensor, asi_clip: Tensor) -> Tensor:
        ts_norm = self._normalize_ts(ts_inputs)
        asi_norm = self._normalize_images(asi_clip)
        return self.model(ts_norm, asi_norm)

    def predict(self, ts_inputs: Tensor, asi_clip: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(ts_inputs, asi_clip)


class SolarNowcastingTrainer:
    """Orchestrates training, evaluation, and logging."""

    def __init__(
        self,
        system: SolarNowcastingSystem,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        scaler: Optional[GradScaler] = None,
        log_interval: int = 50,
        grad_clip: Optional[float] = 1.0,
        metrics_eval_mode: bool = True,
        bias_penalty_weight: float = 0.0,
    ) -> None:
        self.system = system.to(device or torch.device("cpu"))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")
        self.scaler = scaler
        self.log_interval = log_interval
        self.grad_clip = grad_clip
        self.criterion = nn.MSELoss()
        self.metrics_eval_mode = metrics_eval_mode
        self.bias_penalty_weight = bias_penalty_weight

    def _compute_loss(self, preds: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        preds_f32 = preds.float()
        targets_f32 = targets.float()
        loss = self.criterion(preds_f32, targets_f32)
        if self.bias_penalty_weight > 0.0:
            bias_penalty = (preds_f32 - targets_f32).mean(dim=0).abs().mean()
            loss = loss + self.bias_penalty_weight * bias_penalty
        return loss, preds_f32

    def _move_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _compute_baseline(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        baseline = batch.get("persistence")
        if baseline is not None:
            return baseline
        ghi_hist = batch.get("ghi_history")
        if ghi_hist is None:
            return None
        last = ghi_hist[:, -1].unsqueeze(-1)
        return last.repeat(1, batch["targets"].shape[-1])

    def train_epoch(self, loader: DataLoader, epoch: int = 0, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.system.train()
        aggregator = MetricAggregator()
        for step, batch in enumerate(loader):
            batch = self._move_batch(batch)
            ts_inputs = batch["ts_inputs"]
            asi_clip = batch["asi_clip"]
            targets = batch["targets"]

            self.optimizer.zero_grad(set_to_none=True)
            prev_step_count = getattr(self.optimizer, "_step_count", 0)

            if self.scaler is not None:
                with autocast():
                    preds = self.system(ts_inputs, asi_clip)
                loss, preds_float = self._compute_loss(preds, targets)
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    logger.error(
                        "Non-finite loss encountered (AMP) at epoch %d step %d; skipping batch. "
                        "preds range=[%.4f, %.4f] targets range=[%.4f, %.4f]",
                        epoch,
                        step + 1,
                        float(torch.nan_to_num(preds).min().item()),
                        float(torch.nan_to_num(preds).max().item()),
                        float(targets.min().item()),
                        float(targets.max().item()),
                    )
                    continue
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.system(ts_inputs, asi_clip)
                loss, preds_float = self._compute_loss(preds, targets)
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    logger.error(
                        "Non-finite loss encountered at epoch %d step %d; skipping batch. "
                        "preds range=[%.4f, %.4f] targets range=[%.4f, %.4f]",
                        epoch,
                        step + 1,
                        float(torch.nan_to_num(preds).min().item()),
                        float(torch.nan_to_num(preds).max().item()),
                        float(targets.min().item()),
                        float(targets.max().item()),
                    )
                    continue
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None and getattr(self.optimizer, "_step_count", 0) > prev_step_count:
                self.scheduler.step()

            baseline = self._compute_baseline(batch)
            if baseline is not None:
                baseline = baseline.detach().cpu()
            metrics_preds = preds_float
            if self.metrics_eval_mode:
                prev_mode = self.system.training
                self.system.eval()
                with torch.no_grad():
                    metrics_preds = self.system(ts_inputs, asi_clip).float()
                self.system.train(prev_mode)

            aggregator.update(metrics_preds.detach().cpu(), targets.detach().cpu(), baseline=baseline)

            if (step + 1) % self.log_interval == 0:
                logger.info("Epoch %d Step %d Loss %.4f", epoch, step + 1, loss.item())

            if max_steps is not None and (step + 1) >= max_steps:
                break

        if not aggregator.has_data():
            return {"rmse_mean": float("nan")}
        metrics = aggregator.compute()
        summary = MetricAggregator.summarize_per_horizon(metrics)
        logger.info("Epoch %d Train RMSE mean %.4f Skill %.4f", epoch, summary["rmse_mean"], summary.get("skill_rmse_mean", float("nan")))
        return summary

    def evaluate(self, loader: DataLoader, tag: str = "val", epoch: int = 0, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.system.eval()
        aggregator = MetricAggregator()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch = self._move_batch(batch)
                preds = self.system(batch["ts_inputs"], batch["asi_clip"])
                loss = self.criterion(preds, batch["targets"])
                total_loss += loss.item()
                batches += 1
                baseline = self._compute_baseline(batch)
                if baseline is not None:
                    baseline = baseline.detach().cpu()
                aggregator.update(preds.cpu(), batch["targets"].cpu(), baseline=baseline)

                if max_steps is not None and (step + 1) >= max_steps:
                    break

        if not aggregator.has_data():
            return {"loss": float("nan")}
        metrics = aggregator.compute()
        summary = MetricAggregator.summarize_per_horizon(metrics)
        avg_loss = total_loss / max(1, batches)
        logger.info("Epoch %d %s Loss %.4f RMSE mean %.4f Skill %.4f", epoch, tag, avg_loss, summary["rmse_mean"], summary.get("skill_rmse_mean", float("nan")))
        summary["loss"] = avg_loss
        return summary

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> None:
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            if val_loader is not None:
                self.evaluate(val_loader, tag="val", epoch=epoch)

    def inference(self, loader: DataLoader) -> Iterable[Tensor]:
        self.system.eval()
        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch(batch)
                preds = self.system(batch["ts_inputs"], batch["asi_clip"])
                yield preds.cpu()


def compute_normalization_bundle(
    dataset,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> NormalizationBundle:
    """Estimate normalization stats from a dataset."""
    device = device or torch.device("cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ts_sum = None
    ts_sq = None
    ts_count = None
    img_sum = None
    img_sq = None
    img_count = None
    ts_invalid = 0
    img_invalid = 0

    for batch_idx, batch in enumerate(loader):
        ts_inputs = batch["ts_inputs"].to(device)
        asi_clip = batch["asi_clip"].to(device)

        ts_flat = ts_inputs.view(-1, ts_inputs.size(-1))
        ts_mask = torch.isfinite(ts_flat)
        ts_clean = torch.nan_to_num(ts_flat, nan=0.0, posinf=0.0, neginf=0.0)
        ts_sum_batch = ts_clean.sum(dim=0)
        ts_sq_batch = (ts_clean ** 2).sum(dim=0)
        ts_count_batch = ts_mask.sum(dim=0).to(ts_inputs.dtype)
        ts_invalid += int((~ts_mask).sum().item())

        if ts_sum is None:
            ts_sum = ts_sum_batch
            ts_sq = ts_sq_batch
            ts_count = ts_count_batch
        else:
            ts_sum = ts_sum + ts_sum_batch
            ts_sq = ts_sq + ts_sq_batch
            ts_count = ts_count + ts_count_batch

        img_flat = asi_clip.view(-1, asi_clip.size(2), asi_clip.size(3), asi_clip.size(4))
        img_mask = torch.isfinite(img_flat)
        img_clean = torch.nan_to_num(img_flat, nan=0.0, posinf=0.0, neginf=0.0)
        img_sum_batch = img_clean.sum(dim=(0, 2, 3))
        img_sq_batch = (img_clean ** 2).sum(dim=(0, 2, 3))
        img_count_batch = img_mask.sum(dim=(0, 2, 3)).to(asi_clip.dtype)
        img_invalid += int((~img_mask).sum().item())

        if img_sum is None:
            img_sum = img_sum_batch
            img_sq = img_sq_batch
            img_count = img_count_batch
        else:
            img_sum = img_sum + img_sum_batch
            img_sq = img_sq + img_sq_batch
            img_count = img_count + img_count_batch

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    if ts_sum is None or img_sum is None:
        raise RuntimeError("Failed to compute normalization statistics: dataset produced no batches")

    if torch.any(ts_count == 0):
        raise RuntimeError("Encountered feature(s) with no finite values while computing time-series normalization stats")
    if torch.any(img_count == 0):
        raise RuntimeError("Encountered channel(s) with no finite values while computing image normalization stats")

    if ts_invalid:
        logger.warning("Ignored %d non-finite time-series values while computing normalization stats", ts_invalid)
    if img_invalid:
        logger.warning("Ignored %d non-finite image values while computing normalization stats", img_invalid)

    ts_mean = ts_sum / ts_count
    ts_var = ts_sq / ts_count - ts_mean ** 2
    ts_std = torch.sqrt(torch.clamp(ts_var, min=1e-8))

    img_mean = img_sum / img_count
    img_var = img_sq / img_count - img_mean ** 2
    img_std = torch.sqrt(torch.clamp(img_var, min=1e-8))

    ts_stats = NormalizationStats(mean=ts_mean.view(1, 1, -1), std=ts_std.view(1, 1, -1))
    img_stats = NormalizationStats(mean=img_mean.view(1, 1, -1, 1, 1), std=img_std.view(1, 1, -1, 1, 1))

    return NormalizationBundle(ts=ts_stats, images=img_stats)


__all__ = [
    "SolarNowcastingSystem",
    "SolarNowcastingTrainer",
    "NormalizationStats",
    "NormalizationBundle",
    "compute_normalization_bundle",
]
