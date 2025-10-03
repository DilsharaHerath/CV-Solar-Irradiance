import pathlib
import sys

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v_transformer import (
    NormalizationBundle,
    NormalizationStats,
    SolarNowcastingModel,
    SolarNowcastingSystem,
    SolarNowcastingTrainer,
)


class DummyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        ts = torch.randn(30, 5)
        asi = torch.randn(5, 3, 128, 128)
        targets = torch.randn(20)
        baseline = targets.clone()
        return {
            "ts_inputs": ts,
            "asi_clip": asi,
            "targets": targets,
            "persistence": baseline,
        }


def test_trainer_train_epoch_runs():
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)

    ts_stats = NormalizationStats(mean=torch.zeros(1, 1, 5), std=torch.ones(1, 1, 5))
    img_stats = NormalizationStats(mean=torch.zeros(1, 1, 3, 1, 1), std=torch.ones(1, 1, 3, 1, 1))
    normalization = NormalizationBundle(ts=ts_stats, images=img_stats)

    system = SolarNowcastingSystem(model=SolarNowcastingModel(), normalization=normalization)
    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)
    trainer = SolarNowcastingTrainer(system=system, optimizer=optimizer, log_interval=10)

    summary = trainer.train_epoch(loader)
    assert "rmse_mean" in summary

    val_summary = trainer.evaluate(loader, tag="val")
    assert "loss" in val_summary

    preds = list(trainer.inference(loader))
    assert preds[0].shape[1] == 20
