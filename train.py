"""Training script for the solar nowcasting transformer pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.cuda.amp import GradScaler

from v_transformer import (
    NormalizationBundle,
    NormalizationStats,
    SampleConfig,
    SolarNowcastingDataset,
    SolarNowcastingModel,
    SolarNowcastingSystem,
    SolarNowcastingTrainer,
    TimeSformerEncoder,
    TimeSeriesTransformerEncoder,
    compute_normalization_bundle,
    create_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the solar nowcasting transformer.")
    parser.add_argument("--image-root", type=Path, default=Path("/storage2/CV_Irradiance/datasets/full_dataset"), help="Root directory containing ASI images")
    parser.add_argument("--csv-path", type=Path, default=Path("/storage2/CV_Irradiance/datasets/full_dataset/PSA_timeSeries_Metas.csv"), help="Path to the irradiance/time-series CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("/storage2/CV_Irradiance/datasets/runs/default"), help="Directory for logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer initial learning rate")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="One-cycle maximum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pct-start", type=float, default=0.1, help="One-cycle warmup percentage")
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--norm-batches", type=int, default=None, help="Limit batches when computing normalization stats")
    parser.add_argument("--device", type=str, default=None, help="Force training device (cpu or cuda)")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--save-normalization", type=Path, default=None, help="Path to save normalization statistics (.pt)")
    parser.add_argument("--log-file", type=str, default="train.log")
    parser.add_argument("--dry-run", action="store_true", help="Run a single-batch sanity check and exit")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save a checkpoint every N epochs (0 to disable)")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--metrics-file", type=str, default="metrics.csv", help="Filename for per-epoch metrics log")
    parser.add_argument("--csv-encoding", type=str, default="utf-8", help="Encoding for the time-series CSV")
    parser.add_argument("--csv-encoding-errors", type=str, default="ignore", help="How to handle CSV encoding errors (e.g. ignore, replace)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (set <=0 to disable)")
    parser.add_argument("--bias-penalty-weight", type=float, default=0.0, help="Weight for absolute MBE penalty added to the training loss")
    return parser.parse_args()


def setup_logging(output_dir: Path, log_file: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def maybe_parse_time(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.to_datetime(value)



def infer_date_ranges(dataset: SolarNowcastingDataset) -> Tuple[Optional[Tuple[pd.Timestamp, pd.Timestamp]], Optional[Tuple[pd.Timestamp, pd.Timestamp]]]:
    index = dataset.df.index.sort_values()
    if index.empty:
        return None, None
    periods = index.to_period("M")
    unique_months = periods.unique()
    if len(unique_months) < 2:
        return (index.min(), index.max()), None

    train_mask = periods != unique_months[-1]
    if train_mask.sum() == 0:
        train_range = (index.min(), index.max())
    else:
        train_range = (index[train_mask].min(), index[train_mask].max())

    val_mask = periods == unique_months[-1]
    if val_mask.sum() == 0:
        val_range = None
    else:
        val_range = (index[val_mask].min(), index[val_mask].max())

    return train_range, val_range


def resolve_time_ranges(args: argparse.Namespace, base_dataset: SolarNowcastingDataset) -> Tuple[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]], Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    train_start = maybe_parse_time(args.train_start)
    train_end = maybe_parse_time(args.train_end)
    val_start = maybe_parse_time(args.val_start)
    val_end = maybe_parse_time(args.val_end)

    if (args.train_start is None) != (args.train_end is None):
        raise ValueError("Specify both --train-start and --train-end or neither")
    if (args.val_start is None) != (args.val_end is None):
        raise ValueError("Specify both --val-start and --val-end or neither")

    if args.train_start is None and args.train_end is None:
        inferred_train, inferred_val = infer_date_ranges(base_dataset)
        if inferred_train is not None:
            train_start, train_end = inferred_train
            logging.info("Inferred train window: %s to %s", train_start, train_end)
        if args.val_start is None and args.val_end is None and inferred_val is not None:
            val_start, val_end = inferred_val
            logging.info("Inferred val window: %s to %s", val_start, val_end)
    else:
        inferred_train, inferred_val = infer_date_ranges(base_dataset)
        if inferred_train is not None:
            logging.info("Using user-defined train window; dataset spans %s to %s", inferred_train[0], inferred_train[1])
        if inferred_val is not None:
            logging.info("Validation month suggestion: %s to %s", inferred_val[0], inferred_val[1])

    return (train_start, train_end), (val_start, val_end)


def verify_dataset(dataset: SolarNowcastingDataset, sample_count: int = 3) -> None:
    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no samples after applying filters/time range")
    sample_count = min(sample_count, len(dataset))
    for idx in range(sample_count):
        sample = dataset[idx]
        clip = sample["asi_clip"]
        ts_inputs = sample["ts_inputs"]
        if clip.shape[0] != dataset.sample_config.clip_minutes:
            raise RuntimeError("Unexpected clip length for sample at index %d" % idx)
        if ts_inputs.shape[0] != dataset.sample_config.history_minutes:
            raise RuntimeError("Unexpected time-series history length for sample at index %d" % idx)
    logging.info("Verified %d sample(s) for image/series alignment", sample_count)


def save_checkpoint(output_dir: Path, epoch: int, system: SolarNowcastingSystem, optimizer, scheduler, scaler) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
    state = {
        "epoch": epoch,
        "model_state": system.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, ckpt_path)
    logging.info("Saved checkpoint: %s", ckpt_path)
    return ckpt_path


def load_checkpoint(path: Path, system: SolarNowcastingSystem, optimizer, scheduler, scaler, device: torch.device) -> int:
    logging.info("Loading checkpoint from %s", path)
    state = torch.load(path, map_location=device)
    system.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    if scaler is not None and "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    return state.get("epoch", 0)


def write_metrics(history: List[Dict[str, float]], path: Path) -> None:
    df = pd.DataFrame(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote metrics to %s", path)

def build_datasets(args: argparse.Namespace) -> tuple[SolarNowcastingDataset, Optional[SolarNowcastingDataset]]:
    sample_cfg = SampleConfig()
    base_dataset = SolarNowcastingDataset(
        csv_path=args.csv_path,
        image_root=args.image_root,
        sample_config=sample_cfg,
        csv_encoding=args.csv_encoding,
        encoding_errors=args.csv_encoding_errors,
    )

    (train_start, train_end), (val_start, val_end) = resolve_time_ranges(args, base_dataset)

    train_dataset = base_dataset.with_time_range(start_time=train_start, end_time=train_end)
    verify_dataset(train_dataset)

    if val_start is None or val_end is None:
        return train_dataset, None

    val_dataset = base_dataset.with_time_range(start_time=val_start, end_time=val_end)
    verify_dataset(val_dataset, sample_count=1)
    return train_dataset, val_dataset


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir, args.log_file)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    train_dataset, val_dataset = build_datasets(args)
    logging.info("Train samples: %d", len(train_dataset))
    if val_dataset is not None:
        logging.info("Validation samples: %d", len(val_dataset))

    norm_bundle = compute_normalization_bundle(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.norm_batches,
        device=device,
    )
    logging.info("Computed normalization statistics")

    if norm_bundle.ts is None or norm_bundle.images is None:
        raise RuntimeError("Normalization stats could not be computed for both modalities")

    if args.save_normalization is None:
        args.save_normalization = args.output_dir / "normalization.pt"
    torch.save(
        {
            "ts_mean": norm_bundle.ts.mean.cpu(),
            "ts_std": norm_bundle.ts.std.cpu(),
            "image_mean": norm_bundle.images.mean.cpu(),
            "image_std": norm_bundle.images.std.cpu(),
        },
        args.save_normalization,
    )
    logging.info("Saved normalization stats to %s", args.save_normalization)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=not args.no_pin_memory and device.type == "cuda",
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty. Reduce batch size or adjust time window filters.")
    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=not args.no_pin_memory and device.type == "cuda",
        )

    ts_encoder = TimeSeriesTransformerEncoder()
    asi_encoder = TimeSformerEncoder()
    model = SolarNowcastingModel(ts_encoder=ts_encoder, asi_encoder=asi_encoder)
    system = SolarNowcastingSystem(model=model, normalization=norm_bundle)

    optimizer = torch.optim.AdamW(system.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
        anneal_strategy="cos",
    )

    scaler = GradScaler() if args.amp and device.type == "cuda" else None

    trainer = SolarNowcastingTrainer(
        system=system,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        log_interval=args.log_interval,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        bias_penalty_weight=args.bias_penalty_weight,
    )

    start_epoch = 1
    history: List[Dict[str, float]] = []
    metrics_path = args.output_dir / args.metrics_file

    if args.resume is not None:
        if args.resume.exists():
            last_epoch = load_checkpoint(args.resume, trainer.system, optimizer, scheduler, scaler, device)
            start_epoch = last_epoch + 1
            logging.info("Resuming training from epoch %d", start_epoch)
        else:
            logging.warning("Checkpoint not found at %s; starting fresh", args.resume)

    if start_epoch > args.epochs:
        logging.info("Configured epochs (%d) already completed in checkpoint; nothing to do.", args.epochs)
        return

    if args.dry_run:
        logging.info("Running dry-run (single batch)")
        trainer.train_epoch(train_loader, epoch=start_epoch, max_steps=1)
        if val_loader is not None:
            trainer.evaluate(val_loader, tag="val", epoch=start_epoch, max_steps=1)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        record: Dict[str, float] = {"epoch": float(epoch)}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})

        if val_loader is not None:
            val_metrics = trainer.evaluate(val_loader, tag="val", epoch=epoch)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})

        history.append(record)
        write_metrics(history, metrics_path)

        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0):
            save_checkpoint(args.output_dir, epoch, trainer.system, optimizer, scheduler, scaler)

    logging.info("Training complete")


if __name__ == "__main__":
    main()
