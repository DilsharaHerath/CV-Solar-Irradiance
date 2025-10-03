"""Dataset and dataloader utilities for the solar nowcasting pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

try:
    RESAMPLE = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:  # pragma: no cover
    RESAMPLE = Image.BILINEAR

TIMESTAMP_FORMATS: Sequence[str] = (
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
    "%Y-%m-%d_%H%M%S",
    "%Y-%m-%d_%H-%M-%S",
    "%Y-%m-%d_%H-%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)


def _parse_timestamp_from_name(name: str) -> Optional[pd.Timestamp]:
    stem = Path(name).stem
    candidates = {stem}
    sanitized = stem.replace("-", "").replace("_", "").replace(":", "")
    candidates.add(sanitized)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        candidates.add(digits)
    for candidate in candidates:
        try:
            ts_inferred = pd.to_datetime(candidate, errors="raise")
            if not pd.isna(ts_inferred):
                return ts_inferred.floor("min")
        except (ValueError, TypeError):
            ts_inferred = None
        for fmt in TIMESTAMP_FORMATS:
            try:
                ts = pd.to_datetime(candidate, format=fmt, errors="raise")
                return ts.floor("min")
            except (ValueError, TypeError):
                continue
    return None


def _build_image_index(image_root: Path) -> Dict[pd.Timestamp, Path]:
    index: Dict[pd.Timestamp, Path] = {}
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for pattern in patterns:
        for path in image_root.rglob(pattern):
            ts = _parse_timestamp_from_name(path.name)
            if ts is None:
                continue
            if ts not in index:
                index[ts] = path
    if not index:
        raise RuntimeError(f"No images indexed under {image_root}")
    return index


@dataclass
class SampleConfig:
    history_minutes: int = 30
    clip_minutes: int = 5
    horizon_minutes: int = 20


class SolarNowcastingDataset(Dataset):
    """Loads paired irradiance time-series and all-sky image clips."""

    def __init__(
        self,
        csv_path: Path | str,
        image_root: Path | str,
        *,
        metadata_df: Optional[pd.DataFrame] = None,
        image_index: Optional[Dict[pd.Timestamp, Path]] = None,
        timestamp_col: str = "Date",
        feature_cols: Sequence[str] = ("GHI", "DNI", "DHI", "temperature", "pressure"),
        target_col: str = "GHI",
        sample_config: SampleConfig = SampleConfig(),
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        min_sun_elevation: Optional[float] = 10.0,
        image_size: int = 128,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        augmentation: Optional[Callable[[Tensor], Tensor]] = None,
        apply_filters: bool = True,
        csv_encoding: Optional[str] = "utf-8",
        encoding_errors: str = "ignore",
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.timestamp_col = timestamp_col
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.sample_config = sample_config
        self.image_size = image_size
        self.transform = transform
        self.augmentation = augmentation
        self.min_sun_elevation = min_sun_elevation
        self.csv_encoding = csv_encoding
        self.encoding_errors = encoding_errors

        if metadata_df is None:
            read_kwargs = {"parse_dates": [self.timestamp_col]}
            if self.csv_encoding is not None:
                read_kwargs["encoding"] = self.csv_encoding
            read_kwargs["encoding_errors"] = self.encoding_errors
            try:
                df = pd.read_csv(self.csv_path, **read_kwargs)
            except ValueError as exc:
                if "Missing column provided to 'parse_dates'" not in str(exc):
                    raise
                col_names = [self.timestamp_col] + list(self.feature_cols)
                fallback_kwargs = {"names": col_names, "header": None, "comment": "#", "parse_dates": [0], "encoding_errors": self.encoding_errors}
                if self.csv_encoding is not None:
                    fallback_kwargs["encoding"] = self.csv_encoding
                df = pd.read_csv(self.csv_path, **fallback_kwargs)
                df.rename(columns={col_names[0]: self.timestamp_col}, inplace=True)
            df.sort_values(self.timestamp_col, inplace=True)
            df.set_index(self.timestamp_col, inplace=True)
            if apply_filters and self.min_sun_elevation is not None and "sun_elev_deg" in df.columns:
                df = df[df["sun_elev_deg"] >= self.min_sun_elevation]
            self.df = df
        else:
            self.df = metadata_df

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        self.image_index = image_index or _build_image_index(self.image_root)
        self.available_times = set(self.df.index)
        self.start_time = start_time
        self.end_time = end_time
        self.sample_timestamps = self._build_samples()

    def _build_samples(self) -> List[pd.Timestamp]:
        cfg = self.sample_config
        dt = pd.Timedelta(minutes=1)
        history_offset = pd.timedelta_range(end=0, periods=cfg.history_minutes, freq=dt)
        future_offset = pd.timedelta_range(start=dt, periods=cfg.horizon_minutes, freq=dt)
        clip_offset = pd.timedelta_range(end=0, periods=cfg.clip_minutes, freq=dt)
        samples: List[pd.Timestamp] = []
        skipped_nonfinite = 0

        for ts in self.df.index:
            if self.start_time is not None and ts < self.start_time:
                continue
            if self.end_time is not None and ts > self.end_time:
                continue
            if self.min_sun_elevation is not None and "sun_elev_deg" in self.df.columns:
                if self.df.at[ts, "sun_elev_deg"] < self.min_sun_elevation:
                    continue

            history_times = ts + history_offset
            future_times = ts + future_offset
            clip_times = ts + clip_offset

            if not all(t in self.available_times for t in history_times):
                continue
            if not all(t in self.available_times for t in future_times):
                continue
            if not all(t.floor("min") in self.image_index for t in clip_times):
                continue

            history_block = self.df.loc[history_times, self.feature_cols].to_numpy(dtype="float32", copy=False)
            future_targets = self.df.loc[future_times, self.target_col].to_numpy(dtype="float32", copy=False)
            if not np.isfinite(history_block).all() or not np.isfinite(future_targets).all():
                skipped_nonfinite += 1
                continue
            samples.append(ts)

        if skipped_nonfinite:
            logger.info("Skipped %d sample(s) with non-finite feature or target values", skipped_nonfinite)
        return samples

    def __len__(self) -> int:
        return len(self.sample_timestamps)

    def _load_clip(self, timestamps: Iterable[pd.Timestamp]) -> Tensor:
        frames: List[Tensor] = []
        for ts in timestamps:
            path = self.image_index[ts.floor("min")]
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.image_size is not None:
                    img = img.resize((self.image_size, self.image_size), RESAMPLE)
                tensor = TF.to_tensor(img)
            frames.append(tensor)
        clip = torch.stack(frames, dim=0)
        if self.augmentation is not None:
            clip = self.augmentation(clip)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str]:
        ts = self.sample_timestamps[idx]
        cfg = self.sample_config
        dt = pd.Timedelta(minutes=1)

        history_times = ts + pd.timedelta_range(end=0, periods=cfg.history_minutes, freq=dt)
        future_times = ts + pd.timedelta_range(start=dt, periods=cfg.horizon_minutes, freq=dt)
        clip_times = ts + pd.timedelta_range(end=0, periods=cfg.clip_minutes, freq=dt)

        ts_features = self.df.loc[history_times, self.feature_cols].to_numpy(dtype="float32")
        ghi_history = self.df.loc[history_times, self.target_col].to_numpy(dtype="float32")
        targets = self.df.loc[future_times, self.target_col].to_numpy(dtype="float32")

        asi_clip = self._load_clip(clip_times)

        sample: Dict[str, Tensor | str] = {
            "ts_inputs": torch.from_numpy(ts_features),
            "asi_clip": asi_clip,
            "targets": torch.from_numpy(targets),
            "ghi_history": torch.from_numpy(ghi_history),
            "timestamp": ts.isoformat(),
        }
        persistence = torch.tensor([ghi_history[-1]] * cfg.horizon_minutes, dtype=torch.float32)
        sample["persistence"] = persistence
        return sample

    def with_time_range(
        self,
        *,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        apply_filters: bool = False,
    ) -> "SolarNowcastingDataset":
        return SolarNowcastingDataset(
            csv_path=self.csv_path,
            image_root=self.image_root,
            metadata_df=self.df,
            image_index=self.image_index,
            timestamp_col=self.timestamp_col,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            sample_config=self.sample_config,
            start_time=start_time,
            end_time=end_time,
            min_sun_elevation=self.min_sun_elevation,
            image_size=self.image_size,
            transform=self.transform,
            augmentation=self.augmentation,
            apply_filters=apply_filters,
            csv_encoding=self.csv_encoding,
            encoding_errors=self.encoding_errors,
        )


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


__all__ = ["SolarNowcastingDataset", "SampleConfig", "create_dataloader"]
