import pathlib

import pandas as pd
from PIL import Image

from v_transformer import SampleConfig, SolarNowcastingDataset, compute_normalization_bundle, create_dataloader


def build_dummy_dataset(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2024-01-01 12:00", periods=40, freq="1min")
    records = []
    for ts in timestamps:
        filename = image_root / f"{ts:%Y%m%d%H%M}.png"
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(filename)
        records.append(
            {
                "Date": ts,
                "GHI": 500.0,
                "DNI": 450.0,
                "DHI": 120.0,
                "temperature": 25.0,
                "pressure": 1013.0,
            }
        )

    csv_path = tmp_path / "timeseries.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path, image_root


def test_dataset_shapes(tmp_path):
    csv_path, image_root = build_dummy_dataset(tmp_path)
    config = SampleConfig(history_minutes=6, clip_minutes=3, horizon_minutes=2)
    dataset = SolarNowcastingDataset(
        csv_path=csv_path,
        image_root=image_root,
        sample_config=config,
        min_sun_elevation=0.0,
        image_size=64,
    )
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["ts_inputs"].shape == (6, 5)
    assert sample["asi_clip"].shape == (3, 3, 64, 64)
    assert sample["targets"].shape == (2,)
    assert sample["persistence"].shape == (2,)

    subset = dataset.with_time_range(start_time=pd.Timestamp("2024-01-01 12:05"))
    assert len(subset) <= len(dataset)


def test_normalization_bundle(tmp_path):
    csv_path, image_root = build_dummy_dataset(tmp_path)
    config = SampleConfig(history_minutes=6, clip_minutes=3, horizon_minutes=2)
    dataset = SolarNowcastingDataset(
        csv_path=csv_path,
        image_root=image_root,
        sample_config=config,
        min_sun_elevation=0.0,
        image_size=64,
    )

    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["ts_inputs"].shape[-1] == 5

    bundle = compute_normalization_bundle(dataset, batch_size=4, num_workers=0, max_batches=1)
    assert bundle.ts.mean.shape == (1, 1, 5)
    assert bundle.images.mean.shape[0] == 1
