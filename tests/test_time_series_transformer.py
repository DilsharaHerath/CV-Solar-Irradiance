import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v_transformer import TimeSeriesTransformerEncoder


def test_time_series_transformer_encoder_forward():
    model = TimeSeriesTransformerEncoder()
    dummy = torch.randn(2, 30, 5)
    out = model(dummy)
    assert out.shape == (2, 768)
