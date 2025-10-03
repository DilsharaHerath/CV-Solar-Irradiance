import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v_transformer import SolarNowcastingModel


def test_solar_nowcasting_model_forward():
    model = SolarNowcastingModel()
    ts = torch.randn(2, 30, 5)
    asi = torch.randn(2, 5, 3, 128, 128)
    out = model(ts, asi)
    assert out.shape == (2, 20)
