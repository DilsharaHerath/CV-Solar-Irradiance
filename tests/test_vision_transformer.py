import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v_transformer import TimeSformerEncoder


def test_timesformer_encoder_forward():
    model = TimeSformerEncoder()
    dummy = torch.randn(2, 5, 3, 128, 128)
    out = model(dummy)
    assert out.shape == (2, model.embed_dim)
