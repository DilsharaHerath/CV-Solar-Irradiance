import torch

from v_transformer.metrics import MetricAggregator


def test_metric_aggregator_shapes():
    aggregator = MetricAggregator(horizon=3)
    preds = torch.tensor([[1.0, 2.0, 3.0]])
    targets = torch.tensor([[0.0, 1.0, 2.0]])
    baseline = torch.tensor([[0.5, 1.5, 2.5]])
    aggregator.update(preds, targets, baseline=baseline)
    result = aggregator.compute()
    assert result.rmse.shape == (3,)
    assert result.skill_rmse.shape == (3,)
