from __future__ import annotations

import pytest
import torch

from mvbrdf_shr.world.endonerf_protocol import (
    aggregate_metrics,
    endonerf_holdout,
    image_metrics,
    masked_psnr,
)
from scripts.run_endonerf_holdout_twin import interpolate_displacement


def test_holdout_matches_endogaussian_rule():
    split = endonerf_holdout(20)
    assert split.holdout_ids == (1, 9, 17)
    assert 0 in split.train_ids
    assert not set(split.train_ids) & set(split.holdout_ids)


def test_holdout_rejects_invalid_input():
    with pytest.raises(ValueError):
        endonerf_holdout(1)


def test_masked_psnr_ignores_unselected_pixels():
    target = torch.zeros(3, 4, 4)
    prediction = target.clone()
    prediction[:, 3, 3] = 1.0
    mask = torch.ones(4, 4)
    mask[3, 3] = 0.0
    assert masked_psnr(prediction, target, mask) == pytest.approx(99.0)


def test_image_metrics_and_aggregation():
    target = torch.zeros(3, 12, 12)
    prediction = torch.full_like(target, 0.1)
    mask = torch.ones(12, 12)
    row = image_metrics(prediction, target, mask)
    summary = aggregate_metrics([row, row])
    assert row["full_psnr"] == pytest.approx(20.0, rel=1e-5)
    assert summary["n"] == 2
    assert summary["tissue_psnr"] == pytest.approx(20.0, rel=1e-5)


def test_displacement_interpolation_uses_adjacent_training_frames():
    history = {
        0: torch.zeros(2, 3).numpy(),
        2: torch.full((2, 3), 2.0).numpy(),
    }
    result = interpolate_displacement(1, history)
    assert result == pytest.approx(torch.ones(2, 3).numpy())
