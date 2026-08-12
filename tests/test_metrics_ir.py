"""Unit tests for intrinsic-reflectance metrics."""
from __future__ import annotations

import builtins
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mvbrdf_shr.metrics_ir import (
    aggregate_metrics,
    decomposition_metrics,
    linear_aligned_albedo_metrics,
    masked_psnr,
    masked_ssim,
    multiview_albedo_consistency,
    normal_angular_metrics,
    optional_lpips,
    bidirectional_chamfer,
    orb_image_similarity,
    orb_normal_cosine_distance,
    orb_scale_invariant_albedo_psnr,
    orb_scene_depth_mse,
    reconstruction_residual,
    scale_aligned_albedo_metrics,
    world_normal_to_orb_camera,
)


def test_masked_image_metrics_respect_mask() -> None:
    target = torch.zeros(3, 8, 8)
    pred = target.clone()
    pred[:, 0, 0] = 1.0
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    mask[:, 0, 0] = False

    assert math.isinf(masked_psnr(pred, target, mask))
    assert masked_ssim(target, target, mask, window_size=3) == pytest.approx(1.0)
    assert masked_psnr(pred, target, torch.zeros_like(mask)) is None


def test_albedo_scale_and_affine_alignment() -> None:
    pred = torch.linspace(0.1, 0.9, 24).reshape(3, 2, 4)
    scale_target = pred * 2.5
    scale_result = scale_aligned_albedo_metrics(pred, scale_target)
    assert scale_result["scale"] == pytest.approx(2.5)
    assert scale_result["rmse"] == pytest.approx(0.0, abs=1e-6)

    affine_target = pred * 1.7 + 0.2
    affine_result = linear_aligned_albedo_metrics(pred, affine_target)
    assert affine_result["scale"] == pytest.approx(1.7, rel=1e-5)
    assert affine_result["bias"] == pytest.approx(0.2, rel=1e-5)
    assert affine_result["mae"] == pytest.approx(0.0, abs=1e-6)

    broadcast_mask = torch.ones(1, 2, 4)
    masked_result = scale_aligned_albedo_metrics(
        pred, scale_target, broadcast_mask
    )
    assert masked_result["scale"] == pytest.approx(2.5)


def test_optional_lpips_returns_none_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def reject_lpips(name: str, *args: object, **kwargs: object) -> object:
        if name == "lpips":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_lpips)
    image = torch.zeros(3, 8, 8)
    assert optional_lpips(image, image) is None


def test_normal_angular_metrics_and_thresholds() -> None:
    target = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1).expand(3, 1, 2)
    pred = target.clone()
    pred[:, 0, 1] = torch.tensor([0.0, 1.0, 0.0])
    result = normal_angular_metrics(pred, target, thresholds=(45.0, 90.0))

    assert result["mean"] == pytest.approx(45.0)
    assert result["median"] == pytest.approx(0.0)
    assert result["acc_45"] == pytest.approx(0.5)
    assert result["acc_90"] == pytest.approx(1.0)


def test_orb_compatible_metrics_align_scale_and_erode_mask() -> None:
    mask = torch.ones(1, 9, 9)
    target_albedo = torch.stack(
        [
            torch.full((9, 9), 0.2),
            torch.full((9, 9), 0.4),
            torch.full((9, 9), 0.6),
        ]
    )
    pred_albedo = target_albedo / torch.tensor([2.0, 4.0, 3.0])[:, None, None]
    assert orb_scale_invariant_albedo_psnr(
        pred_albedo, target_albedo, mask
    ) > 120.0

    normal = torch.tensor([0.0, 0.0, 1.0])[:, None, None].expand(3, 9, 9)
    assert orb_normal_cosine_distance(normal, normal, mask) == pytest.approx(0.0)

    pred_depth = torch.ones(1, 9, 9)
    target_depth = torch.full((1, 9, 9), 3.0)
    assert orb_scene_depth_mse(
        [pred_depth], [target_depth], [mask]
    ) == pytest.approx(0.0)


def test_orb_image_metrics_match_official_psnr_and_require_real_hdr() -> None:
    pytest.importorskip("cv2")
    official_root = (
        Path(__file__).parents[1] / "baselines_ext" / "Stanford-ORB"
    )
    sys.path.insert(0, str(official_root))
    try:
        from orb.utils.metrics import calc_PSNR

        target = torch.linspace(0.02, 1.4, 3 * 9 * 9).reshape(3, 9, 9)
        pred = target * torch.tensor([0.7, 1.2, 0.9])[:, None, None] + 0.01
        mask = torch.ones(1, 9, 9)
        ours = orb_image_similarity(
            pred,
            target,
            mask,
            scale_invariant=True,
            hdr_available=True,
        )
        pred_np = pred.permute(1, 2, 0).numpy()
        target_np = target.permute(1, 2, 0).numpy()
        mask_np = mask[0].numpy().astype(np.float32)
        official_h, _, _ = calc_PSNR(
            pred_np.copy(),
            target_np.copy(),
            mask_np.copy(),
            max_value=4,
            use_gt_median=True,
            tonemapping=False,
            divide_mask=False,
            scale_invariant=True,
        )
        official_l, _, _ = calc_PSNR(
            pred_np.copy(),
            target_np.copy(),
            mask_np.copy(),
            max_value=1,
            use_gt_median=False,
            tonemapping=True,
            divide_mask=False,
            scale_invariant=True,
        )
        assert ours["psnr_h"] == pytest.approx(official_h, abs=2e-5)
        assert ours["psnr_l"] == pytest.approx(official_l, abs=2e-5)
    finally:
        sys.path.remove(str(official_root))

    ldr_only = orb_image_similarity(
        pred.clamp(0, 1),
        target.clamp(0, 1),
        mask,
        scale_invariant=False,
        hdr_available=False,
    )
    assert ldr_only["psnr_h"] is None
    assert ldr_only["psnr_l"] is not None


def test_orb_scene_depth_matches_official_evaluator() -> None:
    pytest.importorskip("cv2")
    official_root = Path(__file__).parents[1] / "baselines_ext" / "Stanford-ORB"
    sys.path.insert(0, str(official_root))
    try:
        from orb.utils.metrics import calc_depth_distance_per_scene

        predictions = [
            torch.linspace(1.0, 2.0, 121).reshape(1, 11, 11),
            torch.linspace(1.5, 2.5, 121).reshape(1, 11, 11),
        ]
        targets = [prediction * 1.7 + 0.05 for prediction in predictions]
        masks = [torch.ones(1, 11, 11) for _ in predictions]
        ours = orb_scene_depth_mse(predictions, targets, masks)
        official = calc_depth_distance_per_scene(
            [value[0].numpy() for value in predictions],
            [value[0].numpy() for value in targets],
            [value[0].numpy().astype(np.float32) for value in masks],
        )
        assert ours == pytest.approx(official, abs=1e-7)
    finally:
        sys.path.remove(str(official_root))


def test_orb_camera_normal_export_and_shape_chamfer() -> None:
    normal_world = torch.tensor([0.0, 1.0, 0.0])[:, None, None]
    normal_camera = world_normal_to_orb_camera(normal_world, torch.eye(4))
    assert torch.allclose(
        normal_camera[:, 0, 0], torch.tensor([0.0, -1.0, 0.0])
    )

    target = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    predicted = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # pred->target: (0 + 1)/2; target->pred: (0 + 1)/2; average = 0.5.
    assert bidirectional_chamfer(predicted, target) == pytest.approx(0.5)


def test_decomposition_and_reconstruction_metrics() -> None:
    diffuse = torch.full((3, 8, 8), 0.4)
    specular = torch.full((3, 8, 8), 0.1)
    metrics = decomposition_metrics(diffuse, diffuse, specular, specular)
    assert metrics["diffuse_mae"] == 0.0
    assert metrics["specular_ssim"] == pytest.approx(1.0)

    residual = reconstruction_residual(diffuse + specular, diffuse, specular)
    assert residual["mae"] == pytest.approx(0.0, abs=1e-7)
    assert residual["rmse"] == pytest.approx(0.0, abs=1e-7)
    assert residual["bias"] == pytest.approx(0.0, abs=1e-7)


def test_multiview_consistency_ignores_invalid_views() -> None:
    samples = torch.tensor(
        [
            [[1.0], [3.0], [100.0]],
            [[2.0], [2.0], [2.0]],
            [[7.0], [8.0], [9.0]],
        ]
    )
    valid = torch.tensor(
        [
            [[True], [True], [False]],
            [[True], [True], [True]],
            [[True], [False], [False]],
        ]
    )
    result = multiview_albedo_consistency(samples, valid.squeeze(-1))
    assert result["robust_variance"] == pytest.approx(1.0)
    assert result["mad"] == pytest.approx(0.5)


def test_metric_aggregation_ignores_missing_and_nonfinite() -> None:
    result = aggregate_metrics(
        [
            {"psnr": 20.0, "lpips": None, "bad": float("nan")},
            {"psnr": 30.0, "lpips": 0.2},
        ]
    )
    assert result == {"bad": None, "lpips": 0.2, "psnr": 25.0}
