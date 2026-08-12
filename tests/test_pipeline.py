"""Unit tests for MVBRDF-SHR (no large weights)."""
from __future__ import annotations

import torch

from mvbrdf_shr.data.synthetic_toy import SyntheticMultiViewToy, collate_batch
from mvbrdf_shr.losses import total_loss
from mvbrdf_shr.models.brdf import energy_violation, f0_from_metalness
from mvbrdf_shr.models.pipeline import MVBRDFSHR


def test_brdf_energy_and_f0():
    albedo = torch.ones(2, 3, 8, 8) * 0.4
    spec = torch.ones(2, 1, 8, 8) * 0.3
    abs_ = torch.ones(2, 1, 8, 8) * 0.5
    viol = energy_violation(albedo, spec, abs_)
    assert viol.shape == (2, 1, 8, 8)
    assert viol.mean() > 0
    f0 = f0_from_metalness(albedo, torch.zeros(2, 1, 8, 8))
    assert torch.allclose(f0, torch.full_like(f0, 0.04))


def test_pipeline_single_view_forward():
    model = MVBRDFSHR(use_generative=True, resolution=64)
    img = torch.rand(2, 3, 64, 64)
    out = model(img, use_refine=True)
    assert out["pred"].shape == img.shape
    assert out["render"].diffuse.shape == img.shape
    assert out["render"].mask.shape == (2, 1, 64, 64)
    assert out["mat"].absorption.shape == (2, 1, 64, 64)


def test_pipeline_multiview_forward():
    model = MVBRDFSHR(use_generative=True, resolution=32)
    img = torch.rand(1, 3, 3, 32, 32)  # B,V,C,H,W
    out = model(img, use_refine=True)
    assert out["pred"].shape == (1, 3, 32, 32)
    assert out["render"].full.shape == (1, 3, 32, 32)


def test_toy_dataset_and_loss_backward():
    ds = SyntheticMultiViewToy(size=4, resolution=32, n_views=3)
    batch = collate_batch([ds[0], ds[1]])
    model = MVBRDFSHR(use_generative=True, resolution=32)
    out = model(batch.image, light_int=batch.light_ints[:, 0], use_refine=True)
    loss, logs = total_loss(out, image_clean=batch.image_clean, mask_gt=batch.mask_gt)
    assert "photo" in logs
    loss.backward()
    grads = [p.grad is not None for p in model.material.parameters() if p.requires_grad]
    assert any(grads)


def test_inference_shapes_no_ckpt():
    model = MVBRDFSHR(use_generative=False, resolution=48)
    img = torch.rand(1, 3, 48, 48)
    out = model(img, use_refine=False)
    assert out["pred"].shape == (1, 3, 48, 48)
