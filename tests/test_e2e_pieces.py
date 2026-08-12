"""End-to-end and hash-encoder tests."""
from __future__ import annotations

from pathlib import Path

import torch

from mvbrdf_shr.models.hash_encoder import HashGridEncoder, HashMaterialMLP
from mvbrdf_shr.metrics import psnr_torch, ssim_torch


def test_hash_encoder():
    enc = HashGridEncoder(in_dim=3, n_levels=3, features=2, table_size=1024)
    x = torch.rand(4, 8, 3) * 2 - 1
    y = enc(x)
    assert y.shape == (4, 8, enc.out_dim)
    mlp = HashMaterialMLP(enc)
    out = mlp(x)
    assert out.shape[-1] == 7


def test_metrics():
    a = torch.rand(1, 3, 32, 32)
    assert psnr_torch(a, a) > 50
    assert ssim_torch(a, a) > 0.99


def test_demo_dataset_if_present():
    from mvbrdf_shr.data import DemoMultiViewDataset

    root = Path(__file__).resolve().parents[1] / "data" / "demo_mv"
    if not root.exists():
        return
    ds = DemoMultiViewDataset(root=root, split="train", resolution=64, group_views=True)
    if len(ds) == 0:
        return
    b = ds[0]
    assert b.image.dim() in (3, 4)
