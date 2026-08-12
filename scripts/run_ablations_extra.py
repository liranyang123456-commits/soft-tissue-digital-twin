"""Quick ablations for the CMPB paper: Tier-1 #Gaussians and Tier-2 knn_k.

Runs on EndoNeRF pulling and writes results to
``outputs/ablations_extra.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene, _depth_to_points  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light  # noqa: E402

SCENE = ROOT.parent / "datasets" / "endonerf" / "pulling_soft_tissues"
OUT = ROOT / "outputs" / "ablations_extra.json"


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float(20 * np.log10(1.0 / max(np.sqrt(mse), 1e-6)))


def run_tier1(scene, device, n_points, steps=600):
    ref = scene.frames[0]
    pts, cols = _depth_to_points(ref, max_points=n_points)
    pts, cols = pts.to(device), cols.to(device)
    field = GaussianBRDFField(pts, cols, initial_scale=0.8).to(device)
    with torch.no_grad():
        to_cam = cam.center.to(device).view(1, 3) - field.means if False else None
    cam = ref.camera.to(device)
    with torch.no_grad():
        to_cam = cam.center.view(1, 3) - field.means
        field.normal_raw.copy_(torch.nn.functional.normalize(to_cam, dim=-1))
        field.metalness_logits.fill_(-5.0)
        field.roughness_logits.fill_(2.0)
        field.material_budget_logits[:, 0] = 5.0
        field.material_budget_logits[:, 1] = -2.0
        field.material_budget_logits[:, 2] = -5.0
    renderer = GaussianBRDFRenderer(backend="gsplat")
    log_exposure = torch.nn.Parameter(torch.tensor(0.5, device=device))
    opt = torch.optim.Adam(
        [{"params": [field.means], "lr": 1e-4},
         {"params": [field.log_scales], "lr": 1e-3},
         {"params": [field.quaternions], "lr": 1e-3},
         {"params": [field.normal_raw], "lr": 5e-4},
         {"params": [field.opacity_logits], "lr": 5e-3},
         {"params": [field.base_color_logits], "lr": 2e-2},
         {"params": [field.material_budget_logits], "lr": 5e-3},
         {"params": [log_exposure], "lr": 1e-2}]
    )
    target = ref.image.to(device).permute(1, 2, 0)
    tissue = (ref.depth_gt.to(device) > 0.01).float()
    light = make_endoscopic_light(cam, device)
    for step in range(steps):
        opt.zero_grad()
        out = renderer(field, cam, light=light)
        shaded = out.full[:3].permute(1, 2, 0)
        alb = out.albedo
        albedo = alb[:3].permute(1, 2, 0) if alb.shape[0] >= 3 else alb.permute(1, 2, 0)
        pred = 0.5 * shaded * log_exposure.exp() + 0.5 * albedo * log_exposure.exp()
        if pred.shape[:2] != target.shape[:2]:
            pred = F.interpolate(pred.permute(2, 0, 1).unsqueeze(0), size=target.shape[:2],
                                 mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        m = tissue.unsqueeze(-1)
        loss = ((pred - target).abs() * m).sum() / m.sum().clamp_min(1)
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = renderer(field, cam, light=light)
        shaded = out.full[:3].permute(1, 2, 0)
        alb = out.albedo
        albedo = alb[:3].permute(1, 2, 0) if alb.shape[0] >= 3 else alb.permute(1, 2, 0)
        pred = (0.5 * shaded * log_exposure.exp() + 0.5 * albedo * log_exposure.exp()).clamp(0, 1).cpu().numpy()
    tgt = target.cpu().numpy()
    tis = tissue.cpu().numpy() > 0.5
    return psnr(tgt[tis], pred[tis])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_endonerf_scene(SCENE, max_frames=2, image_scale=0.5)
    results = {"tier1_n_gaussians": {}}
    for n in [4000, 6498, 10000]:
        p = run_tier1(scene, device, n, steps=600)
        results["tier1_n_gaussians"][str(n)] = p
        print(f"  Tier-1 #Gaussians {n}: tissue-PSNR {p:.2f} dB")
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
