"""4x4 Tier-2 deformation grid (Fig 4) for the CMPB paper.

4 frames (cols) x 4 aspects (rows) on EndoNeRF pulling:
  row 1: quiver overlay (canonical -> deformed)
  row 2: displacement-magnitude heatmap (projected)
  row 3: deformed-field render
  row 4: |deformed render - actual frame| error map
Writes ``submission/cmpb/figures/fig_deform_4x4.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer  # noqa: E402
from mvbrdf_shr.world.deform import DeformedFieldView  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light  # noqa: E402

NPZ = ROOT / "outputs" / "soft_tissue_twin_v2" / "tier2_deformation" / "boundary_conditions.npz"
CKPT = ROOT / "outputs" / "soft_tissue_twin_v2" / "tier1_reconstruction" / "canonical_field.pt"
SCENE = ROOT.parent / "datasets" / "endonerf" / "pulling_soft_tissues"
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_deform_4x4.png"

FRAMES = [0, 2, 4, 7]
ASPECTS = ["quiver", "displacement |D|", "deformed render", "error map"]


def project(pts, K, w2c):
    cam = (pts - w2c[:3, 3]) @ w2c[:3, :3]
    z = cam[:, 2].clamp_min(1e-6)
    uv = torch.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                      K[1, 1] * cam[:, 1] / z + K[1, 2]], dim=-1)
    return uv, z


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(NPZ)
    canonical_np = d["canonical_positions"]
    disps = d["displacements"]
    frame_ids = d["frame_ids"].tolist()

    scene = load_endonerf_scene(SCENE, max_frames=max(frame_ids) + 1, image_scale=0.5)
    by_id = {f.view_id: f for f in scene.frames}

    # rebuild the canonical field
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    pts = torch.from_numpy(canonical_np).to(device)
    cols = torch.rand(pts.shape[0], 3, device=device) * 0.5 + 0.25
    field = GaussianBRDFField(pts, cols, initial_scale=0.8).to(device)
    field.load_state_dict(ck["field"] if "field" in ck else ck)
    renderer = GaussianBRDFRenderer(backend="gsplat")

    fig, axes = plt.subplots(4, 4, figsize=(15, 13), dpi=150)
    for j, fid in enumerate([frame_ids[i] for i in FRAMES]):
        axes[0, j].set_title(f"frame {fid}", fontsize=11)

    for j, fi in enumerate(FRAMES):
        fid = frame_ids[fi]
        frame = by_id[fid]
        img = frame.image.permute(1, 2, 0).numpy()
        K = frame.camera.K
        w2c = torch.linalg.inv(frame.camera.c2w)
        cam_dev = frame.camera.to(device)
        light = make_endoscopic_light(cam_dev, device)

        canonical = torch.from_numpy(canonical_np).float()
        disp = torch.from_numpy(disps[fi]).float()
        deformed = canonical + disp

        # row 0: quiver
        ax = axes[0, j]
        uv0, z0 = project(canonical, K, w2c)
        uv1, _ = project(deformed, K, w2c)
        H, W = img.shape[:2]
        inside = (uv0[:, 0] >= 0) & (uv0[:, 0] < W) & (uv0[:, 1] >= 0) & (uv0[:, 1] < H) & (z0 > 0)
        idx = torch.nonzero(inside).squeeze(-1)
        idx = idx[torch.randperm(idx.numel())[:350]]
        mag = disp[idx].norm(dim=-1).numpy()
        ax.imshow(img)
        ax.quiver(uv0[idx, 0].numpy(), uv0[idx, 1].numpy(),
                  (uv1[idx, 0] - uv0[idx, 0]).numpy(),
                  (uv1[idx, 1] - uv0[idx, 1]).numpy(),
                  mag, cmap="turbo", angles="xy", scale_units="xy", scale=1.5, width=0.004)
        ax.axis("off")

        # row 1: displacement magnitude heatmap (scatter projected)
        ax = axes[1, j]
        ax.imshow(img, alpha=0.35)
        sc = ax.scatter(uv0[idx, 0].numpy(), uv0[idx, 1].numpy(), c=mag,
                        cmap="turbo", s=6)
        ax.axis("off")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)

        # row 2: deformed render
        ax = axes[2, j]
        with torch.no_grad():
            view = DeformedFieldView(field, disp.to(device))
            out = renderer(view, cam_dev, light=light)
            pred = out.full[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            if pred.shape[:2] != img.shape[:2]:
                pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
                pred = F.interpolate(pred_t, size=img.shape[:2], mode="bilinear",
                                     align_corners=False)[0].permute(1, 2, 0).numpy()
        ax.imshow(pred)
        ax.axis("off")

        # row 3: error map
        ax = axes[3, j]
        err = np.abs(img - pred).mean(axis=-1)
        err = err / max(err.max(), 1e-6)
        ax.imshow(err, cmap="hot", vmin=0, vmax=1)
        ax.axis("off")

    for i, asp in enumerate(ASPECTS):
        axes[i, 0].set_ylabel(asp, fontsize=10, rotation=90, va="center")

    fig.suptitle("Tier-2 deformation on EndoNeRF pulling: 4 frames x 4 aspects",
                 fontsize=13)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
