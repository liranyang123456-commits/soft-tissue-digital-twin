"""Deformation-sequence figure (Fig 4 expansion): 2x2 grid of the tracked
deformation at four frames of the pulling scene, showing the deformation
evolution over time. Writes ``submission/cmpb/figures/fig_deform_seq.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402

NPZ = ROOT / "outputs" / "soft_tissue_twin_v2" / "tier2_deformation" / "boundary_conditions.npz"
SCENE = ROOT.parent / "datasets" / "endonerf" / "pulling_soft_tissues"
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_deform_seq.png"

FRAMES = [0, 2, 4, 7]  # indices into the tracked sequence (early->late)


def project(pts, K, w2c):
    cam = (pts - w2c[:3, 3]) @ w2c[:3, :3]
    z = cam[:, 2].clamp_min(1e-6)
    uv = torch.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                      K[1, 1] * cam[:, 1] / z + K[1, 2]], dim=-1)
    return uv, z


def quiver_ax(ax, canonical, disp, image, K, w2c, frame_id, max_arrows=350):
    deformed = canonical + disp
    uv0, z0 = project(canonical, K, w2c)
    uv1, _ = project(deformed, K, w2c)
    H, W = image.shape[:2]
    inside = (uv0[:, 0] >= 0) & (uv0[:, 0] < W) & (uv0[:, 1] >= 0) & (uv0[:, 1] < H) & (z0 > 0)
    idx = torch.nonzero(inside).squeeze(-1)
    if idx.numel() > max_arrows:
        idx = idx[torch.randperm(idx.numel())[:max_arrows]]
    mag = disp[idx].norm(dim=-1).numpy()
    ax.imshow(image)
    q = ax.quiver(uv0[idx, 0].numpy(), uv0[idx, 1].numpy(),
                  (uv1[idx, 0] - uv0[idx, 0]).numpy(),
                  (uv1[idx, 1] - uv0[idx, 1]).numpy(),
                  mag, cmap="turbo", angles="xy", scale_units="xy",
                  scale=1.5, width=0.004)
    ax.set_title(f"frame {frame_id}, mean|D| {mag.mean():.2f}", fontsize=10)
    ax.axis("off")
    return q


def main():
    d = np.load(NPZ)
    canonical = torch.from_numpy(d["canonical_positions"]).float()
    disps = torch.from_numpy(d["displacements"]).float()
    frame_ids = d["frame_ids"].tolist()

    scene = load_endonerf_scene(SCENE, max_frames=max(frame_ids) + 1, image_scale=0.5)
    by_id = {f.view_id: f for f in scene.frames}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    last_q = None
    for ax, fi in zip(axes.flat, FRAMES):
        fid = frame_ids[fi]
        frame = by_id[fid]
        img = frame.image.permute(1, 2, 0).numpy()
        K = frame.camera.K
        w2c = torch.linalg.inv(frame.camera.c2w)
        last_q = quiver_ax(ax, canonical, disps[fi], img, K, w2c, fid)
    fig.suptitle("Tier-2 deformation sequence (EndoNeRF pulling): "
                 "displacement grows and localizes at the tool contact",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if last_q is not None:
        fig.colorbar(last_q, ax=axes, label="displacement (scene units)",
                     fraction=0.025, pad=0.01)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
