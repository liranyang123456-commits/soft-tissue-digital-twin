"""Render the simulable twin in a deformed (rehearsal) configuration.

Loads the canonical Gaussian field and applies a tracked displacement frame,
then renders the deformed twin -- this is the "rehearsal" output: the twin
driven forward to a new pose. Writes fig_twin_rehearsal.png.
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
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light  # noqa: E402

FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_twin_rehearsal.png"
CKPT = ROOT / "outputs/soft_tissue_twin_v2/tier1_reconstruction/canonical_field.pt"
BC = ROOT / "outputs/soft_tissue_twin_v2/tier2_deformation/boundary_conditions.npz"
SCENE = ROOT.parent / "datasets/endonerf/pulling_soft_tissues"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_endonerf_scene(SCENE, max_frames=1, image_scale=0.5)
    cam = scene.frames[0].camera.to(device)
    light = make_endoscopic_light(cam, device)

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    state = ck["field"] if "field" in ck else ck
    n = state["means"].shape[0]
    field = GaussianBRDFField(state["means"],
                              torch.rand(n, 3, device=device) * 0.5 + 0.25,
                              initial_scale=0.8).to(device)
    field.load_state_dict(state)

    bc = np.load(BC)
    disp = torch.from_numpy(bc["displacements"]).to(device)  # (F, N, 3)
    frame = int(disp.shape[0]) // 2  # a mid-late frame with clear deformation

    renderer = GaussianBRDFRenderer(backend="gsplat")
    with torch.no_grad():
        # canonical render
        field.means.data = torch.from_numpy(bc["canonical_positions"]).to(device)
        canon = renderer(field, cam, light=light).albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        # deformed (rehearsal) render
        field.means.data = torch.from_numpy(bc["canonical_positions"]).to(device) + disp[frame]
        deformed = renderer(field, cam, light=light).albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    axes[0].imshow(canon)
    axes[0].set_title("canonical twin", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(deformed)
    axes[1].set_title(f"rehearsal (deformed, frame {frame})", fontsize=11)
    axes[1].axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    # also save the single deformed render for the architecture figure
    plt.imsave(str(FIG / "fig_twin_rehearsal_single.png"), deformed)
    print(f"wrote {OUT} and single (frame {frame})")


if __name__ == "__main__":
    main()
