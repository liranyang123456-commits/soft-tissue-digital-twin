"""Complete digital-twin result figure (input -> twin -> twin-in-use).

Two clean rows:
  Row 1 "Building the twin": input frame -> canonical field -> + deformation
         -> + mechanical map
  Row 2 "Using the twin": rehearsal pose A -> rehearsal pose B -> strain
         prediction -> provenance audit
Writes ``submission/cmpb/figures/fig_twin_complete.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light  # noqa: E402

FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_twin_complete.png"
CKPT = ROOT / "outputs/soft_tissue_twin_seeded/tier1_reconstruction/canonical_field.pt"
if not CKPT.exists():
    CKPT = ROOT / "outputs/soft_tissue_twin_v2/tier1_reconstruction/canonical_field.pt"
BC = ROOT / "outputs/soft_tissue_twin_v2/tier2_deformation/boundary_conditions.npz"
SCENE = ROOT.parent / "datasets/endonerf/pulling_soft_tissues"


def render(field, renderer, cam, light):
    with torch.no_grad():
        out = renderer(field, cam, light=light)
        return out.albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_endonerf_scene(SCENE, max_frames=9, image_scale=0.5)
    renderer = GaussianBRDFRenderer(backend="gsplat")
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
    canon = torch.from_numpy(bc["canonical_positions"]).to(device)
    disp = torch.from_numpy(bc["displacements"]).to(device)
    F = disp.shape[0]

    inp0 = scene.frames[0].image.permute(1, 2, 0).numpy()

    field.means.data = canon
    twin_render = render(field, renderer, cam, light)

    # mechanical (roughness) map on the canonical field
    mats = field.materials()
    rough = mats.roughness.detach().reshape(-1).clamp(1e-4, 1 - 1e-4)
    orig = field.base_color_logits.data.clone()
    field.base_color_logits.data = torch.logit(rough[:, None].expand(-1, 3).contiguous())
    field.means.data = canon
    mech_map = render(field, renderer, cam, light)[..., 0]
    field.base_color_logits.data = orig

    # rehearsal poses
    field.means.data = canon + disp[F // 2]
    reh_a = render(field, renderer, cam, light)
    field.means.data = canon + disp[-1]
    reh_b = render(field, renderer, cam, light)

    # strain (displacement magnitude) map at the late pose
    mag = disp[-1].norm(dim=1).cpu().numpy()
    mag = (mag - mag.min()) / (np.ptp(mag) + 1e-6)
    mm = torch.from_numpy(mag).to(device).clamp(1e-4, 1 - 1e-4)
    field.base_color_logits.data = torch.logit(mm[:, None].expand(-1, 3).contiguous())
    field.means.data = canon + disp[-1]
    strain_map = render(field, renderer, cam, light)[..., 0]
    field.base_color_logits.data = orig

    quiv = np.asarray(Image.open(FIG / "fig_tier2_quiver.png").convert("RGB"))

    # ---- assemble: 2 rows x 4 cols ----
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), dpi=150)

    def show(ax, img, title, cmap=None):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor("#555555")
            s.set_linewidth(1.0)

    # Row 1: building the twin
    show(axes[0, 0], inp0, "input endoscopic frame")
    show(axes[0, 1], twin_render, "canonical Gaussian field")
    show(axes[0, 2], quiv, "+ deformation field")
    show(axes[0, 3], mech_map, "+ mechanical map", cmap="viridis")

    # Row 2: using the twin
    show(axes[1, 0], reh_a, "rehearsal: pose A")
    show(axes[1, 1], reh_b, "rehearsal: pose B")
    show(axes[1, 2], strain_map, "prediction: strain map", cmap="inferno")
    axa = axes[1, 3]
    axa.axis("off")
    axa.text(0.0, 0.98, "Provenance audit", fontsize=11, fontweight="bold", va="top")
    for k, t in enumerate([
        "albedo / roughness : MEASURED",
        "tissue class (liver): ESTIMATED",
        "Young's modulus    : ASSUMED",
        "stiffness contrast : ESTIMATED",
        "Poisson / viscosity: ASSUMED",
    ]):
        axa.text(0.0, 0.80 - k * 0.17, t, fontsize=9.5, family="monospace", va="top")

    # row labels
    axes[0, 0].set_ylabel("BUILD THE TWIN", fontsize=11, fontweight="bold",
                          rotation=90, labelpad=18)
    axes[1, 0].set_ylabel("USE THE TWIN", fontsize=11, fontweight="bold",
                          rotation=90, labelpad=18)
    # arrows across row 1
    for j in range(3):
        axes[0, j].annotate("", xy=(1.05, 0.5), xytext=(1.0, 0.5),
                            xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    for j in range(2):
        axes[1, j].annotate("", xy=(1.05, 0.5), xytext=(1.0, 0.5),
                            xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))

    fig.suptitle("The soft-tissue digital twin: built from endoscopic video "
                 "(top), then driven forward for rehearsal and prediction (bottom)",
                 fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
