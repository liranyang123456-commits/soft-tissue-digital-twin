"""Soft-tissue material-recovery figure for the CMPB paper.

Left: input frame, recovered albedo map, roughness map (from the canonical
Gaussian BRDF field). Right: the recovered tissue parameters with their
provenance audit tags. Writes
``submission/cmpb/figures/fig_material.png``.
"""
from __future__ import annotations

import json
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

CKPT = ROOT / "outputs" / "soft_tissue_twin_v2" / "tier1_reconstruction" / "canonical_field.pt"
SCENE = ROOT.parent / "datasets" / "endonerf" / "pulling_soft_tissues"
EST = ROOT / "outputs" / "soft_tissue_twin" / "tier3_elasticity" / "tissue_estimate.json"
OUT = ROOT / "submission" / "cmpb" / "figures" / "fig_material.png"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = load_endonerf_scene(SCENE, max_frames=1, image_scale=0.5)
    frame = scene.frames[0]
    img = frame.image.permute(1, 2, 0).numpy()
    cam = frame.camera.to(device)
    light = make_endoscopic_light(cam, device)

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    state = ck["field"] if "field" in ck else ck
    n = state["means"].shape[0]
    field = GaussianBRDFField(state["means"], torch.rand(n, 3, device=device) * 0.5 + 0.25,
                              initial_scale=0.8).to(device)
    field.load_state_dict(state)
    renderer = GaussianBRDFRenderer(backend="gsplat")
    with torch.no_grad():
        out = renderer(field, cam, light=light)
        albedo = out.albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        rough_1d = field.materials().roughness.detach().cpu().numpy()

    # roughness map: temporarily set base color to roughness (grayscale) and re-render
    with torch.no_grad():
        orig = field.base_color_logits.data.clone()
        r = torch.from_numpy(rough_1d).to(device).reshape(-1).clamp(1e-4, 1 - 1e-4)
        field.base_color_logits.data = torch.logit(r[:, None].expand(-1, 3).contiguous())
        out_r = renderer(field, cam, light=light)
        rough_map = out_r.albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., 0]
        field.base_color_logits.data = orig

    est = json.loads(EST.read_text(encoding="utf-8")) if EST.exists() else None

    fig = plt.figure(figsize=(15, 5.2), dpi=150)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
    ax = fig.add_subplot(gs[0])
    ax.imshow(img); ax.set_title("input", fontsize=11); ax.axis("off")
    ax = fig.add_subplot(gs[1])
    ax.imshow(albedo); ax.set_title("recovered albedo", fontsize=11); ax.axis("off")
    ax = fig.add_subplot(gs[2])
    im = ax.imshow(rough_map, cmap="viridis")
    ax.set_title("recovered roughness", fontsize=11); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[3])
    ax.axis("off")
    if est:
        rows = [
            ("tissue class", est["tissue_class"], "estimated"),
            ("constitutive model", est["constitutive_model"], "—"),
            ("Young's E", f"{est['youngs_median_kpa']:.1f} kPa "
             f"[{est['youngs_ci95_kpa'][0]:.1f}, {est['youngs_ci95_kpa'][1]:.1f}]",
             est["audit"]["youngs_modulus"]),
            ("Poisson $\\nu$", f"{est['poisson']['mean']:.2f}", est["poisson"]["source"]),
            ("density", f"{est['density']['mean']:.0f} kg/m$^3$", est["density"]["source"]),
            ("viscosity", f"{est['viscosity']['mean']:.1f} Pa$\\cdot$s", est["viscosity"]["source"]),
            ("friction", f"{est['friction']['mean']:.2f}", est["friction"]["source"]),
        ]
        cell = [["parameter", "value", "provenance"]] + [list(r) for r in rows]
        tbl = ax.table(cellText=cell, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        ax.set_title("recovered tissue parameters + audit", fontsize=11)

    fig.suptitle("Soft-tissue material recovery (EndoNeRF pulling): optical "
                 "BRDF maps + mechanical parameters with provenance audit",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
