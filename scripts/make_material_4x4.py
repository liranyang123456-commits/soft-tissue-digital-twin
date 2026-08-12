"""4x4 material-recovery grid (Fig 11) for the CMPB paper.

4 scenes (rows) x 4 maps (cols): input | albedo | roughness | specular.
Rows: EndoNeRF pulling, EndoNeRF cutting, SCARED kf1, SCARED kf2.
Writes ``submission/cmpb/figures/fig_material_4x4.png``.
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
from scripts.run_scared_tier1 import load_scared_keyframe  # noqa: E402

FIG = ROOT / "submission" / "cmpb" / "figures"
OUT = FIG / "fig_material_4x4.png"

SCENES = [
    ("EndoNeRF pulling", ROOT / "outputs/soft_tissue_twin_v2/tier1_reconstruction/canonical_field.pt",
     ROOT.parent / "datasets/endonerf/pulling_soft_tissues", "endonerf"),
    ("EndoNeRF cutting", ROOT / "outputs/soft_tissue_twin_cutting/tier1_reconstruction/canonical_field.pt",
     ROOT.parent / "datasets/endonerf/cutting_tissues_twice", "endonerf"),
    ("SCARED kf1", ROOT / "outputs/scared_tier1/dataset_1_keyframe_1/tier1_reconstruction/canonical_field.pt",
     None, "scared_kf1"),
    ("SCARED kf2", ROOT / "outputs/scared_tier1/dataset_1_keyframe_2/tier1_reconstruction/canonical_field.pt",
     None, "scared_kf2"),
]


def render_maps(field, renderer, cam, light, device):
    with torch.no_grad():
        out = renderer(field, cam, light=light)
        albedo = out.albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        mats = field.materials()
        rough_1d = mats.roughness.detach().reshape(-1).cpu().numpy()
        spec_1d = mats.specular.detach().reshape(-1).cpu().numpy()
        orig = field.base_color_logits.data.clone()
        # roughness map
        r = torch.from_numpy(rough_1d).to(device).reshape(-1).clamp(1e-4, 1 - 1e-4)
        field.base_color_logits.data = torch.logit(r[:, None].expand(-1, 3).contiguous())
        rough_map = renderer(field, cam, light=light).albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., 0]
        # specular map
        sp = torch.from_numpy(spec_1d).to(device).reshape(-1).clamp(1e-4, 1 - 1e-4)
        field.base_color_logits.data = torch.logit(sp[:, None].expand(-1, 3).contiguous())
        spec_map = renderer(field, cam, light=light).albedo[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., 0]
        field.base_color_logits.data = orig
    return albedo, rough_map, spec_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = GaussianBRDFRenderer(backend="gsplat")
    n_rows = len(SCENES)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.4 * n_rows), dpi=150)
    col_titles = ["input", "albedo", "roughness", "specular"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=12)

    for i, (label, ckpt, scene_dir, kind) in enumerate(SCENES):
        if not ckpt.exists():
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f"{label}\n(pending)", ha="center",
                                va="center", transform=axes[i, j].transAxes)
                axes[i, j].axis("off")
            continue
        if kind == "endonerf":
            scene = load_endonerf_scene(scene_dir, max_frames=1, image_scale=0.5)
        else:
            kf = 1 if kind == "scared_kf1" else 2
            scene = load_scared_keyframe(
                Path(r"E:\MIS_Datasets\SCARED\dataset_1") / f"keyframe_{kf}",
                image_scale=0.5)
        frame = scene.frames[0]
        img = frame.image.permute(1, 2, 0).numpy()
        cam = frame.camera.to(device)
        light = make_endoscopic_light(cam, device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        state = ck["field"] if "field" in ck else ck
        n = state["means"].shape[0]
        field = GaussianBRDFField(state["means"],
                                  torch.rand(n, 3, device=device) * 0.5 + 0.25,
                                  initial_scale=0.8).to(device)
        field.load_state_dict(state)
        albedo, rough_map, spec_map = render_maps(field, renderer, cam, light, device)
        panels = [img, albedo, rough_map, spec_map]
        for j, p in enumerate(panels):
            if j >= 2:
                axes[i, j].imshow(p, cmap="viridis")
            else:
                axes[i, j].imshow(p)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=9, rotation=0,
                                      ha="right", va="center", labelpad=8)

    fig.suptitle("Soft-tissue material recovery across scenes (input | "
                 "albedo | roughness | specular)", fontsize=13)
    fig.tight_layout(rect=[0.05, 0, 1, 0.98])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
