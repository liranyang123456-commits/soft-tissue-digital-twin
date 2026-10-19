"""Fixed-identity tracking figure for MBEC (EndoNeRF pulling, four frames).

Rows: displacement vectors, displacement magnitude, rendered deformed field,
and photometric error.  Magnitude and error use one colour scale across all
frames so that the columns are directly comparable.
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

from mvbrdf_shr.world.deform import DeformedFieldView  # noqa: E402
from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer  # noqa: E402
from mvbrdf_shr.world.scene import GaussianBRDFField  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light, twin_prediction  # noqa: E402

RUN = ROOT / "outputs" / "twin_pulling_flow"
NPZ = RUN / "tier2_deformation" / "boundary_conditions.npz"
CKPT = RUN / "tier1_reconstruction" / "canonical_field.pt"
SCENE = ROOT.parent / "datasets" / "endonerf" / "pulling_soft_tissues"
OUT = ROOT / "submission" / "mbec" / "figures" / "fig_tracking.png"
COLUMNS = [0, 3, 6, 9]
ROWS = ["displacement", r"$\|\Delta_t\|$", "deformed render", "photometric error"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.4})


def project(points: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor):
    camera = (points - w2c[:3, 3]) @ w2c[:3, :3]
    depth = camera[:, 2].clamp_min(1e-6)
    uv = torch.stack([K[0, 0] * camera[:, 0] / depth + K[0, 2], K[1, 1] * camera[:, 1] / depth + K[1, 2]], dim=-1)
    return uv, depth


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracked = np.load(NPZ)
    canonical = torch.from_numpy(tracked["canonical_positions"]).float()
    displacements = tracked["displacements"]
    frame_ids = tracked["frame_ids"].tolist()

    scene = load_endonerf_scene(SCENE, max_frames=max(frame_ids) + 1, image_scale=0.5)
    frames = {frame.view_id: frame for frame in scene.frames}
    state = torch.load(CKPT, map_location=device, weights_only=False)
    field = GaussianBRDFField(canonical.to(device), torch.full((len(canonical), 3), 0.5, device=device), initial_scale=0.8).to(device)
    field.load_state_dict(state["field"] if "field" in state else state)
    exposure = torch.tensor(float(state.get("log_exposure", 0.0)), device=device).exp()
    renderer = GaussianBRDFRenderer(backend="gsplat")

    magnitude_max = float(np.quantile(np.linalg.norm(displacements[COLUMNS], axis=-1), 0.98))
    figure, axes = plt.subplots(4, 4, figsize=(4.72, 3.72), dpi=300)
    figure.subplots_adjust(left=0.045, right=0.905, top=0.955, bottom=0.005, wspace=0.03, hspace=0.04)
    rng = np.random.default_rng(0)
    magnitude_artist = error_artist = None

    for column, index in enumerate(COLUMNS):
        frame_id = frame_ids[index]
        frame = frames[frame_id]
        image = frame.image.permute(1, 2, 0).numpy()
        height, width = image.shape[:2]
        K = frame.camera.K
        w2c = torch.linalg.inv(frame.camera.c2w)
        displacement = torch.from_numpy(displacements[index]).float()
        uv0, depth0 = project(canonical, K, w2c)
        uv1, _ = project(canonical + displacement, K, w2c)
        inside = (uv0[:, 0] >= 0) & (uv0[:, 0] < width) & (uv0[:, 1] >= 0) & (uv0[:, 1] < height) & (depth0 > 0)
        chosen = rng.permutation(np.flatnonzero(inside.numpy()))[:380]
        magnitude = displacement.norm(dim=-1).numpy()

        axis = axes[0, column]
        axis.imshow(image)
        axis.quiver(
            uv0[chosen, 0], uv0[chosen, 1],
            (uv1[chosen, 0] - uv0[chosen, 0]), (uv1[chosen, 1] - uv0[chosen, 1]),
            magnitude[chosen], cmap="cividis", clim=(0, magnitude_max),
            angles="xy", scale_units="xy", scale=1.0 / 3.0, width=0.005,
        )
        axis.set_title(f"frame {frame_id}", fontsize=6.6, pad=2)

        axis = axes[1, column]
        axis.imshow(image, alpha=0.35)
        magnitude_artist = axis.scatter(
            uv0[chosen, 0], uv0[chosen, 1], c=magnitude[chosen], cmap="cividis",
            vmin=0, vmax=magnitude_max, s=1.6, linewidths=0,
        )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)

        with torch.no_grad():
            view = DeformedFieldView(field, displacement.to(device))
            output = renderer(view, frame.camera.to(device), light=make_endoscopic_light(frame.camera.to(device), device))
            render = twin_prediction(output, exposure).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        if render.shape[:2] != image.shape[:2]:
            tensor = torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0)
            render = F.interpolate(tensor, size=image.shape[:2], mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
        axes[2, column].imshow(render)
        error = np.abs(image - render).mean(axis=-1)
        error_artist = axes[3, column].imshow(error, cmap="magma", vmin=0, vmax=0.5)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for row, name in enumerate(ROWS):
        axes[row, 0].set_ylabel(name, fontsize=6.4, labelpad=2)

    def colorbar(artist, row: int, label: str) -> None:
        box = axes[row, 3].get_position()
        cax = figure.add_axes([0.915, box.y0 + 0.1 * box.height, 0.012, 0.8 * box.height])
        bar = figure.colorbar(artist, cax=cax)
        bar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
        bar.set_label(label, fontsize=5.6, labelpad=1.5)

    colorbar(magnitude_artist, 1, "scene units")
    colorbar(error_artist, 3, "|error|")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300, facecolor="white")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
