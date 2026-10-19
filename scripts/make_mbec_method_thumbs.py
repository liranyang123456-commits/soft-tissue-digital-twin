"""Thumbnails of real intermediate outputs for the MBEC method figure.

Every image is produced from stored results of the pipeline:
    input      EndoNeRF frame with the excluded pixels (tools or invalid depth)
    render     canonical Gaussian-BRDF render
    track      fixed-identity displacement at the last tracked frame
    contrast   four-region relative stiffness on the tracked Gaussians
    volume     CT-derived liver boundary with the aligned Gaussian surface
    fem        tetrahedral benchmark mesh, inclusion, and imposed load response
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402

RUN = ROOT / "outputs" / "twin_pulling_flow"
SCENE = ROOT.parent / "datasets" / "endonerf" / "pulling_soft_tissues"
MAP = ROOT / "outputs" / "gaussian_volume_mapping"
FEM = Path(r"E:\Diff_Rending_Re_3D\dataset\ion_ct_synthetic_mechanics60\geom_58db559b24da6eb7cc0d_scenario_000\press_left\gt.pt")
OUT = ROOT / "submission" / "mbec" / "figures" / "method_thumbs"
ASPECT = (4.0, 3.0)


def save(figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"{name}.png", dpi=260, facecolor="white", pad_inches=0)
    plt.close(figure)


def canvas():
    figure = plt.figure(figsize=(ASPECT[0] / 2.54 * 1.2, ASPECT[1] / 2.54 * 1.2))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    return figure, axis


def project(points: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    camera = (points - w2c[:3, 3]) @ w2c[:3, :3]
    depth = camera[:, 2].clamp_min(1e-6)
    uv = torch.stack(
        [K[0, 0] * camera[:, 0] / depth + K[0, 2], K[1, 1] * camera[:, 1] / depth + K[1, 2]],
        dim=-1,
    )
    return uv.numpy(), depth.numpy()


def endoscopic_thumbs() -> None:
    scene = load_endonerf_scene(SCENE, max_frames=10, image_scale=0.5)
    frame = scene.frames[0]
    image = frame.image.permute(1, 2, 0).numpy()
    height, width = image.shape[:2]

    tool = np.asarray(Image.open(SCENE / "masks" / "frame-000000.mask.png").convert("L").resize((width, height))) > 127
    depth = np.asarray(Image.open(SCENE / "depth" / "frame-000000.depth.png").resize((width, height)))
    excluded = tool | (depth <= 0)
    overlay = image.copy()
    overlay[excluded] = 0.4 * overlay[excluded] + 0.6 * np.asarray([0.23, 0.36, 0.48])
    figure, axis = canvas()
    axis.imshow(overlay)
    axis.contour(~excluded, levels=[0.5], colors="white", linewidths=0.6)
    save(figure, "input")

    render = np.asarray(Image.open(RUN / "tier1_reconstruction" / "final_render.png").convert("RGB"))
    figure, axis = canvas()
    axis.imshow(render)
    save(figure, "render")

    tracked = np.load(RUN / "tier2_deformation" / "boundary_conditions.npz")
    canonical = torch.from_numpy(tracked["canonical_positions"]).float()
    displacement = torch.from_numpy(tracked["displacements"][-1]).float()
    K = frame.camera.K
    w2c = torch.linalg.inv(frame.camera.c2w)
    uv0, depth0 = project(canonical, K, w2c)
    uv1, _ = project(canonical + displacement, K, w2c)
    inside = (uv0[:, 0] >= 0) & (uv0[:, 0] < width) & (uv0[:, 1] >= 0) & (uv0[:, 1] < height) & (depth0 > 0)
    rng = np.random.default_rng(0)
    index = rng.permutation(np.flatnonzero(inside))[:260]
    figure, axis = canvas()
    axis.imshow(image, alpha=0.55)
    axis.quiver(
        uv0[index, 0], uv0[index, 1],
        uv1[index, 0] - uv0[index, 0], uv1[index, 1] - uv0[index, 1],
        color="#1F3A56", angles="xy", scale_units="xy", scale=1.0 / 3.0, width=0.0045,
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    save(figure, "track")

    stiffness = json.loads((RUN / "tier3_elasticity" / "relative_stiffness.json").read_text(encoding="utf-8"))
    region = np.asarray(stiffness["assignments"])
    kappa = np.asarray(stiffness["relative_stiffness"])[region]
    figure, axis = canvas()
    axis.imshow(image, alpha=0.4)
    points = axis.scatter(uv0[inside, 0], uv0[inside, 1], c=kappa[inside], s=1.4, cmap="cividis", vmin=1.0, vmax=float(kappa.max()), linewidths=0)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    figure.patches.append(FancyBboxPatch(
        (0.835, 0.07), 0.15, 0.6, boxstyle="round,pad=0.0,rounding_size=0.02",
        transform=figure.transFigure, facecolor="white", alpha=0.85, edgecolor="none", zorder=1,
    ))
    colorbar_axis = figure.add_axes([0.86, 0.12, 0.035, 0.5])
    colorbar_axis.set_zorder(2)
    colorbar = figure.colorbar(points, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=4.5, length=1, pad=0.5)
    colorbar.set_ticks([1.0, 1.3, 1.6])
    save(figure, "contrast")


def volume_thumb() -> None:
    mesh = np.load(MAP / "patient_volume_mesh.npz")
    mapping = np.load(MAP / "gaussian_volume_map.npz")
    boundary = mesh["nodes"][mesh["boundary_nodes"]]
    gaussians = mapping["transformed_gaussians"]
    figure = plt.figure(figsize=(ASPECT[0] / 2.54 * 1.2, ASPECT[1] / 2.54 * 1.2))
    axis = figure.add_axes([-0.1, -0.12, 1.2, 1.3], projection="3d")
    step = max(1, len(boundary) // 5000)
    axis.scatter(*boundary[::step].T, s=0.3, c="#9CA3AF", alpha=0.35, linewidths=0, depthshade=False)
    step = max(1, len(gaussians) // 3000)
    axis.scatter(*gaussians[::step].T, s=0.35, c="#3B5B7A", alpha=0.85, linewidths=0, depthshade=False)
    low, high = boundary.min(axis=0), boundary.max(axis=0)
    centre, half = 0.5 * (low + high), 0.5 * (high - low).max()
    for setter, value in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), centre):
        setter(value - half, value + half)
    axis.set_box_aspect((1, 1, 1), zoom=1.45)
    axis.view_init(elev=22, azim=-125)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    save(figure, "volume")


def fem_thumb() -> None:
    data = torch.load(FEM, map_location="cpu", weights_only=False)
    nodes = data["nodes"].numpy()
    triangles = data["surface_tris"].numpy()
    sequence = data["u_seq"].numpy()
    displacement = sequence[int(np.argmax(np.abs(sequence).max(axis=1)))].reshape(-1, 3)
    centre = data["inclusion_center"].numpy()
    radius = float(data["inclusion_radius"])
    fixed = np.unique(data["fixed"].numpy() // 3)
    observed = data["surface_node_ids"].numpy()

    figure = plt.figure(figsize=(ASPECT[0] / 2.54 * 1.2, ASPECT[1] / 2.54 * 1.2))
    axis = figure.add_axes([-0.1, -0.12, 1.2, 1.3], projection="3d")
    elements = data["elems"].numpy()
    edges = np.concatenate([elements[:, [a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    for a, b in edges[:: max(1, len(edges) // 900)]:
        axis.plot(*nodes[[a, b]].T, color="#94A3B8", linewidth=0.15, alpha=0.6)
    axis.scatter(*nodes[fixed].T, s=1.0, c="#334155", linewidths=0, depthshade=False)
    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
    axis.plot_surface(
        centre[0] + radius * np.cos(u) * np.sin(v),
        centre[1] + radius * np.sin(u) * np.sin(v),
        centre[2] + radius * np.cos(v),
        color="#C9A36A", alpha=0.85, linewidth=0,
    )
    magnitude = np.linalg.norm(displacement[observed], axis=1)
    scale = 0.08 * np.ptp(nodes, axis=0).max() / max(magnitude.max(), 1e-9)
    axis.quiver(
        *nodes[observed].T, *(scale * displacement[observed]).T,
        color="#3B5B7A", linewidth=0.5, arrow_length_ratio=0.35,
    )
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    mid, half = 0.5 * (low + high), 0.5 * (high - low).max()
    for setter, value in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), mid):
        setter(value - half, value + half)
    axis.set_box_aspect((1, 1, 1), zoom=1.15)
    axis.view_init(elev=20, azim=-60)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    save(figure, "fem")


def main() -> None:
    endoscopic_thumbs()
    volume_thumb()
    fem_thumb()
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
