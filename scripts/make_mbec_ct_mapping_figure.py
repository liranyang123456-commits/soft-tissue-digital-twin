"""CT-derived liver volume and Gaussian-to-volume mapping figure for MBEC.

Reads the rule-based liver mask from ``run_ct_liver_dicom.py`` and the mapping
from ``build_gaussian_volume_map.py``.  Only DICOM pixel data are displayed,
so no viewer annotation or patient identifier appears in the figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
CT = ROOT / "outputs" / "hospital_ct_dicom"
MAP = ROOT / "outputs" / "gaussian_volume_mapping"
OUT = ROOT / "submission" / "mbec" / "figures" / "fig_ct_volume_mapping.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.titlesize": 6.6,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
    }
)


def label(figure, x: float, y: float, text: str) -> None:
    figure.text(x, y, text, fontsize=7.5, fontweight="bold", va="top", ha="left")


def body_crop(plane: np.ndarray) -> tuple[slice, slice]:
    labels, count = ndi.label(ndi.binary_fill_holes(plane > 0.15))
    sizes = ndi.sum(np.ones_like(plane), labels, range(1, count + 1))
    rows, cols = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    return slice(max(0, rows.min() - 2), rows.max() + 3), slice(max(0, cols.min() - 2), cols.max() + 3)


def boundary_cloud(axis, boundary: np.ndarray) -> None:
    step = max(1, len(boundary) // 6000)
    axis.scatter(
        boundary[::step, 0], boundary[::step, 1], boundary[::step, 2],
        s=0.25, c="#BDBDBD", alpha=0.35, linewidths=0, depthshade=False,
    )


def frame_3d(axis, points: np.ndarray, zoom: float) -> None:
    low, high = points.min(axis=0), points.max(axis=0)
    centre, half = 0.5 * (low + high), 0.5 * (high - low).max()
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_box_aspect((1, 1, 1), zoom=zoom)
    axis.view_init(elev=22, azim=-125)
    axis.set_proj_type("ortho")
    axis.set_axis_off()


def main() -> None:
    hu_data = np.load(CT / "hu_working.npz")
    hu = hu_data["hu"].astype(np.float32)
    spacing = hu_data["spacing_zyx_mm"]
    liver = np.load(CT / "liver_mask.npz")["mask"]
    mesh = np.load(MAP / "patient_volume_mesh.npz")
    mapping = np.load(MAP / "gaussian_volume_map.npz")
    transferred = np.load(MAP / "transferred_boundary_displacement.npz")
    summary = json.loads((MAP / "mapping_summary.json").read_text(encoding="utf-8"))

    nodes = mesh["nodes"]
    boundary = nodes[mesh["boundary_nodes"]]
    gaussians = mapping["transformed_gaussians"]
    covered = nodes[transferred["volume_node_ids"]]
    magnitude = np.linalg.norm(transferred["displacement_mm"], axis=1)
    distance = mapping["nearest_distance"]

    display = np.clip((hu + 160.0) / 400.0, 0, 1)
    rows = np.flatnonzero(liver.reshape(len(liver), -1).any(axis=1))
    span = rows.max() - rows.min()
    levels = np.linspace(rows.min() + 0.22 * span, rows.max() - 0.12 * span, 3).astype(int)[::-1]

    figure = plt.figure(figsize=(4.72, 3.3), dpi=300)
    top_y, top_h, width, gap = 0.53, 0.40, 0.235, 0.012
    for column, index in enumerate(levels):
        x0 = 0.005 + column * (width + gap)
        axis = figure.add_axes([x0, top_y, width, top_h])
        rr, cc = body_crop(display[index])
        axis.imshow(display[index][rr, cc], cmap="gray", interpolation="bilinear")
        axis.contour(liver[index][rr, cc], levels=[0.5], colors="#D9A441", linewidths=0.75)
        axis.set_title(f"axial, z = {index * spacing[0]:.0f} mm", pad=2)
        axis.axis("off")
        label(figure, x0, 0.995, "abc"[column])

    x0 = 0.005 + 3 * (width + gap)
    axis = figure.add_axes([x0, top_y, width, top_h])
    y_index = int(np.median(np.flatnonzero(liver.any(axis=(0, 2)))))
    z0, z1 = max(0, rows.min() - 20), min(len(liver), rows.max() + 20)
    plane = display[z0:z1, y_index, :][::-1]
    cols = np.flatnonzero((plane > 0.15).sum(axis=0) > 0.1 * plane.shape[0])
    plane = plane[:, cols.min() : cols.max() + 1]
    contour = liver[z0:z1, y_index, cols.min() : cols.max() + 1][::-1]
    axis.imshow(plane, cmap="gray", aspect=spacing[0] / spacing[2], interpolation="bilinear")
    axis.contour(contour, levels=[0.5], colors="#D9A441", linewidths=0.75)
    axis.set_title("coronal", pad=2)
    axis.axis("off")
    label(figure, x0, 0.995, "d")

    axis = figure.add_axes([-0.02, 0.0, 0.40, 0.50], projection="3d")
    boundary_cloud(axis, boundary)
    step = max(1, len(gaussians) // 4000)
    axis.scatter(
        gaussians[::step, 0], gaussians[::step, 1], gaussians[::step, 2],
        s=0.35, c="#3B5B7A", alpha=0.85, linewidths=0, depthshade=False,
    )
    frame_3d(axis, boundary, zoom=1.55)
    figure.text(0.18, 0.47, "aligned Gaussian surface", ha="center", fontsize=6.6)
    label(figure, 0.005, 0.50, "e")

    axis = figure.add_axes([0.33, 0.0, 0.36, 0.50], projection="3d")
    boundary_cloud(axis, boundary)
    points = axis.scatter(
        covered[:, 0], covered[:, 1], covered[:, 2],
        s=0.5, c=magnitude, cmap="cividis", linewidths=0, depthshade=False,
    )
    frame_3d(axis, boundary, zoom=1.55)
    figure.text(0.51, 0.47, "transferred displacement", ha="center", fontsize=6.6)
    label(figure, 0.37, 0.50, "f")
    colorbar_axis = figure.add_axes([0.655, 0.12, 0.010, 0.26])
    colorbar = figure.colorbar(points, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    colorbar.set_label(r"$|u^{obs}|$ (mm)", fontsize=5.6, labelpad=1.5)

    axis = figure.add_axes([0.79, 0.12, 0.17, 0.30])
    axis.hist(distance, bins=26, color="#3B5B7A", alpha=0.85)
    axis.axvline(np.median(distance), color="#B5651D", linewidth=0.8, linestyle="--")
    axis.set_xlabel("distance (mm)", labelpad=1)
    axis.set_ylabel("boundary nodes", labelpad=1)
    axis.set_title(f"coverage {100 * summary['boundary_coverage']:.1f}%", pad=2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=1.8, pad=1)
    label(figure, 0.755, 0.50, "g")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300, facecolor="white")
    plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
