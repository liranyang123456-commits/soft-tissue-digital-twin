"""Liver geometry from the abdominal DICOM series of the hospital CT case.

The earlier JPEG export of this case contained a cranial series with burnt-in
viewer annotations.  This script reads the contrast-enhanced abdominal DICOM
series directly, so physical voxel spacing is available and no annotation
overlay is rendered.  The liver is segmented with an intensity window, a
location prior relative to the lung base, morphological opening, and
connected-component selection; the result is meant to provide a
patient-specific volume for the surface-to-volume interface, not a clinical
segmentation.

Outputs (``outputs/hospital_ct_dicom``):
    liver_mask.npz      boolean mask (z, y, x) and spacing in millimetres
    ct_liver_summary.json
    ct_liver_qc.png     axial and coronal overlays for visual verification
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
CT_ROOT = Path(r"E:\ct_extract")
OUT = ROOT / "outputs" / "hospital_ct_dicom"


def _case_dir(root: Path, case_prefix: str) -> Path:
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(case_prefix)]
    if not matches:
        raise FileNotFoundError(f"no case directory starting with {case_prefix} under {root}")
    return matches[0]


def load_series(case_dir: Path, description: str, min_slices: int) -> tuple[np.ndarray, np.ndarray]:
    """Return HU volume ordered inferior-to-superior and spacing (z, y, x) in mm."""
    series: dict[str, list[tuple[float, Path]]] = defaultdict(list)
    for path in case_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() == ".jpg":
            continue
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(header, "Modality", "") != "CT" or not hasattr(header, "ImagePositionPatient"):
            continue
        if str(getattr(header, "SeriesDescription", "")) != description:
            continue
        series[str(header.SeriesInstanceUID)].append((float(header.ImagePositionPatient[2]), path))
    candidates = [items for items in series.values() if len(items) >= min_slices]
    if not candidates:
        raise FileNotFoundError(f"no CT series '{description}' with at least {min_slices} slices")
    items = sorted(max(candidates, key=len))
    slices = []
    for _, path in items:
        dataset = pydicom.dcmread(str(path))
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(dataset.pixel_array.astype(np.float32) * slope + intercept)
    first = pydicom.dcmread(str(items[0][1]), stop_before_pixels=True)
    z_positions = np.asarray([position for position, _ in items])
    dz = float(np.median(np.diff(z_positions)))
    dy, dx = (float(value) for value in first.PixelSpacing)
    return np.stack(slices), np.asarray([dz, dy, dx])


def downsample(volume: np.ndarray, spacing: np.ndarray, factors: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    fz, fy, fx = factors
    z, y, x = (dim - dim % factor for dim, factor in zip(volume.shape, factors))
    cropped = volume[:z, :y, :x]
    reduced = cropped.reshape(z // fz, fz, y // fy, fy, x // fx, fx).mean(axis=(1, 3, 5))
    return reduced.astype(np.float32), spacing * np.asarray(factors, dtype=np.float64)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask)
    sizes = ndi.sum(mask, labels, range(1, count + 1))
    return labels == (1 + int(np.argmax(sizes)))


def ball(radius: int) -> np.ndarray:
    grid = np.mgrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    return (np.square(grid).sum(axis=0) <= radius * radius)


def segment_liver(
    hu: np.ndarray,
    spacing: np.ndarray,
    window: tuple[float, float],
    opening_mm: float,
    min_component_mm2: float = 600.0,
    dome_mm: float = 90.0,
) -> tuple[np.ndarray, dict]:
    smooth = ndi.median_filter(hu, size=3)
    body = largest_component(smooth > -300)
    body = np.stack([ndi.binary_fill_holes(plane) for plane in body])
    lung = largest_component((smooth < -500) & body)
    lung_rows = np.flatnonzero(lung.reshape(len(lung), -1).sum(axis=1) > 0.002 * lung[0].size)
    if not len(lung_rows):
        raise RuntimeError("lung base not found")
    lung_base = int(lung_rows.min())
    dz = spacing[0]
    z_low = max(0, lung_base - int(round(120.0 / dz)))
    z_high = min(len(hu), lung_base + int(round(dome_mm / dz)))

    _, height, width = hu.shape
    yy, xx = np.mgrid[0:height, 0:width]
    body_columns = np.flatnonzero(body[lung_base].any(axis=0))
    body_center_x = 0.5 * (body_columns.min() + body_columns.max())
    body_width = body_columns.max() - body_columns.min()
    lateral = xx < body_center_x + 0.18 * body_width

    def candidate_mask(low: float, high: float) -> np.ndarray:
        mask = (smooth > low) & (smooth < high) & body & ~lung
        mask[:z_low] = False
        mask[z_high:] = False
        return mask & lateral[None]

    radius = max(1, int(round(opening_mm / float(np.mean(spacing[1:])))))
    first = largest_component(ndi.binary_opening(candidate_mask(*window), structure=ball(radius)))
    core = ndi.binary_erosion(first, structure=ball(3))
    mu, sigma = float(np.median(smooth[core])), float(np.std(smooth[core]))
    adaptive = (max(window[0], mu - 2.5 * sigma), min(window[1], mu + 2.5 * sigma))
    candidate = candidate_mask(*adaptive)
    liver = largest_component(ndi.binary_opening(candidate, structure=ball(radius)))
    liver = ndi.binary_closing(liver, structure=ball(2))
    liver = np.stack([ndi.binary_fill_holes(plane) for plane in liver])

    minimum_area = min_component_mm2 / float(spacing[1] * spacing[2])
    dome_start = lung_base + int(round(40.0 / dz))
    right_limit = body_center_x - 0.05 * body_width
    for index in range(len(liver)):
        labels, count = ndi.label(liver[index])
        if count == 0:
            continue
        components = range(1, count + 1)
        sizes = ndi.sum(liver[index], labels, components)
        keep_ids = 1 + np.flatnonzero(sizes >= minimum_area)
        if index >= dome_start and len(keep_ids):
            centroids = ndi.center_of_mass(liver[index], labels, keep_ids)
            keep_ids = [label for label, (_, cx) in zip(keep_ids, centroids) if cx < right_limit]
        liver[index] = np.isin(labels, keep_ids)
    liver = largest_component(liver)
    info = {
        "lung_base_index": lung_base,
        "search_range_indices": [z_low, z_high],
        "opening_radius_voxels": radius,
        "initial_window_hu": list(window),
        "liver_core_hu_median": round(mu, 1),
        "liver_core_hu_std": round(sigma, 1),
        "adaptive_window_hu": [round(adaptive[0], 1), round(adaptive[1], 1)],
        "minimum_slice_component_mm2": min_component_mm2,
    }
    return liver, info


def quality_figure(hu: np.ndarray, liver: np.ndarray, spacing: np.ndarray, output: Path) -> None:
    rows = np.flatnonzero(liver.reshape(len(liver), -1).any(axis=1))
    axial = np.linspace(rows.min() + 2, rows.max() - 2, 6).astype(int)[::-1]
    display = np.clip((hu + 160.0) / 400.0, 0, 1)
    figure, axes = plt.subplots(2, 4, figsize=(12, 6.4), dpi=150)
    for axis, index in zip(axes.flat[:6], axial):
        axis.imshow(display[index], cmap="gray")
        axis.contour(liver[index], levels=[0.5], colors="#22CC22", linewidths=1.0)
        axis.set_title(f"axial z={index * spacing[0]:.0f} mm", fontsize=8)
        axis.axis("off")
    columns = np.flatnonzero(liver.any(axis=(0, 2)))
    coronal_y = int(np.median(columns))
    sagittal_x = int(np.median(np.flatnonzero(liver.any(axis=(0, 1)))))
    aspect = spacing[0] / spacing[1]
    axes.flat[6].imshow(display[:, coronal_y, :][::-1], cmap="gray", aspect=aspect)
    axes.flat[6].contour(liver[:, coronal_y, :][::-1], levels=[0.5], colors="#22CC22", linewidths=1.0)
    axes.flat[6].set_title("coronal", fontsize=8)
    axes.flat[6].axis("off")
    axes.flat[7].imshow(display[:, :, sagittal_x][::-1], cmap="gray", aspect=aspect)
    axes.flat[7].contour(liver[:, :, sagittal_x][::-1], levels=[0.5], colors="#22CC22", linewidths=1.0)
    axes.flat[7].set_title("sagittal", fontsize=8)
    axes.flat[7].axis("off")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-root", type=Path, default=CT_ROOT)
    parser.add_argument("--case", default="002")
    parser.add_argument("--series", default="1.0 x 1.0_stnd")
    parser.add_argument("--min-slices", type=int, default=500)
    parser.add_argument("--factors", type=int, nargs=3, default=(2, 3, 3))
    parser.add_argument("--window", type=float, nargs=2, default=(70.0, 200.0))
    parser.add_argument("--opening-mm", type=float, default=7.0)
    parser.add_argument("--probe", action="store_true", help="print HU statistics and exit")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    case_dir = _case_dir(args.ct_root, args.case)
    hu_full, spacing_full = load_series(case_dir, args.series, args.min_slices)
    hu, spacing = downsample(hu_full, spacing_full, tuple(args.factors))
    print(f"series {args.series}: {hu_full.shape} at {spacing_full} mm -> {hu.shape} at {spacing} mm")

    if args.probe:
        smooth = ndi.median_filter(hu, size=3)
        body = largest_component(smooth > -300)
        body = np.stack([ndi.binary_fill_holes(plane) for plane in body])
        lung = largest_component((smooth < -500) & body)
        rows = np.flatnonzero(lung.reshape(len(lung), -1).sum(axis=1) > 0.002 * lung[0].size)
        base = int(rows.min())
        print("lung base index", base, "z mm", base * spacing[0])
        for offset_mm in (-150, -100, -60, -30, 0, 30):
            index = base + int(round(offset_mm / spacing[0]))
            plane = smooth[index]
            columns = np.flatnonzero(body[index].any(axis=0))
            cx = 0.5 * (columns.min() + columns.max())
            width = columns.max() - columns.min()
            _, height, w = hu.shape
            roi = plane[int(0.35 * height) : int(0.6 * height), int(cx - 0.35 * width) : int(cx - 0.12 * width)]
            print(f"offset {offset_mm:+} mm: right-ROI HU median {np.median(roi):.0f} p10 {np.percentile(roi, 10):.0f} p90 {np.percentile(roi, 90):.0f}")
        return

    liver, info = segment_liver(hu, spacing, tuple(args.window), args.opening_mm)
    args.output.mkdir(parents=True, exist_ok=True)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    rows = np.flatnonzero(liver.reshape(len(liver), -1).any(axis=1))
    summary = {
        "case": f"case_{args.case}",
        "series": args.series,
        "contrast": "contrast-enhanced",
        "source_shape": list(hu_full.shape),
        "source_spacing_mm": spacing_full.tolist(),
        "working_shape": list(hu.shape),
        "working_spacing_mm": spacing.tolist(),
        "liver_voxels": int(liver.sum()),
        "liver_volume_ml": round(float(liver.sum()) * voxel_ml, 1),
        "craniocaudal_extent_mm": round(float((rows.max() - rows.min() + 1) * spacing[0]), 1),
        **info,
        "interpretation": (
            "Rule-based liver mask for constructing a patient-specific test volume; "
            "not a clinically validated segmentation."
        ),
    }
    np.savez_compressed(args.output / "liver_mask.npz", mask=liver, spacing_zyx_mm=spacing)
    (args.output / "ct_liver_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "hu_working.npz", hu=hu.astype(np.int16), spacing_zyx_mm=spacing)
    quality_figure(hu, liver, spacing, args.output / "ct_liver_qc.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
