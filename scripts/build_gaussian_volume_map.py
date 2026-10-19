"""Build a direct Gaussian-surface to patient-specific volume mapping."""
from __future__ import annotations

import argparse
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

from mvbrdf_shr.world.volume_mapping import (  # noqa: E402
    build_surface_observation_map,
    similarity_icp,
    voxel_mask_to_tetrahedra,
)


def _load_ct_mask(mask_npz: Path) -> tuple[str, np.ndarray, np.ndarray]:
    data = np.load(mask_npz)
    summary_path = mask_npz.with_name("ct_liver_summary.json")
    case_id = "case_deidentified"
    if summary_path.exists():
        case_id = json.loads(summary_path.read_text(encoding="utf-8")).get("case", case_id)
    return case_id, data["mask"].astype(bool), data["spacing_zyx_mm"].astype(np.float64)


def _load_gaussians(checkpoint: Path) -> np.ndarray:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    field = state.get("field", state)
    if "means" not in field:
        raise KeyError(f"{checkpoint} does not contain Gaussian means")
    return field["means"].detach().cpu().numpy().astype(np.float64)


def _subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    index = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[index]


def _plot(
    output: Path,
    volume_nodes: np.ndarray,
    boundary_nodes: np.ndarray,
    transformed_gaussians: np.ndarray,
    observed_nodes: np.ndarray,
    nearest_distance: np.ndarray,
) -> None:
    boundary = volume_nodes[boundary_nodes]
    displayed_boundary = _subsample(boundary, 12000)
    displayed_gaussians = _subsample(transformed_gaussians, 6000)
    observed_mask = np.zeros(len(volume_nodes), dtype=bool)
    observed_mask[observed_nodes] = True
    covered = volume_nodes[observed_mask]
    displayed_covered = _subsample(covered, 6000)

    figure = plt.figure(figsize=(15, 5), dpi=160)
    axis = figure.add_subplot(1, 3, 1, projection="3d")
    axis.scatter(
        displayed_boundary[:, 0],
        displayed_boundary[:, 1],
        displayed_boundary[:, 2],
        s=0.3,
        c="#A9A9A9",
        alpha=0.25,
        label="volume boundary",
    )
    axis.scatter(
        displayed_gaussians[:, 0],
        displayed_gaussians[:, 1],
        displayed_gaussians[:, 2],
        s=1.0,
        c="#C44E52",
        alpha=0.7,
        label="aligned Gaussians",
    )
    axis.set_title("Similarity alignment")
    axis.legend(loc="upper right", fontsize=7)
    axis.set_axis_off()

    axis = figure.add_subplot(1, 3, 2, projection="3d")
    axis.scatter(
        displayed_boundary[:, 0],
        displayed_boundary[:, 1],
        displayed_boundary[:, 2],
        s=0.3,
        c="#D0D0D0",
        alpha=0.2,
    )
    if len(displayed_covered):
        axis.scatter(
            displayed_covered[:, 0],
            displayed_covered[:, 1],
            displayed_covered[:, 2],
            s=1.0,
            c="#4C72B0",
            alpha=0.8,
        )
    axis.set_title("Boundary nodes covered by video surface")
    axis.set_axis_off()

    axis = figure.add_subplot(1, 3, 3)
    axis.hist(nearest_distance, bins=30, color="#4C72B0", alpha=0.85)
    axis.set_xlabel("nearest Gaussian distance (mm)")
    axis.set_ylabel("boundary-node count")
    axis.set_title("Observation-transfer distance")
    figure.tight_layout()
    figure.savefig(output / "gaussian_volume_mapping.png", bbox_inches="tight")
    figure.savefig(output / "gaussian_volume_mapping.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-npz",
        type=Path,
        default=ROOT / "outputs" / "hospital_ct_dicom" / "liver_mask.npz",
    )
    parser.add_argument(
        "--displacements",
        type=Path,
        default=ROOT
        / "outputs"
        / "twin_pulling_flow"
        / "tier2_deformation"
        / "boundary_conditions.npz",
    )
    parser.add_argument(
        "--gaussian-checkpoint",
        type=Path,
        default=ROOT
        / "outputs"
        / "twin_pulling_flow"
        / "tier1_reconstruction"
        / "canonical_field.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "gaussian_volume_mapping",
    )
    parser.add_argument("--block", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--trim-fraction", type=float, default=0.7)
    parser.add_argument("--coverage-fraction", type=float, default=0.05)
    parser.add_argument("--neighbors", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    patient, mask, spacing = _load_ct_mask(args.mask_npz)
    volume = voxel_mask_to_tetrahedra(
        mask,
        block=tuple(args.block),
        spacing=tuple(spacing.tolist()),
    )
    gaussians = _load_gaussians(args.gaussian_checkpoint)
    boundary = volume.nodes[volume.boundary_nodes]
    transform, alignment = similarity_icp(
        gaussians,
        boundary,
        trim_fraction=args.trim_fraction,
    )
    transformed = transform.apply(gaussians)
    observation_map = build_surface_observation_map(
        transformed,
        volume.nodes,
        volume.boundary_nodes,
        neighbors=args.neighbors,
        coverage_fraction=args.coverage_fraction,
    )

    np.savez_compressed(
        args.output / "patient_volume_mesh.npz",
        nodes=volume.nodes,
        elements=volume.elements,
        boundary_nodes=volume.boundary_nodes,
        source_mask_shape=np.asarray(mask.shape),
        block=np.asarray(args.block),
    )
    np.savez_compressed(
        args.output / "gaussian_volume_map.npz",
        transform_scale=np.asarray(transform.scale),
        transform_rotation=transform.rotation,
        transform_translation=transform.translation,
        transformed_gaussians=transformed,
        volume_node_ids=observation_map.volume_node_ids,
        gaussian_ids=observation_map.gaussian_ids,
        weights=observation_map.weights,
        nearest_distance=observation_map.nearest_distance,
    )

    tracked = np.load(args.displacements)
    displacement = tracked["displacements"][-1].astype(np.float64)
    if displacement.shape != gaussians.shape:
        raise ValueError("tracked displacement does not match the Gaussian checkpoint")
    volume_frame = transform.scale * (displacement @ transform.rotation.T)
    transferred = observation_map.transfer(volume_frame)
    transferred_norm = np.linalg.norm(transferred, axis=1)
    np.savez_compressed(
        args.output / "transferred_boundary_displacement.npz",
        volume_node_ids=observation_map.volume_node_ids,
        displacement_mm=transferred,
    )

    coverage = len(observation_map.volume_node_ids) / len(volume.boundary_nodes)
    element_volume = np.abs(
        np.einsum(
            "ij,ij->i",
            np.cross(
                volume.nodes[volume.elements[:, 1]] - volume.nodes[volume.elements[:, 0]],
                volume.nodes[volume.elements[:, 2]] - volume.nodes[volume.elements[:, 0]],
            ),
            volume.nodes[volume.elements[:, 3]] - volume.nodes[volume.elements[:, 0]],
        )
    ) / 6.0
    summary = {
        "patient": patient,
        "ct_series": "contrast-enhanced abdominal CT, DICOM",
        "voxel_spacing_mm_zyx": spacing.tolist(),
        "coordinate_units": "millimetres in the CT patient frame",
        "mesh_volume_ml": float(element_volume.sum() / 1000.0),
        "tracked_frame": int(len(tracked["displacements"])),
        "transferred_displacement_mm": {
            "median": float(np.median(transferred_norm)),
            "p95": float(np.quantile(transferred_norm, 0.95)),
            "max": float(transferred_norm.max()),
        },
        "cross_subject_mapping": True,
        "gaussian_checkpoint": str(args.gaussian_checkpoint),
        "gaussians": int(len(gaussians)),
        "volume_nodes": int(len(volume.nodes)),
        "tetrahedra": int(len(volume.elements)),
        "boundary_nodes": int(len(volume.boundary_nodes)),
        "mapped_boundary_nodes": int(len(observation_map.volume_node_ids)),
        "boundary_coverage": float(coverage),
        "coverage_threshold": observation_map.coverage_threshold,
        "similarity_transform": {
            "scale": transform.scale,
            "rotation": transform.rotation.tolist(),
            "translation": transform.translation.tolist(),
        },
        "alignment": alignment,
        "interpretation": (
            "Computational mapping demonstration only. The endoscopic and CT "
            "surfaces are from different subjects and are not anatomical "
            "correspondences or a clinical validation."
        ),
    }
    (args.output / "mapping_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot(
        args.output,
        volume.nodes,
        volume.boundary_nodes,
        transformed,
        observation_map.volume_node_ids,
        observation_map.nearest_distance,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
