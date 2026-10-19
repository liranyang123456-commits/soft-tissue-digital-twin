"""Shape-only ambiguity of intra-operative surface data on DePoLL.

For each deformed state: distance from the reconstructed intra-operative
surface to the undeformed reference liver surface (no registration and after
rigid trimmed ICP), and the true displacement of the fiducials that lie
close to the observed surface. A small surface distance together with a
large fiducial displacement means that the observed shape is explained by
the undeformed model although the tissue under it has moved, so surface
shape without correspondence cannot identify that motion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.surface_registration import load_medit_mesh, register  # noqa: E402
from scripts.run_depoll_registration import DATA, OUT, REFERENCE, markers, opencv_matrix  # noqa: E402


def surface_distance(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    _, distance, _ = trimesh.proximity.closest_point(mesh, points)
    return distance


def main() -> None:
    nodes, tets, tris = load_medit_mesh(REFERENCE / "nonsegmentedLobes" / "volumeFull.mesh")
    reference_mesh = trimesh.Trimesh(nodes, tris, process=False)
    reference = markers(REFERENCE / "markers" / "markers.yml")
    rng = np.random.default_rng(0)
    rows = []
    for folder in sorted((DATA / "states").iterdir()):
        if not (folder / "surfaces" / "video_reconstruction.ply").exists():
            continue
        state = folder.name
        M = opencv_matrix(DATA / "coordinate_transforms" / state / "M.yml", "M")
        association = opencv_matrix(folder / "markers" / "associations.yml", "Association").ravel().astype(int) - 1
        truth_state = markers(folder / "markers" / "markers.yml") @ M[:3, :3].T + M[:3, 3]
        truth = np.zeros_like(truth_state)
        truth[association] = truth_state
        surface = np.asarray(trimesh.load(folder / "surfaces" / "video_reconstruction.ply").vertices)
        sample = surface[rng.choice(len(surface), min(20000, len(surface)), replace=False)]
        before = surface_distance(sample, reference_mesh)
        rigid = register(nodes, tets, tris, sample, np.eye(3), np.zeros(3), deformable=False, log=lambda *_: None)
        moved = trimesh.Trimesh(nodes @ rigid.R.T + rigid.t, tris, process=False)
        after = surface_distance(sample, moved)
        near, _ = cKDTree(surface).query(truth)
        displacement = np.linalg.norm(truth - reference, axis=1)
        close = near < 15.0
        extent = surface.max(0) - surface.min(0)
        row = {
            "state": state,
            "surface_points": int(len(surface)),
            "surface_extent_mm": extent.round(1).tolist(),
            "surface_to_model_median_mm": float(np.median(before)),
            "surface_to_model_rigid_median_mm": float(np.median(after)),
            "fiducials_within_15mm": int(close.sum()),
            "their_true_displacement_median_mm": float(np.median(displacement[close])) if close.any() else None,
            "all_fiducial_displacement_median_mm": float(np.median(displacement)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shape_ambiguity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
