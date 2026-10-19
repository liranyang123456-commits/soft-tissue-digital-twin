"""Same-subject surface-to-volume validation on the DePoLL porcine dataset.

The reference tetrahedral liver model (pre-operative CT, no insufflation) is
registered to the intra-operative surface reconstructed from the stereo
exploration video of each deformed state, and the 45 parenchymal spheres and
15 surface clips are predicted by barycentric interpolation in the deformed
model. Target registration error (TRE) is the distance to the CT-localized
marker positions of the state, mapped into the common frame with the
provided transform M (used for evaluation only). Observations are the
Eulerian surface (shape without correspondence), the 15 surface clips as
identified (Lagrangian) surface points, optionally perturbed by the tracking
error, or both; the parenchymal spheres are never observed and measure the
interior prediction. Every method starts from the same pose.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.surface_registration import barycentric_embedding, load_medit_mesh, register  # noqa: E402

DATA = Path(r"E:\MIS_Datasets\DePoLL")
REFERENCE = DATA / "reference_x" / "reference" / "referenceModel"
OUT = ROOT / "outputs" / "depoll_registration"


def opencv_matrix(path: Path, key: str) -> np.ndarray:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    return storage.getNode(key).mat()


def markers(path: Path) -> np.ndarray:
    return np.vstack([opencv_matrix(path, "centroidsBalls").T, opencv_matrix(path, "centroidsClips").T])


def tre(predicted: np.ndarray, truth: np.ndarray) -> dict:
    error = np.linalg.norm(predicted - truth, axis=1)
    balls, clips = error[:45], error[45:]
    return {"balls_median": float(np.median(balls)), "balls_mean": float(balls.mean()),
            "clips_median": float(np.median(clips)), "clips_mean": float(clips.mean()),
            "all_median": float(np.median(error)), "per_marker": error.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="*")
    parser.add_argument("--surface", default="video_reconstruction", choices=["video_reconstruction", "surface_CT"])
    parser.add_argument("--points", type=int, default=30000)
    parser.add_argument("--clip-noise", type=float, default=1.8, help="std (mm) added to the clip observations")
    args = parser.parse_args()
    nodes, tets, tris = load_medit_mesh(REFERENCE / "nonsegmentedLobes" / "volumeFull.mesh")
    reference = markers(REFERENCE / "markers" / "markers.yml")
    owner, weights = barycentric_embedding(reference, nodes, tets)
    states = args.states or sorted(p.name for p in (DATA / "states").iterdir() if (p / "surfaces").exists())
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for state in states:
        folder = DATA / "states" / state
        M = opencv_matrix(DATA / "coordinate_transforms" / state / "M.yml", "M")
        association = opencv_matrix(folder / "markers" / "associations.yml", "Association").ravel().astype(int) - 1
        truth_state = markers(folder / "markers" / "markers.yml") @ M[:3, :3].T + M[:3, 3]
        truth = np.zeros_like(truth_state)
        truth[association] = truth_state
        surface = np.asarray(trimesh.load(folder / "surfaces" / f"{args.surface}.ply").vertices)
        if len(surface) > args.points:
            surface = surface[rng.choice(len(surface), args.points, replace=False)]
        print(f"state {state}: {len(surface)} surface points", flush=True)
        result = {"state": state, "surface": args.surface, "surface_points": int(len(surface))}
        result["none"] = tre(reference, truth)
        started = time.time()
        clips = slice(45, 60)
        noisy = truth[clips] + rng.normal(0.0, args.clip_noise, truth[clips].shape)
        lagrangian = (owner[clips], weights[clips], noisy)
        runs = {
            "rigid_surface": dict(observed=surface, deformable=False),
            "elastic_surface": dict(observed=surface, deformable=True),
            "rigid_clips": dict(observed=None, deformable=False, lagrangian=lagrangian),
            "elastic_clips": dict(observed=None, deformable=True, lagrangian=lagrangian),
            "elastic_clips_surface": dict(observed=surface, deformable=True, lagrangian=lagrangian),
        }
        fields = {}
        for method, options in runs.items():
            fit = register(nodes, tets, tris, options.pop("observed"), np.eye(3), np.zeros(3), log=lambda *_: None, **options)
            result[method] = tre(fit.transform(reference, owner, weights, tets), truth)
            fields[method] = fit
        result["runtime_s"] = time.time() - started
        for method in ("none", *runs):
            r = result[method]
            print(f"  {method:22s} balls {r['balls_median']:.1f} mm  clips {r['clips_median']:.1f} mm", flush=True)
        best = fields["elastic_clips_surface"]
        np.savez_compressed(OUT / f"state{state}_{args.surface}.npz", u=best.u, R=best.R, t=best.t)
        (OUT / f"state{state}_{args.surface}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
