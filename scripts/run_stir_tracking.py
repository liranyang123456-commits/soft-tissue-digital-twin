"""Fixed-identity tracking on converted STIR clips.

For each clip the canonical field is fitted to the first frame and the same
primitives are tracked through all sampled frames with the holdout settings
(1,000 canonical iterations, 40 tracking iterations per frame). The projected
positions of every primitive in the first and the last frame are stored for
the common endpoint evaluation in ``evaluate_stir_tracking.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.correspondence import CorrespondenceObserver  # noqa: E402
from mvbrdf_shr.world.deform import track_deformation  # noqa: E402
from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402
from mvbrdf_shr.world.flow_init import make_flow_init  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light, run_tier1, twin_prediction  # noqa: E402

DATA = Path(r"E:\MIS_Datasets\STIRC2024\endonerf_format")
OUT = ROOT / "outputs" / "stir_tracking"


def project(points: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
    depth = camera[:, 2]
    uv = np.stack(
        [K[0, 0] * camera[:, 0] / depth + K[0, 2], K[1, 1] * camera[:, 1] / depth + K[1, 2]], axis=-1
    )
    return uv, depth


def run_clip(clip: Path, out: Path, temporal_model: str, device: torch.device, flow_init: bool = False,
             fused: dict | None = None) -> dict:
    scene = load_endonerf_scene(clip, image_scale=1.0)
    frames = sorted(scene.frames, key=lambda frame: int(frame.view_id))
    started = time.time()
    tier1 = run_tier1(
        SimpleNamespace(max_points=8000, steps=1000, backend="gsplat"), scene, device, out / "tier1"
    )
    exposure = tier1["log_exposure"].exp()
    observer = CorrespondenceObserver(frames[0], device) if fused is not None else None
    tracking = track_deformation(
        tier1["field"],
        frames[1:],
        tier1["renderer"],
        lambda frame: make_endoscopic_light(frame.camera.to(device), device),
        device=device,
        steps_per_frame=40,
        image_fn=lambda rendered: twin_prediction(rendered, exposure),
        temporal_model=temporal_model,
        init_fn=make_flow_init(frames[0], device) if flow_init and fused is None else None,
        observer=observer,
        **(fused or {}),
    )
    canonical = tracking.canonical_positions
    final = canonical + tracking.displacements[-1]
    K0 = frames[0].camera.K.cpu().numpy().astype(np.float64)
    c2w0 = frames[0].camera.c2w.cpu().numpy().astype(np.float64)
    KT = frames[-1].camera.K.cpu().numpy().astype(np.float64)
    c2wT = frames[-1].camera.c2w.cpu().numpy().astype(np.float64)
    uv0, z0 = project(canonical, K0, c2w0)
    uvT, zT = project(final, KT, c2wT)
    # Fused: STIR endpoints follow the Lagrangian 2D tracks (reference-refresh
    # RAFT), not the 3D pose which may roll back on hard frames. 3D depth at
    # the track tip comes from stereo when locally smooth, else from the pose.
    # Query-only RAFT refresh is a fallback for inconsistent neighbourhoods
    # (see fuse_stir_gated.py); it is not the paper-wide tracker.
    if observer is not None:
        uvT = observer.uv.detach().cpu().numpy().astype(np.float64)
        depth_map = frames[-1].depth_gt.cpu().numpy().astype(np.float64)
        h, w = depth_map.shape[-2:]
        u = np.clip(np.rint(uvT[:, 0]), 0, w - 1).astype(int)
        v = np.clip(np.rint(uvT[:, 1]), 0, h - 1).astype(int)
        sampled = depth_map[v, u]
        # Neighbourhood median is stabler than a single stereo pixel for 3D lift.
        z_med = np.zeros(len(uvT))
        for i, (ui, vi) in enumerate(zip(u, v)):
            patch = depth_map[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
            vals = patch[patch > 0.01]
            z_med[i] = np.median(vals) if vals.size else sampled[i]
        offset = observer.offset.detach().cpu().numpy()
        pose_z = zT
        stereo = z_med + offset
        agree = np.abs(stereo - pose_z) <= 0.15 * np.maximum(pose_z, 1e-3)
        zT = np.where((z_med > 0.01) & agree, stereo, np.where(z_med > 0.01, stereo, pose_z))
    opacity = tier1["field"].opacity.detach().reshape(-1).cpu().numpy()
    # Invalidate tracks that the observer marked lost on the last frame.
    if observer is not None:
        lost = ~observer.valid.detach().cpu().numpy()
        opacity = opacity.copy()
        opacity[lost] = 0.0
    np.savez_compressed(
        out / "tracks.npz", uv0=uv0, z0=z0, uvT=uvT, zT=zT, opacity=opacity,
        displacements=tracking.displacements.astype(np.float32), frame_ids=np.asarray(tracking.frame_ids),
    )
    info = {
        "clip": clip.name,
        "frames": len(frames),
        "gaussians": int(canonical.shape[0]),
        "canonical_tissue_psnr": float(tier1["psnr"]),
        "temporal_model": temporal_model,
        "runtime_s": time.time() - started,
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--temporal-model", default="inertia", choices=["inertia", "constant_velocity"])
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--flow-init", action="store_true")
    parser.add_argument("--fused", action="store_true", help="persistent correspondence observations")
    parser.add_argument("--w-corr", type=float, default=0.5)
    parser.add_argument("--w-corr-depth", type=float, default=0.15)
    parser.add_argument("--regularizer", default="laplacian", choices=["laplacian", "arap"])
    parser.add_argument("--method-name", help="output folder name (default derived from options)")
    args = parser.parse_args()
    fused = None
    if args.fused:
        fused = {"w_corr": args.w_corr, "w_corr_depth": args.w_corr_depth, "regularizer": args.regularizer}
    device = torch.device("cuda")
    index = json.loads((args.data / "index.json").read_text())
    names = args.clips or [entry["name"] for entry in index]
    method = "present" if args.temporal_model == "inertia" else "present_cv"
    if args.flow_init:
        method += "_flow"
    if args.fused:
        method = "present_fused"
    method = args.method_name or method
    for name in names:
        out = args.output / method / name
        if (out / "tracks.npz").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        np.random.seed(0)
        info = run_clip(args.data / name, out, args.temporal_model, device, args.flow_init, fused)
        print(f"{name}: {info['frames']} frames, {info['gaussians']} Gaussians, "
              f"fit {info['canonical_tissue_psnr']:.1f} dB, {info['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
