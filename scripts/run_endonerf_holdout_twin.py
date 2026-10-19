"""Run the twin reconstruction under the frozen EndoNeRF holdout protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.appearance_layer import (  # noqa: E402
    AppearanceView,
    appearance_image,
    build_appearance_layer,
    fit_appearance_layer,
)
from mvbrdf_shr.world.correspondence import CorrespondenceObserver  # noqa: E402
from mvbrdf_shr.world.deform import DeformationResult, DeformedFieldView, track_deformation  # noqa: E402
from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402
from mvbrdf_shr.world.endonerf_protocol import endonerf_holdout  # noqa: E402
from mvbrdf_shr.world.flow_init import make_flow_init  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light, run_tier1, twin_prediction  # noqa: E402


def interpolate_displacement(
    frame_id: int,
    displacement_by_frame: dict[int, np.ndarray],
) -> np.ndarray:
    """Linearly interpolate displacement without using the held-out image."""
    available = sorted(displacement_by_frame)
    lower = max((value for value in available if value < frame_id), default=None)
    upper = min((value for value in available if value > frame_id), default=None)
    if lower is None and upper is None:
        raise ValueError("no tracked displacement is available")
    if lower is None:
        return displacement_by_frame[upper].copy()
    if upper is None:
        return displacement_by_frame[lower].copy()
    fraction = (frame_id - lower) / (upper - lower)
    return (
        (1.0 - fraction) * displacement_by_frame[lower]
        + fraction * displacement_by_frame[upper]
    )


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray((255 * array).round().astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--reconstruction-steps", type=int, default=1000)
    parser.add_argument("--tracking-steps", type=int, default=40)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--backend", default="gsplat")
    parser.add_argument("--temporal-model", default="inertia", choices=["constant_velocity", "inertia"])
    parser.add_argument("--flow-init", action="store_true")
    parser.add_argument("--fused", action="store_true", help="persistent correspondence observations")
    parser.add_argument("--appearance-layer", action="store_true", help="dense appearance layer on all training frames")
    parser.add_argument("--appearance-steps", type=int, default=20000)
    parser.add_argument("--appearance-points", type=int, default=50000)
    parser.add_argument("--displacements", type=Path, help="reuse saved identity displacements and skip tracking")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    scene = load_endonerf_scene(args.scene_root, image_scale=args.scale)
    split = endonerf_holdout(len(scene.frames))
    frame_by_id = {int(frame.view_id): frame for frame in scene.frames}
    if 0 not in split.train_ids:
        raise RuntimeError("frame 0 must be a training frame for the canonical field")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tier_args = SimpleNamespace(
        max_points=args.max_points,
        steps=args.reconstruction_steps,
        backend=args.backend,
    )
    tier1 = run_tier1(tier_args, scene, device, args.output)
    exposure = tier1["log_exposure"].exp()
    training_frames = [
        frame_by_id[frame_id] for frame_id in split.train_ids if frame_id != 0
    ]
    if args.displacements is not None:
        packed = np.load(args.displacements)
        tracking = DeformationResult(
            canonical_positions=packed["canonical_positions"],
            displacements=packed["displacements"],
            frame_ids=[int(v) for v in packed["frame_ids"]],
        )
        tracking.save_npz(args.output / "training_displacements.npz")
    else:
        tracking = track_deformation(
            tier1["field"],
            training_frames,
            tier1["renderer"],
            lambda frame: make_endoscopic_light(frame.camera.to(device), device),
            device=device,
            steps_per_frame=args.tracking_steps,
            image_fn=lambda out: twin_prediction(out, exposure),
            temporal_model=args.temporal_model,
            init_fn=make_flow_init(frame_by_id[0], device) if args.flow_init and not args.fused else None,
            observer=CorrespondenceObserver(frame_by_id[0], device) if args.fused else None,
        )
        tracking.save_npz(args.output / "training_displacements.npz")

    zero = np.zeros_like(tracking.canonical_positions)
    displacement_by_frame = {0: zero}
    displacement_by_frame.update(
        {
            int(frame_id): displacement
            for frame_id, displacement in zip(
                tracking.frame_ids,
                tracking.displacements,
            )
        }
    )
    interpolation = {}
    field = tier1["field"]
    renderer = tier1["renderer"]
    light_fn = lambda frame: make_endoscopic_light(frame.camera.to(device), device)
    layer = None
    if args.appearance_layer:
        canonical = torch.as_tensor(tracking.canonical_positions, device=device, dtype=field.means.dtype)
        train_frames = [frame_by_id[i] for i in split.train_ids]
        layer = build_appearance_layer(train_frames, canonical, displacement_by_frame, device,
                                       points_per_frame=args.appearance_points)
        print(f"appearance layer: {layer.field.means.shape[0]} Gaussians, "
              f"{int((layer.birth > 0).sum())} born after the reference frame", flush=True)
        log_exposure = torch.nn.Parameter(tier1["log_exposure"].detach().clone())
        _, residual_by_frame, gain_by_frame, bias_by_frame = fit_appearance_layer(
            layer, train_frames, displacement_by_frame, renderer, light_fn,
            appearance_image, log_exposure, device, steps=args.appearance_steps,
            node_canonical=canonical,
        )
        residual_by_frame.setdefault(0, np.zeros_like(tracking.canonical_positions))
        gain_by_frame.setdefault(0, np.zeros(3, dtype=np.float32))
        bias_by_frame.setdefault(0, np.zeros(3, dtype=np.float32))
        np.savez_compressed(args.output / "appearance_residual.npz", **{str(k): v for k, v in residual_by_frame.items()})
        np.savez_compressed(
            args.output / "appearance_exposure.npz",
            **{f"gain_{k}": v for k, v in gain_by_frame.items()},
            **{f"bias_{k}": v for k, v in bias_by_frame.items()},
            log_exposure=float(log_exposure.detach().cpu()),
        )
        exposure = log_exposure.detach().exp()
        field = layer.field
        torch.save(
            {"hexplane": layer.hexplane.state_dict() if layer.hexplane is not None else None,
             "t_max": layer.t_max},
            args.output / "appearance_hexplane.pt",
        )
    else:
        residual_by_frame, gain_by_frame, bias_by_frame = {}, {}, {}
    with torch.no_grad():
        for frame_id in split.holdout_ids:
            displacement = interpolate_displacement(frame_id, displacement_by_frame)
            frame = frame_by_id[frame_id]
            displacement = torch.from_numpy(displacement).to(device=device, dtype=field.means.dtype)
            rgb_delta = None
            if layer is not None:
                displacement, delta = layer.appearance_motion(displacement, float(frame_id))
                view = AppearanceView(field, displacement, delta)
                image_fn = appearance_image
            else:
                view = DeformedFieldView(field, displacement)
                image_fn = twin_prediction
            rendered = image_fn(
                renderer(
                    view,
                    frame.camera.to(device),
                    light=make_endoscopic_light(frame.camera.to(device), device),
                ),
                exposure,
            ).clamp(0, 1)
            _save_image(rendered, prediction_dir / f"{frame_id:06d}.png")
            lower = max(value for value in split.train_ids if value < frame_id)
            upper = min(
                (value for value in split.train_ids if value > frame_id),
                default=lower,
            )
            interpolation[str(frame_id)] = {"lower": lower, "upper": upper}

    protocol = {
        **split.to_dict(),
        "canonical_frame": 0,
        "holdout_displacement": (
            "appearance: EndoGaussian HGI+SGT; HexPlane D(mu,t) only, identity unused at eval"
            if layer is not None else "linear interpolation of adjacent tracked training frames"
        ),
        "image_formation": (
            "appearance: 0.5 * exposure * (shaded + albedo); HexPlane heads dmu/dR/dS/do"
            if layer is not None else "0.5 * exposure * (shaded + albedo)"
        ),
        "holdout_images_used_for_optimization": False,
        "interpolation": interpolation,
        "reconstruction_steps": args.reconstruction_steps,
        "tracking_steps": args.tracking_steps,
        "temporal_model": args.temporal_model,
        "flow_initialization": bool(args.flow_init),
        "fused_correspondence": bool(args.fused),
        "appearance_layer": (
            {"gaussians": int(layer.field.means.shape[0]), "born_after_reference": int((layer.birth > 0).sum()),
             "steps": args.appearance_steps, "fitted_on": "training frames only"}
            if layer is not None else None
        ),
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(f"predictions: {prediction_dir}")


if __name__ == "__main__":
    main()
