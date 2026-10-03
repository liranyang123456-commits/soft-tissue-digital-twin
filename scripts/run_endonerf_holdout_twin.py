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

from mvbrdf_shr.world.deform import DeformedFieldView, track_deformation  # noqa: E402
from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene  # noqa: E402
from mvbrdf_shr.world.endonerf_protocol import endonerf_holdout  # noqa: E402
from scripts.run_soft_tissue_twin import make_endoscopic_light, run_tier1  # noqa: E402


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
    training_frames = [
        frame_by_id[frame_id] for frame_id in split.train_ids if frame_id != 0
    ]
    tracking = track_deformation(
        tier1["field"],
        training_frames,
        tier1["renderer"],
        lambda frame: make_endoscopic_light(frame.camera.to(device), device),
        device=device,
        steps_per_frame=args.tracking_steps,
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
    with torch.no_grad():
        for frame_id in split.holdout_ids:
            displacement = interpolate_displacement(frame_id, displacement_by_frame)
            frame = frame_by_id[frame_id]
            view = DeformedFieldView(
                field,
                torch.from_numpy(displacement).to(device=device, dtype=field.means.dtype),
            )
            rendered = renderer(
                view,
                frame.camera.to(device),
                light=make_endoscopic_light(frame.camera.to(device), device),
            ).full[:3]
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
        "holdout_displacement": "linear interpolation of adjacent tracked training frames",
        "holdout_images_used_for_optimization": False,
        "interpolation": interpolation,
        "reconstruction_steps": args.reconstruction_steps,
        "tracking_steps": args.tracking_steps,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(f"predictions: {prediction_dir}")


if __name__ == "__main__":
    main()
