"""Compare plausible normal-map coordinate conventions on trained scenes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from mvbrdf_shr.metrics_ir import normal_angular_metrics
from mvbrdf_shr.world.evaluate_ir import load_world_checkpoint


def transform_normal(normal: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,jhw->ihw", rotation.to(normal), normal)


@torch.inference_mode()
def diagnose(checkpoint: Path, max_frames: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, dataset, pipeline = load_world_checkpoint(checkpoint, device)
    selected = [
        index
        for index in state.get("holdout_ids", [])
        if dataset[index].normal_gt is not None
    ][:max_frames]
    if not selected:
        raise ValueError("checkpoint has no holdout normal maps")

    values: dict[str, list[float]] = defaultdict(list)
    for frame_id in selected:
        frame = dataset[frame_id]
        target = frame.normal_gt.to(device)
        camera = frame.camera.to(device)
        pred = pipeline(camera, frame_id, image=frame.image.to(device), refine=False)[
            "render"
        ].normal
        mask = (
            frame.mask_gt.to(device)
            if frame.mask_gt is not None
            else torch.ones_like(target[:1], dtype=torch.bool)
        )
        opengl_flip = torch.diag(target.new_tensor([1.0, -1.0, -1.0]))
        candidates = {
            "stored": target,
            "stored_negated": -target,
            "camera_to_world": transform_normal(target, camera.c2w[:3, :3]),
            "camera_to_world_opengl": transform_normal(
                target, camera.c2w[:3, :3] @ opengl_flip
            ),
            "world_to_camera": transform_normal(target, camera.w2c[:3, :3]),
            "world_to_camera_opengl": transform_normal(
                target, camera.w2c[:3, :3] @ opengl_flip
            ),
        }
        for name, candidate in candidates.items():
            metrics = normal_angular_metrics(pred, candidate, mask)
            assert metrics["mean"] is not None
            values[name].append(float(metrics["mean"]))
            cosine = (
                F.normalize(pred, dim=0) * F.normalize(candidate, dim=0)
            ).sum(dim=0)
            valid = mask.squeeze(0) > 0.5
            unoriented = torch.rad2deg(torch.acos(cosine.abs().clamp(0, 1)))
            values[f"{name}_unoriented"].append(float(unoriented[valid].mean()))

    summary = {
        name: sum(items) / len(items)
        for name, items in sorted(values.items())
    }
    return {
        "checkpoint": str(checkpoint),
        "dataset": dataset.metadata.get("adapter"),
        "frames": selected,
        "mean_angular_error_degrees": summary,
        "best_oriented": min(
            (item for item in summary.items() if not item[0].endswith("_unoriented")),
            key=lambda item: item[1],
        ),
        "best_unoriented": min(
            (item for item in summary.items() if item[0].endswith("_unoriented")),
            key=lambda item: item[1],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [
        diagnose(checkpoint, args.max_frames) for checkpoint in args.checkpoints
    ]
    text = json.dumps(reports, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
