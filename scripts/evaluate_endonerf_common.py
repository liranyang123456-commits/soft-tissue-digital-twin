"""Evaluate EndoNeRF predictions with the official 1-in-8 holdout split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.world.endonerf_protocol import (  # noqa: E402
    aggregate_metrics,
    endonerf_holdout,
    image_metrics,
)


def _files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _source_image(scene: Path, frame_id: int) -> Path:
    candidates = (
        scene / "images" / f"{frame_id:06d}.png",
        scene / "images" / f"frame-{frame_id:06d}.png",
        scene / "images" / f"frame-{frame_id:06d}.color.png",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"image for frame {frame_id} not found")


def _source_auxiliary(scene: Path, frame_id: int, kind: str) -> Path | None:
    patterns = {
        "depth": (
            scene / "depth" / f"frame-{frame_id:06d}.depth.png",
            scene / "depth" / f"{frame_id:06d}.png",
        ),
        "mask": (
            scene / "masks" / f"frame-{frame_id:06d}.mask.png",
            scene / "masks" / f"{frame_id:06d}.png",
        ),
    }
    return next((path for path in patterns[kind] if path.exists()), None)


def _tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _tissue_mask(scene: Path, frame_id: int, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    mask = torch.ones(height, width)
    depth_path = _source_auxiliary(scene, frame_id, "depth")
    if depth_path is not None:
        depth_image = Image.open(depth_path)
        if depth_image.size != (width, height):
            depth_image = depth_image.resize((width, height), Image.Resampling.NEAREST)
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        mask *= torch.from_numpy((depth > 0.01).astype(np.float32))
    tool_path = _source_auxiliary(scene, frame_id, "mask")
    if tool_path is not None:
        tool_image = Image.open(tool_path).convert("L")
        if tool_image.size != (width, height):
            tool_image = tool_image.resize((width, height), Image.Resampling.NEAREST)
        tool = np.asarray(tool_image, dtype=np.float32) / 255.0
        mask *= torch.from_numpy((tool < 0.5).astype(np.float32))
    return mask


def _endogaussian_render_dir(model_path: Path) -> Path:
    test_root = model_path / "test"
    candidates = sorted(test_root.glob("ours_*/renders"))
    if not candidates:
        raise FileNotFoundError(f"no EndoGaussian test renders under {test_root}")
    return candidates[-1]


def prediction_map(
    directory: Path,
    holdout_ids: tuple[int, ...],
    *,
    layout: str,
) -> dict[int, Path]:
    if layout in {"sequential", "endogaussian"}:
        paths = _files(directory)
        if len(paths) != len(holdout_ids):
            raise ValueError(
                f"expected {len(holdout_ids)} predictions, found {len(paths)}"
            )
        return dict(zip(holdout_ids, paths))
    result = {}
    for frame_id in holdout_ids:
        candidates = (
            directory / f"{frame_id:06d}.png",
            directory / f"frame-{frame_id:06d}.png",
            directory / f"{frame_id}.png",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"prediction for frame {frame_id} not found")
        result[frame_id] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--endogaussian-model", type=Path)
    parser.add_argument(
        "--layout",
        choices=("frame-id", "sequential", "endogaussian"),
        default="frame-id",
    )
    parser.add_argument("--method", default="method")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    poses = np.load(args.scene_root / "poses_bounds.npy")
    split = endonerf_holdout(len(poses))
    if args.endogaussian_model is not None:
        prediction_directory = _endogaussian_render_dir(args.endogaussian_model)
        layout = "endogaussian"
    elif args.predictions is not None:
        prediction_directory = args.predictions
        layout = args.layout
    else:
        raise ValueError("provide --predictions or --endogaussian-model")
    predictions = prediction_map(prediction_directory, split.holdout_ids, layout=layout)

    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for frame_id in split.holdout_ids:
        target = _tensor(_source_image(args.scene_root, frame_id))
        prediction = _tensor(predictions[frame_id])
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction.unsqueeze(0),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
        tissue = _tissue_mask(args.scene_root, frame_id, target.shape[-2:])
        row = {
            "frame_id": frame_id,
            "prediction": str(predictions[frame_id]),
            **image_metrics(prediction, target, tissue),
        }
        rows.append(row)

    report = {
        "method": args.method,
        "scene": args.scene_root.name,
        "protocol": split.to_dict(),
        "metrics": aggregate_metrics(rows),
        "per_frame": rows,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    with (args.output / "per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "protocol.json").write_text(
        json.dumps(split.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
