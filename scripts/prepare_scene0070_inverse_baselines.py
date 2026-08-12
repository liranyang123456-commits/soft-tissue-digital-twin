"""Prepare a leakage-free scene_0070 protocol for inverse-rendering baselines.

The exported training directories contain only RGB, alpha masks, and cameras.
Diffuse, normal, depth, and material ground truth remain in the evaluation-only
directory and must not be passed to a training process.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


TRAIN_IDS = (1, 2, 3, 5, 6, 7, 9, 10, 11)
TEST_IDS = (0, 4, 8)


def _rgba(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if rgb.size != mask.size:
        raise ValueError(f"RGB/mask size mismatch: {rgb_path} and {mask_path}")
    array = np.concatenate(
        (np.asarray(rgb), np.asarray(mask, dtype=np.uint8)[..., None]), axis=-1
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGBA").save(output_path)


def _camera_angle_x(metadata: dict) -> float:
    width = float(metadata["w"])
    focal_x = float(metadata["fl_x"])
    return 2.0 * math.atan(width / (2.0 * focal_x))


def _split_metadata(metadata: dict, ids: tuple[int, ...], image_dir: str) -> dict:
    frames = metadata["frames"]
    selected = []
    for view_id in ids:
        frame = dict(frames[view_id])
        frame["file_path"] = f"{image_dir}/{view_id:03d}"
        # Baseline loaders receive no decomposition or geometric ground truth.
        for key in tuple(frame):
            if key.endswith("_path") and key != "file_path":
                frame.pop(key)
        for key in (
            "env_sh",
            "light_position",
            "light_direction",
            "light_color",
            "light_intensity",
            "point_light_weight",
        ):
            frame.pop(key, None)
        frame["view_id"] = view_id
        selected.append(frame)
    return {
        "camera_angle_x": _camera_angle_x(metadata),
        "fl_x": metadata["fl_x"],
        "fl_y": metadata["fl_y"],
        "cx": metadata["cx"],
        "cy": metadata["cy"],
        "w": metadata["w"],
        "h": metadata["h"],
        "frames": selected,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _prepare_nerfactor(
    metadata: dict, common: Path, output: Path
) -> dict[str, list[int]]:
    root = output / "nerfactor"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    split_ids = {
        "train": list(TRAIN_IDS),
        # NeRFactor requires a validation view. Reusing a training view avoids
        # exposing any of the three frozen test images during model selection.
        "val": [TRAIN_IDS[-1]],
        "test": list(TEST_IDS),
    }
    angle_x = _camera_angle_x(metadata)
    for split, ids in split_ids.items():
        for index, view_id in enumerate(ids):
            view_dir = root / f"{split}_{index:03d}"
            view_dir.mkdir(parents=True)
            matrix = metadata["frames"][view_id]["transform_matrix"]
            record = {
                "cam_transform_mat": ",".join(
                    str(float(value)) for row in matrix for value in row
                ),
                "cam_angle_x": angle_x,
                "imh": int(metadata["h"]),
                "imw": int(metadata["w"]),
                "source_view_id": view_id,
            }
            _write_json(view_dir / "metadata.json", record)
            if split != "test":
                source_split = "train"
                shutil.copy2(
                    common / source_split / f"{view_id:03d}.png",
                    view_dir / "rgba.png",
                )
    return split_ids


def prepare(source: Path, output: Path) -> None:
    metadata = json.loads((source / "transforms.json").read_text(encoding="utf-8"))
    if len(metadata.get("frames", [])) != 12:
        raise ValueError("The frozen scene_0070 protocol requires exactly 12 views")

    ours = output / "ours"
    if ours.exists():
        shutil.rmtree(ours)
    ours.mkdir(parents=True)
    ours_frames = []
    for view_id, source_frame in enumerate(metadata["frames"]):
        image_name = f"{view_id:03d}.png"
        (ours / "images").mkdir(exist_ok=True)
        (ours / "mask").mkdir(exist_ok=True)
        shutil.copy2(source / "images" / image_name, ours / "images" / image_name)
        shutil.copy2(source / "mask" / image_name, ours / "mask" / image_name)
        frame = {
            key: value
            for key, value in source_frame.items()
            if not key.endswith("_path")
            and key
            not in {
                "env_sh",
                "light_position",
                "light_direction",
                "light_color",
                "light_intensity",
                "point_light_weight",
            }
        }
        frame["file_path"] = f"images/{view_id:03d}"
        frame["mask_path"] = f"mask/{view_id:03d}.png"
        frame["view_id"] = view_id
        ours_frames.append(frame)
    ours_metadata = {
        key: value for key, value in metadata.items() if key != "frames"
    }
    ours_metadata["frames"] = ours_frames
    _write_json(ours / "transforms.json", ours_metadata)

    common = output / "rgb_mask_camera_only"
    for split, ids in (("train", TRAIN_IDS), ("test", TEST_IDS)):
        for view_id in ids:
            _rgba(
                source / "images" / f"{view_id:03d}.png",
                source / "mask" / f"{view_id:03d}.png",
                common / split / f"{view_id:03d}.png",
            )

    train_meta = _split_metadata(metadata, TRAIN_IDS, "train")
    test_meta = _split_metadata(metadata, TEST_IDS, "test")
    _write_json(common / "transforms_train.json", train_meta)
    _write_json(common / "transforms_test.json", test_meta)

    nvdiffrec = output / "nvdiffrec"
    nvdiffrec.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        target = nvdiffrec / split
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(common / split, target)
    _write_json(nvdiffrec / "transforms_train.json", train_meta)
    _write_json(nvdiffrec / "transforms_test.json", test_meta)
    nerfactor_ids = _prepare_nerfactor(metadata, common, output)

    evaluation = output / "evaluation_only"
    for kind in ("diffuse", "normal", "depth", "albedo", "specular", "mask"):
        destination = evaluation / kind
        destination.mkdir(parents=True, exist_ok=True)
        for view_id in TEST_IDS:
            source_file = source / kind / f"{view_id:03d}.png"
            if source_file.exists():
                shutil.copy2(source_file, destination / source_file.name)

    manifest = {
        "protocol": "scene0070-rgb-mask-camera-only-v1",
        "source_scene": str(source.resolve()),
        "train_view_ids": list(TRAIN_IDS),
        "test_view_ids": list(TEST_IDS),
        "nerfactor_directory_view_ids": nerfactor_ids,
        "training_inputs": [
            "sRGB images",
            "foreground masks",
            "known camera intrinsics",
            "known poses",
        ],
        "ours_training_directory": str(ours.resolve()),
        "withheld_from_training": [
            "diffuse RGB",
            "specular RGB",
            "normal",
            "depth",
            "albedo",
            "BRDF parameters",
            "point cloud",
            "lighting metadata",
        ],
        "evaluation_only": str(evaluation.resolve()),
        "notes": (
            "The alpha channel is the supplied foreground mask. Ground truth is "
            "physically separated from each baseline training directory."
        ),
    }
    _write_json(output / "protocol_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
