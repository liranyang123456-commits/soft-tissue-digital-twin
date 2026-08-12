"""Render the canonical linear multi-view dataset with the Mitsuba 3 Python API.

Mitsuba is imported lazily: argument parsing, ``--dry-run`` and ``--self-test``
work in a plain Python environment without Mitsuba, Dr.Jit, or Blender.

Example:
  python scripts/mitsuba_render_multiview.py \
    --asset model.obj --out data/model_mitsuba --views 60 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


CHANNELS = ("full", "diffuse", "specular", "albedo", "normal", "depth", "mask")
SCHEMA_VERSION = "mvbrdf-world-canonical-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", help="Optional Mitsuba-readable .obj or .ply mesh")
    parser.add_argument("--out", required=True, help="Canonical dataset root")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--variant", default="llvm_ad_rgb", help="Mitsuba variant")
    parser.add_argument("--randomize-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-material", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-light", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests only")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.views < 3 or args.size < 1 or args.samples < 1:
        raise SystemExit("--views must be >= 3; --size and --samples must be positive")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio > 1:
        raise SystemExit("split ratios must be non-negative and sum to at most 1")
    if args.asset and Path(args.asset).suffix.lower() not in {".obj", ".ply"}:
        raise SystemExit("Mitsuba renderer supports --asset .obj and .ply files")
    return args


def split_indices(count: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    n_train = int(count * train_ratio)
    n_val = int(count * val_ratio)
    return {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train : n_train + n_val]),
        "test": sorted(order[n_train + n_val :]),
    }


def frame_path(channel: str, index: int) -> str:
    return f"{channel}/{index:04d}.exr"


def make_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    specs: list[dict[str, Any]] = []
    for index in range(args.views):
        base_azimuth = 2.0 * math.pi * index / args.views
        azimuth = base_azimuth + (rng.uniform(-0.035, 0.035) if args.randomize_camera else 0.0)
        elevation = math.radians(rng.uniform(8.0, 38.0) if args.randomize_camera else 22.0)
        radius = args.radius * (rng.uniform(0.9, 1.1) if args.randomize_camera else 1.0)
        light_azimuth = rng.uniform(0.0, 2.0 * math.pi) if args.randomize_light else base_azimuth + 0.7
        light_elevation = math.radians(rng.uniform(25.0, 70.0) if args.randomize_light else 45.0)
        light_radius = rng.uniform(2.2, 4.0) if args.randomize_light else 3.0
        specs.append(
            {
                "index": index,
                "camera_location": [
                    radius * math.cos(elevation) * math.cos(azimuth),
                    radius * math.cos(elevation) * math.sin(azimuth),
                    radius * math.sin(elevation),
                ],
                "fov": rng.uniform(35.0, 48.0) if args.randomize_camera else 40.0,
                "light_location": [
                    light_radius * math.cos(light_elevation) * math.cos(light_azimuth),
                    light_radius * math.cos(light_elevation) * math.sin(light_azimuth),
                    light_radius * math.sin(light_elevation),
                ],
                "light_power": rng.uniform(80.0, 260.0) if args.randomize_light else 160.0,
                "base_color": [rng.uniform(0.08, 0.8) for _ in range(3)]
                if args.randomize_material else [0.35, 0.12, 0.05],
                "roughness": rng.uniform(0.08, 0.55) if args.randomize_material else 0.22,
                "specular_reflectance": rng.uniform(0.25, 0.8) if args.randomize_material else 0.5,
                "object_rotation_z": rng.uniform(-180.0, 180.0) if args.randomize_scene else 0.0,
                "object_scale": rng.uniform(0.8, 1.15) if args.randomize_scene else 1.0,
                "environment": rng.uniform(0.02, 0.15) if args.randomize_scene else 0.08,
                "floor_color": [rng.uniform(0.08, 0.3) for _ in range(3)]
                if args.randomize_scene else [0.18, 0.18, 0.18],
                "image_size": args.size,
            }
        )
    return specs


def look_at_matrix(origin: list[float], target: list[float] | None = None) -> list[list[float]]:
    target = target or [0.0, 0.0, 0.0]
    forward = [target[i] - origin[i] for i in range(3)]
    norm = math.sqrt(sum(value * value for value in forward))
    forward = [value / norm for value in forward]
    right = [forward[1], -forward[0], 0.0]
    norm = math.sqrt(sum(value * value for value in right))
    right = [value / norm for value in right]
    up = [
        right[1] * forward[2],
        -right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return [
        [right[0], up[0], -forward[0], origin[0]],
        [right[1], up[1], -forward[1], origin[1]],
        [right[2], up[2], -forward[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def make_frame_record(index: int, spec: dict[str, Any]) -> dict[str, Any]:
    focal = 0.5 * spec["image_size"] / math.tan(0.5 * math.radians(spec["fov"]))
    return {
        "frame_id": f"{index:04d}",
        "file_path": frame_path("full", index),
        **{f"{channel}_path": frame_path(channel, index) for channel in CHANNELS[1:]},
        "transform_matrix": look_at_matrix(spec["camera_location"]),
        "fl_x": focal,
        "fl_y": focal,
        "cx": spec["image_size"] / 2,
        "cy": spec["image_size"] / 2,
        "view_id": index,
        "light_id": index,
        "color_space": "linear",
        "camera_location": spec["camera_location"],
        "light_position": spec["light_location"],
        "light_intensity": spec["light_power"],
    }


def write_manifests(root: Path, args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        (root / channel).mkdir(exist_ok=True)
    splits = split_indices(args.views, args.train_ratio, args.val_ratio, args.seed)
    merged_frames = []
    for split, indices in splits.items():
        split_frames = [
            {**make_frame_record(index, specs[index]), "split": split}
            for index in indices
        ]
        merged_frames.extend(split_frames)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "camera_model": "OPENGL",
            "color_space": "linear",
            "normal_encoding": "signed",
            "normal_space": "world",
            "w": args.size,
            "h": args.size,
            "frames": split_frames,
        }
        (root / f"transforms_{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    merged = {
        "schema_version": SCHEMA_VERSION,
        "camera_model": "OPENGL",
        "color_space": "linear",
        "normal_encoding": "signed",
        "normal_space": "world",
        "w": args.size,
        "h": args.size,
        "frames": sorted(merged_frames, key=lambda frame: frame["frame_id"]),
    }
    (root / "transforms.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "renderer": "mitsuba-3",
        "variant": args.variant,
        "linear": True,
        "image_format": "OpenEXR",
        "channels": list(CHANNELS),
        "views": args.views,
        "seed": args.seed,
        "splits": {name: len(values) for name, values in splits.items()},
        "randomization": {
            "scene": args.randomize_scene,
            "material": args.randomize_material,
            "camera": args.randomize_camera,
            "light": args.randomize_light,
        },
        "decomposition": {
            "diffuse": "path tracing with a diffuse counterfactual BSDF",
            "specular": "max(full - diffuse, 0), in linear radiance",
            "albedo_normal_depth": "Mitsuba AOV integrator",
        },
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_mitsuba(variant: str) -> Any:
    try:
        import mitsuba as mi
    except ImportError as exc:
        raise SystemExit(
            "Mitsuba 3 is not installed. Install it with `pip install mitsuba`, "
            "then retry; --dry-run and --self-test do not require Mitsuba."
        ) from exc
    try:
        mi.set_variant(variant)
    except Exception as exc:
        available = ", ".join(mi.variants())
        raise SystemExit(f"Mitsuba variant '{variant}' is unavailable. Available: {available}") from exc
    return mi


def sensor_dict(mi: Any, spec: dict[str, Any], size: int) -> dict[str, Any]:
    return {
        "type": "perspective",
        "fov": spec["fov"],
        "to_world": mi.ScalarTransform4f.look_at(
            origin=spec["camera_location"], target=[0, 0, 0], up=[0, 0, 1]
        ),
        "sampler": {"type": "independent"},
        "film": {
            "type": "hdrfilm",
            "width": size,
            "height": size,
            "pixel_format": "rgb",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        },
    }


def material_dict(spec: dict[str, Any], diffuse_only: bool) -> dict[str, Any]:
    if diffuse_only:
        return {"type": "diffuse", "reflectance": {"type": "rgb", "value": spec["base_color"]}}
    return {
        "type": "roughplastic",
        "distribution": "ggx",
        "alpha": spec["roughness"],
        "diffuse_reflectance": {"type": "rgb", "value": spec["base_color"]},
        "specular_reflectance": {
            "type": "rgb",
            "value": [spec["specular_reflectance"]] * 3,
        },
    }


def scene_dict(
    mi: Any,
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    diffuse_only: bool,
    with_aovs: bool,
) -> dict[str, Any]:
    transform = (
        mi.ScalarTransform4f.rotate([0, 0, 1], spec["object_rotation_z"])
        @ mi.ScalarTransform4f.scale(spec["object_scale"])
    )
    if args.asset:
        asset = Path(args.asset).resolve()
        if not asset.exists():
            raise FileNotFoundError(asset)
        shape: dict[str, Any] = {
            "type": asset.suffix.lower().lstrip("."),
            "filename": str(asset),
            "to_world": transform,
            "bsdf": material_dict(spec, diffuse_only),
        }
    else:
        shape = {
            "type": "sphere",
            "to_world": transform,
            "bsdf": material_dict(spec, diffuse_only),
        }
    nested: dict[str, Any] = {"type": "path", "max_depth": 8}
    integrator = (
        {
            "type": "aov",
            "aovs": "albedo:albedo,normal:sh_normal,depth:depth",
            "integrator": nested,
        }
        if with_aovs
        else nested
    )
    return {
        "type": "scene",
        "integrator": integrator,
        "sensor": sensor_dict(mi, spec, args.size),
        "object": shape,
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, -1.05])
            @ mi.ScalarTransform4f.scale([4, 4, 4]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": spec["floor_color"]},
            },
        },
        "key": {
            "type": "point",
            "position": spec["light_location"],
            "intensity": {"type": "rgb", "value": [spec["light_power"]] * 3},
        },
        "environment": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [spec["environment"]] * 3},
        },
    }


def write_exr(mi: Any, path: Path, array: Any) -> None:
    mi.util.write_bitmap(str(path), array, write_async=False)


def render(args: argparse.Namespace) -> None:
    mi = load_mitsuba(args.variant)
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required for Mitsuba channel decomposition.") from exc

    root = Path(args.out).resolve()
    specs = make_specs(args)
    write_manifests(root, args, specs)
    for spec in specs:
        index = spec["index"]
        full_scene = mi.load_dict(scene_dict(mi, args, spec, diffuse_only=False, with_aovs=True))
        aov_image = np.asarray(mi.render(full_scene, spp=args.samples, seed=args.seed + index))
        if aov_image.ndim != 3 or aov_image.shape[-1] < 10:
            raise RuntimeError(
                f"unexpected Mitsuba AOV shape {aov_image.shape}; expected RGB+albedo+normal+depth"
            )
        full = aov_image[..., 0:3]
        albedo = aov_image[..., 3:6]
        normal = aov_image[..., 6:9]
        depth = aov_image[..., 9:10]

        diffuse_scene = mi.load_dict(
            scene_dict(mi, args, spec, diffuse_only=True, with_aovs=False)
        )
        diffuse = np.asarray(
            mi.render(diffuse_scene, spp=args.samples, seed=args.seed + index)
        )[..., :3]
        specular = np.maximum(full - diffuse, 0.0)
        mask = (np.isfinite(depth) & (depth > 0.0)).astype(np.float32)
        depth = np.where(mask > 0.0, depth, 0.0).astype(np.float32)
        outputs = {
            "full": full,
            "diffuse": diffuse,
            "specular": specular,
            "albedo": albedo,
            "normal": normal,
            "depth": depth,
            "mask": mask,
        }
        for channel, image in outputs.items():
            write_exr(mi, root / frame_path(channel, index), image)
        print(f"[{index + 1}/{args.views}] rendered {index:04d}")
    print(f"Rendered {args.views} canonical Mitsuba views to {root}")


def self_test() -> None:
    root = Path(tempfile.mkdtemp(prefix="mitsuba_layout_"))
    try:
        args = parse_args(["--out", str(root), "--views", "10", "--seed", "17", "--dry-run"])
        specs = make_specs(args)
        write_manifests(root, args, specs)
        assert all((root / channel).is_dir() for channel in CHANNELS)
        assert sum(
            len(json.loads((root / f"transforms_{split}.json").read_text())["frames"])
            for split in ("train", "val", "test")
        ) == 10
        first = json.loads((root / "transforms_train.json").read_text())["frames"][0]
        assert set(f"{channel}_path" for channel in CHANNELS[1:]).issubset(first)
        assert len(first["transform_matrix"]) == 4
        assert json.loads((root / "meta.json").read_text())["linear"] is True
    finally:
        shutil.rmtree(root)
    print("Mitsuba argument/layout self-test passed (Mitsuba not required).")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
    elif args.dry_run:
        root = Path(args.out).resolve()
        write_manifests(root, args, make_specs(args))
        print(f"Wrote dry-run canonical layout to {root}")
    else:
        render(args)


if __name__ == "__main__":
    main()
