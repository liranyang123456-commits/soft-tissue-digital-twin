"""Render a canonical linear multi-view decomposition with Blender Cycles.

The module intentionally imports Blender only when a real render starts, so
``--help``, ``--dry-run`` and ``--self-test`` work with ordinary CPython.

Example:
  blender -b -P scripts/blender_render_multiview_cycles.py -- \
    --asset chair.glb --out data/chair_cycles --views 60 --seed 7
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
    parser.add_argument("--asset", help="Optional .blend/.glb/.gltf/.obj/.fbx asset")
    parser.add_argument("--out", required=True, help="Canonical dataset root")
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--randomize-scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-material", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-light", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests only")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = build_parser().parse_args(argv)
    if args.views < 3 or args.size < 1 or args.samples < 1:
        raise SystemExit("--views must be >= 3; --size and --samples must be positive")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio > 1:
        raise SystemExit("split ratios must be non-negative and sum to at most 1")
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
        spec = {
            "index": index,
            "camera_location": [
                radius * math.cos(elevation) * math.cos(azimuth),
                radius * math.cos(elevation) * math.sin(azimuth),
                radius * math.sin(elevation),
            ],
            "lens_mm": rng.uniform(42.0, 58.0) if args.randomize_camera else 50.0,
            "light_location": [
                light_radius * math.cos(light_elevation) * math.cos(light_azimuth),
                light_radius * math.cos(light_elevation) * math.sin(light_azimuth),
                light_radius * math.sin(light_elevation),
            ],
            "light_energy": rng.uniform(500.0, 1300.0) if args.randomize_light else 900.0,
            "light_size": rng.uniform(0.7, 2.2) if args.randomize_light else 1.5,
            "base_color": [rng.uniform(0.08, 0.8) for _ in range(3)]
            if args.randomize_material else [0.35, 0.12, 0.05],
            "roughness": rng.uniform(0.08, 0.65) if args.randomize_material else 0.22,
            "metallic": rng.uniform(0.0, 0.25) if args.randomize_material else 0.0,
            "object_rotation_z": rng.uniform(-math.pi, math.pi) if args.randomize_scene else 0.0,
            "world_strength": rng.uniform(0.03, 0.18) if args.randomize_scene else 0.08,
        }
        spec["fl_x"] = spec["lens_mm"] / 36.0 * args.size
        spec["image_size"] = args.size
        specs.append(spec)
    return specs


def make_frame_record(index: int, spec: dict[str, Any], matrix: list[list[float]] | None = None) -> dict[str, Any]:
    return {
        "frame_id": f"{index:04d}",
        "file_path": frame_path("full", index),
        **{f"{channel}_path": frame_path(channel, index) for channel in CHANNELS[1:]},
        "transform_matrix": matrix,
        "fl_x": spec["fl_x"],
        "fl_y": spec["fl_x"],
        "cx": spec["image_size"] / 2,
        "cy": spec["image_size"] / 2,
        "view_id": index,
        "light_id": index,
        "color_space": "linear",
        "camera_location": spec["camera_location"],
        "light_position": spec["light_location"],
        "light_intensity": spec["light_energy"],
    }


def write_manifests(
    root: Path,
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    matrices: dict[int, list[list[float]]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        (root / channel).mkdir(exist_ok=True)
    splits = split_indices(args.views, args.train_ratio, args.val_ratio, args.seed)
    merged_frames = []
    for split, indices in splits.items():
        split_frames = [
            {
                **make_frame_record(
                    i, specs[i], None if matrices is None else matrices.get(i)
                ),
                "split": split,
            }
            for i in indices
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
        "renderer": "blender-cycles",
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
            "diffuse": "Cycles Diffuse Direct + Diffuse Indirect",
            "specular": "Cycles Glossy Direct + Glossy Indirect",
            "albedo": "Cycles Diffuse Color",
        },
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def import_asset(bpy: Any, path: str | None) -> list[Any]:
    if path is None:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48)
        return [bpy.context.object]
    asset = Path(path)
    if not asset.exists():
        raise FileNotFoundError(asset)
    before = set(bpy.data.objects)
    suffix = asset.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(asset))
        return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(asset))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(asset))
        else:
            bpy.ops.import_scene.obj(filepath=str(asset))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(asset))
    else:
        raise ValueError(f"unsupported asset format: {suffix}")
    return [obj for obj in set(bpy.data.objects) - before if obj.type == "MESH"]


def look_at(obj: Any, target: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    from mathutils import Vector

    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def ensure_materials(bpy: Any, meshes: list[Any]) -> list[Any]:
    materials: list[Any] = []
    for obj in meshes:
        if not obj.data.materials:
            material = bpy.data.materials.new(f"CanonicalMaterial_{obj.name}")
            material.use_nodes = True
            obj.data.materials.append(material)
        for material in obj.data.materials:
            if material is not None:
                material.use_nodes = True
                materials.append(material)
    return list(dict.fromkeys(materials))


def setup_outputs(bpy: Any, scene: Any, root: Path) -> dict[str, Any]:
    layer = scene.view_layers[0]
    layer.use_pass_diffuse_color = True
    layer.use_pass_diffuse_direct = True
    layer.use_pass_diffuse_indirect = True
    layer.use_pass_glossy_direct = True
    layer.use_pass_glossy_indirect = True
    layer.use_pass_normal = True
    layer.use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render = tree.nodes.new("CompositorNodeRLayers")

    def add(left: str, right: str, name: str) -> Any:
        node = tree.nodes.new("CompositorNodeMixRGB")
        node.name = name
        node.blend_type = "ADD"
        node.inputs[0].default_value = 1.0
        tree.links.new(render.outputs[left], node.inputs[1])
        tree.links.new(render.outputs[right], node.inputs[2])
        return node.outputs[0]

    sources = {
        "full": render.outputs["Image"],
        "diffuse": add("Diffuse Direct", "Diffuse Indirect", "CanonicalDiffuse"),
        "specular": add("Glossy Direct", "Glossy Indirect", "CanonicalSpecular"),
        "albedo": render.outputs["Diffuse Color"],
        "normal": render.outputs["Normal"],
        "depth": render.outputs["Depth"],
        "mask": render.outputs["Alpha"],
    }
    outputs: dict[str, Any] = {}
    for channel, source in sources.items():
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.name = f"Canonical_{channel}"
        node.base_path = str(root / channel)
        node.format.file_format = "OPEN_EXR"
        node.format.color_depth = "32"
        node.format.color_mode = "BW" if channel in {"depth", "mask"} else "RGBA"
        tree.links.new(source, node.inputs[0])
        outputs[channel] = node
    return outputs


def finalize_outputs(root: Path, index: int) -> None:
    stem = f"{index:04d}"
    for channel in CHANNELS:
        directory = root / channel
        target = directory / f"{stem}.exr"
        candidates = sorted(
            (path for path in directory.glob(f"{stem}_*.exr") if path != target),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError(f"Cycles compositor did not produce {channel}/{stem}.exr")
        candidates[-1].replace(target)
        for stale in candidates[:-1]:
            stale.unlink()


def render(args: argparse.Namespace) -> None:
    try:
        import bpy
    except ImportError as exc:
        raise SystemExit(
            "Blender Python module 'bpy' is unavailable. Run this script with "
            "`blender -b -P ... -- [arguments]`, or use --dry-run/--self-test."
        ) from exc

    root = Path(args.out).resolve()
    specs = make_specs(args)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meshes = import_asset(bpy, args.asset)
    if not meshes:
        raise RuntimeError("asset contains no mesh objects")
    materials = ensure_materials(bpy, meshes)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.cycles.device = args.device
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("CanonicalWorld")
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes.get("Background")

    camera_data = bpy.data.cameras.new("CanonicalCamera")
    camera = bpy.data.objects.new("CanonicalCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_width = 36.0

    light_data = bpy.data.lights.new("CanonicalKey", type="AREA")
    light = bpy.data.objects.new("CanonicalKey", light_data)
    scene.collection.objects.link(light)
    outputs = setup_outputs(bpy, scene, root)
    matrices: dict[int, list[list[float]]] = {}

    for spec in specs:
        index = spec["index"]
        scene.frame_set(index)
        camera.location = spec["camera_location"]
        camera.data.lens = spec["lens_mm"]
        look_at(camera)
        light.location = spec["light_location"]
        light_data.energy = spec["light_energy"]
        light_data.shape = "DISK"
        light_data.size = spec["light_size"]
        look_at(light)
        background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        background.inputs["Strength"].default_value = spec["world_strength"]
        for obj in meshes:
            obj.rotation_euler.z = spec["object_rotation_z"]
        for material in materials:
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                continue
            bsdf.inputs["Base Color"].default_value = (*spec["base_color"], 1.0)
            bsdf.inputs["Roughness"].default_value = spec["roughness"]
            bsdf.inputs["Metallic"].default_value = spec["metallic"]
        for node in outputs.values():
            node.file_slots[0].path = f"{index:04d}_"
        bpy.ops.render.render(write_still=False)
        finalize_outputs(root, index)
        matrices[index] = [[float(value) for value in row] for row in camera.matrix_world]

    write_manifests(root, args, specs, matrices)
    print(f"Rendered {args.views} canonical Cycles views to {root}")


def self_test() -> None:
    root = Path(tempfile.mkdtemp(prefix="cycles_layout_"))
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
        assert json.loads((root / "meta.json").read_text())["linear"] is True
    finally:
        shutil.rmtree(root)
    print("Blender argument/layout self-test passed (bpy not required).")


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
