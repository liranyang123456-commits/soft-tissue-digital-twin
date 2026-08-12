"""Blender-side synthetic multiview/full-diffuse dataset renderer.

Example:
  blender -b -P scripts/blender_render_multiview.py -- \
    --asset chair.glb --out data/blender_chair --views 60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def import_asset(path: str | None):
    if path is None:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64)
        material = bpy.data.materials.new("GlossyMaterial")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.35, 0.12, 0.05, 1)
        bsdf.inputs["Roughness"].default_value = 0.18
        bpy.context.object.data.materials.append(material)
        return
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported asset format {suffix}")


def look_at(obj, target=(0, 0, 0)):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def principled_nodes():
    for material in bpy.data.materials:
        if material.use_nodes:
            node = material.node_tree.nodes.get("Principled BSDF")
            if node is not None:
                yield node


def set_diffuse_only(enabled: bool, saved: list | None = None):
    if enabled:
        state = []
        for node in principled_nodes():
            values = {}
            for name in ("Metallic", "Roughness", "Specular IOR Level", "Specular"):
                if name in node.inputs:
                    values[name] = node.inputs[name].default_value
            state.append((node, values))
            if "Metallic" in node.inputs:
                node.inputs["Metallic"].default_value = 0.0
            if "Roughness" in node.inputs:
                node.inputs["Roughness"].default_value = 1.0
            for name in ("Specular IOR Level", "Specular"):
                if name in node.inputs:
                    node.inputs[name].default_value = 0.0
        return state
    assert saved is not None
    for node, values in saved:
        for name, value in values.items():
            node.inputs[name].default_value = value
    return None


def main():
    args = arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_asset(args.asset)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.04, 0.04, 0.04)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 50
    camera.data.sensor_width = 36

    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = 900
    light_data.shape = "DISK"
    light_data.size = 2.0
    light = bpy.data.objects.new("KeyLight", light_data)
    scene.collection.objects.link(light)

    output = Path(args.out)
    full_dir, diffuse_dir = output / "images", output / "diffuse"
    full_dir.mkdir(parents=True, exist_ok=True)
    diffuse_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(args.views):
        azimuth = 2 * math.pi * index / args.views
        elevation = math.radians(12 + 20 * math.sin(index * 0.37))
        camera.location = (
            args.radius * math.cos(elevation) * math.cos(azimuth),
            args.radius * math.cos(elevation) * math.sin(azimuth),
            args.radius * math.sin(elevation),
        )
        look_at(camera)

        light_angle = azimuth * 1.7 + 0.4
        light.location = (
            2.5 * math.cos(light_angle),
            2.5 * math.sin(light_angle),
            2.0 + 0.5 * math.sin(index * 0.51),
        )
        look_at(light)
        light_data.energy = 650 + 450 * (0.5 + 0.5 * math.sin(index * 0.73))

        full_path = full_dir / f"{index:04d}.png"
        scene.render.filepath = str(full_path)
        bpy.ops.render.render(write_still=True)

        saved = set_diffuse_only(True)
        diffuse_path = diffuse_dir / f"{index:04d}.png"
        scene.render.filepath = str(diffuse_path)
        bpy.ops.render.render(write_still=True)
        set_diffuse_only(False, saved)

        direction = Vector(light.location).normalized()
        frames.append(
            {
                "file_path": f"images/{index:04d}.png",
                "diffuse_path": f"diffuse/{index:04d}.png",
                "transform_matrix": [list(row) for row in camera.matrix_world],
                "light_position": list(light.location),
                "light_direction": list(direction),
                "light_intensity": light_data.energy / 900.0,
                "env_sh": [[0.8, 0.8, 0.8]] + [[0, 0, 0]] * 8,
            }
        )

    angle_x = 2 * math.atan(camera.data.sensor_width / (2 * camera.data.lens))
    metadata = {
        "camera_angle_x": angle_x,
        "w": args.size,
        "h": args.size,
        "frames": frames,
    }
    (output / "transforms.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Rendered {args.views} full/diffuse multiview pairs to {output}")


if __name__ == "__main__":
    main()

