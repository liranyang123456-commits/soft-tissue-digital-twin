"""Render scene_0070 cameras under one fixed learned illumination.

The resulting RGB sequence is suitable for inverse-rendering methods that
assume one scene illumination across all views. Training exports still contain
only RGB, masks, and cameras; decomposition buffers remain evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import torch

from mvbrdf_shr.viz import save_image
from mvbrdf_shr.world.evaluate_ir import load_world_checkpoint


@torch.inference_mode()
def render_constant_light(
    checkpoint: Path,
    source: Path,
    output: Path,
    light_frame: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dataset, pipeline = load_world_checkpoint(
        checkpoint, device, data_root_override=source
    )
    metadata = json.loads((source / "transforms.json").read_text(encoding="utf-8"))
    if len(dataset.frames) != len(metadata["frames"]):
        raise ValueError("Checkpoint and source camera counts differ")

    if output.exists():
        shutil.rmtree(output)
    for kind in (
        "images",
        "diffuse",
        "specular",
        "albedo",
        "normal",
        "depth",
        "mask",
    ):
        (output / kind).mkdir(parents=True, exist_ok=True)

    source_light = pipeline.frame_lighting(light_frame)
    fixed_light = replace(
        source_light,
        point_weight=torch.zeros_like(source_light.point_weight),
    )
    for index, frame in enumerate(dataset.frames):
        result = pipeline(
            frame.camera.to(device),
            light_frame,
            image=None,
            refine=False,
            apply_sensor=True,
            light_override=fixed_light,
        )
        render = result["render"]
        name = f"{index:03d}.png"
        save_image(render.full.clamp(0, 1), output / "images" / name)
        save_image(render.diffuse.clamp(0, 1), output / "diffuse" / name)
        save_image(render.specular.clamp(0, 1), output / "specular" / name)
        save_image(render.albedo.clamp(0, 1), output / "albedo" / name)
        save_image(((render.normal + 1) * 0.5).clamp(0, 1), output / "normal" / name)
        for kind in ("depth", "mask"):
            source_file = source / kind / name
            if source_file.exists():
                shutil.copy2(source_file, output / kind / name)

    generated = dict(metadata)
    generated["constant_light_source_frame"] = light_frame
    generated["generation_checkpoint"] = str(checkpoint.resolve())
    generated["frames"] = []
    for index, source_frame in enumerate(metadata["frames"]):
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
        frame.update(
            {
                "file_path": f"images/{index:03d}",
                "diffuse_path": f"diffuse/{index:03d}.png",
                "specular_path": f"specular/{index:03d}.png",
                "albedo_path": f"albedo/{index:03d}.png",
                "normal_path": f"normal/{index:03d}.png",
                "depth_path": f"depth/{index:03d}.png",
                "mask_path": f"mask/{index:03d}.png",
                "view_id": index,
            }
        )
        generated["frames"].append(frame)
    (output / "transforms.json").write_text(
        json.dumps(generated, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--light-frame", type=int, default=1)
    args = parser.parse_args()
    render_constant_light(
        args.checkpoint.resolve(),
        args.source.resolve(),
        args.output.resolve(),
        args.light_frame,
    )


if __name__ == "__main__":
    main()
