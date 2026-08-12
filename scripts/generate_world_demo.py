"""Generate a tiny calibrated world-space multiview scene without Blender."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.lighting import PerFrameLighting
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.scene import GaussianBRDFField, inverse_sigmoid


def look_at(position: torch.Tensor) -> torch.Tensor:
    forward = F.normalize(-position, dim=0)
    up_hint = torch.tensor([0.0, -1.0, 0.0])
    right = F.normalize(torch.cross(forward, up_hint, dim=0), dim=0)
    down = F.normalize(torch.cross(forward, right, dim=0), dim=0)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.stack([right, down, forward], dim=1)
    c2w[:3, 3] = position
    return c2w


def save(tensor: torch.Tensor, path: Path):
    array = (
        tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/world_demo")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.out)
    images, diffuse = output / "images", output / "diffuse"
    images.mkdir(parents=True, exist_ok=True)
    diffuse.mkdir(parents=True, exist_ok=True)

    # Colored Gaussian shell.
    count = 24
    i = torch.arange(count, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / count
    radius = torch.sqrt(1 - z.square())
    theta = i * math.pi * (3 - math.sqrt(5))
    points = 0.7 * torch.stack(
        [radius * torch.cos(theta), z, radius * torch.sin(theta)], dim=-1
    )
    colors = (points / 1.4 + 0.5).clamp(0.05, 0.95)
    scene = GaussianBRDFField(points, colors, initial_scale=0.22)
    with torch.no_grad():
        scene.normal_raw.copy_(F.normalize(points, dim=-1))
        scene.roughness_logits.copy_(
            inverse_sigmoid(torch.linspace(0.12, 0.5, count).view(-1, 1))
        )
    lights = PerFrameLighting(args.views)
    renderer = GaussianBRDFRenderer(chunk_size=8)
    fx = 0.85 * args.size
    K = torch.tensor(
        [[fx, 0, args.size / 2], [0, fx, args.size / 2], [0, 0, 1.0]]
    )
    convert = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    metadata = {
        "fl_x": fx,
        "fl_y": fx,
        "cx": args.size / 2,
        "cy": args.size / 2,
        "w": args.size,
        "h": args.size,
        "points_path": "points.npy",
        "point_colors_path": "point_colors.npy",
        "frames": [],
    }
    for view in range(args.views):
        angle = 2 * math.pi * view / args.views
        position = torch.tensor(
            [2.8 * math.cos(angle), 0.5 * math.sin(angle * 0.5), 2.8 * math.sin(angle)]
        )
        c2w_cv = look_at(position)
        camera = PerspectiveCamera(K, c2w_cv, args.size, args.size, view)
        with torch.no_grad():
            lights.direction_raw[view].copy_(
                F.normalize(torch.tensor([math.cos(angle + 0.8), -0.4, math.sin(angle + 0.8)]), dim=0)
            )
            lights.log_intensity[view].fill_(math.log(1.0 + 0.3 * math.sin(angle) ** 2))
            rendered = renderer(scene, camera, light=lights(view))
        save(rendered.full, images / f"{view:04d}.png")
        save(rendered.diffuse, diffuse / f"{view:04d}.png")
        c2w_gl = c2w_cv @ convert
        metadata["frames"].append(
            {
                "file_path": f"images/{view:04d}.png",
                "diffuse_path": f"diffuse/{view:04d}.png",
                "transform_matrix": c2w_gl.tolist(),
                "light_direction": lights(view).direction.detach().tolist(),
                "light_intensity": float(lights(view).intensity.detach()),
                "env_sh": lights(view).env_sh.detach().tolist(),
            }
        )
    np.save(output / "points.npy", points.numpy())
    np.save(output / "point_colors.npy", colors.numpy())
    (output / "transforms.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote calibrated world demo to {output}")


if __name__ == "__main__":
    main()

