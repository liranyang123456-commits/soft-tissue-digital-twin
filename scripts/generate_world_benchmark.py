"""Generate a calibrated multi-scene 3D Gaussian BRDF benchmark locally."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.lighting import WorldLight
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.scene import GaussianBRDFField, inverse_sigmoid


def look_at(position: torch.Tensor) -> torch.Tensor:
    forward = F.normalize(-position, dim=0)
    up_hint = torch.tensor([0.0, -1.0, 0.0], device=position.device)
    right = F.normalize(torch.cross(forward, up_hint, dim=0), dim=0)
    down = F.normalize(torch.cross(forward, right, dim=0), dim=0)
    c2w = torch.eye(4, device=position.device)
    c2w[:3, :3] = torch.stack([right, down, forward], dim=1)
    c2w[:3, 3] = position
    return c2w


def save_rgb(tensor: torch.Tensor, path: Path):
    array = (
        tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_gray(tensor: torch.Tensor, path: Path):
    array = (tensor.detach().cpu().clamp(0, 1).squeeze().numpy() * 255).astype(
        np.uint8
    )
    Image.fromarray(array, mode="L").save(path)


def make_geometry(
    shape: str, count: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if shape in {"sphere", "ellipsoid", "bumpy"}:
        index = torch.arange(count, device=device, dtype=torch.float32) + 0.5
        z = 1 - 2 * index / count
        radius = torch.sqrt((1 - z.square()).clamp(min=0))
        theta = index * math.pi * (3 - math.sqrt(5))
        unit = torch.stack(
            [radius * torch.cos(theta), z, radius * torch.sin(theta)], dim=-1
        )
        axes = 0.55 + 0.45 * torch.rand(
            3, generator=generator, device=device
        )
        if shape == "sphere":
            axes[:] = axes.mean()
        bump = (
            1
            + 0.12 * torch.sin(theta * 3 + z * 5)
            if shape == "bumpy"
            else torch.ones_like(z)
        )
        points = unit * axes * bump[:, None]
        normals = F.normalize(unit / axes, dim=-1)
        return points, normals
    # Torus.
    major = 0.55 + 0.1 * torch.rand((), generator=generator, device=device)
    minor = 0.18 + 0.08 * torch.rand((), generator=generator, device=device)
    u = 2 * math.pi * torch.rand(count, generator=generator, device=device)
    v = 2 * math.pi * torch.rand(count, generator=generator, device=device)
    points = torch.stack(
        [
            (major + minor * torch.cos(v)) * torch.cos(u),
            minor * torch.sin(v),
            (major + minor * torch.cos(v)) * torch.sin(u),
        ],
        dim=-1,
    )
    normals = F.normalize(
        torch.stack(
            [torch.cos(v) * torch.cos(u), torch.sin(v), torch.cos(v) * torch.sin(u)],
            dim=-1,
        ),
        dim=-1,
    )
    return points, normals


def make_scene(
    seed: int, count: int, device: torch.device
) -> tuple[GaussianBRDFField, str]:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = ["sphere", "ellipsoid", "bumpy", "torus"][seed % 4]
    points, normals = make_geometry(shape, count, generator, device)
    phase = torch.rand(3, generator=generator, device=device) * math.pi * 2
    frequency = 2 + seed % 5
    colors = (
        0.18
        + 0.72
        * (
            0.5
            + 0.5
            * torch.sin(
                points[:, :1] * frequency
                + points[:, 1:2] * (frequency + 1)
                + phase.view(1, 3)
            )
        )
    ).clamp(0.03, 0.97)
    scale = 0.055 if shape != "torus" else 0.065
    scene = GaussianBRDFField(points, colors, initial_scale=scale).to(device)
    with torch.no_grad():
        scene.normal_raw.copy_(normals)
        scene.opacity_logits.fill_(2.8)
        roughness = 0.04 + 0.24 * torch.rand(
            count, 1, generator=generator, device=device
        )
        metallic = 0.15 + 0.75 * torch.rand(
            count, 1, generator=generator, device=device
        )
        specular = 0.35 + 0.50 * torch.rand(
            count, 1, generator=generator, device=device
        )
        absorption = 0.01 + 0.06 * torch.rand(
            count, 1, generator=generator, device=device
        )
        diffuse = (1 - specular - absorption).clamp(min=0.15)
        budget = torch.cat([diffuse, specular, absorption], dim=-1)
        budget = budget / budget.sum(-1, keepdim=True)
        scene.material_budget_logits.copy_(budget.log())
        scene.roughness_logits.copy_(inverse_sigmoid((roughness - 0.04) / 0.96))
        scene.metalness_logits.copy_(inverse_sigmoid(metallic.clamp(1e-4, 1 - 1e-4)))
    return scene, shape


def scene_arrays(scene: GaussianBRDFField) -> dict[str, np.ndarray]:
    material = scene.materials()
    return {
        "position": scene.means.detach().cpu().numpy(),
        "scale": scene.scales.detach().cpu().numpy(),
        "quaternion": F.normalize(scene.quaternions, dim=-1).detach().cpu().numpy(),
        "opacity": scene.opacity.detach().cpu().numpy(),
        "normal": scene.normals.detach().cpu().numpy(),
        "albedo": material.albedo.detach().cpu().numpy(),
        "specular": material.specular.detach().cpu().numpy(),
        "roughness": material.roughness.detach().cpu().numpy(),
        "metalness": material.metalness.detach().cpu().numpy(),
        "absorption": material.absorption.detach().cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/world_benchmark_v1")
    parser.add_argument("--train-scenes", type=int, default=60)
    parser.add_argument("--val-scenes", type=int, default=10)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--gaussians", type=int, default=384)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark generation")
    device = torch.device("cuda")
    output = Path(args.out)
    flat = output / "flat"
    for split in ("train", "val", "test"):
        for kind in ("input", "gt", "mask", "specular"):
            (flat / split / kind).mkdir(parents=True, exist_ok=True)
    for kind in ("input", "gt", "mask", "specular"):
        (output / "flat_hard" / "test" / kind).mkdir(parents=True, exist_ok=True)
    renderer = GaussianBRDFRenderer(backend="gsplat")
    focal = args.size * 0.9
    K = torch.tensor(
        [[focal, 0, args.size / 2], [0, focal, args.size / 2], [0, 0, 1]],
        device=device,
    )
    convert = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    split_counts = {
        "train": args.train_scenes,
        "val": args.val_scenes,
        "test": args.test_scenes,
    }
    manifest = {
        "name": "WorldGaussianBRDF-Benchmark-v1",
        "resolution": args.size,
        "views_per_scene": args.views,
        "gaussians_per_scene": args.gaussians,
        "splits": split_counts,
        "frames": [],
        # Provenance declaration (do not remove). The GT diffuse images in this
        # benchmark are produced by the SAME GaussianBRDFRenderer (analytic GGX
        # Cook-Torrance, diffuse lobe only) used by the method under test. This
        # makes the benchmark a CONTROLLED PROBE / oracle upper bound (when
        # geometry/albedo/lighting priors are supplied) rather than an
        # independent third-party measurement. Numbers on this benchmark must be
        # reported with that caveat; the blind inverse-rendering setting
        # (no priors) is the honest lower bound. See also the geometry gate on
        # real captured data (Stanford-ORB / DiLiGenT-MV) for an
        # renderer-independent check.
        "gt_provenance": {
            "gt_diffuse_source": "GaussianBRDFRenderer (method's own analytic GGX renderer)",
            "gt_diffuse_pass": "diffuse lobe only (specular lobe disabled)",
            "independence": "NOT independent of the method renderer",
            "intended_use": "controlled probe / oracle upper bound; not a SOTA claim",
            "honest_lower_bound": "blind inverse-rendering (no priors) reported separately",
            "renderer_independent_check": "Stanford-ORB / DiLiGenT-MV geometry gate",
        },
    }
    global_scene = 0
    hard_test_frames = 0
    for split, scene_count in split_counts.items():
        for local_scene in range(scene_count):
            seed = args.seed + global_scene * 101
            scene, shape = make_scene(seed, args.gaussians, device)
            scene_dir = output / "scenes" / split / f"scene_{global_scene:04d}"
            for kind in (
                "images",
                "diffuse",
                "specular",
                "mask",
                "normal",
                "depth",
                "albedo",
            ):
                (scene_dir / kind).mkdir(parents=True, exist_ok=True)
            transforms = {
                "fl_x": focal,
                "fl_y": focal,
                "cx": args.size / 2,
                "cy": args.size / 2,
                "w": args.size,
                "h": args.size,
                "shape": shape,
                "frames": [],
            }
            arrays = scene_arrays(scene)
            np.savez(scene_dir / "gaussian_brdf_gt.npz", **arrays)
            np.save(scene_dir / "points.npy", arrays["position"])
            np.save(scene_dir / "point_colors.npy", arrays["albedo"])
            for view in range(args.views):
                angle = 2 * math.pi * view / args.views
                elevation = 0.10 + 0.32 * math.sin(angle * 1.7 + seed)
                camera_position = torch.tensor(
                    [
                        1.65 * math.cos(elevation) * math.cos(angle),
                        1.65 * math.sin(elevation),
                        1.65 * math.cos(elevation) * math.sin(angle),
                    ],
                    device=device,
                )
                c2w = look_at(camera_position)
                camera = PerspectiveCamera(K, c2w, args.size, args.size, view)
                light_angle = angle + 0.18 * math.sin(view * 1.7 + seed)
                point_mode = False
                mirror_direction = F.normalize(camera_position, dim=0)
                light = WorldLight(
                    env_sh=torch.cat(
                        [
                            torch.full((1, 3), 0.35 + 0.2 * (seed % 5) / 4, device=device),
                            torch.zeros(8, 3, device=device),
                        ]
                    ),
                    direction=mirror_direction,
                    intensity=torch.tensor(
                        [12.0 + 8.0 * (0.5 + 0.5 * math.sin(view * 1.31))],
                        device=device,
                    ),
                    color=torch.tensor(
                        [1.0, 0.82 + 0.18 * ((seed + view) % 2), 0.78 + 0.22 * (view % 3) / 2],
                        device=device,
                    ),
                    position=torch.stack(
                        [
                            torch.tensor(2.2 * math.cos(light_angle), device=device),
                            camera_position[1] - 0.2,
                            torch.tensor(2.2 * math.sin(light_angle), device=device),
                        ]
                    ),
                    point_weight=torch.tensor([1.0 if point_mode else 0.0], device=device),
                )
                with torch.no_grad():
                    rendered = renderer(scene, camera, light=light)
                stem = f"scene_{global_scene:04d}_v{view:03d}"
                paths = {
                    "input": scene_dir / "images" / f"{view:03d}.png",
                    "gt": scene_dir / "diffuse" / f"{view:03d}.png",
                    "specular": scene_dir / "specular" / f"{view:03d}.png",
                    "mask": scene_dir / "mask" / f"{view:03d}.png",
                    "normal": scene_dir / "normal" / f"{view:03d}.png",
                    "depth": scene_dir / "depth" / f"{view:03d}.png",
                    "albedo": scene_dir / "albedo" / f"{view:03d}.png",
                }
                save_rgb(rendered.full, paths["input"])
                save_rgb(rendered.diffuse, paths["gt"])
                save_rgb(rendered.specular, paths["specular"])
                save_gray(rendered.mask, paths["mask"])
                save_rgb((rendered.normal + 1) * 0.5, paths["normal"])
                valid_depth = rendered.depth[rendered.alpha > 1e-3]
                if len(valid_depth):
                    lo, hi = valid_depth.min(), valid_depth.max()
                    depth = (rendered.depth - lo) / (hi - lo + 1e-6)
                else:
                    depth = torch.zeros_like(rendered.depth)
                save_gray(depth * (rendered.alpha > 1e-3), paths["depth"])
                save_rgb(rendered.albedo, paths["albedo"])
                for key in ("input", "gt", "specular", "mask"):
                    destination = flat / split / key / f"{stem}.png"
                    destination.write_bytes(paths[key].read_bytes())
                mse = (rendered.full - rendered.diffuse).square().mean().clamp(min=1e-12)
                input_psnr = float(20 * torch.log10(1 / mse.sqrt()))
                highlight_coverage = float((rendered.mask > 0.2).float().mean())
                is_hard = split == "test" and input_psnr < 35.0
                if is_hard:
                    hard_test_frames += 1
                    for key in ("input", "gt", "specular", "mask"):
                        destination = (
                            output / "flat_hard" / "test" / key / f"{stem}.png"
                        )
                        destination.write_bytes(paths[key].read_bytes())
                c2w_gl = c2w @ convert
                transforms["frames"].append(
                    {
                        "file_path": f"images/{view:03d}.png",
                        "diffuse_path": f"diffuse/{view:03d}.png",
                        "transform_matrix": c2w_gl.detach().cpu().tolist(),
                        "light_position": light.position.detach().cpu().tolist(),
                        "light_direction": light.direction.detach().cpu().tolist(),
                        "light_color": light.color.detach().cpu().tolist(),
                        "point_light_weight": float(light.point_weight),
                        "light_intensity": float(light.intensity),
                        "env_sh": light.env_sh.detach().cpu().tolist(),
                    }
                )
                manifest["frames"].append(
                    {
                        "split": split,
                        "scene": global_scene,
                        "view": view,
                        "shape": shape,
                        "input_psnr": input_psnr,
                        "highlight_coverage": highlight_coverage,
                        "hard_test": is_hard,
                        **{
                            key: str(path.relative_to(output)).replace("\\", "/")
                            for key, path in paths.items()
                        },
                    }
                )
            (scene_dir / "transforms.json").write_text(
                json.dumps(transforms, indent=2), encoding="utf-8"
            )
            print(
                f"[{split}] scene {local_scene + 1}/{scene_count} "
                f"(global {global_scene}, {shape})",
                flush=True,
            )
            global_scene += 1
    manifest["total_scenes"] = global_scene
    manifest["total_frames"] = len(manifest["frames"])
    manifest["hard_test_frames"] = hard_test_frames
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {global_scene} scenes / {len(manifest['frames'])} frames to {output}"
    )


if __name__ == "__main__":
    main()

