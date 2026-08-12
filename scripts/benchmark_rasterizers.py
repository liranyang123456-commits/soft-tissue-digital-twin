"""Benchmark PyTorch reference splatting against gsplat CUDA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.lighting import PerFrameLighting
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.scene import GaussianBRDFField


def benchmark(
    renderer, scene, camera, light, iterations: int, backward: bool = False
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if backward:
        with torch.no_grad():
            renderer(scene, camera, light=light)
    else:
        with torch.no_grad():
            renderer(scene, camera, light=light)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        if backward:
            renderer(scene, camera, light=light).full.mean().backward()
            scene.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                renderer(scene, camera, light=light)
        torch.cuda.synchronize()
    return {
        "milliseconds": (time.perf_counter() - start) * 1000 / iterations,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=int, default=2000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--gsplat-iters", type=int, default=20)
    parser.add_argument("--torch-iters", type=int, default=1)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", default="outputs/world_raster_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    torch.manual_seed(7)
    points = torch.randn(args.gaussians, 3, device=device) * 0.6
    points[:, 2] = torch.rand(args.gaussians, device=device) * 2 + 2
    colors = torch.rand(args.gaussians, 3, device=device)
    scene = GaussianBRDFField(points, colors, initial_scale=0.025).to(device)
    focal = 0.8 * args.size
    K = torch.tensor(
        [[focal, 0, args.size / 2], [0, focal, args.size / 2], [0, 0, 1]],
        device=device,
    )
    camera = PerspectiveCamera(K, torch.eye(4, device=device), args.size, args.size)
    light = PerFrameLighting(1).to(device)(0)
    torch_result = (
        None
        if args.skip_torch
        else benchmark(
            GaussianBRDFRenderer(chunk_size=32, backend="torch").to(device),
            scene,
            camera,
            light,
            args.torch_iters,
            args.backward,
        )
    )
    gsplat_result = benchmark(
        GaussianBRDFRenderer(backend="gsplat").to(device),
        scene,
        camera,
        light,
        args.gsplat_iters,
        args.backward,
    )
    result = {
        "device": torch.cuda.get_device_name(),
        "gaussians": args.gaussians,
        "resolution": [args.size, args.size],
        "backward": args.backward,
        "torch": torch_result,
        "gsplat": gsplat_result,
        "speedup": (
            torch_result["milliseconds"] / gsplat_result["milliseconds"]
            if torch_result is not None
            else None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

