"""Export projected EndoGaussian primitive positions at the first and last frame.

Executed from the EndoGaussian repository root after training. The canonical
Gaussians are deformed by the trained deformation network at the time stamps
of the first and the last video camera, exactly as in the official renderer,
and projected with the same cameras.
"""
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from arguments import ModelHiddenParams, ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel  # noqa: E402
from scene import Scene  # noqa: E402


def deformed_means(gaussians, camera):
    means = gaussians.get_xyz
    time = torch.tensor(camera.time, device=means.device).repeat(means.shape[0], 1)
    table = gaussians._deformation_table
    out = means.clone()
    moved, _, _, _ = gaussians._deformation(
        means[table], gaussians._scaling[table], gaussians._rotation[table],
        gaussians._opacity[table], time[table],
    )
    out[table] = moved
    return out


def project(camera, points):
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    clip = homogeneous @ camera.full_proj_transform.to(points.device)
    ndc = clip[:, :2] / clip[:, 3:4]
    u = ((ndc[:, 0] + 1.0) * camera.image_width - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * camera.image_height - 1.0) * 0.5
    depth = (homogeneous @ camera.world_view_transform.to(points.device))[:, 2]
    return torch.stack([u, v], dim=1).cpu().numpy(), depth.cpu().numpy()


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    hyper = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--out", required=True)
    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    dataset = model.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyper.extract(args))
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cameras = sorted(scene.getVideoCameras(), key=lambda camera: camera.time)
        first, last = cameras[0], cameras[-1]
        uv0, z0 = project(first, deformed_means(gaussians, first))
        uvT, zT = project(last, deformed_means(gaussians, last))
        opacity = gaussians.get_opacity.reshape(-1).cpu().numpy()
    np.savez_compressed(args.out, uv0=uv0, z0=z0, uvT=uvT, zT=zT, opacity=opacity,
                        times=np.array([first.time, last.time]))
    print(f"saved {args.out}: {uv0.shape[0]} Gaussians")


if __name__ == "__main__":
    main()
