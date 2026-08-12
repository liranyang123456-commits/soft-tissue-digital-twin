"""Trial: Stanford-ORB scene → reconstruct → export digital twin.

Uses a real multi-view scene with known camera poses and PBR materials
to validate the full digital-twin pipeline end-to-end. Stanford-ORB
provides Blender-rendered images with ground-truth transforms, making
it ideal for verifying the reconstruction → export → physics chain.

Usage:
    python scripts/run_orb_twin_trial.py --scene ball_scene002 --frames 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.lighting import WorldLight
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.export import to_mitsuba_xml
from mvbrdf_shr.world.material_db import classify_from_pbr
from mvbrdf_shr.world.physics_est import estimate_physics, to_instance_physics
from mvbrdf_shr.world.segment import InstanceSegmentation


def load_orb_scene(scene_dir: str, max_frames: int = 20, target_size: int = 512):
    """Load a Stanford-ORB Blender-format scene."""
    import json as _json
    scene_dir = Path(scene_dir)
    tf_path = scene_dir / "transforms_train.json"
    with open(tf_path) as f:
        tf = _json.load(f)

    angle_x = tf["camera_angle_x"]
    frames_data = tf["frames"][:max_frames]

    images = []
    cameras = []
    for fd in frames_data:
        img_path = scene_dir / fd["file_path"]
        if not img_path.suffix:
            img_path = img_path.with_suffix(".png")
        if not img_path.exists():
            img_path = img_path.with_suffix(".png")
        if not img_path.exists():
            continue
        img = Image.open(str(img_path)).convert("RGB")
        # Downscale to target_size to avoid GPU OOM
        if img.size[0] != target_size:
            img = img.resize((target_size, target_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        W, H = img.size
        focal = 0.5 * W / np.tan(0.5 * angle_x)
        K = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=torch.float32)
        # Blender → OpenCV convention
        c2w = np.array(fd["transform_matrix"], dtype=np.float32)
        c2w_cv = c2w.copy()
        c2w_cv[:3, 1] *= -1  # flip y
        c2w_cv[:3, 2] *= -1  # flip z
        cam = PerspectiveCamera(K=K, c2w=torch.from_numpy(c2w_cv), width=W, height=H,
                                frame_id=len(cameras), view_id=len(cameras))
        images.append(torch.from_numpy(img_np).permute(2, 0, 1))
        cameras.append(cam)

    return images, cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="ball_scene002")
    parser.add_argument("--orb-base", default=str(ROOT.parent / "mvbrdf_shr_next" / "data" / "external" / "stanford_orb" / "blender_LDR"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "orb_twin_trial"))
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load scene ---
    scene_dir = Path(args.orb_base) / args.scene
    print(f"[1/6] Loading Stanford-ORB scene: {args.scene}")
    images, cameras = load_orb_scene(str(scene_dir), max_frames=args.frames)
    print(f"  {len(images)} frames, {images[0].shape[-2]}x{images[0].shape[-1]}")
    # Save sample images
    for i in range(min(3, len(images))):
        Image.fromarray((images[i].permute(1,2,0).numpy()*255).astype(np.uint8)).save(
            str(output / f"input_{i:03d}.png"))

    # --- Build point cloud from camera centers + forward points ---
    print(f"[2/6] Building initial point cloud")
    points_list = []
    colors_list = []
    for i, (img, cam) in enumerate(zip(images, cameras)):
        # Sample pixels and back-project
        H, W = img.shape[-2], img.shape[-1]
        step = 8
        ys, xs = torch.meshgrid(
            torch.arange(0, H, step), torch.arange(0, W, step), indexing="ij")
        # Use center depth estimate (2.0m typical for ORB)
        depth = 2.0
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        cx, cy = cam.K[0, 2], cam.K[1, 2]
        dirs_cam = torch.stack([(xs-cx)/fx, (ys-cy)/fy, torch.ones_like(xs)], dim=-1) * depth
        pts = (dirs_cam.reshape(-1, 3) @ cam.c2w[:3, :3].T) + cam.c2w[:3, 3]
        cols = img[:, ys.flatten(), xs.flatten()].T
        points_list.append(pts)
        colors_list.append(cols)
    points = torch.cat(points_list, dim=0).to(device)
    colors = torch.cat(colors_list, dim=0).to(device)
    # Subsample
    n = points.shape[0]
    if n > 2000:
        idx = torch.randperm(n)[:2000]
        points = points[idx]
        colors = colors[idx]
    print(f"  {points.shape[0]} points")

    # --- Build Gaussian field + optimize ---
    print(f"[3/6] Building Gaussian BRDF field + {args.steps}-step optimization")
    field = GaussianBRDFField(points, colors, initial_scale=0.05).to(device)
    renderer = GaussianBRDFRenderer(backend="torch")
    optimizer = torch.optim.Adam([
        {"params": [field.means], "lr": 1e-3},
        {"params": [field.opacity_logits], "lr": 5e-3},
        {"params": [field.base_color_logits], "lr": 1e-3},
        {"params": [field.normal_raw], "lr": 1e-3},
    ], lr=1e-3)

    light = WorldLight(
        env_sh=torch.zeros(9, 3, device=device),
        direction=torch.tensor([0.3, -0.5, -0.8], device=device),
        intensity=torch.tensor([3.0], device=device),
        color=torch.tensor([1.0, 0.95, 0.9], device=device),
        position=torch.tensor([2.0, 2.0, 2.0], device=device),
        point_weight=torch.tensor([0.5], device=device),
        exposure=torch.tensor([0.0], device=device),
        white_balance=torch.tensor([1.0, 1.0, 1.0], device=device),
    )

    losses = []
    for step in range(args.steps):
        fi = step % len(images)
        cam = cameras[fi].to(device)
        img_target = images[fi].to(device)
        optimizer.zero_grad()
        out = renderer(field, cam, light=light)
        pred = out.full[:3]
        if pred.shape != img_target.shape:
            pred = torch.nn.functional.interpolate(pred.unsqueeze(0), img_target.shape[-2:],
                                                    mode="bilinear", align_corners=False)[0]
        loss = torch.nn.functional.l1_loss(pred, img_target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}: loss = {loss.item():.6f}")

    with open(output / "loss_curve.json", "w") as f:
        json.dump(losses, f)

    # --- Render comparison ---
    print(f"[4/6] Rendering comparison")
    with torch.no_grad():
        cam_eval = cameras[0].to(device)
        out = renderer(field, cam_eval, light=light)
        rendered = out.full[:3].clamp(0, 1).cpu()
        target = images[0].cpu()
        h, w = min(rendered.shape[-2], target.shape[-2]), min(rendered.shape[-1], target.shape[-1])
        comparison = torch.cat([target[:, :h, :w], rendered[:, :h, :w]], dim=-1)
        Image.fromarray((comparison.permute(1,2,0).numpy()*255).astype(np.uint8)).save(
            str(output / "comparison.png"))
        mse = torch.mean((target[:, :h, :w] - rendered[:, :h, :w])**2).item()
        psnr = float(20 * np.log10(1.0 / max(np.sqrt(mse), 1e-6)))
        print(f"  PSNR: {psnr:.2f} dB")

    # --- Export optical twin ---
    print(f"[5/6] Exporting optical twin (Mitsuba + USD)")
    with torch.no_grad():
        mat = field.materials()
        values = {
            "position": field.means.detach().cpu().numpy(),
            "scale": field.scales.detach().cpu().numpy(),
            "rotation_quaternion": torch.nn.functional.normalize(field.quaternions, dim=-1).detach().cpu().numpy(),
            "opacity": field.opacity.detach().cpu().numpy(),
            "normal": field.normals.detach().cpu().numpy(),
            "albedo": mat.albedo.detach().cpu().numpy(),
            "specular": mat.specular.detach().cpu().numpy(),
            "diffuse_weight": mat.diffuse_weight.detach().cpu().numpy(),
            "roughness": mat.roughness.detach().cpu().numpy(),
            "metalness": mat.metalness.detach().cpu().numpy(),
            "absorption": mat.absorption.detach().cpu().numpy(),
            "projected_texture": field.texture.detach().cpu().numpy(),
        }
    mesh_path = output / "surface.ply"
    _write_ply(mesh_path, values)
    to_mitsuba_xml(values, mesh_path, output)
    try:
        from mvbrdf_shr.world.usd_export import to_usd
        to_usd(values, mesh_path, output)
        print(f"  USD: {output / 'scene.usda'}")
    except ImportError:
        print("  USD skipped (pxr not available)")
    print(f"  Mitsuba: {output / 'scene.xml'}")

    # --- Export physics twin ---
    print(f"[6/6] Exporting physics twin")
    avg_metal = float(mat.metalness.mean().cpu())
    avg_rough = float(mat.roughness.mean().cpu())
    avg_alb = mat.albedo.mean(0).cpu().numpy()
    mat_class = classify_from_pbr(avg_metal, avg_rough, tuple(avg_alb))
    print(f"  PBR: metal={avg_metal:.3f} rough={avg_rough:.3f} albedo={avg_alb}")
    print(f"  Material class: {mat_class}")
    inst = InstanceSegmentation(0, np.arange(field.n_gaussians), "object", mat_class, 1.0)
    est = estimate_physics(inst, field)
    print(f"  density={est.density.mean:.0f}±{est.density.std:.0f} friction={est.friction.mean:.3f} restitution={est.restitution.mean:.3f}")
    with open(output / "physics.json", "w") as f:
        json.dump(est.to_dict(), f, indent=2, default=str)
    try:
        from mvbrdf_shr.world.usd_export import to_usd_with_physics
        to_usd_with_physics(values, [mesh_path], [to_instance_physics(est)], output)
        print(f"  Physics twin: {output / 'twin.usda'}")
    except Exception as e:
        print(f"  Physics twin failed: {e}")

    summary = {"scene": args.scene, "frames": len(images), "points": points.shape[0],
               "steps": args.steps, "psnr_dB": psnr, "material_class": mat_class,
               "final_loss": losses[-1]}
    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone! PSNR={psnr:.2f}dB. Results in {output}")


def _write_ply(path, values):
    n = len(values["position"])
    lines = ["ply","format ascii 1.0",f"element vertex {n}",
             "property float x","property float y","property float z",
             "property float nx","property float ny","property float nz",
             "property uchar red","property uchar green","property uchar blue","end_header"]
    rgb = (np.clip(values["albedo"], 0, 1) * 255).astype(np.uint8)
    for i in range(n):
        p, nm, c = values["position"][i], values["normal"][i], rgb[i]
        lines.append(f"{p[0]} {p[1]} {p[2]} {nm[0]} {nm[1]} {nm[2]} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
