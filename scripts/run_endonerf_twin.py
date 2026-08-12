"""End-to-end trial: EndoNeRF soft-tissue images → digital-twin scene export.

This script runs the full pipeline on a real EndoNeRF soft-tissue sequence:
1. Load the EndoNeRF scene (cutting_tissues_twice or pulling_soft_tissues).
2. Build a GaussianBRDFField from the depth-derived point cloud.
3. Run a short optimization (geometry warm-up).
4. Export the optical twin (Mitsuba XML + USD).
5. Segment instances (color-based fallback, no SAM needed).
6. Estimate physics parameters (Bayesian).
7. Export the physics twin (USD with UsdPhysics).
8. Save intermediate results (images, point cloud, checkpoints) for inspection.

Usage:
    python scripts/run_endonerf_twin.py --scene cutting_tissues_twice --frames 20
    python scripts/run_endonerf_twin.py --scene pulling_soft_tissues --frames 30 --steps 200
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

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene, list_endonerf_scenes
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.lighting import WorldLight
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.export import to_mitsuba_xml
from mvbrdf_shr.world.material_db import classify_from_pbr
from mvbrdf_shr.world.physics_est import estimate_physics, to_instance_physics
from mvbrdf_shr.world.segment import InstanceSegmentation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="cutting_tissues_twice",
                   help="EndoNeRF scene name.")
    p.add_argument("--datasets-dir", default=str(ROOT.parent / "datasets"),
                   help="Datasets directory (contains endonerf/).")
    p.add_argument("--output", default=str(ROOT / "outputs" / "endonerf_twin"),
                   help="Output directory for results.")
    p.add_argument("--frames", type=int, default=20,
                   help="Number of frames to load.")
    p.add_argument("--steps", type=int, default=100,
                   help="Optimization steps (geometry warm-up).")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Image scale factor (0.5 = half resolution).")
    return p


def main():
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: Load EndoNeRF scene ---
    scene_path = Path(args.datasets_dir) / "endonerf" / args.scene
    print(f"\n[1/7] Loading EndoNeRF scene: {scene_path}")
    scene = load_endonerf_scene(
        scene_path, max_frames=args.frames, image_scale=args.scale
    )
    print(f"  Loaded {len(scene)} frames, {scene.points.shape[0] if scene.points is not None else 0} points")

    # Save sample images for inspection
    sample_dir = output / "sample_images"
    sample_dir.mkdir(exist_ok=True)
    from PIL import Image
    for i, frame in enumerate(scene.frames[:5]):
        img_np = (frame.image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(str(sample_dir / f"frame_{i:03d}.png"))
        if frame.depth_gt is not None:
            depth_np = frame.depth_gt.numpy()
            d_min, d_max = depth_np[depth_np > 0].min(), depth_np[depth_np > 0].max()
            depth_vis = ((depth_np - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(str(sample_dir / f"depth_{i:03d}.png"))
    print(f"  Saved sample images to {sample_dir}")

    # --- Step 2: Build Gaussian BRDF field ---
    print(f"\n[2/7] Building Gaussian BRDF field from point cloud")
    points = scene.points.to(device)
    colors = scene.point_colors.to(device) if scene.point_colors is not None else None
    field = GaussianBRDFField(points, colors, initial_scale=0.5).to(device)
    print(f"  {field.n_gaussians} Gaussians initialized")

    # Set reasonable initial material for soft tissue (non-metal, rough)
    with torch.no_grad():
        field.metalness_logits.fill_(-5.0)  # non-metal
        field.roughness_logits.fill_(2.0)    # moderate roughness
        field.material_budget_logits[:, 0] = 5.0   # diffuse dominant
        field.material_budget_logits[:, 1] = -2.0   # low specular
        field.material_budget_logits[:, 2] = -5.0   # low absorption

    # --- Step 3: Quick geometry warm-up optimization ---
    print(f"\n[3/7] Running {args.steps}-step geometry warm-up")
    renderer = GaussianBRDFRenderer(backend="torch")
    optimizer = torch.optim.Adam([
        {"params": [field.means], "lr": 1e-3},
        {"params": [field.normal_raw], "lr": 1e-3},
        {"params": [field.opacity_logits], "lr": 5e-3},
        {"params": [field.base_color_logits], "lr": 1e-3},
    ])

    # Use a simple lighting setup (endoscopic = point light near camera)
    cam = scene.frames[0].camera.to(device)
    light = WorldLight(
        env_sh=torch.zeros(9, 3, device=device),
        direction=torch.tensor([0.0, 0.0, -1.0], device=device),
        intensity=torch.tensor([2.0], device=device),
        color=torch.tensor([1.0, 0.95, 0.9], device=device),  # warm endoscopic light
        position=cam.center.clone(),
        point_weight=torch.tensor([1.0], device=device),
        exposure=torch.tensor([0.0], device=device),
        white_balance=torch.tensor([1.0, 1.0, 1.0], device=device),
    )

    losses = []
    for step in range(args.steps):
        # Pick a random frame for supervision
        fi = step % len(scene.frames)
        frame = scene.frames[fi]
        cam_step = frame.camera.to(device)
        light.position = cam_step.center.clone()

        optimizer.zero_grad()
        out = renderer(field, cam_step, light=light)
        pred = out.full[:3].permute(1, 2, 0)  # (H, W, 3)
        target = frame.image.to(device).permute(1, 2, 0)  # (H, W, 3)

        # Resize pred to match target if needed
        if pred.shape[:2] != target.shape[:2]:
            pred = torch.nn.functional.interpolate(
                pred.permute(2, 0, 1).unsqueeze(0),
                target.shape[:2][::-1], mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)

        loss = torch.nn.functional.l1_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}: loss = {loss.item():.6f}")

    # Save loss curve
    import json
    with open(output / "loss_curve.json", "w") as f:
        json.dump(losses, f)

    # --- Step 4: Export optical twin (Mitsuba XML + USD) ---
    print(f"\n[4/7] Exporting optical twin (Mitsuba + USD)")
    with torch.no_grad():
        material = field.materials()
        values = {
            "position": field.means.detach().cpu().numpy(),
            "scale": field.scales.detach().cpu().numpy(),
            "rotation_quaternion": torch.nn.functional.normalize(
                field.quaternions, dim=-1
            ).detach().cpu().numpy(),
            "opacity": field.opacity.detach().cpu().numpy(),
            "normal": field.normals.detach().cpu().numpy(),
            "albedo": material.albedo.detach().cpu().numpy(),
            "specular": material.specular.detach().cpu().numpy(),
            "diffuse_weight": material.diffuse_weight.detach().cpu().numpy(),
            "roughness": material.roughness.detach().cpu().numpy(),
            "metalness": material.metalness.detach().cpu().numpy(),
            "absorption": material.absorption.detach().cpu().numpy(),
            "projected_texture": field.texture.detach().cpu().numpy(),
        }

    # Write a simple mesh (point cloud as PLY for now)
    mesh_path = output / "surface_mesh.ply"
    _write_point_ply(mesh_path, values)
    to_mitsuba_xml(values, mesh_path, output, light=None, camera=None)
    print(f"  Mitsuba scene: {output / 'scene.xml'}")

    # USD export (optional, needs pxr)
    try:
        from mvbrdf_shr.world.usd_export import to_usd
        to_usd(values, mesh_path, output, light=None, camera=None)
        print(f"  USD scene: {output / 'scene.usda'}")
    except ImportError:
        print("  USD export skipped (pxr not available)")

    # --- Step 5: Render a comparison image ---
    print(f"\n[5/7] Rendering comparison")
    with torch.no_grad():
        cam_eval = scene.frames[0].camera.to(device)
        light.position = cam_eval.center.clone()
        out = renderer(field, cam_eval, light=light)
        rendered = out.full[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        target = scene.frames[0].image.permute(1, 2, 0).numpy()
        h = min(rendered.shape[0], target.shape[0])
        w = min(rendered.shape[1], target.shape[1])
        comparison = np.concatenate([target[:h, :w], rendered[:h, :w]], axis=1)
        Image.fromarray((comparison * 255).astype(np.uint8)).save(
            str(output / "comparison_input_vs_rendered.png")
        )
        # Compute PSNR
        mse = np.mean((target[:h, :w] - rendered[:h, :w]) ** 2)
        psnr = float(20 * np.log10(1.0 / max(np.sqrt(mse), 1e-6)))
        print(f"  PSNR (input vs rendered): {psnr:.2f} dB")
        print(f"  Comparison saved: {output / 'comparison_input_vs_rendered.png'}")

    # --- Step 6: Instance segmentation + physics estimation ---
    print(f"\n[6/7] Instance segmentation + physics estimation")
    # For soft tissue, treat the whole scene as one "tissue" instance
    # (real segmentation would use SAM, but we demonstrate the physics pipeline)
    with torch.no_grad():
        mats = field.materials()
        avg_metalness = float(mats.metalness.mean().cpu())
        avg_roughness = float(mats.roughness.mean().cpu())
        avg_albedo = mats.albedo.mean(dim=0).cpu().numpy()

    from mvbrdf_shr.world.tissue_db import classify_tissue_from_pbr
    material_class = classify_tissue_from_pbr(
        avg_metalness, avg_roughness,
        (float(avg_albedo[0]), float(avg_albedo[1]), float(avg_albedo[2]))
    )
    print(f"  PBR: metalness={avg_metalness:.3f}, roughness={avg_roughness:.3f}, albedo={avg_albedo}")
    print(f"  Classified tissue: {material_class} (kPa-scale prior, not engineering material)")

    # Soft tissue: use the tissue-aware estimator (kPa-scale, viscoelastic).
    # No deformation observation in this static trial → E falls back to the
    # broad class prior and is honestly tagged 'assumed_table'.
    from mvbrdf_shr.world.physics_est import estimate_tissue_physics
    tissue_est = estimate_tissue_physics(material_class, instance_id=0)
    print(f"  Tissue estimate:")
    print(f"    class: {tissue_est.tissue_class} ({tissue_est.constitutive_model})")
    print(f"    Young's E: {tissue_est.youngs_median_pa/1e3:.1f} kPa "
          f"95%CI={tissue_est.youngs_ci95_kpa} ({tissue_est.audit['youngs_modulus']})")
    print(f"    Poisson: {tissue_est.poisson.mean:.3f} ± {tissue_est.poisson.std:.3f}")
    print(f"    viscosity: {tissue_est.viscosity.mean:.1f} ± {tissue_est.viscosity.std:.1f} Pa·s")
    print(f"    density: {tissue_est.density.mean:.1f} ± {tissue_est.density.std:.1f} kg/m³")
    print(f"    friction: {tissue_est.friction.mean:.3f} ± {tissue_est.friction.std:.3f}")

    # Rigid-body-style estimate kept only for USD assembly compatibility.
    instance = InstanceSegmentation(
        instance_id=0,
        gaussian_indices=np.arange(field.n_gaussians),
        semantic_label="soft_tissue",
        material_class="rubber",  # closest rigid-sim proxy; audit records truth
        confidence=1.0,
    )
    est = estimate_physics(instance, field)
    est.audit["tissue_class"] = material_class
    est.audit["youngs_modulus"] = tissue_est.audit["youngs_modulus"]
    est.youngs_modulus.mean = tissue_est.youngs_median_pa  # kPa, not GPa

    # Save physics estimates (rigid-proxy + tissue-native)
    with open(output / "physics_estimate.json", "w") as f:
        json.dump(est.to_dict(), f, indent=2, default=str)
    with open(output / "tissue_estimate.json", "w") as f:
        json.dump(tissue_est.to_dict(), f, indent=2, default=str)
    print(f"  Saved: {output / 'physics_estimate.json'}")
    print(f"  Saved: {output / 'tissue_estimate.json'}")

    # --- Step 7: Export physics twin (USD with UsdPhysics) ---
    print(f"\n[7/7] Exporting physics twin (USD + UsdPhysics)")
    try:
        from mvbrdf_shr.world.usd_export import to_usd_with_physics
        ip = to_instance_physics(est)
        to_usd_with_physics(values, [mesh_path], [ip], output)
        print(f"  Physics twin: {output / 'twin.usda'}")
    except ImportError:
        print("  USD physics export skipped (pxr not available)")
    except Exception as e:
        print(f"  USD physics export failed: {e}")

    # --- Summary ---
    summary = {
        "scene": args.scene,
        "n_frames": len(scene),
        "n_gaussians": field.n_gaussians,
        "optimization_steps": args.steps,
        "final_loss": losses[-1],
        "psnr_dB": psnr,
        "tissue_class": material_class,
        "tissue_physics": tissue_est.to_dict(),
        "physics": est.to_dict(),
        "output_dir": str(output),
    }
    with open(output / "trial_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Trial complete! Results in: {output}")
    print(f"  - Sample images: {sample_dir}/")
    print(f"  - Loss curve: {output / 'loss_curve.json'}")
    print(f"  - Mitsuba scene: {output / 'scene.xml'}")
    print(f"  - Comparison: {output / 'comparison_input_vs_rendered.png'}")
    print(f"  - Physics: {output / 'physics_estimate.json'}")
    print(f"  - Summary: {output / 'trial_summary.json'}")
    print(f"  - PSNR: {psnr:.2f} dB")
    print(f"{'='*60}")


def _write_point_ply(path: Path, values: dict):
    """Write Gaussians as a point PLY (for mesh-based export fallback)."""
    n = len(values["position"])
    lines = ["ply", "format ascii 1.0",
             f"element vertex {n}",
             "property float x", "property float y", "property float z",
             "property float nx", "property float ny", "property float nz",
             "property uchar red", "property uchar green", "property uchar blue",
             "end_header"]
    albedo = (np.clip(values["albedo"], 0, 1) * 255).astype(np.uint8)
    for i in range(n):
        p = values["position"][i]
        nrm = values["normal"][i]
        c = albedo[i]
        lines.append(f"{p[0]} {p[1]} {p[2]} {nrm[0]} {nrm[1]} {nrm[2]} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
