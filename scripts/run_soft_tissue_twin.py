"""End-to-end soft-tissue digital twin: endoscopic video -> simulable scene.

Three tiers, each saving a small set of decisive intermediate results:

- **Tier 1 (optical-geometric twin)**: canonical Gaussian BRDF field from the
  reference frame (depth-supervised, mask-supervised), with checkpoint
  comparisons so reconstruction quality is auditable, not just final PSNR.
- **Tier 2 (kinematic twin)**: per-frame displacement tracking on the
  canonical topology -> replayable boundary conditions (NPZ) + quiver overlay.
- **Tier 3 (mechanical twin)**:
    a) synthetic closed loop with KNOWN E (uniform + stiff-inclusion "tumor",
       surface-only observations) -> verifies the inversion machinery;
    b) real-data RELATIVE stiffness contrast from Tier-2 deformation (no
       tool-force sensor -> absolute E not identifiable; audit says so);
    c) tissue physics estimate with full provenance (tissue_db prior +
       deformation evidence).

Usage:
    python scripts/run_soft_tissue_twin.py --scene pulling_soft_tissues
    python scripts/run_soft_tissue_twin.py --scene cutting_tissues_twice \
        --frames 24 --steps 800 --track-frames 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvbrdf_shr.world.endonerf_dataset import load_endonerf_scene, _depth_to_points
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.lighting import WorldLight
from mvbrdf_shr.world.renderer import GaussianBRDFRenderer
from mvbrdf_shr.world.deform import (
    track_deformation,
    visualize_deformation,
    save_deformation_summary,
)
from mvbrdf_shr.world.tissue_sim import (
    run_synthetic_inversion,
    invert_relative_field,
)
from mvbrdf_shr.world.tissue_db import (
    classify_tissue_from_pbr,
    get_tissue_prior,
)
from mvbrdf_shr.world.physics_est import estimate_tissue_physics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="pulling_soft_tissues")
    p.add_argument("--datasets-dir", default=str(ROOT.parent / "datasets"))
    p.add_argument("--output", default=str(ROOT / "outputs" / "soft_tissue_twin"))
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--steps", type=int, default=800, help="Tier-1 steps")
    p.add_argument("--track-frames", type=int, default=10, help="Tier-2 frames")
    p.add_argument("--track-steps", type=int, default=60, help="Tier-2 steps/frame")
    p.add_argument("--max-points", type=int, default=8000)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "gsplat"])
    p.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    p.add_argument("--skip-tier3-synthetic", action="store_true")
    return p


def make_endoscopic_light(cam, device) -> WorldLight:
    """Endoscope-co-located illumination for shading the frontal tissue.

    Uses a directional light along the view axis (no 1/r^2 falloff) plus a
    small ambient SH lobe. A true point light at the camera would require
    intensity ~ r^2 (~5000 here) and is numerically noisier for a demo.
    """
    # View axis in world: camera +z mapped through c2w.
    forward = cam.c2w[:3, 2].to(device).clone()  # OpenCV +z forward
    env = torch.zeros(9, 3, device=device)
    env[0] = 1.2  # strong ambient → appearance ≈ albedo under endoscope
    return WorldLight(
        env_sh=env,
        direction=-forward,  # surface-to-light = toward camera
        intensity=torch.tensor([2.0], device=device),
        color=torch.tensor([1.0, 0.95, 0.9], device=device),
        position=cam.center.to(device).clone(),
        point_weight=torch.tensor([0.0], device=device),  # directional
        exposure=torch.tensor([0.0], device=device),
        white_balance=torch.tensor([1.0, 1.0, 1.0], device=device),
    )


def tissue_mask_from_frame(frame) -> torch.Tensor:
    """Binary tissue mask. EndoNeRF ``masks/`` store TOOL masks, not tissue.

    Prefer ``depth > 0`` (tools have no depth). Fall back to ``1 - tool_mask``.
    """
    if frame.depth_gt is not None:
        return (frame.depth_gt > 0.01).float()
    if frame.mask_gt is not None:
        return 1.0 - frame.mask_gt.clamp(0, 1)
    h, w = frame.image.shape[-2:]
    return torch.ones(h, w)


def save_image(tensor_chw_or_hwc, path: Path) -> None:
    if hasattr(tensor_chw_or_hwc, "detach"):
        arr = tensor_chw_or_hwc.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_chw_or_hwc)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr).save(str(path))
    else:
        Image.fromarray(arr).save(str(path))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float(20 * np.log10(1.0 / max(np.sqrt(mse), 1e-6)))


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------


def run_tier1(args, scene, device, out_dir: Path) -> dict:
    print("\n" + "=" * 64)
    print("TIER 1: optical-geometric twin (canonical Gaussian BRDF field)")
    print("=" * 64)
    t1_dir = out_dir / "tier1_reconstruction"
    t1_dir.mkdir(parents=True, exist_ok=True)

    # Denser canonical point cloud from the reference frame's depth.
    ref = scene.frames[0]
    pts, cols = _depth_to_points(ref, max_points=args.max_points)
    pts, cols = pts.to(device), cols.to(device)
    # Scene-adaptive Gaussian scale: median nearest-neighbor spacing.
    with torch.no_grad():
        if pts.shape[0] > 1:
            # subsample for speed
            sample = pts[:: max(1, pts.shape[0] // 500)]
            d = torch.cdist(sample, sample)
            d.fill_diagonal_(float("inf"))
            nn = d.min(dim=1).values.median().item()
            init_scale = max(nn * 0.35, 1e-3)  # sharper than 0.6×nn
        else:
            init_scale = 0.5
    print(f"  canonical points: {pts.shape[0]} (init_scale={init_scale:.4f})")

    field = GaussianBRDFField(pts, cols, initial_scale=init_scale).to(device)
    renderer = GaussianBRDFRenderer(backend=args.backend)
    print(f"  renderer backend: {renderer.backend}")

    cam = ref.camera.to(device)
    with torch.no_grad():
        to_cam = cam.center.to(device).view(1, 3) - field.means
        field.normal_raw.copy_(torch.nn.functional.normalize(to_cam, dim=-1))
        field.metalness_logits.fill_(-5.0)
        field.roughness_logits.fill_(2.0)
        field.material_budget_logits[:, 0] = 5.0
        field.material_budget_logits[:, 1] = -2.0
        field.material_budget_logits[:, 2] = -5.0

    log_exposure = torch.nn.Parameter(torch.tensor(0.5, device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": [field.means], "lr": 1e-4},
            {"params": [field.log_scales], "lr": 1e-3},
            {"params": [field.quaternions], "lr": 1e-3},
            {"params": [field.normal_raw], "lr": 5e-4},
            {"params": [field.opacity_logits], "lr": 5e-3},
            {"params": [field.base_color_logits], "lr": 2e-2},
            {"params": [field.material_budget_logits], "lr": 5e-3},
            {"params": [log_exposure], "lr": 1e-2},
        ]
    )

    target = ref.image.to(device).permute(1, 2, 0)
    depth_gt = ref.depth_gt.to(device) if ref.depth_gt is not None else None
    tissue = tissue_mask_from_frame(ref).to(device)  # NOT the tool mask
    print(f"  tissue coverage: {float(tissue.mean())*100:.1f}% (depth>0)")
    light = make_endoscopic_light(cam, device)

    loss_curve: list[float] = []
    checkpoints = {0, args.steps // 4, args.steps // 2,
                   3 * args.steps // 4, args.steps - 1}
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        out = renderer(field, cam, light=light)
        shaded = out.full[:3].permute(1, 2, 0)
        alb = out.albedo
        albedo = alb[:3].permute(1, 2, 0) if alb.shape[0] >= 3 else alb.permute(1, 2, 0)
        exposure = log_exposure.exp()
        pred = 0.5 * shaded * exposure + 0.5 * albedo * exposure
        if pred.shape[:2] != target.shape[:2]:
            pred = F.interpolate(
                pred.permute(2, 0, 1).unsqueeze(0), size=target.shape[:2],
                mode="bilinear", align_corners=False,
            )[0].permute(1, 2, 0)
        m = tissue.unsqueeze(-1)
        photo_l1 = ((pred - target).abs() * m).sum() / m.sum().clamp_min(1)
        photo_l2 = (((pred - target) ** 2) * m).sum() / m.sum().clamp_min(1)
        loss = photo_l1 + 0.5 * photo_l2
        if depth_gt is not None and out.depth is not None:
            rd = out.depth[0] if out.depth.ndim == 3 else out.depth
            if rd.shape != depth_gt.shape:
                rd = F.interpolate(
                    rd.unsqueeze(0).unsqueeze(0), size=depth_gt.shape,
                    mode="bilinear", align_corners=False,
                )[0, 0]
            valid = tissue > 0.5
            if valid.any():
                loss = loss + 0.2 * F.l1_loss(rd[valid], depth_gt[valid])
        if out.alpha is not None:
            ra = out.alpha[0] if out.alpha.ndim == 3 else out.alpha
            if ra.shape != tissue.shape:
                ra = F.interpolate(
                    ra.unsqueeze(0).unsqueeze(0), size=tissue.shape,
                    mode="bilinear", align_corners=False,
                )[0, 0]
            loss = loss + 0.05 * F.binary_cross_entropy(
                ra.clamp(1e-4, 1 - 1e-4), tissue
            )
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))

        if step in checkpoints:
            with torch.no_grad():
                pred_c = pred.detach().clamp(0, 1)
                mse = ((pred_c - target) ** 2 * m).sum() / m.sum().clamp_min(1)
                cur_psnr = float(
                    20 * torch.log10(1.0 / mse.sqrt().clamp_min(1e-6))
                )
            pred_np = pred_c.cpu().numpy()
            tgt_np = target.cpu().numpy()
            masked = pred_np * tissue.cpu().numpy()[..., None]
            comp = np.concatenate([tgt_np, pred_np, masked], axis=1)
            save_image(comp, t1_dir / f"compare_step{step:05d}.png")
            print(
                f"  step {step:5d}: loss {loss.item():.5f}  "
                f"tissue-PSNR {cur_psnr:.2f} dB  "
                f"exposure {float(exposure.detach()):.2f}"
            )

    with torch.no_grad():
        out = renderer(field, cam, light=light)
        exposure = log_exposure.exp()
        shaded = out.full[:3].permute(1, 2, 0)
        alb = out.albedo
        albedo = alb[:3].permute(1, 2, 0) if alb.shape[0] >= 3 else alb.permute(1, 2, 0)
        pred_np = (0.5 * shaded * exposure + 0.5 * albedo * exposure).clamp(0, 1).cpu().numpy()
    tgt_np = target.cpu().numpy()
    tissue_np = tissue.cpu().numpy() > 0.5
    final_psnr_all = psnr(tgt_np, pred_np)
    final_psnr_tissue = psnr(tgt_np[tissue_np], pred_np[tissue_np])
    elapsed = time.time() - t0
    print(
        f"  Tier-1 done: tissue-PSNR {final_psnr_tissue:.2f} dB  "
        f"all-PSNR {final_psnr_all:.2f} dB  ({elapsed:.1f}s)"
    )

    (t1_dir / "loss_curve.json").write_text(json.dumps(loss_curve))
    save_image(pred_np, t1_dir / "final_render.png")
    save_image(tgt_np, t1_dir / "reference.png")
    save_image(
        np.concatenate([tgt_np, pred_np, pred_np * tissue_np[..., None]], axis=1),
        t1_dir / "final_compare.png",
    )
    if out.depth is not None:
        d = out.depth[0] if out.depth.ndim == 3 else out.depth
        d = d.detach().cpu().numpy()
        d_vis = (d - d.min()) / (d.max() - d.min() + 1e-6)
        save_image(d_vis, t1_dir / "final_depth.png")
    torch.save(
        {"field": field.state_dict(), "log_exposure": float(log_exposure.detach())},
        str(t1_dir / "canonical_field.pt"),
    )

    return {
        "field": field,
        "renderer": renderer,
        "log_exposure": log_exposure.detach(),
        "psnr": final_psnr_tissue,
        "psnr_all": final_psnr_all,
        "n_gaussians": field.n_gaussians,
        "loss_final": loss_curve[-1],
        "elapsed_s": elapsed,
        "dir": str(t1_dir),
    }


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------


def run_tier2(args, scene, tier1: dict, device, out_dir: Path) -> dict:
    print("\n" + "=" * 64)
    print("TIER 2: kinematic twin (deformation tracking -> boundary conditions)")
    print("=" * 64)
    t2_dir = out_dir / "tier2_deformation"
    t2_dir.mkdir(parents=True, exist_ok=True)

    n_track = min(args.track_frames, len(scene) - 1)
    track_frames = scene.frames[1 : 1 + n_track]
    result = track_deformation(
        tier1["field"],
        track_frames,
        tier1["renderer"],
        lambda f: make_endoscopic_light(f.camera.to(device), device),
        device=device,
        steps_per_frame=args.track_steps,
    )
    result.save_npz(t2_dir / "boundary_conditions.npz")
    save_deformation_summary(result, t2_dir / "deformation_summary.json")

    # Quiver overlay on the last tracked frame.
    last = track_frames[-1]
    img_np = last.image.permute(1, 2, 0).numpy()
    visualize_deformation(
        result, img_np, last.camera, t2_dir / "deformation_quiver.png"
    )
    print(f"  saved: boundary_conditions.npz, deformation_summary.json, "
          f"deformation_quiver.png")
    return {"result": result, "dir": str(t2_dir)}


# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------


def _plot_synthetic_result(res, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    ax = axes[0]
    ax.plot(res.loss_curve)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(f"{title} — inversion loss")

    ax = axes[1]
    E_gt = res.E_gt_pa
    E_est = res.E_est_pa
    if E_gt is not None and E_est.shape[0] == E_gt.shape[0] and E_est.shape[0] > 2:
        ax.scatter(E_gt / 1e3, E_est / 1e3, s=8, alpha=0.6)
        lim = [0, max(E_gt.max(), E_est.max()) / 1e3 * 1.1]
        ax.plot(lim, lim, "r--", label="ideal")
        ax.set_xlabel("E ground truth (kPa)")
        ax.set_ylabel("E estimated (kPa)")
        ax.legend()
        err = np.abs(E_est - E_gt).mean() / 1e3
        # Also annotate recovered contrast if bimodal.
        stiff = E_gt > np.median(E_gt) * 1.5
        if stiff.any() and (~stiff).any():
            c_gt = float(E_gt[stiff].mean() / E_gt[~stiff].mean())
            c_est = float(E_est[stiff].mean() / max(E_est[~stiff].mean(), 1.0))
            ax.set_title(f"{title} — E field (MAE {err:.2f} kPa, "
                         f"contrast {c_gt:.1f}x→{c_est:.1f}x)")
        else:
            ax.set_title(f"{title} — per-node E (MAE {err:.2f} kPa)")
    else:
        # Uniform or 2-param: plot the first parameter's convergence.
        curve = np.asarray([np.atleast_1d(c)[0] for c in res.E_curve]) / 1e3
        its = np.arange(len(curve)) * 10
        ax.plot(its, curve, label="estimated")
        if E_gt is not None:
            ax.axhline(E_gt.flat[0] / 1e3, color="r", linestyle="--",
                       label=f"ground truth {E_gt.flat[0]/1e3:.1f} kPa")
        if res.E_curve and len(np.atleast_1d(res.E_curve[-1])) == 2:
            curve2 = np.asarray([c[1] for c in res.E_curve]) / 1e3
            ax.plot(its, curve2, label="inclusion est")
            if E_gt is not None:
                stiff = E_gt > np.median(E_gt) * 1.5
                if stiff.any():
                    ax.axhline(E_gt[stiff].mean() / 1e3, color="orange",
                               linestyle="--", label="inclusion GT")
        ax.set_xlabel("iteration")
        ax.set_ylabel("E (kPa)")
        ax.legend()
        ax.set_title(f"{title} — E convergence")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def run_tier3(args, scene, tier1: dict, tier2: dict, device, out_dir: Path) -> dict:
    print("\n" + "=" * 64)
    print("TIER 3: mechanical twin (elasticity inversion + tissue estimate)")
    print("=" * 64)
    t3_dir = out_dir / "tier3_elasticity"
    t3_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # --- (a) synthetic closed loop: the machinery validation ---------------
    if not args.skip_tier3_synthetic:
        prior = get_tissue_prior("generic_soft_tissue")

        print("  [3a] Experiment A: uniform E, absolute recovery (force known)")
        res_a = run_synthetic_inversion(
            inclusion=False, n_iters=200, device=device, prior=prior
        )
        E_est_a = float(res_a.E_est_pa[0]) / 1e3
        E_gt_a = float(res_a.E_gt_pa[0]) / 1e3
        err_a = abs(E_est_a - E_gt_a) / E_gt_a * 100
        print(f"       E_gt {E_gt_a:.2f} kPa -> E_est {E_est_a:.2f} kPa "
              f"(error {err_a:.1f}%)")
        _plot_synthetic_result(res_a, t3_dir / "synthetic_uniform.png",
                               "Exp A (uniform)")
        summary["synthetic_uniform"] = {
            "E_gt_kpa": E_gt_a, "E_est_kpa": E_est_a,
            "relative_error_pct": err_a,
        }

        print("  [3b] Experiment B: stiff inclusion (tumor), 2-region, surface-only")
        res_b = run_synthetic_inversion(
            inclusion=True, inclusion_E_kpa=32.0, surface_only=True,
            region_based=True, n_iters=200, device=device, prior=prior,
        )
        E_gt_b = res_b.E_gt_pa / 1e3
        E_est_b = res_b.E_est_pa / 1e3
        incl = E_gt_b > 20
        bg = ~incl
        contrast_gt = float(E_gt_b[incl].mean() / E_gt_b[bg].mean())
        contrast_est = float(E_est_b[incl].mean() / max(E_est_b[bg].mean(), 1e-9))
        mae_b = float(np.abs(E_est_b - E_gt_b).mean())
        print(f"       inclusion contrast: GT {contrast_gt:.2f}x -> "
              f"est {contrast_est:.2f}x; field MAE {mae_b:.2f} kPa")
        _plot_synthetic_result(res_b, t3_dir / "synthetic_inclusion.png",
                               "Exp B (inclusion, surface-only)")
        summary["synthetic_inclusion"] = {
            "contrast_gt": contrast_gt, "contrast_est": contrast_est,
            "field_mae_kpa": mae_b,
            "surface_only": True,
        }

    # --- (b) real-data relative stiffness (no force sensor) -----------------
    print("  [3c] Real-data relative stiffness from Tier-2 deformation")
    deform = tier2["result"]
    if deform.displacements.shape[0] >= 1:
        positions = torch.from_numpy(deform.canonical_positions).float().to(device)
        disp = torch.from_numpy(deform.displacements[-1]).float().to(device)
        rel = invert_relative_field(positions, disp, n_regions=4)
        (t3_dir / "relative_stiffness.json").write_text(
            json.dumps(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in rel.items()},
                indent=2,
            )
        )
        print(f"       relative stiffness per region: "
              f"{np.round(rel['relative_stiffness'], 2)}")
        summary["real_relative_stiffness"] = {
            "relative_stiffness": rel["relative_stiffness"].tolist(),
            "audit": rel["audit"],
        }
    else:
        print("       skipped (no tracked deformation)")

    # --- (c) tissue estimate with full provenance ---------------------------
    print("  [3d] Tissue physics estimate (prior + evidence + audit)")
    field = tier1["field"]
    with torch.no_grad():
        mats = field.materials()
        avg_metal = float(mats.metalness.mean())
        avg_rough = float(mats.roughness.mean())
        avg_alb = mats.albedo.mean(dim=0).cpu().numpy()
    tissue_class = classify_tissue_from_pbr(
        avg_metal, avg_rough, (float(avg_alb[0]), float(avg_alb[1]), float(avg_alb[2]))
    )
    tissue_est = estimate_tissue_physics(tissue_class, instance_id=0)
    (t3_dir / "tissue_estimate.json").write_text(
        json.dumps(tissue_est.to_dict(), indent=2)
    )
    lo, hi = tissue_est.youngs_ci95_kpa
    print(f"       class: {tissue_est.tissue_class} "
          f"({tissue_est.constitutive_model})")
    print(f"       E: {tissue_est.youngs_median_pa/1e3:.1f} kPa "
          f"95%CI [{lo:.1f}, {hi:.1f}] ({tissue_est.audit['youngs_modulus']})")
    print(f"       nu: {tissue_est.poisson.mean:.3f}, "
          f"viscosity: {tissue_est.viscosity.mean:.1f} Pa·s, "
          f"rho: {tissue_est.density.mean:.0f} kg/m3")
    summary["tissue_estimate"] = tissue_est.to_dict()
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = build_parser().parse_args()
    # Seed for reproducibility: every reported number derives from a seeded run.
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {output}")
    print(f"Seed: {args.seed}")

    scene_path = Path(args.datasets_dir) / "endonerf" / args.scene
    print(f"\nLoading EndoNeRF scene: {scene_path}")
    scene = load_endonerf_scene(
        scene_path, max_frames=args.frames, image_scale=args.scale
    )
    print(f"  {len(scene)} frames loaded")

    tier1 = run_tier1(args, scene, device, output)
    tier2 = run_tier2(args, scene, tier1, device, output)
    tier3 = run_tier3(args, scene, tier1, tier2, device, output)

    summary = {
        "scene": args.scene,
        "tier1": {
            k: (float(v) if hasattr(v, "item") else v)
            for k, v in tier1.items()
            if k not in {"field", "renderer", "log_exposure"}
        },
        "tier2_dir": tier2["dir"],
        "tier3": tier3,
        "output_dir": str(output),
    }
    (output / "twin_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("\n" + "=" * 64)
    print("Twin complete. Intermediate results:")
    print(f"  Tier 1: {tier1['dir']}  (PSNR {tier1['psnr']:.2f} dB)")
    print(f"  Tier 2: {tier2['dir']}")
    print(f"  Tier 3: {output / 'tier3_elasticity'}")
    print(f"  Summary: {output / 'twin_summary.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
