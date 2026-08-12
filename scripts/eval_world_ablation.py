"""Systematic world/fusion ablations on scene_0070 strict holdout.

Uses existing checkpoints where possible; reports physical / refined / fusion
variants without fabricating numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.train import load_dataset


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.clamp(0, 1) - target.clamp(0, 1)).square().mean().clamp(min=1e-12)
    return float((-10 * torch.log10(mse)).item())


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float().clamp(0, 1).unsqueeze(0)
    target = target.float().clamp(0, 1).unsqueeze(0)
    pred = F.pad(pred, (5, 5, 5, 5), mode="reflect")
    target = F.pad(target, (5, 5, 5, 5), mode="reflect")
    mp = F.avg_pool2d(pred, 11, 1)
    mt = F.avg_pool2d(target, 11, 1)
    vp = F.avg_pool2d(pred.square(), 11, 1) - mp.square()
    vt = F.avg_pool2d(target.square(), 11, 1) - mt.square()
    cov = F.avg_pool2d(pred * target, 11, 1) - mp * mt
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp.square() + mt.square() + c1) * (vp + vt + c2) + 1e-12
    )
    return float(score.mean().item())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-checkpoint",
        default="outputs/world_benchmark_v2_scene_0070_strict/last.pt",
    )
    parser.add_argument(
        "--single-checkpoint",
        default="outputs/world_benchmark_single/best.pt",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        default="outputs/world_benchmark_v2_scene_0070_hybrid/fusion.pt",
    )
    parser.add_argument("--output", default="outputs/world_ablation_scene_0070")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.world_checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    holdout = state.get("holdout_ids", [0, 4, 8])
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device),
        state["colors"].to(device) if state.get("colors") is not None else None,
    )
    pipe = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=True,
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
    ).to(device)
    pipe.load_state_dict(state["model"], strict=False)
    pipe.eval()

    pro = MVBRDFSHRPro(128, 48, 96).to(device)
    pstate = torch.load(args.single_checkpoint, map_location=device, weights_only=False)
    pro.load_state_dict(pstate.get("model", pstate), strict=False)
    pro.eval()

    # Hybrid owns its own Pro backbone; must load the same single checkpoint
    # used during fusion training (not a randomly initialized single).
    hybrid = HybridMVBRDFSHR(resolution=128).to(device)
    hybrid.load_single_checkpoint(args.single_checkpoint)
    if Path(args.fusion_checkpoint).exists():
        fstate = torch.load(args.fusion_checkpoint, map_location=device, weights_only=False)
        # fusion.pt stores gate/correction modules, not a full model state_dict.
        if "gate" in fstate and "correction" in fstate:
            hybrid.gate.load_state_dict(fstate["gate"])
            hybrid.correction.load_state_dict(fstate["correction"])
        else:
            hybrid.load_state_dict(fstate.get("model", fstate), strict=False)
    hybrid.eval()

    rows = {
        "ours_phys": [],
        "ours_refined": [],
        "ours_pro": [],
        "fusion_full": [],
        "fusion_analytic": [],
        "fusion_no_correction": [],
        "input": [],
    }

    for fid in holdout:
        frame = dataset[fid]
        image = frame.image.to(device)
        gt = frame.diffuse_gt.to(device)
        out = pipe(frame.camera.to(device), fid, image=image, refine=True)
        phys = out["render"].diffuse
        refined = out["pred"]
        x = F.interpolate(image.unsqueeze(0), (128, 128), mode="bilinear", align_corners=False)
        pro_pred = pro(x, use_refine=True, detach_physics=True)["pred"]
        pro_pred = F.interpolate(pro_pred, size=image.shape[-2:], mode="bilinear", align_corners=False)[0]

        fout = hybrid(image.unsqueeze(0), out, detach_backbones=True)
        full = fout["pred"][0]
        # Analytic-only gate: zero the learned residual by using sigmoid(analytic_logit)
        world_d = out["render"].diffuse.unsqueeze(0)
        world_s = out["render"].specular.unsqueeze(0)
        alpha = out["render"].alpha.unsqueeze(0)
        err = (image.unsqueeze(0) - (world_d + world_s).clamp(0, 1)).abs().mean(1, keepdim=True)
        c = (alpha * torch.exp(-err / hybrid.reconstruction_temperature)).clamp(1e-4, 1 - 1e-4)
        g_an = c  # already in (0,1)
        analytic = (g_an * world_d + (1 - g_an) * pro_pred.unsqueeze(0))[0]
        # Learned gate, no correction
        no_corr = fout["fused"][0]

        rows["ours_phys"].append((psnr(phys, gt), ssim(phys, gt)))
        rows["ours_refined"].append((psnr(refined, gt), ssim(refined, gt)))
        rows["ours_pro"].append((psnr(pro_pred, gt), ssim(pro_pred, gt)))
        rows["fusion_full"].append((psnr(full, gt), ssim(full, gt)))
        rows["fusion_analytic"].append((psnr(analytic, gt), ssim(analytic, gt)))
        rows["fusion_no_correction"].append((psnr(no_corr, gt), ssim(no_corr, gt)))
        rows["input"].append((psnr(image, gt), ssim(image, gt)))

    # Pull inverse numbers from existing JSON if present.
    inv = Path("outputs/world_benchmark_v2_scene_0070_inverse/world_metrics_holdout.json")
    invp = Path("outputs/world_benchmark_v2_scene_0070_inverse_pro/world_metrics_holdout.json")
    summary = {}
    for key, vals in rows.items():
        summary[key] = {
            "psnr": float(np.mean([v[0] for v in vals])),
            "ssim": float(np.mean([v[1] for v in vals])),
            "per_view": [
                {"frame": holdout[i], "psnr": vals[i][0], "ssim": vals[i][1]}
                for i in range(len(vals))
            ],
        }
    if inv.exists():
        d = json.loads(inv.read_text())
        summary["inverse"] = {
            "psnr": d["summary"]["diffuse_psnr"],
            "ssim": d["summary"]["diffuse_ssim"],
            "source": str(inv),
        }
    if invp.exists():
        d = json.loads(invp.read_text())
        summary["inverse_pro"] = {
            "psnr": d["summary"]["diffuse_psnr"],
            "ssim": d["summary"]["diffuse_ssim"],
            "source": str(invp),
        }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_metrics.json").write_text(
        json.dumps(
            {
                "protocol": "scene_0070 strict holdout; local checkpoints only",
                "holdout_ids": holdout,
                "variants": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
