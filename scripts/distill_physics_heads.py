"""Pre-train the single-image physics heads on world oracle GT (path E).

Trains ScreenGeometryField (image->normal) and MaterialField (image->albedo/
material) using the world_benchmark oracle normal/albedo GT, so the decomposition
prior fed to the restorer is more accurate than a from-scratch random init.

The resulting physics checkpoint can be loaded by train_shiq_pro.py via --resume
so SHIQ fine-tuning starts from a physics-informed init rather than random.

NOTE: world GT is from the method's own renderer (gt_provenance). This only
improves the prior; the final SHR number is measured independently on SHIQ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvbrdf_shr.data.world_distill import WorldDistillDataset, collate_distill
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = ROOT / "data" / "world_benchmark_v2"


def cosine_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, the standard normal/angle loss."""
    return (1.0 - F.cosine_similarity(pred, gt, dim=1)).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-root", default=str(DEFAULT_WORLD))
    ap.add_argument("--resolution", type=int, default=200,
                    help="Match the SHIQ training resolution so the transferred "
                         "heads see the same input scale.")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "physics_distilled.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = WorldDistillDataset(args.world_root, args.resolution)
    print(f"world distill samples: {len(ds)} (resolution {args.resolution})")
    assert len(ds) > 0, f"no world distill samples at {args.world_root}"
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_distill, drop_last=True, num_workers=0,
    )

    # Build the full pipeline so the physics heads match the production
    # architecture exactly; we only optimize geometry + material here.
    model = MVBRDFSHRPro(resolution=args.resolution, base_ch=48, restorer_base=96).to(device)
    phys_params = (
        list(model.geometry.parameters()) + list(model.material.parameters())
    )
    opt = torch.optim.AdamW(phys_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best_loss = float("inf")
    pbar = tqdm(total=args.steps, desc="distill-physics")
    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            image = batch["image"].to(device)
            normal_gt = batch["normal_gt"].to(device)
            albedo_gt = batch["albedo_gt"].to(device)
            geom = model.geometry(image)
            mat = model.material(image, geom.normal)
            l_normal = cosine_loss(geom.normal, normal_gt)
            l_albedo = F.l1_loss(mat.albedo, albedo_gt)
            # Encourage albedo to also match in gradients (texture fidelity).
            l_albedo_grad = (
                (mat.albedo[:, :, :, 1:] - mat.albedo[:, :, :, :-1]).abs()
                - (albedo_gt[:, :, :, 1:] - albedo_gt[:, :, :, :-1]).abs()
            ).abs().mean()
            loss = l_normal + 2.0 * l_albedo + 0.5 * l_albedo_grad
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            pbar.set_postfix(
                n=f"{float(l_normal):.3f}", alb=f"{float(l_albedo):.3f}",
                tot=f"{float(loss):.3f}",
            )
            pbar.update(1)
            if step % 200 == 0 and float(loss) < best_loss:
                best_loss = float(loss)
                # Save the FULL pipeline state so train_shiq_pro.py can resume
                # it (it expects {"model": state_dict}).
                torch.save(
                    {"model": model.state_dict(), "step": step,
                     "loss": float(loss), "cfg": vars(args)},
                    args.output,
                )
            step += 1
    pbar.close()
    # Final save.
    torch.save(
        {"model": model.state_dict(), "step": step,
         "loss": float(loss), "cfg": vars(args)},
        args.output,
    )
    print(f"physics heads distilled -> {args.output} (best_loss={best_loss:.4f})")
    print(f"Use with: train_shiq_pro.py --resume {args.output}")


if __name__ == "__main__":
    main()
