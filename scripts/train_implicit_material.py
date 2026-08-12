"""Train a bounded implicit material image-formation residual on train views."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mvbrdf_shr.world.train import train


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(campaign: Path, scene: str) -> Path:
    path = campaign / "stanford_orb" / scene / "brdf" / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"strict BRDF checkpoint not found: {path}")
    return path


def _config(
    checkpoint: Path,
    output_dir: Path,
    steps: int,
    hidden: int,
    residual_limit: float,
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(state["config"])
    loss = dict(cfg.get("loss", {}))
    loss.update(
        photo_l1=1.0,
        photo_ssim=0.2,
        implicit_residual=0.05,
        albedo_gt=0.0,
        normal_gt=0.0,
        depth_gt=0.0,
    )
    cfg.update(
        output_dir=str(output_dir),
        init_from=str(checkpoint),
        steps=int(steps),
        auto_resume=False,
        include_novel=False,
        use_dataset_split=True,
        use_refiner=False,
        freeze_geometry=True,
        freeze_normals=True,
        freeze_materials=True,
        optimize_normals=False,
        freeze_lighting=True,
        use_implicit_material=True,
        implicit_material_hidden=int(hidden),
        implicit_material_layers=3,
        implicit_material_residual_limit=float(residual_limit),
        implicit_material_lr=5e-4,
        densify_every=0,
        checkpoint_every=max(steps // 2, 1),
        loss=loss,
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--residual-limit", type=float, default=0.10)
    args = parser.parse_args()
    campaign = (ROOT / args.campaign).resolve()
    output = (ROOT / args.output).resolve()
    for scene in args.scenes:
        checkpoint = _checkpoint(campaign, scene)
        destination = output / "stanford_orb" / scene / "brdf"
        print(f"[{scene}] implicit material <- {checkpoint}", flush=True)
        train(
            _config(
                checkpoint,
                destination,
                args.steps,
                args.hidden,
                args.residual_limit,
            )
        )


if __name__ == "__main__":
    main()

