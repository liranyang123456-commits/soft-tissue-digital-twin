"""Two-stage train-only adaptation of the dual-latent neural BRDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mvbrdf_shr.world.train import train


ROOT = Path(__file__).resolve().parents[1]


def _strict_checkpoint(campaign: Path, scene: str) -> Path:
    path = campaign / "stanford_orb" / scene / "brdf" / "last.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _base_config(
    checkpoint: Path,
    output_dir: Path,
    pretrained: Path,
    steps: int,
    strength: float,
    residual_reg: float,
    neural_lr: float,
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(state["config"])
    loss = dict(cfg.get("loss", {}))
    loss.update(
        photo_l1=1.0,
        photo_ssim=0.30,
        photo_log=0.15,
        photo_highlight=0.25,
        albedo_gt=0.0,
        normal_gt=0.0,
        depth_gt=0.0,
        material_smooth=0.0,
    )
    cfg.update(
        output_dir=str(output_dir),
        init_from=str(checkpoint),
        steps=int(steps),
        auto_resume=False,
        include_novel=False,
        use_dataset_split=True,
        views_per_step=4,
        view_sampling="distinct_views",
        use_refiner=False,
        freeze_geometry=True,
        freeze_normals=True,
        freeze_materials=True,
        optimize_normals=False,
        freeze_lighting=True,
        use_implicit_material=False,
        use_neural_brdf=True,
        neural_brdf_pretrained=str(pretrained),
        neural_brdf_mode="log_residual",
        neural_brdf_residual_limit=0.35,
        neural_brdf_highlight_gate=True,
        neural_brdf_boundary_fallback=True,
        neural_brdf_strength=float(strength),
        neural_brdf_residual_reg=float(residual_reg),
        neural_brdf_lr=float(neural_lr),
        densify_every=0,
        checkpoint_every=max(steps // 2, 1),
        loss=loss,
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--light-steps", type=int, default=200)
    parser.add_argument("--material-steps", type=int, default=500)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--residual-reg", type=float, default=1e-3)
    parser.add_argument("--neural-lr", type=float, default=1e-4)
    args = parser.parse_args()
    campaign = (ROOT / args.campaign).resolve()
    pretrained = (ROOT / args.pretrained).resolve()
    output = (ROOT / args.output).resolve()
    for scene in args.scenes:
        source = _strict_checkpoint(campaign, scene)
        illumination_dir = output / "stanford_orb" / scene / "illumination"
        illumination_cfg = _base_config(
            source,
            illumination_dir,
            pretrained,
            args.light_steps,
            args.strength,
            args.residual_reg,
            args.neural_lr,
        )
        illumination_cfg.update(
            freeze_neural_brdf_material_encoder=True,
            freeze_neural_brdf_light_encoder=False,
            freeze_neural_brdf_decoder=True,
        )
        print(f"[{scene}] stage 1/2 illumination latent", flush=True)
        illumination_checkpoint = train(illumination_cfg)

        material_dir = output / "stanford_orb" / scene / "brdf"
        material_cfg = _base_config(
            illumination_checkpoint,
            material_dir,
            pretrained,
            args.material_steps,
            args.strength,
            args.residual_reg,
            args.neural_lr,
        )
        material_cfg.update(
            freeze_neural_brdf_material_encoder=False,
            freeze_neural_brdf_light_encoder=True,
            freeze_neural_brdf_decoder=False,
        )
        print(f"[{scene}] stage 2/2 shared surface material latent", flush=True)
        train(material_cfg)


if __name__ == "__main__":
    main()

