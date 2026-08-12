"""Supervised synthetic pretraining for the factorized neural BRDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mvbrdf_shr.models.dual_latent_brdf import DualLatentNeuralBRDF
from mvbrdf_shr.world.renderer import _ggx_d, _schlick_fresnel, _smith_g1


def _samples(count: int, device: torch.device) -> dict[str, torch.Tensor]:
    normal = F.normalize(torch.randn(count, 3, device=device), dim=-1)

    def hemisphere() -> torch.Tensor:
        direction = F.normalize(torch.randn(count, 3, device=device), dim=-1)
        sign = torch.where(
            (direction * normal).sum(-1, keepdim=True) < 0, -1.0, 1.0
        )
        return direction * sign

    view, light = hemisphere(), hemisphere()
    base_color = torch.rand(count, 3, device=device) * 0.9 + 0.03
    budget = torch.softmax(torch.randn(count, 3, device=device), dim=-1)
    albedo = base_color * budget[:, :1]
    roughness = torch.rand(count, 1, device=device) * 0.85 + 0.08
    metalness = torch.rand(count, 1, device=device) * 0.95
    material = torch.cat(
        (
            albedo,
            budget[:, :1],
            budget[:, 1:2],
            roughness,
            metalness,
            budget[:, 2:3],
        ),
        -1,
    )
    half = F.normalize(light + view, dim=-1)
    n_dot_l = (normal * light).sum(-1, keepdim=True).clamp_min(0)
    n_dot_v = (normal * view).sum(-1, keepdim=True).clamp_min(1e-4)
    n_dot_h = (normal * half).sum(-1, keepdim=True).clamp_min(0)
    h_dot_v = (half * view).sum(-1, keepdim=True).clamp_min(0)
    alpha = roughness.square()
    f0 = 0.04 * (1 - metalness) + albedo * metalness
    diffuse = albedo / torch.pi
    specular = (
        budget[:, 1:2]
        * _ggx_d(n_dot_h, alpha)
        * _smith_g1(n_dot_l, alpha)
        * _smith_g1(n_dot_v, alpha)
        * _schlick_fresnel(h_dot_v, f0)
        / (4 * n_dot_l * n_dot_v + 1e-6)
    )
    # Smooth, bounded departures from isotropic GGX emulate coating and
    # wavelength-dependent effects while preserving GGX as the anchor.
    chroma = base_color - base_color.mean(-1, keepdim=True)
    target_residual = (
        0.20
        * (1 - roughness)
        * n_dot_h.square()
        * (2 * metalness - 0.5)
        + 0.08 * chroma
    ).clamp(-0.30, 0.30)
    identity = torch.rand(count, 1, device=device) < 0.25
    target_residual = torch.where(
        identity.expand_as(target_residual),
        torch.zeros_like(target_residual),
        target_residual,
    )
    return {
        "position": torch.rand(count, 3, device=device) * 2 - 1,
        "normal": normal,
        "view": view,
        "light": light,
        "material": material,
        "intensity": torch.exp(torch.randn(1, 1, device=device) * 0.7),
        "color": torch.rand(1, 3, device=device) * 0.8 + 0.2,
        "environment": torch.rand(1, 3, device=device),
        "diffuse": diffuse,
        "specular": specular,
        "target_residual": target_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(2026)
    model = DualLatentNeuralBRDF(mode="log_residual", residual_limit=0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    for step in range(args.steps):
        sample = _samples(args.batch_size, device)
        material_latent = model.encode_material(
            sample["position"], sample["material"]
        )
        light_latent = model.encode_light(
            sample["light"],
            sample["intensity"].expand(args.batch_size, -1),
            sample["color"].expand(args.batch_size, -1),
            sample["environment"].expand(args.batch_size, -1),
        )
        residual = model.decode_log_specular_residual(
            material_latent,
            light_latent,
            sample["normal"],
            sample["view"],
            sample["light"],
        )
        residual_loss = F.smooth_l1_loss(
            residual, sample["target_residual"]
        )
        predicted_specular = sample["specular"] * torch.exp(residual)
        rendered_loss = F.smooth_l1_loss(
            torch.log1p(predicted_specular),
            torch.log1p(
                sample["specular"] * torch.exp(sample["target_residual"])
            ),
        )
        # Penalize unnecessary changes where the physical anchor is sufficient.
        identity = sample["target_residual"].abs().amax(-1) == 0
        identity_loss = (
            residual[identity].square().mean()
            if identity.any()
            else residual.new_zeros(())
        )
        loss = residual_loss + rendered_loss + 0.25 * identity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(
                f"[synthetic {step}/{args.steps}] total={loss.item():.6f} "
                f"residual={residual_loss.item():.6f} "
                f"rendered_log={rendered_loss.item():.6f}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": 2026,
            "mode": "log_residual",
            "residual_limit": 0.35,
        },
        args.output,
    )
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

