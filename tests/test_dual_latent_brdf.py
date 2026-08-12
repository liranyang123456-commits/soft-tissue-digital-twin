from __future__ import annotations

import torch

from mvbrdf_shr.models.dual_latent_brdf import DualLatentNeuralBRDF


def _inputs(count: int = 32) -> dict[str, torch.Tensor]:
    return {
        "position": torch.rand(count, 3) * 2 - 1,
        "normal": torch.nn.functional.normalize(torch.randn(count, 3), dim=-1),
        "view": torch.nn.functional.normalize(torch.randn(count, 3), dim=-1),
        "light": torch.nn.functional.normalize(torch.randn(count, 3), dim=-1),
        "material": torch.rand(count, 8),
        "intensity": torch.ones(1, 1),
        "color": torch.ones(1, 3),
        "environment": torch.full((1, 3), 0.2),
    }


def test_dual_latent_brdf_is_positive_differentiable_and_factorized():
    model = DualLatentNeuralBRDF(material_latent=16, light_latent=8, hidden=32)
    values = _inputs()
    diffuse, specular, latent = model(
        values["position"],
        values["normal"],
        values["view"],
        values["light"],
        values["material"],
        values["intensity"],
        values["color"],
        values["environment"],
    )
    assert diffuse.shape == specular.shape == (32, 3)
    assert (diffuse >= 0).all() and (specular >= 0).all()
    assert latent["material"].shape == (32, 16)
    assert latent["light"].shape == (32, 8)
    # A capture has one shared illumination latent across all surface points.
    assert torch.allclose(latent["light"], latent["light"][:1].expand_as(latent["light"]))
    (diffuse.mean() + specular.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_material_latent_does_not_depend_on_view_or_light():
    model = DualLatentNeuralBRDF(material_latent=16, light_latent=8, hidden=32)
    values = _inputs(8)
    first = model.encode_material(values["position"], values["material"])
    values["view"].normal_()
    values["light"].normal_()
    second = model.encode_material(values["position"], values["material"])
    assert torch.equal(first, second)


def test_log_residual_mode_is_identity_at_initialization_and_bounded():
    model = DualLatentNeuralBRDF(
        material_latent=16,
        light_latent=8,
        hidden=32,
        mode="log_residual",
        residual_limit=0.25,
    )
    values = _inputs(16)
    material = model.encode_material(values["position"], values["material"])
    light = model.encode_light(
        values["light"],
        values["intensity"].expand(16, -1),
        values["color"].expand(16, -1),
        values["environment"].expand(16, -1),
    )

    residual = model.decode_log_specular_residual(
        material,
        light,
        values["normal"],
        values["view"],
        values["light"],
    )

    assert torch.equal(residual, torch.zeros_like(residual))
    with torch.no_grad():
        model.decoder[-1].bias.fill_(100)
    saturated = model.decode_log_specular_residual(
        material,
        light,
        values["normal"],
        values["view"],
        values["light"],
    )
    assert saturated.abs().max() <= 0.25


def test_directional_illumination_samples_have_distinct_latents():
    model = DualLatentNeuralBRDF(
        material_latent=16,
        light_latent=8,
        hidden=32,
        mode="log_residual",
    )
    values = _inputs(8)
    latents = model.encode_light(
        values["light"],
        torch.ones(8, 1),
        torch.ones(8, 3),
        torch.full((8, 3), 0.2),
    )

    assert latents.var(dim=0).sum() > 0

