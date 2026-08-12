"""Material physics database for digital-twin parameter inference.

Each material class stores prior distributions (mean + std) for the
physical parameters a rigid-body simulator needs:

- ``density`` (kg/m³)
- ``youngs_modulus`` (Pa) — informational; not used by rigid-body sim
- ``poisson_ratio`` — informational
- ``friction`` (static ≈ dynamic, unitless coefficient)
- ``restitution`` (unitless, 0–1)

Values are drawn from engineering handbooks (Callister, MatWeb) and the
Autodesk material library. They are PRIORS for Bayesian update, not
calibrated measurements — every value carries an ``assumed_table`` audit
source unless updated by PBR evidence (see :mod:`physics_est`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class MaterialPrior:
    """Gaussian prior over physical parameters for one material class."""

    name: str
    density_mean: float       # kg/m³
    density_std: float
    youngs_modulus_mean: float  # Pa
    youngs_modulus_std: float
    poisson_ratio_mean: float
    poisson_ratio_std: float
    friction_mean: float      # static friction coefficient
    friction_std: float
    restitution_mean: float
    restitution_std: float
    # Heuristic PBR signature used as likelihood evidence.
    expected_metalness: float  # 0 or 1 typical
    expected_roughness: float  # typical surface roughness
    notes: str = ""


# ---------------------------------------------------------------------------
# Database: 8 common material classes
# ---------------------------------------------------------------------------

MATERIAL_DATABASE: dict[str, MaterialPrior] = {
    "wood": MaterialPrior(
        name="wood",
        density_mean=700.0, density_std=150.0,
        youngs_modulus_mean=11e9, youngs_modulus_std=4e9,
        poisson_ratio_mean=0.40, poisson_ratio_std=0.05,
        friction_mean=0.50, friction_std=0.15,
        restitution_mean=0.30, restitution_std=0.10,
        expected_metalness=0.0, expected_roughness=0.7,
        notes="Generic hardwood/softwood blend. Density varies 400–900.",
    ),
    "steel": MaterialPrior(
        name="steel",
        density_mean=7850.0, density_std=200.0,
        youngs_modulus_mean=200e9, youngs_modulus_std=10e9,
        poisson_ratio_mean=0.30, poisson_ratio_std=0.01,
        friction_mean=0.60, friction_std=0.10,
        restitution_mean=0.25, restitution_std=0.05,
        expected_metalness=1.0, expected_roughness=0.3,
        notes="Carbon steel. Stainless has lower friction ~0.4.",
    ),
    "aluminum": MaterialPrior(
        name="aluminum",
        density_mean=2700.0, density_std=100.0,
        youngs_modulus_mean=70e9, youngs_modulus_std=5e9,
        poisson_ratio_mean=0.33, poisson_ratio_std=0.01,
        friction_mean=0.45, friction_std=0.10,
        restitution_mean=0.30, restitution_std=0.05,
        expected_metalness=1.0, expected_roughness=0.25,
        notes="Aluminum alloy 6061. Anodized surfaces rougher.",
    ),
    "copper": MaterialPrior(
        name="copper",
        density_mean=8960.0, density_std=50.0,
        youngs_modulus_mean=110e9, youngs_modulus_std=5e9,
        poisson_ratio_mean=0.34, poisson_ratio_std=0.01,
        friction_mean=0.40, friction_std=0.08,
        restitution_mean=0.25, restitution_std=0.05,
        expected_metalness=1.0, expected_roughness=0.2,
        notes="Pure copper. Distinctive reddish albedo.",
    ),
    "plastic": MaterialPrior(
        name="plastic",
        density_mean=1100.0, density_std=200.0,
        youngs_modulus_mean=3e9, youngs_modulus_std=1.5e9,
        poisson_ratio_mean=0.40, poisson_ratio_std=0.05,
        friction_mean=0.30, friction_std=0.10,
        restitution_mean=0.50, restitution_std=0.10,
        expected_metalness=0.0, expected_roughness=0.4,
        notes="Generic thermoplastic (ABS/PP/PE). Wide variance.",
    ),
    "glass": MaterialPrior(
        name="glass",
        density_mean=2500.0, density_std=100.0,
        youngs_modulus_mean=70e9, youngs_modulus_std=5e9,
        poisson_ratio_mean=0.22, poisson_ratio_std=0.02,
        friction_mean=0.40, friction_std=0.10,
        restitution_mean=0.60, restitution_std=0.10,
        expected_metalness=0.0, expected_roughness=0.05,
        notes="Soda-lime glass. Very low roughness, high restitution.",
    ),
    "ceramic": MaterialPrior(
        name="ceramic",
        density_mean=2300.0, density_std=300.0,
        youngs_modulus_mean=300e9, youngs_modulus_std=50e9,
        poisson_ratio_mean=0.25, poisson_ratio_std=0.03,
        friction_mean=0.40, friction_std=0.10,
        restitution_mean=0.35, restitution_std=0.08,
        expected_metalness=0.0, expected_roughness=0.3,
        notes="Porcelain/stoneware. Brittle; restitution moderate.",
    ),
    "rubber": MaterialPrior(
        name="rubber",
        density_mean=1100.0, density_std=100.0,
        youngs_modulus_mean=0.05e9, youngs_modulus_std=0.03e9,
        poisson_ratio_mean=0.48, poisson_ratio_std=0.01,
        friction_mean=0.80, friction_std=0.10,
        restitution_mean=0.75, restitution_std=0.10,
        expected_metalness=0.0, expected_roughness=0.6,
        notes="Natural rubber. High friction, high restitution, near-incompressible.",
    ),
    "fabric": MaterialPrior(
        name="fabric",
        density_mean=300.0, density_std=150.0,
        youngs_modulus_mean=0.001e9, youngs_modulus_std=0.001e9,
        poisson_ratio_mean=0.30, poisson_ratio_std=0.10,
        friction_mean=0.50, friction_std=0.15,
        restitution_mean=0.10, restitution_std=0.05,
        expected_metalness=0.0, expected_roughness=0.8,
        notes="Cloth/upholstery. Very low density; soft-body in reality.",
    ),
    "stone": MaterialPrior(
        name="stone",
        density_mean=2700.0, density_std=400.0,
        youngs_modulus_mean=50e9, youngs_modulus_std=20e9,
        poisson_ratio_mean=0.25, poisson_ratio_std=0.05,
        friction_mean=0.60, friction_std=0.15,
        restitution_mean=0.20, restitution_std=0.08,
        expected_metalness=0.0, expected_roughness=0.7,
        notes="Granite/marble generic. Brittle, high friction.",
    ),
}


def get_prior(material_class: str) -> MaterialPrior:
    """Look up a material prior by class name.

    Falls back to ``plastic`` (the broadest generic class) for unknown
    names, recording the fallback in notes.
    """
    if material_class in MATERIAL_DATABASE:
        return MATERIAL_DATABASE[material_class]
    return MaterialPrior(
        name=f"unknown({material_class})",
        density_mean=1500.0, density_std=500.0,
        youngs_modulus_mean=10e9, youngs_modulus_std=10e9,
        poisson_ratio_mean=0.35, poisson_ratio_std=0.10,
        friction_mean=0.40, friction_std=0.15,
        restitution_mean=0.30, restitution_std=0.10,
        expected_metalness=0.0, expected_roughness=0.5,
        notes=f"Generic fallback for unknown class '{material_class}'.",
    )


def list_material_classes() -> list[str]:
    """Return all available material class names."""
    return sorted(MATERIAL_DATABASE.keys())


def classify_from_pbr(
    metalness: float, roughness: float, albedo: tuple[float, float, float]
) -> str:
    """Heuristic material classification from PBR optical parameters.

    This is a simple rule-based classifier used when no semantic label is
    available. It maps metalness/roughness/albedo to the closest material
    class in the database. Used as the prior source in
    :func:`physics_est.estimate_physics` when CLIP labels are absent.
    """
    if metalness > 0.7:
        # Distinguish copper by reddish albedo (r high, r > b, r > g).
        r, g, b = albedo
        if r > 0.7 and r > b + 0.15 and r > g:
            return "copper"
        if roughness < 0.2:
            return "aluminum"
        return "steel"
    if roughness < 0.1 and metalness < 0.1:
        return "glass"
    if roughness > 0.7 and metalness < 0.1:
        # Could be wood, fabric, or stone; distinguish by density proxy (albedo uniformity).
        r, g, b = albedo
        uniformity = 1.0 - (max(r, g, b) - min(r, g, b))
        if uniformity < 0.7:
            return "wood"
        return "fabric" if (r + g + b) / 3 < 0.3 else "stone"
    if metalness < 0.1 and 0.3 < roughness < 0.6:
        return "plastic"
    return "plastic"  # generic fallback
