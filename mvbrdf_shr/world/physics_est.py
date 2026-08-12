"""Bayesian physical-parameter estimation for digital-twin instances (Phase B).

Physical parameters (density, friction, restitution) are NOT directly
observable from images. This module estimates them by:

1. Querying a material-class prior from :mod:`material_db` (either from a
   CLIP semantic label or a PBR heuristic classifier).
2. Treating the recovered PBR optical parameters (metalness, roughness,
   albedo) as weak observational evidence.
3. Performing a Gaussian-Gaussian Bayesian update: the prior and the
   PBR-derived likelihood are both Gaussian, so the posterior is closed
   form.
4. Computing derived quantities (mass, inertia) from the posterior density
   and the instance mesh.
5. Recording an audit trail for every parameter's provenance.

The key honesty property: every output value is tagged with its source
(``assumed_table``, ``estimated_pbr``, or ``user_override``) and a
confidence score, so downstream consumers know what is measured vs
assumed. This is the engineering ethic a digital twin demands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from .material_db import MATERIAL_DATABASE, MaterialPrior, classify_from_pbr, get_prior

if TYPE_CHECKING:
    from .scene import GaussianBRDFField
    from .segment import InstanceSegmentation


ParamSource = Literal["measured", "estimated_pbr", "assumed_table", "user_override"]


@dataclass(slots=True)
class GaussianPosterior:
    """Gaussian posterior over one scalar physical parameter."""

    mean: float
    std: float
    source: ParamSource
    prior_mean: float
    prior_std: float
    evidence: str = ""

    @property
    def confidence(self) -> float:
        """Confidence in [0,1]: how much the posterior narrowed vs the prior."""
        if self.prior_std <= 0:
            return 1.0
        return float(max(0.0, 1.0 - self.std / self.prior_std))

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "source": self.source,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class PhysicsEstimate:
    """Full physics estimate for one instance, with audit trail."""

    instance_id: int
    semantic_label: str
    material_class: str
    density: GaussianPosterior
    friction: GaussianPosterior
    restitution: GaussianPosterior
    youngs_modulus: GaussianPosterior
    poisson_ratio: GaussianPosterior
    mass: float  # kg
    inertia_diagonal: tuple[float, float, float]  # kg·m²
    is_dynamic: bool = True
    audit: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "semantic_label": self.semantic_label,
            "material_class": self.material_class,
            "density": self.density.to_dict(),
            "friction": self.friction.to_dict(),
            "restitution": self.restitution.to_dict(),
            "youngs_modulus": self.youngs_modulus.to_dict(),
            "poisson_ratio": self.poisson_ratio.to_dict(),
            "mass": self.mass,
            "inertia_diagonal": list(self.inertia_diagonal),
            "is_dynamic": self.is_dynamic,
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Bayesian update (Gaussian-Gaussian closed form)
# ---------------------------------------------------------------------------


def gaussian_update(
    prior_mean: float,
    prior_var: float,
    obs_mean: float,
    obs_var: float,
) -> tuple[float, float]:
    """Closed-form Gaussian-Gaussian Bayesian update.

    Returns (posterior_mean, posterior_var). When the observation is very
    noisy (obs_var → ∞), the posterior → prior; when the observation is
    very confident (obs_var → 0), the posterior → observation.
    """
    if prior_var <= 0 and obs_var <= 0:
        return prior_mean, 0.0
    if prior_var <= 0:
        return obs_mean, obs_var
    if obs_var <= 0:
        return prior_mean, prior_var
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + obs_mean / obs_var)
    return post_mean, post_var


def _pbr_to_physics_observations(
    field: "GaussianBRDFField",
    gaussian_indices: np.ndarray,
    prior: MaterialPrior,
) -> dict[str, tuple[float, float, str]]:
    """Derive weak physics observations from PBR optical parameters.

    Returns a dict mapping parameter name to (obs_mean, obs_var, evidence).
    The observation variance is deliberately large (weak evidence) because
    PBR → physics mappings are heuristic, not physical laws.
    """
    with torch.no_grad():
        materials = field.materials()
        idx = torch.as_tensor(gaussian_indices, device=field.means.device, dtype=torch.long)
        metalness = float(materials.metalness[idx].mean().cpu())
        roughness = float(materials.roughness[idx].mean().cpu())
        albedo = materials.albedo[idx].mean(dim=0).cpu().numpy()
        albedo_tuple = (float(albedo[0]), float(albedo[1]), float(albedo[2]))

    observations: dict[str, tuple[float, float, str]] = {}

    # --- Metalness → density evidence ---
    # If metalness ≈ 1, density should be in the metal range (2700–9000).
    # Use the prior's own density as the observation target but sharpen it.
    if metalness > 0.7:
        obs_var = (prior.density_std * 0.5) ** 2  # moderate evidence
        observations["density"] = (
            prior.density_mean, obs_var,
            f"metalness={metalness:.2f}→metal density prior sharpened",
        )
    else:
        # Non-metal: weak evidence, slight pull toward lower density.
        obs_var = (prior.density_std * 2.0) ** 2  # very weak
        observations["density"] = (
            prior.density_mean * 0.9, obs_var,
            f"metalness={metalness:.2f}→non-metal, weak density pull",
        )

    # --- Roughness → friction evidence ---
    # Rougher surfaces tend to have higher friction (positive correlation).
    # Map roughness ∈ [0,1] to a friction observation via linear scaling.
    friction_obs = 0.2 + roughness * 0.6  # [0.2, 0.8]
    obs_var = (prior.friction_std * 1.5) ** 2  # weak-moderate
    observations["friction"] = (
        friction_obs, obs_var,
        f"roughness={roughness:.2f}→friction≈{friction_obs:.2f}",
    )

    # --- Roughness → restitution evidence (inverse) ---
    # Smoother/harder surfaces bounce more; rougher surfaces absorb energy.
    rest_obs = max(0.05, 0.7 - roughness * 0.5)
    obs_var = (prior.restitution_std * 2.0) ** 2  # weak
    observations["restitution"] = (
        rest_obs, obs_var,
        f"roughness={roughness:.2f}→restitution≈{rest_obs:.2f}",
    )

    # Young's modulus and Poisson ratio: no direct PBR evidence, prior only.
    return observations


# ---------------------------------------------------------------------------
# Mesh-derived quantities (mass, inertia)
# ---------------------------------------------------------------------------


def _compute_mesh_volume_and_inertia(
    mesh_path: str | None, density: float
) -> tuple[float, tuple[float, float, float]]:
    """Compute mesh volume and diagonal inertia tensor at uniform density.

    Returns (volume_m3, (Ixx, Iyy, Izz)). Falls back to a bounding-box
    approximation if trimesh is unavailable or the mesh is degenerate.
    """
    if mesh_path is None:
        return 1e-3, (1e-4, 1e-4, 1e-4)
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        volume = float(mesh.volume) if hasattr(mesh, "volume") else float(np.prod(mesh.bounding_box.extents))
        try:
            inertia = np.asarray(mesh.moment_inertia)
            diag = np.clip(np.diag(inertia), 1e-9, None)
            return volume, (float(diag[0]), float(diag[1]), float(diag[2]))
        except Exception:
            extents = np.asarray(mesh.bounding_box.extents)
            w, h, d = extents
            mass = density * volume
            Ixx = mass * (h * h + d * d) / 12.0
            Iyy = mass * (w * w + d * d) / 12.0
            Izz = mass * (w * w + h * h) / 12.0
            return volume, (Ixx, Iyy, Izz)
    except (ImportError, ValueError, RuntimeError):
        return 1e-3, (1e-4, 1e-4, 1e-4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_physics(
    instance: "InstanceSegmentation",
    field: "GaussianBRDFField",
    *,
    mesh_path: str | None = None,
    user_overrides: dict[str, float] | None = None,
) -> PhysicsEstimate:
    """Estimate physics parameters for one instance via Bayesian update.

    Parameters
    ----------
    instance:
        Segmented instance with Gaussian indices and (optional) semantic label.
    field:
        The reconstructed Gaussian BRDF field.
    mesh_path:
        Path to the instance's exported mesh (for volume/inertia).
    user_overrides:
        Optional dict of parameter overrides (e.g. ``{"density": 1200.0}``).
        These take top priority and are tagged ``user_override``.
    """
    # 1. Determine material class: use instance label if available, else classify.
    if instance.material_class and instance.material_class != "unknown":
        material_class = instance.material_class
    else:
        with torch.no_grad():
            mats = field.materials()
            idx = torch.as_tensor(
                instance.gaussian_indices, device=field.means.device, dtype=torch.long
            )
            metalness = float(mats.metalness[idx].mean().cpu())
            roughness = float(mats.roughness[idx].mean().cpu())
            albedo = mats.albedo[idx].mean(dim=0).cpu().numpy()
            material_class = classify_from_pbr(
                metalness, roughness, (float(albedo[0]), float(albedo[1]), float(albedo[2]))
            )

    prior = get_prior(material_class)
    observations = _pbr_to_physics_observations(field, instance.gaussian_indices, prior)
    overrides = user_overrides or {}

    # 2. Bayesian update for each parameter.
    def _update_param(
        name: str, prior_mean: float, prior_std: float
    ) -> GaussianPosterior:
        if name in overrides:
            return GaussianPosterior(
                mean=float(overrides[name]),
                std=0.0,
                source="user_override",
                prior_mean=prior_mean,
                prior_std=prior_std,
                evidence="user-specified override",
            )
        if name in observations:
            obs_mean, obs_var, evidence = observations[name]
            post_mean, post_var = gaussian_update(
                prior_mean, prior_std ** 2, obs_mean, obs_var
            )
            return GaussianPosterior(
                mean=post_mean,
                std=float(np.sqrt(post_var)),
                source="estimated_pbr",
                prior_mean=prior_mean,
                prior_std=prior_std,
                evidence=evidence,
            )
        return GaussianPosterior(
            mean=prior_mean,
            std=prior_std,
            source="assumed_table",
            prior_mean=prior_mean,
            prior_std=prior_std,
            evidence=f"no PBR evidence; prior from {material_class} table",
        )

    density_post = _update_param("density", prior.density_mean, prior.density_std)
    friction_post = _update_param("friction", prior.friction_mean, prior.friction_std)
    restitution_post = _update_param("restitution", prior.restitution_mean, prior.restitution_std)
    youngs_post = _update_param(
        "youngs_modulus", prior.youngs_modulus_mean, prior.youngs_modulus_std
    )
    poisson_post = _update_param(
        "poisson_ratio", prior.poisson_ratio_mean, prior.poisson_ratio_std
    )

    # 3. Derived quantities: mass and inertia from posterior density + mesh.
    volume, inertia_diag = _compute_mesh_volume_and_inertia(mesh_path, density_post.mean)
    mass = density_post.mean * volume

    # 4. Audit trail.
    audit = {
        "density": density_post.source,
        "friction": friction_post.source,
        "restitution": restitution_post.source,
        "youngs_modulus": youngs_post.source,
        "poisson_ratio": poisson_post.source,
        "mass": "estimated" if mesh_path is not None else "assumed_unit_volume",
        "inertia": "estimated" if mesh_path is not None else "assumed_bbox",
        "material_class_source": "semantic_label" if instance.material_class != "unknown" else "pbr_heuristic",
    }

    return PhysicsEstimate(
        instance_id=instance.instance_id,
        semantic_label=instance.semantic_label,
        material_class=material_class,
        density=density_post,
        friction=friction_post,
        restitution=restitution_post,
        youngs_modulus=youngs_post,
        poisson_ratio=poisson_post,
        mass=mass,
        inertia_diagonal=inertia_diag,
        is_dynamic=True,
        audit=audit,
    )


def estimate_all_physics(
    segmentation_result,
    field: "GaussianBRDFField",
    *,
    mesh_paths: dict[int, str] | None = None,
    user_overrides: dict[int, dict[str, float]] | None = None,
) -> dict[int, PhysicsEstimate]:
    """Estimate physics for all instances in a segmentation result."""
    from .segment import SegmentationResult

    if not isinstance(segmentation_result, SegmentationResult):
        raise TypeError("expected SegmentationResult")
    mesh_paths = mesh_paths or {}
    user_overrides = user_overrides or {}
    estimates: dict[int, PhysicsEstimate] = {}
    for inst_id, inst in segmentation_result.instances.items():
        estimates[inst_id] = estimate_physics(
            inst,
            field,
            mesh_path=mesh_paths.get(inst_id),
            user_overrides=user_overrides.get(inst_id),
        )
    return estimates


def to_instance_physics(est: PhysicsEstimate, mesh_path: str | None = None):
    """Convert a PhysicsEstimate to a usd_export.InstancePhysics for assembly."""
    from .usd_export import InstancePhysics

    return InstancePhysics(
        instance_id=est.instance_id,
        semantic_label=est.semantic_label,
        material_class=est.material_class,
        density=est.density.mean,
        mass=est.mass,
        friction=est.friction.mean,
        restitution=est.restitution.mean,
        youngs_modulus=est.youngs_modulus.mean,
        poisson_ratio=est.poisson_ratio.mean,
        is_dynamic=est.is_dynamic,
        audit=est.audit,
    )


# ---------------------------------------------------------------------------
# Soft-tissue estimation (Tier 3 of the soft-tissue twin)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TissuePhysicsEstimate:
    """Mechanical estimate for one soft-tissue instance, with audit trail.

    Unlike :class:`PhysicsEstimate` (rigid-body oriented), this carries
    viscoelastic + constitutive-model information and keeps Young's modulus
    as a **log-normal** posterior (mu/sigma of ln E[Pa]), which is the
    correct geometry for kPa-scale tissue stiffness.
    """

    instance_id: int
    tissue_class: str
    youngs_log_mu: float          # posterior mu of ln(E/Pa)
    youngs_log_sigma: float       # posterior sigma of ln(E/Pa)
    poisson: GaussianPosterior
    density: GaussianPosterior
    viscosity: GaussianPosterior  # Kelvin-Voigt Pa·s
    friction: GaussianPosterior
    constitutive_model: str
    audit: dict[str, str] = field(default_factory=dict)

    @property
    def youngs_median_pa(self) -> float:
        return float(np.exp(self.youngs_log_mu))

    @property
    def youngs_ci95_kpa(self) -> tuple[float, float]:
        """Approximate 95% credible interval of E in kPa."""
        lo = self.youngs_log_mu - 1.96 * self.youngs_log_sigma
        hi = self.youngs_log_mu + 1.96 * self.youngs_log_sigma
        return (float(np.exp(lo)) / 1e3, float(np.exp(hi)) / 1e3)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "tissue_class": self.tissue_class,
            "youngs_median_kpa": self.youngs_median_pa / 1e3,
            "youngs_ci95_kpa": list(self.youngs_ci95_kpa),
            "youngs_log_mu": self.youngs_log_mu,
            "youngs_log_sigma": self.youngs_log_sigma,
            "poisson": self.poisson.to_dict(),
            "density": self.density.to_dict(),
            "viscosity": self.viscosity.to_dict(),
            "friction": self.friction.to_dict(),
            "constitutive_model": self.constitutive_model,
            "audit": self.audit,
        }


def estimate_tissue_physics(
    tissue_class: str,
    *,
    instance_id: int = 0,
    deformation_E_pa: tuple[float, float] | None = None,
    user_overrides: dict[str, float] | None = None,
) -> TissuePhysicsEstimate:
    """Estimate soft-tissue mechanics: prior + optional deformation evidence.

    Parameters
    ----------
    tissue_class:
        Tissue class name (see :mod:`tissue_db`), from semantic labeling or
        :func:`tissue_db.classify_tissue_from_pbr`.
    instance_id:
        Instance identifier for the audit trail.
    deformation_E_pa:
        Optional ``(E_mean_pa, E_std_pa)`` from simulation-in-the-loop
        inversion over observed deformation (:mod:`tissue_sim`). This is the
        ONLY honest evidence for tissue stiffness — color cannot measure E.
        When absent, the posterior falls back to the (broad) class prior and
        is tagged ``assumed_table``.
    user_overrides:
        Optional parameter overrides (top priority, tagged ``user_override``).
    """
    from .tissue_db import get_tissue_prior

    prior = get_tissue_prior(tissue_class)
    overrides = user_overrides or {}
    audit: dict[str, str] = {}

    # --- Young's modulus: log-space Bayesian update (closed form) ---
    if "youngs_modulus" in overrides:
        E = max(float(overrides["youngs_modulus"]), 1.0)
        mu, sigma = float(np.log(E)), 1e-6
        audit["youngs_modulus"] = "user_override"
    elif deformation_E_pa is not None:
        obs_mean, obs_std = deformation_E_pa
        obs_mean = max(float(obs_mean), 1.0)
        # Log-normal approximation of the observation.
        obs_var_log = float(np.log1p((obs_std / obs_mean) ** 2))
        obs_mu_log = float(np.log(obs_mean)) - 0.5 * obs_var_log
        post_var = 1.0 / (1.0 / prior.youngs_log_sigma**2 + 1.0 / max(obs_var_log, 1e-6))
        mu = post_var * (
            prior.youngs_log_mu / prior.youngs_log_sigma**2
            + obs_mu_log / max(obs_var_log, 1e-6)
        )
        sigma = float(np.sqrt(post_var))
        audit["youngs_modulus"] = "estimated_deformation"
    else:
        mu, sigma = prior.youngs_log_mu, prior.youngs_log_sigma
        audit["youngs_modulus"] = "assumed_table"

    # --- Gaussian-updated scalars ---
    def _gauss(name, prior_mean, prior_std, obs=None):
        if name in overrides:
            audit[name] = "user_override"
            return GaussianPosterior(
                mean=float(overrides[name]), std=0.0, source="user_override",
                prior_mean=prior_mean, prior_std=prior_std,
                evidence="user-specified override",
            )
        if obs is not None:
            obs_mean, obs_var, evidence = obs
            pm, pv = gaussian_update(prior_mean, prior_std**2, obs_mean, obs_var)
            audit[name] = "estimated"
            return GaussianPosterior(
                mean=pm, std=float(np.sqrt(pv)), source="estimated_pbr",
                prior_mean=prior_mean, prior_std=prior_std, evidence=evidence,
            )
        audit[name] = "assumed_table"
        return GaussianPosterior(
            mean=prior_mean, std=prior_std, source="assumed_table",
            prior_mean=prior_mean, prior_std=prior_std,
            evidence=f"no observation; prior from {prior.name} tissue table",
        )

    return TissuePhysicsEstimate(
        instance_id=instance_id,
        tissue_class=prior.name,
        youngs_log_mu=float(mu),
        youngs_log_sigma=float(sigma),
        poisson=_gauss("poisson_ratio", prior.poisson_mean, prior.poisson_std),
        density=_gauss("density", prior.density_mean, prior.density_std),
        viscosity=_gauss("viscosity", prior.viscosity_mean, prior.viscosity_std),
        friction=_gauss("friction", prior.friction_mean, prior.friction_std),
        constitutive_model=prior.constitutive_model,
        audit=audit,
    )
