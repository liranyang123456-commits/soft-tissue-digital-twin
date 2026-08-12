"""Soft-tissue material database for digital-twin parameter inference.

Soft tissues are fundamentally different from the engineering materials in
:mod:`material_db`:

- **Stiffness is in kPa, not GPa.** Liver is ~2-10 kPa; the hardest entries
  here (tumor, skin) are ~10^2 kPa. The engineering table starts at 10^9 Pa.
- **Near-incompressible**: Poisson ratio nu ≈ 0.45-0.49, NOT 0.3.
- **Hyperelastic, not linear-elastic**: large strains are the norm, so each
  entry names a constitutive model family (Neo-Hookean / Mooney-Rivlin /
  Ogden) whose parameters the simulator should consume.
- **Viscoelastic**: tissues exhibit rate-dependent damping; we store a
  Kelvin-Voigt viscosity prior (Pa·s).
- **Log-normal elasticity prior**: E spans orders of magnitude across
  tissue types and pathologies, so the prior is stored as
  (mu, sigma) of ln(E[Pa]) rather than a Gaussian in E. This avoids
  negative-stiffness posterior tails.

Values are compiled from the biomechanics measurement literature
(ex-vivo indentation / shear / elastography surveys, e.g. the tissue
property tables used by SOFA/IMSTK surgical simulators). They are PRIORS
for Bayesian update, not calibrated measurements — every value carries an
``assumed_table`` audit source unless updated by deformation evidence
(see :mod:`physics_est` and :mod:`tissue_sim`).
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True, frozen=True)
class TissuePrior:
    """Prior over mechanical + optical parameters for one tissue class."""

    name: str
    # --- Elasticity: log-normal prior on Young's modulus E [Pa] ---
    youngs_log_mu: float       # mu of ln(E/Pa)
    youngs_log_sigma: float    # sigma of ln(E/Pa)
    # --- Compressibility ---
    poisson_mean: float        # near 0.5 = incompressible
    poisson_std: float
    # --- Density ---
    density_mean: float        # kg/m³
    density_std: float
    # --- Viscoelasticity (Kelvin-Voigt damping) ---
    viscosity_mean: float      # Pa·s
    viscosity_std: float
    # --- Contact (vs surgical tools / other tissue) ---
    friction_mean: float
    friction_std: float
    # --- Constitutive model family for the simulator ---
    constitutive_model: str    # "neo_hookean" | "mooney_rivlin" | "ogden"
    # --- Optical signature (for PBR-based classification) ---
    expected_roughness: float  # wet tissue: moderate specular
    albedo_hint: tuple[float, float, float]  # typical linear albedo
    notes: str = ""

    @property
    def youngs_median_pa(self) -> float:
        """Median of the log-normal E prior, in Pa."""
        return math.exp(self.youngs_log_mu)

    @property
    def youngs_median_kpa(self) -> float:
        return self.youngs_median_pa / 1e3


# ln(E[Pa]) helpers: 1 kPa => ln(1e3) = 6.9078
_LN_1KPA = math.log(1e3)

# ---------------------------------------------------------------------------
# Database: 6 soft-tissue classes
# ---------------------------------------------------------------------------

TISSUE_DATABASE: dict[str, TissuePrior] = {
    "liver": TissuePrior(
        name="liver",
        youngs_log_mu=_LN_1KPA + math.log(6.0),    # median 6 kPa
        youngs_log_sigma=0.5,                       # ~3-12 kPa at ±1σ
        poisson_mean=0.48, poisson_std=0.01,
        density_mean=1060.0, density_std=30.0,
        viscosity_mean=15.0, viscosity_std=8.0,
        friction_mean=0.30, friction_std=0.10,
        constitutive_model="neo_hookean",
        expected_roughness=0.45,
        albedo_hint=(0.45, 0.16, 0.13),  # dark red-brown (perfused)
        notes="Healthy liver, ex-vivo indentation surveys. Tumor-bearing "
              "liver is locally 2-5x stiffer (see 'tumor').",
    ),
    "fat": TissuePrior(
        name="fat",
        youngs_log_mu=_LN_1KPA + math.log(2.0),    # median 2 kPa
        youngs_log_sigma=0.5,
        poisson_mean=0.45, poisson_std=0.02,
        density_mean=950.0, density_std=30.0,
        viscosity_mean=8.0, viscosity_std=4.0,
        friction_mean=0.25, friction_std=0.10,
        constitutive_model="neo_hookean",
        expected_roughness=0.55,
        albedo_hint=(0.75, 0.62, 0.45),  # pale yellow
        notes="Adipose tissue. Softer than parenchyma; often surrounds organs.",
    ),
    "muscle": TissuePrior(
        name="muscle",
        youngs_log_mu=_LN_1KPA + math.log(20.0),   # median 20 kPa
        youngs_log_sigma=0.5,
        poisson_mean=0.49, poisson_std=0.01,
        density_mean=1060.0, density_std=30.0,
        viscosity_mean=25.0, viscosity_std=12.0,
        friction_mean=0.35, friction_std=0.10,
        constitutive_model="mooney_rivlin",
        expected_roughness=0.50,
        albedo_hint=(0.50, 0.14, 0.12),  # deep red
        notes="Skeletal/cardiac muscle, transverse direction. Anisotropic "
              "along fibers; isotropic prior is the conservative choice.",
    ),
    "skin": TissuePrior(
        name="skin",
        youngs_log_mu=_LN_1KPA + math.log(80.0),   # median 80 kPa
        youngs_log_sigma=0.6,
        poisson_mean=0.48, poisson_std=0.02,
        density_mean=1100.0, density_std=50.0,
        viscosity_mean=30.0, viscosity_std=15.0,
        friction_mean=0.50, friction_std=0.15,
        constitutive_model="ogden",
        expected_roughness=0.60,
        albedo_hint=(0.66, 0.45, 0.36),  # pink-tan
        notes="Dermis, low-strain regime. Strongly strain-stiffening (Ogden).",
    ),
    "tumor": TissuePrior(
        name="tumor",
        youngs_log_mu=_LN_1KPA + math.log(30.0),   # median 30 kPa
        youngs_log_sigma=0.7,                       # wide: 2-5x host tissue
        poisson_mean=0.48, poisson_std=0.01,
        density_mean=1050.0, density_std=30.0,
        viscosity_mean=20.0, viscosity_std=10.0,
        friction_mean=0.30, friction_std=0.10,
        constitutive_model="neo_hookean",
        expected_roughness=0.45,
        albedo_hint=(0.50, 0.25, 0.22),  # paler than perfused liver
        notes="Generic carcinoma. Stiffness CONTRAST vs host tissue is the "
              "clinical signal (elastography); absolute value varies widely.",
    ),
    "generic_soft_tissue": TissuePrior(
        name="generic_soft_tissue",
        youngs_log_mu=_LN_1KPA + math.log(10.0),   # median 10 kPa
        youngs_log_sigma=0.8,                       # deliberately broad
        poisson_mean=0.47, poisson_std=0.02,
        density_mean=1040.0, density_std=50.0,
        viscosity_mean=15.0, viscosity_std=10.0,
        friction_mean=0.35, friction_std=0.15,
        constitutive_model="neo_hookean",
        expected_roughness=0.50,
        albedo_hint=(0.55, 0.25, 0.22),
        notes="Fallback when the tissue type is unknown. Broad prior; the "
              "posterior must be driven by deformation evidence.",
    ),
}


def get_tissue_prior(tissue_class: str) -> TissuePrior:
    """Look up a tissue prior by class name (falls back to generic)."""
    if tissue_class in TISSUE_DATABASE:
        return TISSUE_DATABASE[tissue_class]
    return TISSUE_DATABASE["generic_soft_tissue"]


def list_tissue_classes() -> list[str]:
    """Return all available tissue class names."""
    return sorted(TISSUE_DATABASE.keys())


def classify_tissue_from_pbr(
    metalness: float, roughness: float, albedo: tuple[float, float, float]
) -> str:
    """Heuristic tissue-class guess from PBR optical parameters.

    Tissue is always non-metal; the useful optical signal is the albedo
    hue (perfusion level) and the specular wetness (roughness). This only
    selects the PRIOR class — mechanical values must come from deformation
    evidence (:mod:`tissue_sim`), never from color alone.
    """
    if metalness > 0.3:
        # Surgical tool in view, not tissue.
        return "generic_soft_tissue"
    r, g, b = albedo
    perfusion = r - 0.5 * (g + b)  # redness above green/blue
    brightness = (r + g + b) / 3.0
    if brightness > 0.55 and perfusion < 0.15:
        return "fat"
    if perfusion > 0.30 and brightness < 0.45:
        return "muscle" if roughness > 0.5 else "liver"
    if perfusion > 0.15:
        return "liver"
    return "generic_soft_tissue"
