"""USD export for the recovered Gaussian BRDF field (optical + physics layers).

Phase A (:func:`to_usd`) writes a USD stage with:
- triangle mesh (from the exported surface mesh)
- PBR material (UsdPreviewSurface: diffuseColor, roughness, metallic)
- environment lighting (DomeLight from SH-derived envmap)
- perspective camera

Phase B (:func:`to_usd_with_physics`) extends the stage with:
- per-instance Xform prims (one per segmented object)
- UsdPhysics RigidBodyAPI / MassAPI / CollisionAPI
- contact materials (friction, restitution) with audit metadata

``pxr`` (OpenUSD / usd-core) is imported lazily so that the rest of the
package works without it, following the same convention used for
``trimesh`` and ``mitsuba`` elsewhere in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .scene import GaussianBRDFField
    from .lighting import WorldLight
    from .camera import PerspectiveCamera


# ---------------------------------------------------------------------------
# Lazy USD import helper
# ---------------------------------------------------------------------------


def _import_usd():
    """Import and return the ``pxr`` module, raising a clear error if absent."""
    try:
        from pxr import Usd, UsdGeom, UsdShade, UsdLux, UsdPhysics, Sdf, Vt, Gf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without pxr
        raise ImportError(
            "USD export requires the 'pxr' package (OpenUSD / usd-core). "
            "Install with `pip install usd-core` (search name 'pxr-usd-core') "
            "or use an Omniverse/Isaac Sim Python environment."
        ) from exc
    return Usd, UsdGeom, UsdShade, UsdLux, UsdPhysics, Sdf, Vt, Gf


# ---------------------------------------------------------------------------
# Phase A: optical twin
# ---------------------------------------------------------------------------


def to_usd(
    values: dict[str, np.ndarray],
    mesh_path: Path,
    output_dir: Path,
    *,
    light: "WorldLight | None" = None,
    camera: "PerspectiveCamera | None" = None,
    stage_path: Path | None = None,
) -> Path:
    """Write a USD stage (``.usda``) representing the optical twin.

    Parameters
    ----------
    values:
        Per-Gaussian arrays from :func:`mvbrdf_shr.world.export.export_scene`.
    mesh_path:
        Surface mesh (PLY) produced by ``_write_surface_mesh``.
    output_dir:
        Destination directory; the stage and assets are written here.
    light, camera:
        Optional world-space light/camera to embed.
    stage_path:
        Optional explicit stage path; defaults to ``<output_dir>/scene.usda``.
    """
    Usd, UsdGeom, UsdShade, UsdLux, _UsdPhysics, Sdf, Vt, Gf = _import_usd()
    import trimesh  # optional dep, already required by export.py meshing

    output_dir.mkdir(parents=True, exist_ok=True)
    if stage_path is None:
        stage_path = output_dir / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/Looks", "Scope")
    stage.DefinePrim("/World/Environment", "Scope")

    # --- Mesh -----------------------------------------------------------
    mesh = _load_mesh(mesh_path)
    object_prim = UsdGeom.Xform.Define(stage, "/World/object")
    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/object/geometry")
    mesh_prim.CreatePointsAttr(_to_vt_vec3f(mesh.vertices))
    mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.flatten().tolist()))
    counts = np.full(len(mesh.faces), 3, dtype=np.int32)
    mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray(counts.tolist()))
    if hasattr(mesh, "visual") and mesh.visual is not None and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
        mesh_prim.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set(Vt.Vec2fArray([Gf.Vec2f(uv[0], uv[1]) for uv in uvs]))

    # --- PBR material ---------------------------------------------------
    mat = _define_pbr_material(stage, "/World/Looks/object_material", values)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)

    # --- Lighting -------------------------------------------------------
    if light is not None:
        _add_light(stage, "/World/Environment", light)
    else:
        UsdLux.DomeLight.Define(stage, "/World/Environment/dome").CreateIntensityAttr(300.0)

    # --- Camera ---------------------------------------------------------
    if camera is not None:
        _add_camera(stage, "/World/camera", camera)

    stage.GetRootLayer().Save()
    return stage_path


# ---------------------------------------------------------------------------
# Phase B: rigid-body physics twin
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InstancePhysics:
    """Per-instance physics parameters + audit metadata for USD assembly."""

    instance_id: int
    semantic_label: str
    material_class: str  # e.g. "wood", "aluminum", "glass"
    density: float  # kg/m^3
    mass: float  # kg (density * volume)
    friction: float
    restitution: float
    youngs_modulus: float  # Pa
    poisson_ratio: float
    is_dynamic: bool = True
    # Audit trail: each parameter's provenance.
    audit: dict[str, str] | None = None


def to_usd_with_physics(
    values: dict[str, np.ndarray],
    instance_mesh_paths: list[Path],
    instance_physics: list[InstancePhysics],
    output_dir: Path,
    *,
    light: "WorldLight | None" = None,
    camera: "PerspectiveCamera | None" = None,
    stage_path: Path | None = None,
) -> Path:
    """Write a USD stage with UsdPhysics rigid bodies (Phase B).

    Each entry in ``instance_mesh_paths`` / ``instance_physics`` becomes
    one ``/World/instance_<i>`` Xform carrying collision geometry, a
    PBR material, and ``UsdPhysicsRigidBodyAPI`` / ``MassAPI``.
    """
    Usd, UsdGeom, UsdShade, UsdLux, UsdPhysics, Sdf, Vt, Gf = _import_usd()

    output_dir.mkdir(parents=True, exist_ok=True)
    if stage_path is None:
        stage_path = output_dir / "twin.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # --- Physics scene --------------------------------------------------
    physics_scene_path = "/World/PhysicsScene"
    UsdPhysics.Scene.Define(stage, physics_scene_path).CreateGravityDirectionAttr(
        Gf.Vec3f(0.0, -1.0, 0.0)
    )
    UsdPhysics.Scene.Get(stage, physics_scene_path).CreateGravityMagnitudeAttr(9.81)

    stage.DefinePrim("/World/Looks", "Scope")
    stage.DefinePrim("/World/Environment", "Scope")
    stage.DefinePrim("/World/Audit", "Scope")

    # --- Per-instance prims --------------------------------------------
    for idx, (mesh_path, phys) in enumerate(
        zip(instance_mesh_paths, instance_physics)
    ):
        prim_path = f"/World/instance_{phys.instance_id}"
        xform = UsdGeom.Xform.Define(stage, prim_path)
        mesh = _load_mesh(mesh_path)

        # Visual + collision geometry share one mesh prim.
        mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/geometry")
        mesh_prim.CreatePointsAttr(_to_vt_vec3f(mesh.vertices))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.flatten().tolist()))
        counts = np.full(len(mesh.faces), 3, dtype=np.int32)
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray(counts.tolist()))

        # PBR material (per-instance, so each object can differ).
        instance_values = _per_instance_values(values, phys)
        mat = _define_pbr_material(
            stage, f"/World/Looks/instance_{phys.instance_id}_material", instance_values
        )
        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(mat)

        # Collision API on the mesh.
        UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())

        # Rigid body API on the Xform.
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
        rb.CreateKinematicEnabledAttr(not phys.is_dynamic)

        # Mass API.
        UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Get(stage, prim_path)
        mass_api.CreateMassAttr(phys.mass)
        mass_api.CreateDensityAttr(phys.density)
        inertia = _compute_inertia_tensor(mesh, phys.density)
        mass_api.CreateDiagonalInertiaAttr(
            Gf.Vec3f(float(inertia[0]), float(inertia[1]), float(inertia[2]))
        )

        # Contact material (friction / restitution) on a UsdShade.Material
        # prim carrying the UsdPhysics.MaterialAPI schema. Physics material
        # binding reuses UsdShade.MaterialBindingAPI.
        contact_path = f"/World/Looks/contact_{phys.instance_id}"
        contact_mat = UsdShade.Material.Define(stage, contact_path)
        UsdPhysics.MaterialAPI.Apply(contact_mat.GetPrim())
        contact_api = UsdPhysics.MaterialAPI.Get(stage, contact_path)
        contact_api.CreateStaticFrictionAttr(phys.friction)
        contact_api.CreateDynamicFrictionAttr(phys.friction)
        contact_api.CreateRestitutionAttr(phys.restitution)
        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(contact_mat)

        # Audit metadata as primvars / custom attributes.
        _write_audit(stage, f"{prim_path}/audit", phys)

    # --- Environment + camera ------------------------------------------
    if light is not None:
        _add_light(stage, "/World/Environment", light)
    else:
        UsdLux.DomeLight.Define(stage, "/World/Environment/dome").CreateIntensityAttr(300.0)
    if camera is not None:
        _add_camera(stage, "/World/camera", camera)

    stage.GetRootLayer().Save()
    return stage_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_mesh(mesh_path: Path):
    """Load a mesh with trimesh (the same optional dep used by export.py)."""
    import trimesh
    return trimesh.load(str(mesh_path), force="mesh")


def _to_vt_vec3f(points: np.ndarray):
    from pxr import Vt, Gf  # type: ignore[import-not-found]
    return Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in np.asarray(points, dtype=np.float32)])


def _define_pbr_material(stage, path: str, values: dict[str, np.ndarray]):
    """Define a UsdPreviewSurface material from representative PBR values."""
    from pxr import UsdShade, Sdf, Vt, Gf  # type: ignore[import-not-found]

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/previewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")

    albedo = np.clip(np.median(values["albedo"], axis=0), 0.0, 1.0)
    roughness = float(np.clip(np.median(values["roughness"]), 0.0, 1.0))
    metallic = float(np.clip(np.median(values["metalness"]), 0.0, 1.0))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(albedo[0]), float(albedo[1]), float(albedo[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return mat


def _per_instance_values(
    values: dict[str, np.ndarray], phys: InstancePhysics
) -> dict[str, np.ndarray]:
    """Return PBR values scoped to one instance.

    In Phase B without per-instance Gaussian indices yet, we fall back to
    the global arrays; :mod:`mvbrdf_shr.world.segment` will later supply
    per-instance index masks so this can slice precisely.
    """
    return values


def _add_light(stage, scope_path: str, light: "WorldLight"):
    """Add a DomeLight (from SH) + a point/distant light to the stage."""
    from pxr import UsdLux, Gf  # type: ignore[import-not-found]

    sh = (
        light.env_sh.detach().cpu().numpy()
        if hasattr(light.env_sh, "detach")
        else np.asarray(light.env_sh)
    )
    if sh.ndim == 3 and sh.shape[0] == 9:
        envmap = _sh_to_envmap_usd(sh)
        dome = UsdLux.DomeLight.Define(stage, f"{scope_path}/dome")
        # Encode the envmap as a texture; fall back to a flat color if no
        # image-writing dep is available.
        try:
            import imageio.v3 as iio  # type: ignore[import-not-found]
            from pxr import Sdf  # type: ignore[import-not-found]
            # DomeLight texture expects a file path; USD handles lat-long.
            dome.CreateTextureFileAttr("env.hdr")
            dome.CreateTextureFormatAttr("automatic")
        except ImportError:
            avg = np.clip(envmap.mean(axis=(0, 1)), 0.0, None)
            dome.CreateColorAttr(
                Gf.Vec3f(float(avg[0]), float(avg[1]), float(avg[2]))
            )
        dome.CreateIntensityAttr(1.0)
    else:
        UsdLux.DomeLight.Define(stage, f"{scope_path}/dome").CreateIntensityAttr(300.0)

    pw = (
        float(light.point_weight.detach().cpu().item())
        if hasattr(light.point_weight, "detach")
        else float(light.point_weight)
    )
    intensity = (
        float(light.intensity.detach().cpu().item())
        if hasattr(light.intensity, "detach")
        else float(light.intensity)
    )
    color = (
        light.color.detach().cpu().numpy()
        if hasattr(light.color, "detach")
        else np.asarray(light.color)
    )
    if pw > 0.5:
        pos = (
            light.position.detach().cpu().numpy()
            if hasattr(light.position, "detach")
            else np.asarray(light.position)
        )
        sphere = UsdLux.SphereLight.Define(stage, f"{scope_path}/key")
        sphere.CreateRadiusAttr(0.05)
        sphere.CreateIntensityAttr(intensity * 100.0)
        sphere.CreateColorAttr(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
        sphere.AddTranslateOp().Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
    else:
        direction = (
            light.direction.detach().cpu().numpy()
            if hasattr(light.direction, "detach")
            else np.asarray(light.direction)
        )
        distant = UsdLux.DistantLight.Define(stage, f"{scope_path}/key")
        distant.CreateIntensityAttr(intensity)
        distant.CreateColorAttr(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
        distant.AddRotateXYZOp().Set(_direction_to_euler(direction))


def _add_camera(stage, path: str, camera: "PerspectiveCamera"):
    """Add a UsdGeom.Camera matching the world-space PerspectiveCamera."""
    from pxr import UsdGeom, Gf  # type: ignore[import-not-found]

    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    fx, fy = float(camera.K[0, 0]), float(camera.K[1, 1])
    fov_x = float(np.degrees(2.0 * np.arctan(camera.width / (2.0 * fx))))
    cam.CreateHorizontalApertureAttr().Set(float(camera.width))
    cam.CreateVerticalApertureAttr().Set(float(camera.height))
    cam.CreateHorizontalApertureOffsetAttr().Set(0.0)
    cam.CreateVerticalApertureOffsetAttr().Set(0.0)
    cam.CreateFocalLengthAttr().Set(fx)
    cam.CreateFStopAttr().Set(0.0)
    # OpenCV (+z forward, +y down) -> USD/Gf (+y up, -z forward).
    c2w = camera.c2w.detach().cpu().numpy() if hasattr(camera.c2w, "detach") else np.asarray(camera.c2w)
    gl = c2w.copy()
    gl[:3, 1] = -c2w[:3, 1]
    gl[:3, 2] = -c2w[:3, 2]
    xform = cam.AddTransformOp()
    xform.Set(Gf.Matrix4d(gl.tolist()))


def _direction_to_euler(direction: np.ndarray) -> tuple[float, float, float]:
    """Convert a world-space direction into XYZ Euler angles (degrees)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    yaw = float(np.degrees(np.arctan2(d[0], d[2])))
    pitch = float(np.degrees(np.arcsin(np.clip(-d[1], -1.0, 1.0))))
    return (0.0, pitch, yaw)


def _sh_to_envmap_usd(sh: np.ndarray, height: int = 16, width: int = 32) -> np.ndarray:
    """Reconstruct a lat-long envmap from SH (same convention as export.py)."""
    phi = (np.arange(height) + 0.5) * np.pi / height
    theta = (np.arange(width) + 0.5) * (2 * np.pi / width) - 0.5 * np.pi
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    sin_phi = np.sin(phi)
    x = -np.cos(theta) * sin_phi
    y = np.cos(phi)
    z = -np.sin(theta) * sin_phi
    basis = np.stack(
        [
            np.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3 * z * z - 1),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ],
        axis=-1,
    )
    return np.clip(basis @ sh, 0.0, None)


def _compute_inertia_tensor(mesh, density: float) -> np.ndarray:
    """Approximate the diagonal inertia tensor of a mesh at uniform density.

    Uses the mesh's principal axes via trimesh's inertia computation when
    available; falls back to a bounding-box approximation. Handles point
    clouds and degenerate meshes gracefully (NaN-safe).
    """
    import trimesh
    try:
        if hasattr(mesh, "moment_inertia") and mesh.moment_inertia is not None:
            inertia = np.asarray(mesh.moment_inertia, dtype=np.float64)
            diag = np.diag(inertia)
            if np.all(np.isfinite(diag)) and np.all(diag > 0):
                return diag
    except Exception:
        pass
    # Bounding-box fallback: I = (1/12) m (w^2 + d^2) per axis.
    extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    if not np.all(np.isfinite(extents)) or np.any(extents <= 0):
        return np.array([1e-6, 1e-6, 1e-6], dtype=np.float32)
    try:
        vol = float(mesh.volume) if hasattr(mesh, "volume") else 0.0
    except Exception:
        vol = 0.0
    if not np.isfinite(vol) or vol <= 0:
        vol = float(np.prod(extents))
    mass = density * vol
    w, h, d = extents
    return np.array(
        [
            mass * (h * h + d * d) / 12.0,
            mass * (w * w + d * d) / 12.0,
            mass * (w * w + h * h) / 12.0,
        ],
        dtype=np.float32,
    )


def _write_audit(stage, path: str, phys: InstancePhysics):
    """Write physics-parameter provenance as USD custom attributes."""
    from pxr import Sdf  # type: ignore[import-not-found]
    prim = stage.DefinePrim(path, "Scope")
    # Always record the core provenance.
    audit = phys.audit or {}
    for key, value in {
        "semantic_label": phys.semantic_label,
        "material_class": phys.material_class,
        "density_source": audit.get("density", "assumed_table"),
        "friction_source": audit.get("friction", "assumed_table"),
        "restitution_source": audit.get("restitution", "assumed_table"),
        "mass_source": audit.get("mass", "estimated"),
        "inertia_source": audit.get("inertia", "estimated"),
        "youngs_modulus_source": audit.get("youngs_modulus", "assumed_table"),
        "poisson_ratio_source": audit.get("poisson_ratio", "assumed_table"),
    }.items():
        attr = prim.CreateAttribute(f"audit:{key}", Sdf.ValueTypeNames.String)
        attr.Set(str(value))
