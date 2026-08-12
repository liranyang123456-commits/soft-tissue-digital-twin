"""Export the recovered 3D Gaussian geometry and BRDF parameters.

Supports three output formats:
- ``npz`` / ``ply``: raw per-Gaussian fields (legacy, for evaluation).
- ``mitsuba``: a Mitsuba 3 ``scene.xml`` + mesh + envmap for optical
  re-rendering and physically-based relighting.
- ``usd``: a USD stage with PBR materials for downstream tools and (in
  Phase B) UsdPhysics rigid-body simulation. See ``usd_export.py``.

The Mitsuba and USD paths are optional dependencies imported lazily so
that the legacy NPZ/PLY export works in a minimal environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .scene import GaussianBRDFField


def export_scene(
    checkpoint: str,
    output_dir: str,
    *,
    fmt: str = "npz",
    light=None,
    camera=None,
) -> Path:
    """Export a trained world-space Gaussian BRDF field.

    Parameters
    ----------
    checkpoint:
        Path to a ``.pt`` produced by ``world.train``.
    output_dir:
        Destination directory; created if missing.
    fmt:
        One of ``npz`` (also writes PLY + surface mesh), ``mitsuba``
        (writes ``scene.xml`` + mesh + envmap), or ``usd`` (writes a
        ``.usda`` stage; see :mod:`mvbrdf_shr.world.usd_export`).
    light, camera:
        Optional :class:`~mvbrdf_shr.world.lighting.WorldLight` and
        :class:`~mvbrdf_shr.world.camera.PerspectiveCamera` to embed in
        the Mitsuba/USD scene. Ignored for ``npz``.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    colors = state.get("colors")
    field = GaussianBRDFField.from_point_cloud(
        state["points"],
        colors,
        normal_mode=str(state.get("config", {}).get("normal_mode", "learned")),
    )
    # Extract only the nested scene state from the full pipeline.
    scene_state = {
        key.removeprefix("scene."): value
        for key, value in state["model"].items()
        if key.startswith("scene.")
    }
    field.load_state_dict(scene_state)
    field.eval()
    with torch.no_grad():
        material = field.materials()
        def as_numpy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        values = {
            "position": as_numpy(field.means),
            "scale": as_numpy(field.scales),
            "rotation_quaternion": as_numpy(
                torch.nn.functional.normalize(field.quaternions, dim=-1)
            ),
            "opacity": as_numpy(field.opacity),
            "normal": as_numpy(field.normals),
            "albedo": as_numpy(material.albedo),
            "specular": as_numpy(material.specular),
            "diffuse_weight": as_numpy(material.diffuse_weight),
            "roughness": as_numpy(material.roughness),
            "metalness": as_numpy(material.metalness),
            "absorption": as_numpy(material.absorption),
            "projected_texture": as_numpy(field.texture),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "gaussian_brdf.npz", **values)
    _write_ply(output / "gaussian_brdf.ply", values)
    mesh_path = _write_surface_mesh(output / "surface_mesh.ply", values)
    fmt_lower = fmt.lower()
    if fmt_lower == "mitsuba":
        mesh_path = to_mitsuba_xml(
            values, mesh_path, output, light=light, camera=camera
        )
    elif fmt_lower == "usd":
        from .usd_export import to_usd

        to_usd(values, mesh_path, output, light=light, camera=camera)
    print(f"exported 3D Gaussian BRDF model to {output} (format={fmt_lower})")
    return mesh_path


def _write_surface_mesh(
    path: Path,
    values: dict[str, np.ndarray],
    *,
    method: str = "auto",
) -> Path:
    """Convert the optimized Gaussian surface samples into an evaluator mesh.

    ``method`` selects the meshing strategy:
    - ``auto`` (default): Poisson surface reconstruction when the optional
      ``pymeshlab``/``open3d`` dependency is available, falling back to
      marching cubes. Poisson produces watertight, simulation-ready
      meshes; marching cubes is the legacy evaluator mesh.
    - ``marching_cubes``: always use the trimesh voxel marching cubes path.
    - ``convex_hull``: always use the convex hull (cheapest, lossy).
    """
    import trimesh

    opacity = values["opacity"][:, 0]
    keep = opacity > 0.05
    points = values["position"][keep]
    normals = values["normal"][keep]
    scales = values["scale"][keep]
    if len(points) < 4:
        raise ValueError("not enough visible Gaussians to export a surface mesh")
    tangent = np.partition(scales, 1, axis=1)[:, 1:].mean(axis=1)
    extent_floor = np.ptp(points, axis=0).max() / 256
    pitch = max(float(np.median(tangent) * 0.75), float(extent_floor), 1e-4)

    mesh: trimesh.Trimesh | None = None
    if method in {"auto", "poisson"}:
        mesh = _try_poisson_mesh(points, normals)
    if mesh is None and method != "convex_hull":
        try:
            mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=pitch)
        except (ValueError, RuntimeError, MemoryError):
            mesh = None
    if mesh is None:
        mesh = trimesh.points.PointCloud(points).convex_hull
    components = mesh.split(only_watertight=False)
    if components:
        mesh = max(components, key=lambda component: len(component.faces))
    mesh.export(path)
    return path


def _try_poisson_mesh(
    points: np.ndarray, normals: np.ndarray
) -> "trimesh.Trimesh | None":
    """Attempt Poisson surface reconstruction; return None on any failure.

    Tries Open3D first (weighted Poisson), then PyMeshLab. Both are
    optional; absence is silent so the caller can fall back.
    """
    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except ImportError:
        o3d = None
    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )
            import trimesh  # noqa: F811
            return trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                process=True,
            )
        except (ValueError, RuntimeError, MemoryError):
            return None
    try:
        import pymeshlab as pml  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        ms = pml.MeshSet()
        m = pml.Mesh(
            vertex_matrix=points.astype(np.float64),
            v_normals_matrix=normals.astype(np.float64),
        )
        ms.add_mesh(m, "gaussian_surface")
        ms.apply_filter("poisson_reconstruction", visiblelayer=False)
        out = ms.current_mesh()
        import trimesh  # noqa: F811
        return trimesh.Trimesh(
            vertices=np.asarray(out.vertex_matrix()),
            faces=np.asarray(out.face_matrix()),
            process=True,
        )
    except (ValueError, RuntimeError, MemoryError):
        return None


# ---------------------------------------------------------------------------
# Mitsuba 3 XML export (Phase A — optical twin)
# ---------------------------------------------------------------------------


def _sh_to_envmap(
    sh: np.ndarray, height: int = 32, width: int = 64
) -> np.ndarray:
    """Reconstruct a lat-long HDR envmap from 3rd-order SH coefficients.

    ``sh`` has shape (9, 3). Returns (H, W, 3) linear radiance, matching
    the convention of :func:`mvbrdf_shr.world.lighting.latlong_to_sh`
    so that a round-trip SH→envmap→SH is consistent.
    """
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
    )  # (H, W, 9)
    return np.clip(basis @ sh, 0.0, None)  # (H, W, 3)


def _bsdf_xml_for_gaussian(
    albedo: np.ndarray,
    diffuse_weight: float,
    specular: float,
    roughness: float,
    metalness: float,
    absorption: float,
) -> str:
    """Map one Gaussian's PBR parameters to a Mitsuba BSDF XML fragment.

    The Gaussian BRDF field enforces ``diffuse_weight + specular + absorption
    = 1`` via a softmax budget. We translate this to a Mitsuba
    ``principled`` BSDF whose diffuse lobe carries the energy-controlled
    albedo and whose specular lobe is scaled by the specular budget. Metals
    route the base color into the conductor's reflectance.
    """
    albedo_lin = np.clip(albedo, 1e-4, 1.0).tolist()
    alpha = float(np.clip(roughness * roughness, 1e-3, 1.0))
    # Energy reaching the surface (1 - absorption); split into diffuse/spec.
    surface_energy = float(np.clip(1.0 - absorption, 0.0, 1.0))
    if surface_energy < 0.05:
        # Nearly fully absorptive: emit a black diffuse BSDF.
        return (
            '<bsdf type="diffuse">\n'
            '  <rgb name="reflectance" value="0 0 0"/>\n'
            "</bsdf>"
        )
    if metalness > 0.5:
        # Conductor: base color becomes the specular reflectance.
        eta = [0.0, 0.0, 0.0]
        k = albedo_lin
        return (
            '<bsdf type="conductor">\n'
            f'  <rgb name="eta" value="{" ".join(map(str, eta))}"/>\n'
            f'  <rgb name="k" value="{" ".join(map(str, k))}"/>\n'
            "</bsdf>"
        )
    # Dielectric: principled-like mix of diffuse + rough conductor lobe.
    diffuse_scale = float(np.clip(diffuse_weight / surface_energy, 0.0, 1.0))
    spec_scale = float(np.clip(specular / surface_energy, 0.0, 1.0))
    diffuse_rgb = [c * diffuse_scale for c in albedo_lin]
    return (
        '<bsdf type="roughplastic">\n'
        f'  <rgb name="diffuse_reflectance" value="{" ".join(map(str, diffuse_rgb))}"/>\n'
        f'  <float name="alpha" value="{alpha}"/>\n'
        '  <float name="int_ior" value="1.5"/>\n'
        '  <float name="ext_ior" value="1.0"/>\n'
        '  <bool name="nonlinear" value="true"/>\n'
        "</bsdf>"
    )


def to_mitsuba_xml(
    values: dict[str, np.ndarray],
    mesh_path: Path,
    output_dir: Path,
    *,
    light=None,
    camera=None,
) -> Path:
    """Write a Mitsuba 3 scene (``scene.xml`` + assets) for optical re-rendering.

    Parameters
    ----------
    values:
        Per-Gaussian arrays from :func:`export_scene`.
    mesh_path:
        Surface mesh (PLY) produced by :func:`_write_surface_mesh`.
    output_dir:
        Destination directory; assets are written alongside the XML.
    light:
        Optional :class:`~mvbrdf_shr.world.lighting.WorldLight`. When
        provided, an ``envmap`` (from SH) and a point/distant light are
        emitted. When omitted, a uniform grey environment is used.
    camera:
        Optional :class:`~mvbrdf_shr.world.camera.PerspectiveCamera`.
        When omitted, a default orbiting perspective sensor is emitted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_rel = mesh_path.name

    # --- Environment -------------------------------------------------------
    env_xml = ""
    if light is not None:
        sh = (
            light.env_sh.detach().cpu().numpy()
            if torch.is_tensor(light.env_sh)
            else np.asarray(light.env_sh)
        )
        if sh.ndim == 3 and sh.shape[0] == 9:
            envmap = _sh_to_envmap(sh)
        else:
            envmap = np.full((16, 32, 3), 0.3, dtype=np.float32)
        env_path = output_dir / "env.hdr"
        try:
            import imageio.v3 as iio  # type: ignore[import-not-found]
            iio.imwrite(str(env_path), envmap.astype(np.float32))
            env_rel = env_path.name
            env_xml = (
                f'<texture type="bitmap" id="envmap">\n'
                f'  <string name="filename" value="{env_rel}"/>\n'
                "</texture>\n"
                f'<emitter type="envmap">\n'
                f'  <ref id="envmap"/>\n'
                "</emitter>\n"
            )
        except ImportError:
            env_xml = '<emitter type="constant"><rgb name="radiance" value="0.3 0.3 0.3"/></emitter>\n'
        # Dominant light as a point or distant emitter.
        pw = float(light.point_weight.detach().cpu().item()) if torch.is_tensor(light.point_weight) else float(light.point_weight)
        if pw > 0.5:
            pos = light.position.detach().cpu().numpy().tolist() if torch.is_tensor(light.position) else list(light.position)
            intensity = float(light.intensity.detach().cpu().item()) if torch.is_tensor(light.intensity) else float(light.intensity)
            color = light.color.detach().cpu().numpy().tolist() if torch.is_tensor(light.color) else list(light.color)
            env_xml += (
                f'<emitter type="point">\n'
                f'  <point name="position" value="{" ".join(map(str, pos))}"/>\n'
                f'  <rgb name="intensity" value="{" ".join(str(intensity * c) for c in color)}"/>\n'
                "</emitter>\n"
            )
        else:
            direction = light.direction.detach().cpu().numpy().tolist() if torch.is_tensor(light.direction) else list(light.direction)
            intensity = float(light.intensity.detach().cpu().item()) if torch.is_tensor(light.intensity) else float(light.intensity)
            color = light.color.detach().cpu().numpy().tolist() if torch.is_tensor(light.color) else list(light.color)
            env_xml += (
                f'<emitter type="directional">\n'
                f'  <vector name="direction" value="{" ".join(map(str, direction))}"/>\n'
                f'  <rgb name="irradiance" value="{" ".join(str(intensity * c) for c in color)}"/>\n'
                "</emitter>\n"
            )
    else:
        env_xml = '<emitter type="constant"><rgb name="radiance" value="0.3 0.3 0.3"/></emitter>\n'

    # --- Sensor ------------------------------------------------------------
    if camera is not None:
        c2w = camera.c2w.detach().cpu().numpy() if torch.is_tensor(camera.c2w) else np.asarray(camera.c2w)
        # Mitsuba uses +y up, -z forward (OpenGL). Our camera is OpenCV (+z forward, +y down).
        gl = c2w.copy()
        gl[:3, 1] = -c2w[:3, 1]
        gl[:3, 2] = -c2w[:3, 2]
        fx, fy = float(camera.K[0, 0]), float(camera.K[1, 1])
        fov_x = float(2.0 * np.degrees(np.arctan(camera.width / (2.0 * fx))))
        sensor_xml = (
            '<sensor type="perspective">\n'
            f'  <float name="fov" value="{fov_x}"/>\n'
            '  <string name="fov_axis" value="x"/>\n'
            '  <transform name="to_world">\n'
            f'    <matrix value="{" ".join(map(str, gl.T.flatten()))}"/>\n'
            "  </transform>\n"
            f'  <film type="hdrfilm">\n'
            f'    <integer name="width" value="{camera.width}"/>\n'
            f'    <integer name="height" value="{camera.height}"/>\n'
            '    <rfilter type="box"/>\n'
            "  </film>\n"
            '  <sampler type="independent"><integer name="sample_count" value="64"/></sampler>\n'
            "</sensor>\n"
        )
    else:
        sensor_xml = (
            '<sensor type="perspective">\n'
            '  <float name="fov" value="40"/>\n'
            '  <transform name="to_world">\n'
            '    <lookat origin="0 0 3" target="0 0 0" up="0 1 0"/>\n'
            "  </transform>\n"
            '  <film type="hdrfilm"><integer name="width" value="512"/><integer name="height" value="512"/><rfilter type="box"/></film>\n'
            '  <sampler type="independent"><integer name="sample_count" value="64"/></sampler>\n'
            "</sensor>\n"
        )

    # --- Shape + BSDF ------------------------------------------------------
    # Aggregate per-Gaussian material to a single representative BSDF for
    # the exported mesh (per-vertex materials require a Mitsuba plugin;
    # the representative median is sufficient for optical-twin validation).
    med_albedo = np.median(values["albedo"], axis=0)
    med_dw = float(np.median(values["diffuse_weight"]))
    med_spec = float(np.median(values["specular"]))
    med_rough = float(np.median(values["roughness"]))
    med_metal = float(np.median(values["metalness"]))
    med_abs = float(np.median(values["absorption"]))
    bsdf_xml = _bsdf_xml_for_gaussian(
        med_albedo, med_dw, med_spec, med_rough, med_metal, med_abs
    )
    shape_xml = (
        f'<shape type="ply" id="object">\n'
        f'  <string name="filename" value="{mesh_rel}"/>\n'
        f"  {bsdf_xml}\n"
        "</shape>\n"
    )

    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<scene version="3.0.0">\n'
        f"{env_xml}"
        f"{sensor_xml}"
        f'{integrator_xml()}'
        f"{shape_xml}"
        "</scene>\n"
    )
    xml_path = output_dir / "scene.xml"
    xml_path.write_text(xml, encoding="utf-8")
    return xml_path


def integrator_xml(*, max_depth: int = 6) -> str:
    """Default path-tracing integrator with modest bounce count."""
    return (
        f'<integrator type="path">\n'
        f'  <integer name="max_depth" value="{max_depth}"/>\n'
        "</integrator>\n"
    )


def _write_ply(path: Path, values: dict[str, np.ndarray]):
    count = len(values["position"])
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float opacity",
        "property float scale_x",
        "property float scale_y",
        "property float scale_z",
        "property float specular",
        "property float roughness",
        "property float metalness",
        "property float absorption",
        "end_header",
    ]
    lines = header
    rgb = (values["albedo"].clip(0, 1) * 255).astype(np.uint8)
    for index in range(count):
        p, n, c, s = (
            values["position"][index],
            values["normal"][index],
            rgb[index],
            values["scale"][index],
        )
        lines.append(
            f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]} "
            f"{c[0]} {c[1]} {c[2]} {values['opacity'][index, 0]} "
            f"{s[0]} {s[1]} {s[2]} {values['specular'][index, 0]} "
            f"{values['roughness'][index, 0]} {values['metalness'][index, 0]} "
            f"{values['absorption'][index, 0]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained Gaussian BRDF field to NPZ/PLY, Mitsuba, or USD."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/world_export")
    parser.add_argument(
        "--format",
        choices=["npz", "mitsuba", "usd"],
        default="npz",
        help="Output format: npz (legacy), mitsuba (scene.xml), or usd (.usda).",
    )
    args = parser.parse_args()
    export_scene(args.checkpoint, args.output, fmt=args.format)


if __name__ == "__main__":
    main()

