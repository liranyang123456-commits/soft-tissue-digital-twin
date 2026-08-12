"""Differentiable anisotropic 3D Gaussian splatting with BRDF separation."""
from __future__ import annotations

from dataclasses import dataclass
import glob
import math
import os
import shutil
import subprocess
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.envlight import sh_to_irradiance
from ..models.dual_latent_brdf import DualLatentNeuralBRDF
from ..models.implicit_material_field import ImplicitMaterialImagingField
from .camera import PerspectiveCamera
from .lighting import NeuralIncidentLightField, WorldLight, fibonacci_sphere
from .scene import GaussianBRDFField, GaussianMaterials

_PI = 3.141592653589793
_EPS = 1e-6


def _view_locked_opacity(
    scene: GaussianBRDFField, camera: PerspectiveCamera
) -> torch.Tensor:
    """Weight source-owned Gaussians for the active camera.

    ``hard`` preserves the legacy exact/neighbor mask. ``soft`` retains source
    identity while blending observations according to camera angle (or circular
    view distance when source camera centers are unavailable).
    """
    opacity = scene.opacity[:, 0]
    if not bool(getattr(scene, "view_locked_opacity", False)):
        return opacity
    source = getattr(scene, "source_view_id", None)
    view_id = getattr(camera, "view_id", None)
    if source is None or view_id is None:
        return opacity
    source = source.to(opacity.device)
    if int(source.min()) < 0:
        return opacity
    neighbors = int(getattr(scene, "view_lock_neighbors", 0) or 0)
    mode = str(getattr(scene, "view_lock_mode", "hard")).lower()
    if mode not in {"hard", "soft"}:
        raise ValueError(f"unknown view_lock_mode {mode!r}")
    active = {int(view_id)}
    unique_ids = sorted(int(uid) for uid in source.unique().tolist())
    n_views = max(unique_ids) + 1 if unique_ids else 1
    if mode == "soft" and unique_ids:
        temperature = max(
            float(getattr(scene, "view_lock_temperature", 0.45)), 1e-3
        )
        min_weight = max(
            float(getattr(scene, "view_lock_min_weight", 0.02)), 0.0
        )
        centers = getattr(scene, "source_camera_centers", None)
        source_weight = torch.zeros(
            n_views, device=opacity.device, dtype=opacity.dtype
        )
        valid_centers = (
            isinstance(centers, torch.Tensor)
            and centers.ndim == 2
            and centers.shape[-1] == 3
            and int(view_id) < len(centers)
            and torch.isfinite(centers[int(view_id)]).all()
        )
        if valid_centers:
            centers = centers.to(opacity)
            object_center = scene.means.detach().mean(dim=0)
            target_direction = F.normalize(
                centers[int(view_id)] - object_center, dim=0, eps=1e-6
            )
            available = torch.tensor(unique_ids, device=opacity.device)
            source_direction = F.normalize(
                centers[available] - object_center[None], dim=-1, eps=1e-6
            )
            angle = torch.acos(
                (source_direction * target_direction[None])
                .sum(dim=-1)
                .clamp(-1 + 1e-6, 1 - 1e-6)
            )
            source_weight[available] = torch.exp(
                -0.5 * (angle / temperature).square()
            )
        else:
            for uid in unique_ids:
                direct = abs(uid - int(view_id))
                distance = min(direct, max(n_views - direct, 0))
                source_weight[uid] = math.exp(
                    -0.5 * (float(distance) / temperature) ** 2
                )
        selected = source_weight[source].clamp_min(0)
        selected = torch.where(
            selected >= min_weight, selected, torch.zeros_like(selected)
        )
        confidence = getattr(scene, "source_confidence", None)
        if isinstance(confidence, torch.Tensor) and confidence.shape[0] == len(source):
            selected = selected * confidence.to(selected).reshape(-1).clamp(0, 1)
        if bool((selected > 0).any()):
            return opacity * selected
        # A very small temperature can remove every source. Fall through to
        # the deterministic hard nearest-source behavior instead of rendering
        # an empty frame.
    if neighbors > 0 and n_views > 1:
        # DiLiGenT views are ordered around the object; enable adjacent ids.
        for delta in range(1, neighbors + 1):
            active.add((int(view_id) - delta) % n_views)
            active.add((int(view_id) + delta) % n_views)
    active &= set(unique_ids)
    if not active and unique_ids:
        # Strict holdout views have no source-owned Gaussians. Falling back to
        # every source mixes incompatible view-specific PS shells and destroys
        # the rendered normal. Select the closest available source IDs on the
        # circular DiLiGenT camera trajectory instead.
        def circular_distance(uid: int) -> int:
            direct = abs(uid - int(view_id))
            return min(direct, n_views - direct)

        count = max(1, 2 * neighbors)
        active = set(
            sorted(unique_ids, key=lambda uid: (circular_distance(uid), uid))[:count]
        )
    keep = torch.zeros_like(opacity, dtype=torch.bool)
    for uid in active:
        keep |= source == uid
    if not bool(keep.any()):
        return torch.zeros_like(opacity)
    return opacity * keep.to(opacity.dtype)


@dataclass(slots=True)
class WorldRenderOutputs:
    full: torch.Tensor
    diffuse: torch.Tensor
    specular: torch.Tensor
    mask: torch.Tensor
    normal: torch.Tensor
    depth: torch.Tensor
    albedo: torch.Tensor
    projected_texture: torch.Tensor
    alpha: torch.Tensor
    implicit_residual: torch.Tensor | None = None


def _schlick_fresnel(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return f0 + (1 - f0) * (1 - cos_theta).clamp(0, 1).pow(5)


def _ggx_d(n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a2 = alpha.square()
    denom = n_dot_h.square() * (a2 - 1) + 1
    return a2 / (_PI * denom.square() + _EPS)


def _smith_g1(cosine: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    k = (alpha + 1).square() / 8
    return cosine / (cosine * (1 - k) + k + _EPS)


class GaussianBRDFRenderer(nn.Module):
    """Reference PyTorch renderer.

    It uses exact projected anisotropic covariance and front-to-back alpha
    compositing.  `chunk_size` bounds memory while preserving depth order.
    For large production scenes this interface can later be backed by gsplat
    without changing the scene or training API.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        min_covariance: float = 0.3,
        mask_threshold: float = 0.08,
        neural_light_samples: int = 32,
        backend: str = "auto",
        output_clamp: bool = True,
        shadow_mode: str = "none",
        shadow_strength: float = 1.0,
    ):
        super().__init__()
        if backend not in {"auto", "torch", "gsplat"}:
            raise ValueError("backend must be auto, torch, or gsplat")
        if shadow_mode not in {"none", "gaussian"}:
            raise ValueError("shadow_mode must be none or gaussian")
        self.chunk_size = chunk_size
        self.min_covariance = min_covariance
        self.mask_threshold = nn.Parameter(torch.tensor(float(mask_threshold)))
        self.neural_light_samples = neural_light_samples
        self.backend = backend
        self.output_clamp = output_clamp
        self.shadow_mode = shadow_mode
        self.shadow_strength = shadow_strength
        self._gsplat_failed = False

    def _direct_visibility(
        self,
        scene: GaussianBRDFField,
        directions: torch.Tensor,
        maximum_distance: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate light transmittance through anisotropic Gaussian density."""
        if self.shadow_mode == "none" or scene.n_gaussians <= 1:
            return torch.ones(
                scene.n_gaussians,
                1,
                device=scene.means.device,
                dtype=scene.means.dtype,
            )
        visibility = []
        means = scene.means
        radii = scene.scales.mean(dim=-1).clamp_min(1e-4)
        opacity = scene.opacity.squeeze(-1)
        for start in range(0, scene.n_gaussians, self.chunk_size):
            end = min(start + self.chunk_size, scene.n_gaussians)
            delta = means[None, :, :] - means[start:end, None, :]
            ray = directions[start:end, None, :]
            along = (delta * ray).sum(dim=-1)
            perpendicular = delta - along[..., None] * ray
            distance2 = perpendicular.square().sum(dim=-1)
            support = radii[None, :].square() * 2.0
            density = opacity[None, :] * torch.exp(-distance2 / support)
            valid = (along > radii[None, :] * 0.25) & (
                along < maximum_distance[start:end]
            )
            optical_depth = (density * valid).sum(dim=-1, keepdim=True)
            visibility.append(torch.exp(-self.shadow_strength * optical_depth))
        return torch.cat(visibility, dim=0)

    @staticmethod
    def gsplat_available() -> bool:
        try:
            from gsplat import rasterization  # noqa: F401

            return True
        except (ImportError, OSError):
            return False

    def active_backend(self, device: torch.device) -> str:
        if self.backend == "torch":
            return "torch"
        if self.backend == "gsplat":
            if device.type != "cuda":
                raise RuntimeError("gsplat backend requires CUDA tensors")
            if not self.gsplat_available():
                raise RuntimeError("gsplat backend requested but gsplat is unavailable")
            return "gsplat"
        return (
            "gsplat"
            if (
                device.type == "cuda"
                and not self._gsplat_failed
                and self.gsplat_available()
            )
            else "torch"
        )

    def _shade_sh(
        self,
        scene: GaussianBRDFField,
        materials: GaussianMaterials,
        camera: PerspectiveCamera,
        light: WorldLight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = scene.normals
        view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
        directional = F.normalize(light.direction, dim=-1).view(1, 3).expand_as(n)
        point_vector = light.position.view(1, 3) - scene.means
        point_distance = point_vector.norm(dim=-1, keepdim=True).clamp(min=0.1)
        point_direction = point_vector / point_distance
        point_weight = light.point_weight.view(1, 1)
        direction = F.normalize(
            directional * (1 - point_weight) + point_direction * point_weight,
            dim=-1,
        )
        n_dot_l = (n * direction).sum(-1, keepdim=True).clamp(min=0)
        n_dot_v = (n * view).sum(-1, keepdim=True).clamp(min=1e-4)

        # SH diffuse irradiance plus the dominant direct light.
        irradiance = sh_to_irradiance(
            n.T.unsqueeze(0).unsqueeze(-1), light.env_sh.unsqueeze(0)
        )[0, :, :, 0].T
        attenuation = (1 - point_weight) + point_weight / point_distance.square()
        maximum_distance = (
            (1 - point_weight) * torch.full_like(point_distance, float("inf"))
            + point_weight * point_distance
        )
        visibility = self._direct_visibility(scene, direction, maximum_distance)
        direct = (
            light.color.view(1, 3)
            * light.intensity.view(1, 1)
            * attenuation
            * n_dot_l
            * visibility
        )
        diffuse = materials.albedo / _PI * (irradiance + direct)

        half = F.normalize(direction + view, dim=-1)
        n_dot_h = (n * half).sum(-1, keepdim=True).clamp(min=0)
        h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
        alpha = materials.roughness.square()
        d_term = _ggx_d(n_dot_h, alpha)
        g_term = _smith_g1(n_dot_l, alpha) * _smith_g1(n_dot_v, alpha)
        f0 = 0.04 * (1 - materials.metalness) + materials.albedo * materials.metalness
        fresnel = _schlick_fresnel(h_dot_v, f0)
        spec_brdf = (
            materials.specular * d_term * g_term * fresnel
            / (4 * n_dot_l * n_dot_v + _EPS)
        )
        specular = spec_brdf * direct
        return diffuse, specular

    def _shade_neural(
        self,
        scene: GaussianBRDFField,
        materials: GaussianMaterials,
        camera: PerspectiveCamera,
        field: NeuralIncidentLightField,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo rendering-equation integration using Li(x, wi)."""
        g = scene.n_gaussians
        dirs = fibonacci_sphere(
            self.neural_light_samples, scene.means.device, scene.means.dtype
        )
        wi = dirs.view(1, -1, 3).expand(g, -1, -1)
        x = scene.means[:, None, :].expand_as(wi)
        li = field(x, wi)
        n = scene.normals[:, None, :]
        view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
        view = view[:, None, :]
        n_dot_l = (n * wi).sum(-1, keepdim=True).clamp(min=0)
        n_dot_v = (n * view).sum(-1, keepdim=True).clamp(min=1e-4)

        diffuse_brdf = materials.albedo[:, None, :] / _PI
        half = F.normalize(wi + view, dim=-1)
        n_dot_h = (n * half).sum(-1, keepdim=True).clamp(min=0)
        h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
        alpha = materials.roughness[:, None, :].square()
        d_term = _ggx_d(n_dot_h, alpha)
        g_term = _smith_g1(n_dot_l, alpha) * _smith_g1(n_dot_v, alpha)
        f0 = (
            0.04 * (1 - materials.metalness) + materials.albedo * materials.metalness
        )[:, None, :]
        fresnel = _schlick_fresnel(h_dot_v, f0)
        spec_brdf = (
            materials.specular[:, None, :]
            * d_term
            * g_term
            * fresnel
            / (4 * n_dot_l * n_dot_v + _EPS)
        )
        solid_angle = 4 * _PI / self.neural_light_samples
        diffuse = (li * diffuse_brdf * n_dot_l).sum(1) * solid_angle
        specular = (li * spec_brdf * n_dot_l).sum(1) * solid_angle
        return diffuse, specular

    def _shade_environment(
        self,
        scene: GaussianBRDFField,
        materials: GaussianMaterials,
        camera: PerspectiveCamera,
        light: WorldLight,
        neural_brdf: DualLatentNeuralBRDF | None = None,
        neural_brdf_strength: float = 1.0,
        gaussian_chunk: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate a known HDR environment with deterministic quadrature."""
        assert light.env_directions is not None
        assert light.env_radiance is not None
        assert light.env_solid_angle is not None
        directions = F.normalize(light.env_directions, dim=-1)
        radiance = light.env_radiance
        weights = light.env_solid_angle.view(1, -1, 1)
        diffuse_parts: list[torch.Tensor] = []
        specular_parts: list[torch.Tensor] = []
        if neural_brdf is not None:
            gaussian_chunk = min(gaussian_chunk, 256)
        for start in range(0, scene.n_gaussians, gaussian_chunk):
            stop = min(start + gaussian_chunk, scene.n_gaussians)
            n = scene.normals[start:stop, None, :]
            view = F.normalize(
                camera.center.view(1, 3) - scene.means[start:stop], dim=-1
            )[:, None, :]
            wi = directions.view(1, -1, 3)
            li = radiance.view(1, -1, 3)
            n_dot_l = (n * wi).sum(-1, keepdim=True).clamp(min=0)
            n_dot_v = (n * view).sum(-1, keepdim=True).clamp(min=1e-4)
            albedo = materials.albedo[start:stop, None, :]
            diffuse_brdf = albedo / _PI
            half = F.normalize(wi + view, dim=-1)
            n_dot_h = (n * half).sum(-1, keepdim=True).clamp(min=0)
            h_dot_v = (half * view).sum(-1, keepdim=True).clamp(min=0)
            roughness = materials.roughness[start:stop, None, :]
            alpha = roughness.square()
            d_term = _ggx_d(n_dot_h, alpha)
            g_term = _smith_g1(n_dot_l, alpha) * _smith_g1(n_dot_v, alpha)
            metalness = materials.metalness[start:stop, None, :]
            f0 = 0.04 * (1 - metalness) + albedo * metalness
            fresnel = _schlick_fresnel(h_dot_v, f0)
            spec_brdf = (
                materials.specular[start:stop, None, :]
                * d_term
                * g_term
                * fresnel
                / (4 * n_dot_l * n_dot_v + _EPS)
            )
            if neural_brdf is not None:
                material_features = torch.cat(
                    (
                        materials.albedo[start:stop],
                        materials.diffuse_weight[start:stop],
                        materials.specular[start:stop],
                        materials.roughness[start:stop],
                        materials.metalness[start:stop],
                        materials.absorption[start:stop],
                    ),
                    dim=-1,
                )
                material_latent = neural_brdf.encode_material(
                    scene.means[start:stop], material_features
                )
                sample_count = directions.shape[0]
                strength = float(neural_brdf_strength)
                if neural_brdf.mode == "log_residual":
                    luminance = radiance.mean(-1, keepdim=True).clamp_min(1e-6)
                    directional_color = radiance / luminance
                    environment_rgb = (
                        light.env_sh[0] * 0.282095
                    ).view(1, 3).expand(sample_count, -1)
                    light_latent = neural_brdf.encode_light(
                        directions,
                        luminance,
                        directional_color,
                        environment_rgb,
                    )
                    residual = neural_brdf.decode_log_specular_residual(
                        material_latent[:, None, :].expand(-1, sample_count, -1),
                        light_latent[None, :, :].expand(
                            stop - start, sample_count, -1
                        ),
                        n.expand(-1, sample_count, -1),
                        view.expand(-1, sample_count, -1),
                        wi.expand(stop - start, -1, -1),
                    )
                    if neural_brdf.highlight_gate:
                        residual = residual * (
                            n_dot_h.square() * (1 - roughness).clamp(0, 1)
                        )
                    spec_brdf = spec_brdf * torch.exp(strength * residual)
                else:
                    light_latent = neural_brdf.encode_light(
                        light.direction.view(1, 3),
                        light.intensity.view(1, 1),
                        light.color.view(1, 3),
                        (light.env_sh[0] * 0.282095).view(1, 3),
                    )
                    _, neural_specular = neural_brdf.decode(
                        material_latent[:, None, :].expand(-1, sample_count, -1),
                        light_latent.view(1, 1, -1).expand(
                            stop - start, sample_count, -1
                        ),
                        n.expand(-1, sample_count, -1),
                        view.expand(-1, sample_count, -1),
                        wi.expand(stop - start, -1, -1),
                    )
                    # Legacy replacement mode retained for old checkpoints.
                    spec_brdf = (
                        spec_brdf * (1 - strength) + neural_specular * strength
                    )
            diffuse_parts.append((li * diffuse_brdf * n_dot_l * weights).sum(1))
            specular_parts.append((li * spec_brdf * n_dot_l * weights).sum(1))
        return torch.cat(diffuse_parts), torch.cat(specular_parts)

    def _project_covariance(
        self, scene: GaussianBRDFField, camera: PerspectiveCamera
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points_camera = camera.world_to_camera(scene.means)
        uv, depth, valid = camera.project(scene.means)
        rotation_cw = camera.w2c[:3, :3]
        cov_world = scene.covariance_world()
        cov_camera = rotation_cw @ cov_world @ rotation_cw.T
        jacobian = camera.projection_jacobian(points_camera)
        cov_2d = jacobian @ cov_camera @ jacobian.transpose(-1, -2)
        eye = torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype)
        cov_2d = cov_2d + self.min_covariance * eye
        return uv, depth, valid, cov_2d

    def _splat_torch(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        attributes: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        uv, depth, valid, covariance = self._project_covariance(scene, camera)
        ids = torch.where(valid)[0]
        if ids.numel() == 0:
            empty = {
                name: torch.zeros(
                    value.shape[-1],
                    camera.height,
                    camera.width,
                    device=value.device,
                    dtype=value.dtype,
                )
                for name, value in attributes.items()
            }
            alpha = torch.zeros(
                1,
                camera.height,
                camera.width,
                device=scene.means.device,
                dtype=scene.means.dtype,
            )
            return empty, alpha

        ids = ids[torch.argsort(depth[ids])]
        ys, xs = torch.meshgrid(
            torch.arange(
                camera.height, device=scene.means.device, dtype=scene.means.dtype
            ),
            torch.arange(
                camera.width, device=scene.means.device, dtype=scene.means.dtype
            ),
            indexing="ij",
        )
        pixels = torch.stack([xs, ys], dim=-1)
        transmittance = torch.ones_like(xs)
        rendered = {
            name: torch.zeros(
                value.shape[-1],
                camera.height,
                camera.width,
                device=value.device,
                dtype=value.dtype,
            )
            for name, value in attributes.items()
        }

        for start in range(0, ids.numel(), self.chunk_size):
            chunk = ids[start : start + self.chunk_size]
            delta = pixels.unsqueeze(0) - uv[chunk, None, None, :]
            inv_cov = torch.linalg.inv(covariance[chunk])
            mahalanobis = torch.einsum(
                "ghwi,gij,ghwj->ghw", delta, inv_cov, delta
            )
            opacity = _view_locked_opacity(scene, camera)[chunk]
            alpha = (
                opacity[:, None, None]
                * torch.exp(-0.5 * mahalanobis.clamp(min=0))
            ).clamp(0, 0.99)
            prefix = torch.cumprod(
                torch.cat([torch.ones_like(alpha[:1]), 1 - alpha + 1e-7], dim=0),
                dim=0,
            )[:-1]
            weights = transmittance.unsqueeze(0) * prefix * alpha
            for name, value in attributes.items():
                rendered[name] = rendered[name] + torch.einsum(
                    "ghw,gc->chw", weights, value[chunk]
                )
            transmittance = transmittance * torch.prod(1 - alpha + 1e-7, dim=0)

        return rendered, (1 - transmittance).unsqueeze(0)

    def _splat_gsplat(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        attributes: dict[str, torch.Tensor],
        view_lock: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """CUDA tile rasterization of all G-buffers in one gsplat call."""
        # User-site installs on Windows do not always append Scripts to PATH,
        # although the ninja wheel installed ninja.exe there.
        if shutil.which("ninja") is None:
            try:
                import ninja

                os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ["PATH"]
            except ImportError:
                pass
        if os.name == "nt" and shutil.which("cl") is None:
            candidates = (
                glob.glob(
                    r"C:\Program Files\Microsoft Visual Studio\2022"
                    r"\*\VC\Auxiliary\Build\vcvars64.bat"
                )
                + glob.glob(
                    r"C:\Program Files (x86)\Microsoft Visual Studio\2019"
                    r"\*\VC\Auxiliary\Build\vcvars64.bat"
                )
            )
            if candidates:
                environment = subprocess.check_output(
                    f'call "{candidates[0]}" >nul && set',
                    shell=True,
                    text=True,
                    encoding="mbcs",
                )
                for line in environment.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value
        if os.name == "nt":
            compatibility_header = os.path.join(
                os.path.dirname(__file__), "gsplat_windows_compat.h"
            )
            nvcc_flags = os.environ.get("NVCC_PREPEND_FLAGS", "")
            preinclude = f'--pre-include "{compatibility_header}"'
            if preinclude not in nvcc_flags:
                os.environ["NVCC_PREPEND_FLAGS"] = (
                    f"{preinclude} {nvcc_flags}".strip()
                )
            # gsplat 1.5 emits GCC host flags even on Windows. Translate them
            # before its lazy backend imports PyTorch's private JIT helper.
            import torch.utils.cpp_extension as cpp_extension

            if not getattr(cpp_extension._jit_compile, "_mvbrdf_msvc", False):
                original_jit_compile = cpp_extension._jit_compile

                def msvc_jit_compile(*args, **kwargs):
                    positional = list(args)
                    if len(positional) > 2 and positional[2] is not None:
                        flags = []
                        for flag in positional[2]:
                            if flag == "-O3":
                                flags.append("/O2")
                            elif flag == "-O0":
                                flags.append("/Od")
                            elif not flag.startswith("-W"):
                                flags.append(flag)
                        positional[2] = flags
                    return original_jit_compile(*positional, **kwargs)

                msvc_jit_compile._mvbrdf_msvc = True
                cpp_extension._jit_compile = msvc_jit_compile
        from gsplat import rasterization

        names = [name for name in attributes if name != "depth"]
        widths = [attributes[name].shape[-1] for name in names]
        features = torch.cat([attributes[name] for name in names], dim=-1)
        if view_lock is False:
            opacities = scene.opacity[:, 0]
        else:
            opacities = _view_locked_opacity(scene, camera)
        rendered_features, rendered_alpha, _ = rasterization(
            means=scene.means,
            quats=scene.quaternions,
            scales=scene.scales,
            opacities=opacities,
            colors=features,
            viewmats=camera.w2c.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.width,
            height=camera.height,
            eps2d=self.min_covariance,
            packed=True,
            render_mode="RGB+ED",
            channel_chunk=32,
        )
        # gsplat output is (camera, H, W, channels); ED is the last channel.
        channels = rendered_features[0, ..., :-1]
        expected_depth = rendered_features[0, ..., -1:].permute(2, 0, 1)
        rendered: dict[str, torch.Tensor] = {}
        offset = 0
        for name, width in zip(names, widths):
            rendered[name] = channels[..., offset : offset + width].permute(2, 0, 1)
            offset += width
        alpha = rendered_alpha[0].permute(2, 0, 1)
        # gsplat's RGB+ED mode returns depth already normalized by alpha,
        # while the common path below normalizes accumulated attributes.
        rendered["depth"] = expected_depth * alpha
        return rendered, alpha

    def forward(
        self,
        scene: GaussianBRDFField,
        camera: PerspectiveCamera,
        light: WorldLight | None = None,
        neural_light: NeuralIncidentLightField | None = None,
        implicit_material: ImplicitMaterialImagingField | None = None,
        neural_brdf: DualLatentNeuralBRDF | None = None,
        neural_brdf_strength: float = 1.0,
    ) -> WorldRenderOutputs:
        if neural_light is None and light is None:
            raise ValueError("either light or neural_light is required")
        materials = scene.materials()
        if neural_light is not None:
            diffuse_g, specular_g = self._shade_neural(
                scene, materials, camera, neural_light
            )
        else:
            assert light is not None
            if light.env_directions is not None:
                diffuse_g, specular_g = self._shade_environment(
                    scene,
                    materials,
                    camera,
                    light,
                    neural_brdf=neural_brdf,
                    neural_brdf_strength=neural_brdf_strength,
                )
            else:
                diffuse_g, specular_g = self._shade_sh(
                    scene, materials, camera, light
                )
        if (
            neural_brdf is not None
            and light is not None
            and light.env_directions is None
        ):
            normal = scene.normals
            view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
            direction = F.normalize(light.direction, dim=-1).view(1, 3).expand_as(normal)
            material_features = torch.cat(
                (
                    materials.albedo,
                    materials.diffuse_weight,
                    materials.specular,
                    materials.roughness,
                    materials.metalness,
                    materials.absorption,
                ),
                dim=-1,
            )
            half_direction = F.normalize(direction + view, dim=-1)
            n_dot_l = (normal * direction).sum(-1, keepdim=True).clamp_min(0)
            n_dot_v = (normal * view).sum(-1, keepdim=True).clamp_min(1e-4)
            n_dot_h = (normal * half_direction).sum(-1, keepdim=True).clamp_min(0)
            h_dot_v = (half_direction * view).sum(-1, keepdim=True).clamp_min(0)
            alpha = materials.roughness.square()
            f0 = (
                0.04 * (1 - materials.metalness)
                + materials.albedo * materials.metalness
            )
            explicit_specular = (
                materials.specular
                * _ggx_d(n_dot_h, alpha)
                * _smith_g1(n_dot_l, alpha)
                * _smith_g1(n_dot_v, alpha)
                * _schlick_fresnel(h_dot_v, f0)
                / (4 * n_dot_l * n_dot_v + _EPS)
            )
            strength = float(neural_brdf_strength)
            if neural_brdf.mode == "log_residual":
                material_latent = neural_brdf.encode_material(
                    scene.means, material_features
                )
                light_latent = neural_brdf.encode_light(
                    direction,
                    light.intensity.view(1, 1).expand(scene.n_gaussians, -1),
                    light.color.view(1, 3).expand(scene.n_gaussians, -1),
                    (light.env_sh[0] * 0.282095)
                    .view(1, 3)
                    .expand(scene.n_gaussians, -1),
                )
                residual = neural_brdf.decode_log_specular_residual(
                    material_latent,
                    light_latent,
                    normal,
                    view,
                    direction,
                )
                if neural_brdf.highlight_gate:
                    residual = residual * (
                        n_dot_h.square()
                        * (1 - materials.roughness).clamp(0, 1)
                    )
                specular_g = specular_g * torch.exp(strength * residual)
            else:
                _, neural_specular, _ = neural_brdf(
                    scene.means,
                    normal,
                    view,
                    direction,
                    material_features,
                    light.intensity,
                    light.color,
                    light.env_sh[0] * 0.282095,
                )
                specular_ratio = (
                    neural_specular / explicit_specular.clamp_min(1e-4)
                ).clamp(0.5, 1.5)
                specular_g = specular_g * (1 + strength * (specular_ratio - 1))
        implicit_residual = None
        if implicit_material is not None and light is not None:
            view = F.normalize(camera.center.view(1, 3) - scene.means, dim=-1)
            material_features = torch.cat(
                (
                    materials.albedo,
                    materials.diffuse_weight,
                    materials.specular,
                    materials.roughness,
                    materials.metalness,
                    materials.absorption,
                ),
                dim=-1,
            )
            environment_rgb = light.env_sh[0] * 0.282095
            implicit_residual = implicit_material(
                scene.means,
                scene.normals,
                view,
                light.direction,
                material_features,
                environment_rgb,
                torch.log1p(light.intensity.clamp_min(0)),
            )
            diffuse_g = diffuse_g * torch.exp(implicit_residual[:, :3])
            specular_g = specular_g * torch.exp(implicit_residual[:, 3:])
        full_g = diffuse_g + specular_g
        _, depth_g, _, _ = self._project_covariance(scene, camera)
        appearance = {
            "full": full_g,
            "diffuse": diffuse_g,
            "specular": specular_g,
            "depth": depth_g[:, None],
            "albedo": materials.albedo,
            "texture": scene.texture,
        }
        geometry = {"normal": scene.normals}
        source_ids = getattr(scene, "source_view_id", None)
        split_normal = bool(getattr(scene, "view_locked_opacity", False)) and (
            source_ids is not None and int(source_ids.min()) >= 0
        )
        if self.active_backend(scene.means.device) == "gsplat":
            try:
                if split_normal:
                    rendered, alpha = self._splat_gsplat(
                        scene,
                        camera,
                        {**appearance, **geometry},
                        view_lock=True,
                    )
                    normal_denom = alpha.clamp(min=1e-6)
                else:
                    rendered, alpha = self._splat_gsplat(
                        scene, camera, {**appearance, **geometry}
                    )
                    normal_denom = alpha.clamp(min=1e-6)
            except Exception as error:
                if self.backend == "gsplat":
                    raise
                self._gsplat_failed = True
                warnings.warn(
                    f"gsplat unavailable at runtime ({error}); falling back to "
                    "the differentiable PyTorch rasterizer.",
                    RuntimeWarning,
                )
                rendered, alpha = self._splat_torch(
                    scene, camera, {**appearance, **geometry}
                )
                normal_denom = alpha.clamp(min=1e-6)
        else:
            rendered, alpha = self._splat_torch(
                scene, camera, {**appearance, **geometry}
            )
            normal_denom = alpha.clamp(min=1e-6)
        denom = alpha.clamp(min=1e-6)
        normal = F.normalize(rendered["normal"] / normal_denom, dim=0, eps=1e-6)
        depth = rendered["depth"] / denom
        threshold = self.mask_threshold.abs().clamp(min=1e-4)
        mask = (rendered["specular"].mean(0, keepdim=True) / threshold).clamp(0, 1)
        color = (
            (lambda value: value.clamp(0, 1))
            if self.output_clamp
            else (lambda value: value.clamp_min(0))
        )
        return WorldRenderOutputs(
            full=color(rendered["full"]),
            diffuse=color(rendered["diffuse"]),
            specular=color(rendered["specular"]),
            mask=mask,
            normal=normal,
            depth=depth,
            albedo=rendered["albedo"].clamp(0, 1),
            projected_texture=rendered["texture"].clamp(0, 1),
            alpha=alpha,
            implicit_residual=implicit_residual,
        )

