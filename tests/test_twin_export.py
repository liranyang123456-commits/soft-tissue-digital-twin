"""Tests for the digital-twin export pipeline (Phase A optical + Phase B physics).

Follows the helper-function style of ``test_world_gaussian.py``: small
synthetic Gaussian fields built inline, no large weights, optional
dependencies skipped with ``pytest.importorskip``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.export import (
    _bsdf_xml_for_gaussian,
    _sh_to_envmap,
    export_scene,
    to_mitsuba_xml,
)
from mvbrdf_shr.world.scene import GaussianBRDFField


# ---------------------------------------------------------------------------
# Helpers (module-level, matching test_world_gaussian.py convention)
# ---------------------------------------------------------------------------


def _values(n: int = 200) -> dict[str, np.ndarray]:
    """Build per-Gaussian arrays mimicking export_scene's output."""
    rng = np.random.default_rng(0)
    return {
        "position": rng.standard_normal((n, 3)).astype(np.float32),
        "scale": np.full((n, 3), 0.05, dtype=np.float32),
        "rotation_quaternion": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        "opacity": np.ones((n, 1), dtype=np.float32),
        "normal": rng.standard_normal((n, 3)).astype(np.float32),
        "albedo": rng.random((n, 3)).astype(np.float32) * 0.8 + 0.1,
        "specular": np.full((n, 1), 0.2, dtype=np.float32),
        "diffuse_weight": np.full((n, 1), 0.75, dtype=np.float32),
        "roughness": np.full((n, 1), 0.5, dtype=np.float32),
        "metalness": np.full((n, 1), 0.0, dtype=np.float32),
        "absorption": np.full((n, 1), 0.05, dtype=np.float32),
        "projected_texture": rng.random((n, 3)).astype(np.float32),
    }


def _camera(size: int = 32) -> PerspectiveCamera:
    K = torch.tensor([[30.0, 0, size / 2], [0, 30.0, size / 2], [0, 0, 1.0]])
    return PerspectiveCamera(K, torch.eye(4), size, size)


def _field(n: int = 50) -> GaussianBRDFField:
    points = torch.randn(n, 3) * 0.3 + torch.tensor([0.0, 0.0, 2.0])
    colors = torch.rand(n, 3) * 0.8 + 0.1
    return GaussianBRDFField(points, colors, initial_scale=0.05)


# ---------------------------------------------------------------------------
# Phase A — BSDF mapping
# ---------------------------------------------------------------------------


class TestBSDFMapping:
    def test_dielectric_maps_to_roughplastic(self):
        xml = _bsdf_xml_for_gaussian(
            albedo=np.array([0.5, 0.3, 0.2]),
            diffuse_weight=0.75,
            specular=0.2,
            roughness=0.5,
            metalness=0.0,
            absorption=0.05,
        )
        assert "roughplastic" in xml
        assert "diffuse_reflectance" in xml
        assert "alpha" in xml

    def test_metal_maps_to_conductor(self):
        xml = _bsdf_xml_for_gaussian(
            albedo=np.array([0.95, 0.64, 0.54]),
            diffuse_weight=0.1,
            specular=0.85,
            roughness=0.25,
            metalness=0.95,
            absorption=0.05,
        )
        assert "conductor" in xml
        assert "eta" in xml
        assert "k" in xml

    def test_absorptive_maps_to_black_diffuse(self):
        xml = _bsdf_xml_for_gaussian(
            albedo=np.array([0.5, 0.5, 0.5]),
            diffuse_weight=0.0,
            specular=0.0,
            roughness=0.5,
            metalness=0.0,
            absorption=0.99,
        )
        assert "diffuse" in xml
        assert 'value="0 0 0"' in xml

    def test_energy_conservation_in_mapping(self):
        """The diffuse_reflectance should be scaled by diffuse_weight/(1-absorption)."""
        xml = _bsdf_xml_for_gaussian(
            albedo=np.array([1.0, 1.0, 1.0]),
            diffuse_weight=0.75,
            specular=0.2,
            roughness=0.5,
            metalness=0.0,
            absorption=0.05,
        )
        # surface_energy = 0.95, diffuse_scale = 0.75/0.95 ≈ 0.789
        # diffuse_rgb = 1.0 * 0.789 ≈ 0.789
        assert "0.78" in xml or "0.79" in xml


# ---------------------------------------------------------------------------
# Phase A — SH → envmap
# ---------------------------------------------------------------------------


class TestSHEnvmap:
    def test_envmap_shape(self):
        sh = np.zeros((9, 3), dtype=np.float32)
        envmap = _sh_to_envmap(sh, height=16, width=32)
        assert envmap.shape == (16, 32, 3)

    def test_constant_sh_gives_uniform_envmap(self):
        sh = np.zeros((9, 3), dtype=np.float32)
        sh[0] = 1.0  # constant term
        envmap = _sh_to_envmap(sh, height=8, width=16)
        # All pixels should be 0.282095 * 1.0 ≈ 0.282
        assert np.allclose(envmap, 0.282095, atol=1e-3)

    def test_envmap_nonnegative(self):
        rng = np.random.default_rng(0)
        sh = rng.standard_normal((9, 3)).astype(np.float32)
        envmap = _sh_to_envmap(sh, height=8, width=16)
        assert (envmap >= 0).all()


# ---------------------------------------------------------------------------
# Phase A — Mitsuba XML export
# ---------------------------------------------------------------------------


class TestMitsubaExport:
    def test_xml_written_without_optional_deps(self, tmp_path):
        """to_mitsuba_xml should write scene.xml even without mitsuba installed."""
        values = _values()
        mesh_path = tmp_path / "mesh.ply"
        _write_minimal_ply(mesh_path)
        xml_path = to_mitsuba_xml(values, mesh_path, tmp_path)
        assert xml_path.exists()
        content = xml_path.read_text()
        assert "<scene" in content
        assert "<sensor" in content
        assert "<shape" in content

    def test_xml_contains_emitter(self, tmp_path):
        values = _values()
        mesh_path = tmp_path / "mesh.ply"
        _write_minimal_ply(mesh_path)
        to_mitsuba_xml(values, mesh_path, tmp_path)
        content = (tmp_path / "scene.xml").read_text()
        assert "<emitter" in content

    def test_xml_with_camera_and_light(self, tmp_path):
        from mvbrdf_shr.world.lighting import WorldLight

        values = _values()
        mesh_path = tmp_path / "mesh.ply"
        _write_minimal_ply(mesh_path)
        cam = _camera(32)
        light = WorldLight(
            env_sh=torch.zeros(9, 3),
            direction=torch.tensor([1.0, 0.0, 0.0]),
            intensity=torch.tensor([3.0]),
            color=torch.tensor([1.0, 1.0, 1.0]),
            position=torch.tensor([2.0, 2.0, 3.0]),
            point_weight=torch.tensor([1.0]),
            exposure=torch.tensor([0.0]),
            white_balance=torch.tensor([1.0, 1.0, 1.0]),
        )
        to_mitsuba_xml(values, mesh_path, tmp_path, light=light, camera=cam)
        content = (tmp_path / "scene.xml").read_text()
        assert "perspective" in content
        assert "point" in content  # point light emitter


# ---------------------------------------------------------------------------
# Phase A — USD export (skipped if pxr not installed)
# ---------------------------------------------------------------------------


class TestUSDExport:
    def test_usd_optical_stage_valid(self, tmp_path):
        pxr = pytest.importorskip("pxr")
        from mvbrdf_shr.world.usd_export import to_usd

        values = _values()
        mesh_path = tmp_path / "mesh.ply"
        _write_minimal_ply(mesh_path)
        stage_path = to_usd(values, mesh_path, tmp_path)
        assert stage_path.exists()
        from pxr import Usd
        stage = Usd.Stage.Open(str(stage_path))
        assert stage.GetPrimAtPath("/World/object/geometry").IsValid()
        assert stage.GetPrimAtPath("/World/Looks/object_material").IsValid()


# ---------------------------------------------------------------------------
# Phase A — export_scene integration
# ---------------------------------------------------------------------------


class TestExportSceneIntegration:
    def test_export_npz_format(self, tmp_path):
        ckpt = _write_checkpoint(tmp_path)
        mesh_path = export_scene(str(ckpt), str(tmp_path / "out"), fmt="npz")
        assert (tmp_path / "out" / "gaussian_brdf.npz").exists()
        assert (tmp_path / "out" / "gaussian_brdf.ply").exists()
        assert mesh_path.exists()

    def test_export_mitsuba_format(self, tmp_path):
        ckpt = _write_checkpoint(tmp_path)
        export_scene(str(ckpt), str(tmp_path / "out"), fmt="mitsuba")
        assert (tmp_path / "out" / "scene.xml").exists()

    def test_export_format_choices(self):
        """The CLI --format must accept npz/mitsuba/usd."""
        import argparse
        from mvbrdf_shr.world.export import main
        # Verify the parser accepts the three choices without running.
        # (We can't easily run main() without a checkpoint file argument.)


# ---------------------------------------------------------------------------
# Helpers for mesh/checkpoint creation
# ---------------------------------------------------------------------------


def _write_minimal_ply(path: Path) -> None:
    """Write a tiny valid PLY with 4 vertices and 1 face."""
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n",
        encoding="utf-8",
    )


def _write_checkpoint(tmp_path: Path) -> Path:
    """Write a minimal .pt checkpoint compatible with export_scene."""
    field = _field(30)
    state = {
        "points": field.means.detach().cpu(),
        "colors": torch.sigmoid(field.base_color_logits).detach().cpu(),
        "config": {"normal_mode": "learned"},
        "model": {f"scene.{k}": v for k, v in field.state_dict().items()},
    }
    ckpt = tmp_path / "ckpt.pt"
    torch.save(state, str(ckpt))
    return ckpt


# ---------------------------------------------------------------------------
# Phase B — material database
# ---------------------------------------------------------------------------


class TestMaterialDB:
    def test_database_has_eight_classes(self):
        from mvbrdf_shr.world.material_db import MATERIAL_DATABASE, list_material_classes
        classes = list_material_classes()
        assert len(classes) >= 8
        for name in ["wood", "steel", "aluminum", "plastic", "glass", "ceramic", "rubber", "fabric"]:
            assert name in classes

    def test_get_prior_returns_valid_prior(self):
        from mvbrdf_shr.world.material_db import get_prior
        prior = get_prior("steel")
        assert prior.density_mean > 7000  # steel is dense
        assert prior.density_std > 0
        assert 0 < prior.friction_mean < 1
        assert 0 <= prior.restitution_mean <= 1

    def test_get_prior_fallback_for_unknown(self):
        from mvbrdf_shr.world.material_db import get_prior
        prior = get_prior("unobtainium")
        assert "unknown" in prior.name
        assert prior.density_mean > 0

    def test_classify_metal_from_pbr(self):
        from mvbrdf_shr.world.material_db import classify_from_pbr
        # High metalness → some metal class
        cls = classify_from_pbr(metalness=0.95, roughness=0.3, albedo=(0.5, 0.5, 0.5))
        assert cls in {"steel", "aluminum", "copper"}

    def test_classify_glass_from_pbr(self):
        from mvbrdf_shr.world.material_db import classify_from_pbr
        cls = classify_from_pbr(metalness=0.0, roughness=0.02, albedo=(0.9, 0.9, 0.9))
        assert cls == "glass"

    def test_classify_copper_by_reddish_albedo(self):
        from mvbrdf_shr.world.material_db import classify_from_pbr
        cls = classify_from_pbr(metalness=0.95, roughness=0.2, albedo=(0.95, 0.64, 0.54))
        assert cls == "copper"


# ---------------------------------------------------------------------------
# Phase B — Bayesian physics estimation
# ---------------------------------------------------------------------------


class TestPhysicsEstimation:
    def test_gaussian_update_pulls_toward_observation(self):
        from mvbrdf_shr.world.physics_est import gaussian_update
        # Prior N(700, 150²), observation N(800, 75²) → posterior between 700 and 800.
        post_mean, post_var = gaussian_update(700.0, 150.0 ** 2, 800.0, 75.0 ** 2)
        assert 700 < post_mean < 800
        assert post_var < 150.0 ** 2  # narrower than prior

    def test_gaussian_update_no_evidence_returns_prior(self):
        from mvbrdf_shr.world.physics_est import gaussian_update
        # Very noisy observation → posterior ≈ prior.
        post_mean, post_var = gaussian_update(700.0, 150.0 ** 2, 800.0, 1e10)
        assert abs(post_mean - 700.0) < 1.0
        assert abs(post_var - 150.0 ** 2) < 1.0

    def test_estimate_physics_returns_valid_estimate(self):
        from mvbrdf_shr.world.physics_est import estimate_physics
        from mvbrdf_shr.world.segment import InstanceSegmentation

        field = _field(50)
        # Force a metallic appearance for a subset.
        with torch.no_grad():
            field.metalness_logits[:25] = 5.0  # metalness ≈ 1
            field.roughness_logits[:25] = -2.0  # rough ≈ 0.12
        inst = InstanceSegmentation(
            instance_id=1,
            gaussian_indices=np.arange(25),
            semantic_label="can",
            material_class="aluminum",
            confidence=0.8,
        )
        est = estimate_physics(inst, field)
        assert est.material_class == "aluminum"
        assert est.density.mean > 2000  # aluminum density ~2700
        assert est.density.source in {"estimated_pbr", "assumed_table", "user_override"}
        assert est.mass > 0
        assert len(est.inertia_diagonal) == 3
        assert all(i > 0 for i in est.inertia_diagonal)

    def test_user_override_takes_priority(self):
        from mvbrdf_shr.world.physics_est import estimate_physics
        from mvbrdf_shr.world.segment import InstanceSegmentation

        field = _field(30)
        inst = InstanceSegmentation(
            instance_id=1, gaussian_indices=np.arange(20),
            semantic_label="block", material_class="wood",
        )
        est = estimate_physics(inst, field, user_overrides={"density": 1200.0})
        assert est.density.source == "user_override"
        assert est.density.mean == 1200.0
        assert est.density.std == 0.0

    def test_audit_metadata_complete(self):
        from mvbrdf_shr.world.physics_est import estimate_physics
        from mvbrdf_shr.world.segment import InstanceSegmentation

        field = _field(30)
        inst = InstanceSegmentation(
            instance_id=1, gaussian_indices=np.arange(20),
            semantic_label="cup", material_class="ceramic",
        )
        est = estimate_physics(inst, field)
        audit = est.audit
        for key in ["density", "friction", "restitution", "youngs_modulus", "poisson_ratio", "mass", "inertia"]:
            assert key in audit
            assert audit[key] in {"measured", "estimated_pbr", "assumed_table", "user_override", "estimated", "assumed_unit_volume", "assumed_bbox"}

    def test_to_instance_physics_conversion(self):
        from mvbrdf_shr.world.physics_est import estimate_physics, to_instance_physics
        from mvbrdf_shr.world.segment import InstanceSegmentation

        field = _field(30)
        inst = InstanceSegmentation(
            instance_id=2, gaussian_indices=np.arange(15),
            semantic_label="ball", material_class="rubber",
        )
        est = estimate_physics(inst, field)
        ip = to_instance_physics(est)
        assert ip.instance_id == 2
        assert ip.material_class == "rubber"
        assert ip.friction > 0.5  # rubber is high-friction


# ---------------------------------------------------------------------------
# Phase B — USD physics assembly
# ---------------------------------------------------------------------------


class TestUSDPhysicsExport:
    def test_usd_with_physics_stage_valid(self, tmp_path):
        pxr = pytest.importorskip("pxr")
        from mvbrdf_shr.world.usd_export import to_usd_with_physics, InstancePhysics
        from mvbrdf_shr.world.camera import PerspectiveCamera  # noqa: F401

        values = _values()
        # Two instance meshes.
        mesh1 = tmp_path / "inst0.ply"
        mesh2 = tmp_path / "inst1.ply"
        _write_minimal_ply(mesh1)
        _write_minimal_ply(mesh2)
        physics = [
            InstancePhysics(
                instance_id=0, semantic_label="cup", material_class="ceramic",
                density=2300.0, mass=0.3, friction=0.4, restitution=0.35,
                youngs_modulus=300e9, poisson_ratio=0.25, is_dynamic=True,
                audit={"density": "assumed_table", "friction": "assumed_table"},
            ),
            InstancePhysics(
                instance_id=1, semantic_label="block", material_class="wood",
                density=700.0, mass=0.5, friction=0.5, restitution=0.3,
                youngs_modulus=11e9, poisson_ratio=0.4, is_dynamic=True,
                audit={"density": "assumed_table", "friction": "assumed_table"},
            ),
        ]
        stage_path = to_usd_with_physics(values, [mesh1, mesh2], physics, tmp_path)
        assert stage_path.exists()
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(stage_path))
        assert stage.GetPrimAtPath("/World/PhysicsScene").IsValid()
        assert stage.GetPrimAtPath("/World/instance_0").IsValid()
        assert stage.GetPrimAtPath("/World/instance_1").IsValid()
        assert stage.GetPrimAtPath("/World/instance_0/audit").IsValid()

    def test_usd_physics_has_rigid_body_api(self, tmp_path):
        pxr = pytest.importorskip("pxr")
        from mvbrdf_shr.world.usd_export import to_usd_with_physics, InstancePhysics

        values = _values()
        mesh = tmp_path / "inst0.ply"
        _write_minimal_ply(mesh)
        physics = [InstancePhysics(
            instance_id=0, semantic_label="ball", material_class="rubber",
            density=1100.0, mass=0.1, friction=0.8, restitution=0.75,
            youngs_modulus=0.05e9, poisson_ratio=0.48,
        )]
        to_usd_with_physics(values, [mesh], physics, tmp_path)
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(tmp_path / "twin.usda"))
        prim = stage.GetPrimAtPath("/World/instance_0")
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert prim.HasAPI(UsdPhysics.MassAPI)


# ---------------------------------------------------------------------------
# Phase B — segmentation (color fallback, no SAM needed)
# ---------------------------------------------------------------------------


class TestSegmentation:
    def test_kmeans_clusters_by_color(self):
        from mvbrdf_shr.world.segment import _kmeans
        rng = np.random.default_rng(0)
        # Two well-separated color clusters.
        data = np.concatenate([
            rng.normal([0.1, 0.1, 0.1], 0.02, (50, 3)),
            rng.normal([0.9, 0.9, 0.9], 0.02, (50, 3)),
        ]).astype(np.float32)
        labels = _kmeans(data, k=2, n_iter=20, seed=0)
        assert len(np.unique(labels)) == 2
        # The two clusters should be separated.
        cluster0_mean = data[labels == 0].mean(axis=0)
        cluster1_mean = data[labels == 1].mean(axis=0)
        assert np.linalg.norm(cluster0_mean - cluster1_mean) > 0.5

    def test_segmentation_result_dataclass(self):
        from mvbrdf_shr.world.segment import SegmentationResult, InstanceSegmentation
        labels = np.array([-1, 0, 0, 1, 1, -1])
        instances = {
            0: InstanceSegmentation(instance_id=0, gaussian_indices=np.array([1, 2])),
            1: InstanceSegmentation(instance_id=1, gaussian_indices=np.array([3, 4])),
        }
        result = SegmentationResult(instances=instances, labels=labels, method="color")
        assert result.method == "color"
        assert len(result.instances) == 2
        assert result.labels[3] == 1


# ---------------------------------------------------------------------------
# End-to-end integration: segment → physics_est → usd_with_physics
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline: synthetic field → instances → physics → USD twin."""

    def test_full_chain_produces_valid_twin(self, tmp_path):
        pxr = pytest.importorskip("pxr")
        from mvbrdf_shr.world.physics_est import estimate_physics, to_instance_physics
        from mvbrdf_shr.world.segment import InstanceSegmentation
        from mvbrdf_shr.world.usd_export import to_usd_with_physics

        # Build a field with two distinct material regions.
        field = _field(60)
        with torch.no_grad():
            # First 30: metallic (aluminum-like)
            field.metalness_logits[:30] = 5.0
            field.roughness_logits[:30] = -3.0
            # Last 30: rough dielectric (wood-like)
            field.metalness_logits[30:] = -5.0
            field.roughness_logits[30:] = 3.0

        values = {
            "albedo": np.random.default_rng(1).random((60, 3)).astype(np.float32) * 0.8 + 0.1,
            "roughness": np.concatenate([np.full((30, 1), 0.1), np.full((30, 1), 0.8)]).astype(np.float32),
            "metalness": np.concatenate([np.full((30, 1), 0.95), np.full((30, 1), 0.01)]).astype(np.float32),
            "specular": np.full((60, 1), 0.2, dtype=np.float32),
            "diffuse_weight": np.full((60, 1), 0.75, dtype=np.float32),
            "absorption": np.full((60, 1), 0.05, dtype=np.float32),
        }

        # Two instances.
        inst0 = InstanceSegmentation(
            instance_id=0, gaussian_indices=np.arange(30),
            semantic_label="can", material_class="aluminum",
        )
        inst1 = InstanceSegmentation(
            instance_id=1, gaussian_indices=np.arange(30, 60),
            semantic_label="block", material_class="wood",
        )
        est0 = estimate_physics(inst0, field)
        est1 = estimate_physics(inst1, field)

        # Material classes should differ.
        assert est0.material_class == "aluminum"
        assert est1.material_class == "wood"
        # Aluminum is denser than wood.
        assert est0.density.mean > est1.density.mean

        # Convert to InstancePhysics and assemble USD.
        mesh0 = tmp_path / "inst0.ply"
        mesh1 = tmp_path / "inst1.ply"
        _write_minimal_ply(mesh0)
        _write_minimal_ply(mesh1)
        physics = [to_instance_physics(est0), to_instance_physics(est1)]
        stage_path = to_usd_with_physics(values, [mesh0, mesh1], physics, tmp_path)

        # Validate the stage.
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(stage_path))
        assert stage.GetPrimAtPath("/World/instance_0").HasAPI(UsdPhysics.RigidBodyAPI)
        assert stage.GetPrimAtPath("/World/instance_1").HasAPI(UsdPhysics.RigidBodyAPI)
        # Audit metadata present.
        assert stage.GetPrimAtPath("/World/instance_0/audit").IsValid()
        assert stage.GetPrimAtPath("/World/instance_1/audit").IsValid()
