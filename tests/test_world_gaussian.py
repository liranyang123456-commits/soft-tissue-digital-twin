from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from mvbrdf_shr.models.implicit_material_field import ImplicitMaterialImagingField
from mvbrdf_shr.world.camera import PerspectiveCamera
from mvbrdf_shr.world.coords import CameraConvention, convert_camera_pose
from mvbrdf_shr.world.data import MultiViewFrame, MultiViewScene
from mvbrdf_shr.world.datasets.canonical import (
    CanonicalCamera,
    CanonicalDataset,
    CanonicalFrame,
)
from mvbrdf_shr.world.fusion import HybridMVBRDFSHR
from mvbrdf_shr.world.geometry_init import _image_photometric_normal_map
from mvbrdf_shr.world.evaluate_ir import evaluate_inverse_rendering
from mvbrdf_shr.world.lighting import (
    NeuralIncidentLightField,
    PerFrameLighting,
    latlong_dominant_light,
    latlong_to_sh,
)
from mvbrdf_shr.world.pipeline import WorldGaussianBRDFPipeline
from mvbrdf_shr.world.renderer import (
    GaussianBRDFRenderer,
    WorldRenderOutputs,
    _view_locked_opacity,
)
from mvbrdf_shr.world.schema import (
    AssetPaths,
    CameraRecord,
    FrameRecord,
    GroundTruthStatus,
    SceneManifest,
)
from mvbrdf_shr.world.scene import GaussianBRDFField
from mvbrdf_shr.world.train import (
    build_pipeline,
    load_dataset,
    sample_frame_indices,
    split_frame_indices,
    split_scene_indices,
)


def camera(size: int = 32) -> PerspectiveCamera:
    K = torch.tensor(
        [[30.0, 0, size / 2], [0, 30.0, size / 2], [0, 0, 1.0]]
    )
    return PerspectiveCamera(K, torch.eye(4), size, size)


def scene() -> GaussianBRDFField:
    points = torch.tensor(
        [
            [-0.25, -0.2, 2.0],
            [0.25, -0.2, 2.0],
            [-0.2, 0.25, 2.1],
            [0.2, 0.25, 2.1],
        ]
    )
    colors = torch.tensor(
        [[0.8, 0.2, 0.1], [0.2, 0.8, 0.1], [0.1, 0.2, 0.8], [0.7, 0.7, 0.2]]
    )
    return GaussianBRDFField(points, colors, initial_scale=0.12)


def test_view_lock_holdout_uses_nearest_source_instead_of_all():
    field = scene()
    field.register_buffer("source_view_id", torch.tensor([1, 1, 4, 4]))
    field.view_locked_opacity = True
    field.view_lock_neighbors = 0
    cam = camera()
    cam.view_id = 0

    locked = _view_locked_opacity(field, cam)

    assert torch.count_nonzero(locked).item() == 2
    assert torch.allclose(locked[:2], field.opacity[:2, 0])
    assert torch.count_nonzero(locked[2:]).item() == 0


def test_explicitly_disabled_view_lock_preserves_all_opacities():
    field = scene()
    field.register_buffer("source_view_id", torch.tensor([1, 1, 4, 4]))
    field.view_locked_opacity = False
    cam = camera()
    cam.view_id = 0

    unlocked = _view_locked_opacity(field, cam)

    assert torch.allclose(unlocked, field.opacity[:, 0])


def test_soft_view_lock_prefers_angularly_near_source():
    field = scene()
    field.register_buffer("source_view_id", torch.tensor([0, 0, 1, 1]))
    field.register_buffer(
        "source_camera_centers",
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    field.register_buffer("source_confidence", torch.ones(4, 1))
    field.view_locked_opacity = True
    field.view_lock_mode = "soft"
    field.view_lock_temperature = 0.5
    field.view_lock_min_weight = 0.0
    cam = camera()
    cam.view_id = 0

    locked = _view_locked_opacity(field, cam)

    assert torch.all(locked[:2] > locked[2:])
    assert torch.all(locked > 0)
    assert torch.all(locked <= field.opacity[:, 0])


def test_robust_image_ps_rejects_saturated_observation(tmp_path):
    directions = torch.tensor(
        [
            [-0.5, 0.0, -1.0],
            [0.5, 0.0, -1.0],
            [0.0, -0.5, -1.0],
            [0.0, 0.5, -1.0],
            [-0.4, -0.4, -1.0],
            [0.4, -0.4, -1.0],
            [-0.4, 0.4, -1.0],
            [0.4, 0.4, -1.0],
        ],
        dtype=torch.float32,
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    target = torch.tensor([0.0, 0.0, -1.0])
    frames = []
    for index, direction in enumerate(directions):
        intensity = float(torch.dot(direction, target))
        if index == 0:
            intensity = 1.0
        frames.append(
            MultiViewFrame(
                image=torch.full((3, 8, 8), intensity),
                camera=camera(8),
                image_path=tmp_path / f"{index}.png",
                mask_gt=torch.ones(1, 8, 8),
                light_direction=direction,
                view_id=0,
                light_id=index,
            )
        )
    dataset = MultiViewScene(frames)

    normal, confidence, _ = _image_photometric_normal_map(
        dataset,
        list(range(len(frames))),
        torch.device("cpu"),
        face_camera=True,
        lower_quantile=0.0,
        upper_quantile=1.0,
        irls_iterations=3,
        saturation_threshold=0.95,
    )

    assert normal[2].mean() < -0.95
    assert confidence.mean() > 0.5


def test_camera_projection_and_rays():
    cam = camera()
    uv, depth, valid = cam.project(torch.tensor([[0.0, 0.0, 2.0]]))
    assert torch.allclose(uv[0], torch.tensor([16.0, 16.0]))
    assert depth.item() == 2.0
    assert valid.item()
    origins, directions = cam.pixel_rays()
    assert origins.shape == directions.shape == (32, 32, 3)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(32, 32), atol=1e-5)


def test_coordinate_conversion_and_manifest_roundtrip(tmp_path):
    identity = torch.eye(4)
    opencv = convert_camera_pose(
        identity, CameraConvention.OPENGL, CameraConvention.OPENCV
    )
    assert torch.equal(opencv.diag(), torch.tensor([1.0, -1.0, -1.0, 1.0]))
    assert torch.allclose(
        convert_camera_pose(
            opencv, CameraConvention.OPENCV, CameraConvention.OPENGL
        ),
        identity,
    )
    manifest = SceneManifest(
        scene_id="fixture",
        frames=[
            FrameRecord(
                frame_id="v000_l000",
                split="train",
                view_id=0,
                assets=AssetPaths(full="images/000.png"),
                camera=CameraRecord(
                    intrinsic=np.eye(3).tolist(),
                    camera_to_world=np.eye(4).tolist(),
                    width=8,
                    height=8,
                ),
            )
        ],
        gt_status={"albedo": GroundTruthStatus.PSEUDO},
    )
    path = tmp_path / "scene.json"
    manifest.to_json(path)
    restored = SceneManifest.from_json(path)
    assert restored.scene_id == "fixture"
    assert restored.gt_status["albedo"] == GroundTruthStatus.PSEUDO


def test_material_energy_is_conserved():
    field = scene()
    assert torch.allclose(field.energy_sum(), torch.ones(4, 1), atol=1e-6)
    materials = field.materials()
    assert materials.albedo.shape == (4, 3)
    assert materials.roughness.min() >= 0.04


def test_constant_latlong_projects_to_dc_spherical_harmonic():
    coefficients = latlong_to_sh(torch.ones(3, 64, 128))
    assert torch.allclose(
        coefficients[0], torch.full((3,), 3.5449), atol=2e-3
    )
    assert coefficients[1:].abs().max() < 2e-3


def test_latlong_dominant_light_uses_stanford_world_convention():
    envmap = torch.zeros(3, 32, 64)
    envmap[:, 15:17, :2] = torch.tensor([4.0, 2.0, 1.0])[:, None, None]
    direction, color, intensity = latlong_dominant_light(envmap)

    assert direction[2] > 0.9
    assert torch.allclose(color, torch.tensor([1.0, 0.5, 0.25]), atol=1e-4)
    assert intensity > 0


def test_point_cloud_colors_initialize_rendered_albedo():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    colors = torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.2, 0.1]])
    field = GaussianBRDFField.from_point_cloud(points, colors, initial_scale=0.1)
    assert torch.allclose(field.materials().albedo, colors, atol=1e-4)


def test_mesh_normals_initialize_shading_and_gaussian_axes():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    field = GaussianBRDFField.from_point_cloud(points, initial_scale=0.1)
    field.initialize_surface_normals(normals)

    assert torch.allclose(field.normals, normals, atol=1e-6)
    local_z = field.rotations[:, :, 2]
    assert torch.allclose(local_z, normals, atol=1e-6)


def test_covariance_normal_mode_uses_short_axis_and_anisotropic_scale():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    scales = torch.tensor([[0.1, 0.1, 0.01], [0.2, 0.2, 0.02]])
    normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    field = GaussianBRDFField.from_point_cloud(
        points, initial_scale=scales, normal_mode="covariance"
    )
    field.initialize_surface_normals(normals)

    assert torch.allclose(field.scales, scales)
    assert torch.allclose(field.normals, normals, atol=1e-6)


def test_mesh_initialization_samples_surface_with_anisotropic_scales(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.Trimesh(
        vertices=[
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 2.0],
            [-1.0, 1.0, 2.5],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    mesh_path = tmp_path / "surface.ply"
    mesh.export(mesh_path)
    canonical = CanonicalDataset(
        frames=[
            CanonicalFrame(
                image=torch.ones(3, 8, 8),
                camera=CanonicalCamera(
                    torch.tensor([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1.0]]),
                    torch.eye(4),
                    8,
                    8,
                ),
                image_path=tmp_path / "unused.png",
                view_id=0,
                mask=torch.ones(1, 8, 8),
            )
        ],
        name="fixture",
        root=tmp_path,
        mesh_path=mesh_path,
    )
    sampled = MultiViewScene.from_canonical(
        canonical, use_mesh_points=True, max_mesh_points=100
    )

    assert sampled.points is not None and sampled.points.shape == (100, 3)
    assert sampled.point_normals is not None
    assert sampled.point_scales is not None
    assert torch.all(sampled.point_scales[:, 2] < sampled.point_scales[:, 0])
    assert torch.unique(sampled.points, dim=0).shape[0] > len(mesh.vertices)


def test_fixed_budget_densify_replaces_low_opacity_splats():
    field = scene()
    original = field.means.detach().clone()
    with torch.no_grad():
        field.opacity_logits[0].fill_(-20)
    field.means.grad = torch.zeros_like(field.means)
    field.means.grad[-1].fill_(10)
    slots = field.reallocate_gaussians(fraction=0.24)

    assert slots.tolist() == [0]
    assert not torch.allclose(field.means[0], original[0])
    assert torch.linalg.vector_norm(field.means[0] - original[-1]) < 0.2


def test_geometry_npz_initialization_and_frame_split(tmp_path):
    points = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    field = GaussianBRDFField.from_point_cloud(points, initial_scale=0.1)
    scale = np.full((2, 3), 0.03, dtype=np.float32)
    quaternion = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    normal = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
    prior = tmp_path / "prior.npz"
    np.savez(
        prior,
        position=points.numpy(),
        scale=scale,
        quaternion=quaternion,
        normal=normal,
    )
    field.initialize_from_npz(str(prior), mode="geometry")
    assert torch.allclose(field.scales, torch.from_numpy(scale))
    assert torch.allclose(field.normals, torch.from_numpy(normal))
    train, holdout = split_frame_indices(
        12, {"holdout_stride": 4, "holdout_offset": 0}
    )
    assert holdout == [0, 4, 8]
    assert set(train).isdisjoint(holdout)
    assert sorted(train + holdout) == list(range(12))


def test_structured_multiview_sampling(tmp_path):
    frames = [
        MultiViewFrame(
            image=torch.zeros(3, 8, 8),
            camera=camera(8),
            image_path=tmp_path / f"{index}.png",
            view_id=index // 2,
            light_id=index % 2,
        )
        for index in range(6)
    ]
    dataset = MultiViewScene(frames)
    same_view = sample_frame_indices(dataset, list(range(6)), 2, "same_view")
    assert len({dataset[index].view_id for index in same_view}) == 1
    same_light = sample_frame_indices(dataset, list(range(6)), 3, "same_light")
    assert len({dataset[index].light_id for index in same_light}) == 1
    distinct = sample_frame_indices(dataset, list(range(6)), 3, "distinct_views")
    assert len({dataset[index].view_id for index in distinct}) == 3
    train, holdout = split_scene_indices(
        dataset, {"holdout_view_stride": 3, "holdout_view_offset": 0}
    )
    assert {dataset[index].view_id for index in holdout} == {0}
    assert {dataset[index].view_id for index in train} == {1, 2}
    train, holdout = split_scene_indices(
        dataset, {"holdout_light_stride": 2, "holdout_light_offset": 0}
    )
    assert {dataset[index].light_id for index in holdout} == {0}
    assert {dataset[index].light_id for index in train} == {1}
    assert {dataset[index].view_id for index in holdout} == {0, 1, 2}
    assert {dataset[index].view_id for index in train} == {0, 1, 2}


def test_anisotropic_renderer_outputs_and_gradients():
    field = scene()
    cam = camera()
    lighting = PerFrameLighting(1)
    renderer = GaussianBRDFRenderer(chunk_size=2)
    output = renderer(field, cam, light=lighting(0))
    assert output.full.shape == (3, 32, 32)
    assert output.diffuse.shape == output.specular.shape == output.full.shape
    assert output.depth.shape == output.mask.shape == (1, 32, 32)
    assert output.normal.shape == (3, 32, 32)
    loss = output.full.mean() + output.diffuse.mean()
    loss.backward()
    assert field.means.grad is not None
    assert field.base_color_logits.grad is not None


def test_gaussian_shadow_visibility_detects_occluder():
    points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.4]])
    field = GaussianBRDFField.from_point_cloud(
        points, torch.full((2, 3), 0.5), initial_scale=0.2
    )
    renderer = GaussianBRDFRenderer(
        chunk_size=2, shadow_mode="gaussian", shadow_strength=2.0
    )
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    distance = torch.full((2, 1), float("inf"))
    visibility = renderer._direct_visibility(field, directions, distance)
    assert visibility[0] < visibility[1]
    assert 0 < visibility.min() <= visibility.max() <= 1


def test_neilf_renderer_and_pipeline():
    field = scene()
    cam = camera(24)
    neilf = NeuralIncidentLightField(hidden=32, n_freqs=2)
    renderer = GaussianBRDFRenderer(chunk_size=4, neural_light_samples=8)
    output = renderer(field, cam, neural_light=neilf)
    assert output.full.shape == (3, 24, 24)
    pipeline = WorldGaussianBRDFPipeline(
        field, n_frames=1, lighting_mode="sh", use_refiner=False, chunk_size=4
    )
    result = pipeline(cam, 0, image=torch.zeros(3, 24, 24), refine=False)
    assert torch.allclose(result["pred"], result["render"].diffuse)


def test_implicit_material_is_bounded_zero_initialized_and_trainable():
    field = scene()
    cam = camera(24)
    baseline = WorldGaussianBRDFPipeline(
        field, n_frames=1, use_refiner=False, chunk_size=4
    )
    implicit = WorldGaussianBRDFPipeline(
        field,
        n_frames=1,
        use_refiner=False,
        chunk_size=4,
        use_implicit_material=True,
        implicit_material_residual_limit=0.05,
    )
    implicit.frame_lighting.load_state_dict(baseline.frame_lighting.state_dict())
    expected = baseline(cam, 0, refine=False)["render"].full
    result = implicit(cam, 0, refine=False)
    assert torch.allclose(result["render"].full, expected, atol=1e-6)
    residual = result["implicit_residual"]
    assert residual is not None
    assert torch.count_nonzero(residual) == 0

    loss = (result["render"].full - torch.rand_like(expected)).square().mean()
    loss.backward()
    module = implicit.implicit_material
    assert isinstance(module, ImplicitMaterialImagingField)
    output_layer = module.network[-1]
    assert output_layer.weight.grad is not None
    assert output_layer.weight.grad.abs().sum() > 0
    assert field.means.grad is not None


def test_hdr_pipeline_applies_exposure_without_clipping():
    field = scene()
    cam = camera(24)
    pipeline = WorldGaussianBRDFPipeline(
        field,
        n_frames=1,
        lighting_mode="sh",
        use_refiner=False,
        chunk_size=4,
        linear_hdr=True,
    )
    with torch.no_grad():
        pipeline.frame_lighting.log_intensity.fill_(2.0)
        pipeline.frame_lighting.log_exposure.fill_(1.0)
        pipeline.frame_lighting.log_white_balance[0].copy_(
            torch.tensor([0.2, -0.1, -0.1])
        )
    result = pipeline(cam, 0, refine=False)
    assert result["render"].full.max() > 1.0
    exposure, white_balance = pipeline.frame_lighting.sensor(0)
    assert exposure.item() == pytest.approx(np.e)
    assert white_balance.prod().item() == pytest.approx(1.0, rel=1e-5)


def test_repeated_light_ids_share_light_but_not_exposure():
    lighting = PerFrameLighting(3, light_ids=[5, 7, 5])
    assert lighting.env_sh.shape[0] == 2
    assert lighting.light_index(0) == lighting.light_index(2)
    assert lighting.light_index(0) != lighting.light_index(1)
    with torch.no_grad():
        lighting.log_exposure[2].fill_(1.0)
    assert torch.allclose(lighting(0).direction, lighting(2).direction)
    assert lighting(0).exposure != lighting(2).exposure


def test_hybrid_falls_back_to_single_and_rejects_invisible_world():
    class Single(torch.nn.Module):
        def forward(self, image, **_):
            mask = torch.full_like(image[:, :1], 0.2)
            return {
                "pred": image * 0.5,
                "spec_pred": image * 0.1,
                "mask_pred": mask,
            }

    hybrid = HybridMVBRDFSHR(single=Single(), gate_hidden=8)
    image = torch.rand(1, 3, 8, 8)
    single = hybrid(image)
    assert single["mode"] == "single"
    assert torch.allclose(single["pred"], image * 0.5)
    zeros3 = torch.zeros(3, 8, 8)
    zeros1 = torch.zeros(1, 8, 8)
    world = WorldRenderOutputs(
        full=zeros3,
        diffuse=zeros3,
        specular=zeros3,
        mask=zeros1,
        normal=zeros3,
        depth=zeros1,
        albedo=zeros3,
        projected_texture=zeros3,
        alpha=zeros1,
    )
    fused = hybrid(image, world)
    assert fused["mode"] == "hybrid"
    assert float(fused["gate"].detach().max()) < 1e-3
    assert torch.allclose(fused["pred"], image * 0.5, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gsplat_matches_reference_and_has_gradients():
    if not GaussianBRDFRenderer.gsplat_available():
        pytest.skip("gsplat unavailable")
    field = scene().cuda()
    cam = camera().to("cuda")
    lighting = PerFrameLighting(1).cuda()
    reference = GaussianBRDFRenderer(chunk_size=2, backend="torch")(
        field, cam, light=lighting(0)
    )
    accelerated = GaussianBRDFRenderer(backend="gsplat")(
        field, cam, light=lighting(0)
    )
    assert (reference.full - accelerated.full).abs().mean() < 0.01
    assert (reference.diffuse - accelerated.diffuse).abs().mean() < 0.01
    accelerated.full.mean().backward()
    assert field.means.grad is not None
    assert torch.isfinite(field.means.grad).all()


def test_blender_transforms_loader(tmp_path):
    images = tmp_path / "images"
    diffuse = tmp_path / "diffuse"
    specular = tmp_path / "specular"
    albedo = tmp_path / "albedo"
    normal = tmp_path / "normal"
    mask = tmp_path / "mask"
    images.mkdir()
    diffuse.mkdir()
    specular.mkdir()
    albedo.mkdir()
    normal.mkdir()
    mask.mkdir()
    array = np.full((16, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(array).save(images / "0000.png")
    Image.fromarray(array // 2).save(diffuse / "0000.png")
    Image.fromarray(array // 2).save(specular / "0000.png")
    Image.fromarray(array).save(albedo / "0000.png")
    encoded_normal = np.zeros((16, 16, 3), dtype=np.uint8)
    encoded_normal[..., 2] = 255
    encoded_normal[..., :2] = 128
    Image.fromarray(encoded_normal).save(normal / "0000.png")
    Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(mask / "0000.png")
    meta = {
        "camera_angle_x": 0.7,
        "frames": [
            {
                "file_path": "images/0000.png",
                "diffuse_path": "diffuse/0000.png",
                "specular_path": "specular/0000.png",
                "albedo_path": "albedo/0000.png",
                "normal_path": "normal/0000.png",
                "mask_path": "mask/0000.png",
                "transform_matrix": np.eye(4).tolist(),
                "light_direction": [0, 0, 1],
                "light_intensity": 1.0,
                "view_id": 3,
                "light_id": 7,
                "exposure": 1.25,
            }
        ],
    }
    (tmp_path / "transforms.json").write_text(json.dumps(meta), encoding="utf-8")
    dataset = MultiViewScene.from_blender(tmp_path)
    assert len(dataset) == 1
    assert dataset[0].image.shape == (3, 16, 16)
    assert dataset[0].diffuse_gt is not None
    assert dataset[0].specular_gt is not None
    assert dataset[0].albedo_gt is not None
    assert dataset[0].normal_gt is not None
    assert dataset[0].mask_gt is not None
    assert dataset[0].view_id == 3 and dataset[0].light_id == 7
    assert dataset[0].exposure == 1.25
    assert dataset.group_by_view() == {3: [0]}
    assert dataset.group_by_light() == {7: [0]}
    assert dataset[0].camera.width == 16
    pseudo = torch.full((3, 16, 16), 0.25)
    confidence = torch.ones(1, 16, 16)
    origins, directions = dataset[0].camera.pixel_rays()
    point = origins[8, 8] + 2 * directions[8, 8]
    fused = dataset.fuse_projected_texture(
        point.unsqueeze(0),
        [0],
        source_images={0: pseudo},
        confidence_maps={0: confidence},
    )
    assert torch.allclose(fused, torch.full((1, 3), 0.25), atol=1e-5)


def test_unified_inverse_rendering_evaluation_smoke(tmp_path):
    images = tmp_path / "images"
    diffuse = tmp_path / "diffuse"
    images.mkdir()
    diffuse.mkdir()
    frame_records = []
    for index, split in enumerate(("train", "test")):
        array = np.full((8, 8, 3), 96 + index * 16, dtype=np.uint8)
        Image.fromarray(array).save(images / f"{index:04d}.png")
        Image.fromarray(array // 2).save(diffuse / f"{index:04d}.png")
        frame_records.append(
            {
                "file_path": f"images/{index:04d}.png",
                "diffuse_path": f"diffuse/{index:04d}.png",
                "transform_matrix": np.eye(4).tolist(),
                "view_id": index,
                "light_id": 0,
                "split": split,
            }
        )
    (tmp_path / "transforms.json").write_text(
        json.dumps({"camera_angle_x": 0.7, "frames": frame_records}),
        encoding="utf-8",
    )
    cfg = {
        "data_type": "blender",
        "data_root": str(tmp_path),
        "transforms": "transforms.json",
        "n_gaussians": 8,
        "use_refiner": False,
        "raster_backend": "torch",
        "chunk_size": 4,
    }
    dataset = load_dataset(cfg)
    pipeline = build_pipeline(dataset, cfg, torch.device("cpu"), [0])
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "model": pipeline.state_dict(),
            "config": cfg,
            "points": pipeline.scene.means.detach(),
            "colors": pipeline.scene.texture.detach(),
            "train_ids": [0],
            "holdout_ids": [1],
        },
        checkpoint,
    )
    report = evaluate_inverse_rendering(
        checkpoint, tmp_path / "evaluation", split="holdout"
    )
    assert report["protocol"]["nvs"] is True
    assert report["summary"]["full_psnr"] is not None
    assert (tmp_path / "evaluation" / "metrics.json").exists()


def test_mvsnet_loader_unprojects_depth(tmp_path):
    images = tmp_path / "blended_images"
    cameras = tmp_path / "cams"
    depths = tmp_path / "rendered_depth_maps"
    images.mkdir()
    cameras.mkdir()
    depths.mkdir()
    Image.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(
        images / "00000000.jpg"
    )
    camera_text = """extrinsic
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

intrinsic
4 0 4
0 4 3
0 0 1

0.1 0.01 10 2.0
"""
    (cameras / "00000000_cam.txt").write_text(camera_text)
    depth = np.full((6, 8), 2.0, dtype="<f4")
    with (depths / "00000000.pfm").open("wb") as file:
        file.write(b"Pf\n8 6\n-1.0\n")
        np.flipud(depth).tofile(file)
    dataset = MultiViewScene.from_mvsnet(
        tmp_path, scale=1.0, max_points=12, max_point_views=1
    )
    assert len(dataset) == 1
    assert dataset.points is not None and len(dataset.points) == 12
    assert torch.allclose(dataset.points[:, 2], torch.full((12,), 2.0))
    assert dataset.point_colors is not None

