from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from mvbrdf_shr.world.datasets import load_diligent_mv, load_stanford_orb


def _rgb(path, value=128, shape=(6, 8)):
    Image.fromarray(np.full((*shape, 3), value, dtype=np.uint8)).save(path)


def test_stanford_orb_blender_marks_pseudo_albedo(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "pseudo_albedo").mkdir()
    (tmp_path / "masks").mkdir()
    _rgb(tmp_path / "images" / "r_0.png")
    _rgb(tmp_path / "pseudo_albedo" / "r_0.png", value=64)
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(
        tmp_path / "masks" / "r_0.png"
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [1, 2, 3]
    (tmp_path / "transforms.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [
                    {
                        "file_path": "images/r_0.png",
                        "transform_matrix": transform.tolist(),
                        "view_id": "camera-0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_stanford_orb(tmp_path)
    frame = dataset[0]
    assert frame.image.shape == (3, 6, 8)
    assert frame.linear_rgb
    assert frame.albedo is not None and frame.albedo_is_pseudo
    assert frame.mask is not None
    assert frame.normal is frame.depth is frame.envmap is None
    assert torch.allclose(frame.camera.c2w[:3, 3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(
        frame.camera.c2w[:3, :3], torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    )
    assert dataset.metadata["albedo_semantics"] == "pseudo_estimate_not_ground_truth"
    # 8-bit midpoint is decoded from sRGB rather than mislabeled as linear.
    assert 0.21 < float(frame.image.mean()) < 0.22


def test_stanford_orb_minimal_llff_layout(tmp_path):
    (tmp_path / "images").mkdir()
    _rgb(tmp_path / "images" / "000.png", shape=(4, 6))
    pose = np.zeros((3, 5), dtype=np.float32)
    pose[:, :4] = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]], dtype=np.float32
    )
    pose[:, 4] = [4, 6, 5]
    np.save(tmp_path / "poses_bounds.npy", np.concatenate([pose.ravel(), [0.1, 4.0]])[None])

    dataset = load_stanford_orb(tmp_path, source_is_linear=True)
    frame = dataset[0]
    assert dataset.metadata["layout"] == "llff"
    assert torch.allclose(frame.camera.c2w, torch.eye(4))
    assert torch.allclose(frame.camera.K.diag(), torch.tensor([5.0, 5.0, 1.0]))
    assert frame.albedo is None and not frame.albedo_is_pseudo


def test_stanford_orb_merges_official_train_test_and_ground_truth(tmp_path):
    scene = tmp_path / "blender_LDR" / "baking_scene001"
    ground_truth = tmp_path / "ground_truth"
    for directory in ("train", "test", "train_mask", "test_mask"):
        (scene / directory).mkdir(parents=True)
    for split, value in (("train", 64), ("test", 96)):
        _rgb(scene / split / "0000.png", value=value)
        Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(
            scene / f"{split}_mask" / "0000.png"
        )
        (scene / f"transforms_{split}.json").write_text(
            json.dumps(
                {
                    "camera_angle_x": 0.7,
                    "frames": [
                        {
                            "file_path": f"./{split}/0000",
                            "transform_matrix": np.eye(4).tolist(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    gt_scene = ground_truth / scene.name
    for directory in ("pseudo_gt_albedo", "surface_normal", "z_depth"):
        (gt_scene / directory).mkdir(parents=True)
    np.save(gt_scene / "pseudo_gt_albedo" / "0000.npy", np.full((6, 8, 3), 0.4))
    np.save(
        gt_scene / "surface_normal" / "0000.npy",
        np.dstack(
            [
                np.zeros((6, 8)),
                np.zeros((6, 8)),
                np.ones((6, 8)),
            ]
        ),
    )
    np.save(gt_scene / "z_depth" / "0000.npy", np.ones((6, 8)))

    dataset = load_stanford_orb(scene, ground_truth_root=ground_truth)
    assert len(dataset) == 2
    assert [frame.metadata["split"] for frame in dataset] == ["train", "test"]
    assert all(frame.mask is not None for frame in dataset)
    assert not dataset[0].albedo_is_pseudo and dataset[1].albedo_is_pseudo
    assert dataset[0].normal is dataset[0].depth is None
    assert dataset[1].normal is not None and dataset[1].depth is not None


def test_stanford_orb_loads_cross_scene_novel_light_metadata(tmp_path):
    training_scene = tmp_path / "blender_LDR" / "baking_scene001"
    target_scene = training_scene.parent / "baking_scene002"
    (training_scene / "train").mkdir(parents=True)
    (target_scene / "test").mkdir(parents=True)
    (target_scene / "test_mask").mkdir()
    _rgb(training_scene / "train" / "0000.png")
    _rgb(target_scene / "test" / "0012.png", value=96)
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(
        target_scene / "test_mask" / "0012.png"
    )
    (training_scene / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [
                    {
                        "file_path": "./train/0000",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (training_scene / "transforms_novel.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "scene_name": target_scene.name,
                        "file_path": "./test/0012",
                        "camera_angle_x": 0.8,
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    default_dataset = load_stanford_orb(training_scene)
    assert len(default_dataset) == 1  # Never leak novel-light targets into training.

    dataset = load_stanford_orb(training_scene, include_novel=True)
    assert len(dataset) == 2
    novel = dataset[1]
    assert novel.image_path == target_scene / "test" / "0012.png"
    assert novel.mask is not None
    assert novel.metadata["split"] == "novel"
    assert novel.metadata["task"] == "novel_scene_relighting"
    assert novel.metadata["training_scene"] == "baking_scene001"
    assert novel.metadata["target_scene"] == "baking_scene002"
    assert novel.metadata["hdr_available"] is False
    assert novel.light_id == "baking_scene002:0012"
    assert dataset.metadata["has_novel_light"] is True


def test_diligent_mv_minimal_calibrated_fixture(tmp_path):
    view = tmp_path / "view_01"
    view.mkdir()
    K = np.array([[10, 0, 4], [0, 10, 3], [0, 0, 1]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [0, 0, -2]
    np.savez(view / "camera.npz", K=K, c2w=c2w)
    np.savetxt(view / "light_directions.txt", [[0, 0, 1], [1, 0, 1]])
    np.savetxt(view / "light_intensities.txt", [[1, 0.8, 0.7], [0.5, 0.5, 0.5]])
    _rgb(view / "001.png", value=32)
    _rgb(view / "002.png", value=64)
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(view / "mask.png")
    normals = np.zeros((6, 8, 3), dtype=np.float32)
    normals[..., 2] = 1
    np.save(view / "Normal_gt.npy", normals)
    (tmp_path / "object_mesh_Gt.ply").write_text("ply\n", encoding="ascii")

    dataset = load_diligent_mv(tmp_path)
    assert len(dataset) == 2
    assert dataset.mesh_path == tmp_path / "object_mesh_Gt.ply"
    assert dataset.metadata["expected_views"] == 20
    first, second = dataset[0], dataset[1]
    assert (first.view_id, first.light_id) == (0, 1)
    assert (second.view_id, second.light_id) == (0, 2)
    assert first.mask is not None and first.normal is not None
    assert first.albedo is first.depth is first.envmap is None
    assert torch.allclose(first.camera.c2w, torch.from_numpy(c2w))
    assert torch.allclose(first.light_direction, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(first.light_intensity, torch.ones(3))
    assert first.metadata["intensity_normalized"] is True


def test_diligent_mv_strict_counts_rejects_subset(tmp_path):
    (tmp_path / "view_01").mkdir()
    try:
        load_diligent_mv(tmp_path, strict_counts=True)
    except ValueError as error:
        assert "20 views" in str(error)
    else:
        raise AssertionError("strict_counts must reject incomplete DiLiGenT-MV fixtures")
