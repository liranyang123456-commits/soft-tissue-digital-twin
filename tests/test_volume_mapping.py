from __future__ import annotations

import numpy as np
import pytest

from mvbrdf_shr.world.volume_mapping import (
    build_surface_observation_map,
    similarity_icp,
    voxel_mask_to_tetrahedra,
)


def test_single_voxel_builds_shared_tetrahedral_volume():
    mask = np.ones((1, 1, 1), dtype=bool)
    volume = voxel_mask_to_tetrahedra(mask, block=(1, 1, 1))
    assert volume.nodes.shape == (8, 3)
    assert volume.elements.shape == (6, 4)
    assert set(volume.boundary_nodes.tolist()) == set(range(8))


def test_tetrahedral_volume_reuses_nodes_between_voxels():
    mask = np.ones((1, 1, 2), dtype=bool)
    volume = voxel_mask_to_tetrahedra(mask, block=(1, 1, 1))
    assert volume.nodes.shape[0] == 12
    assert volume.elements.shape[0] == 12


def test_similarity_icp_recovers_scale_and_alignment():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(200, 3))
    angle = 0.25
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = 2.3 * (source @ rotation.T) + np.asarray([1.2, -0.7, 0.4])
    transform, metrics = similarity_icp(source, target, trim_fraction=1.0)
    aligned = transform.apply(source)
    assert np.sqrt(np.mean(np.square(aligned - target))) < 1e-5
    assert transform.scale == pytest.approx(2.3, rel=1e-5)
    assert metrics["normalized_median_distance"] < 1e-6


def test_surface_observation_map_transfers_constant_displacement():
    gaussians = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    nodes = gaussians.copy()
    mapping = build_surface_observation_map(
        gaussians,
        nodes,
        np.arange(4),
        neighbors=2,
        coverage_fraction=0.5,
    )
    displacement = np.tile([0.1, -0.2, 0.3], (4, 1))
    transferred = mapping.transfer(displacement)
    assert np.allclose(transferred, displacement[mapping.volume_node_ids])
