"""Map a tracked Gaussian surface to a patient-specific tetrahedral volume.

The mapping has two explicit parts:

1. build a conforming tetrahedral mesh from a binary clinical-image mask;
2. align Gaussian surface points to the volume boundary with a similarity
   transform and transfer observed displacement by local interpolation.

The implementation uses only NumPy and SciPy and therefore does not depend
on an external tetrahedralization package.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation


@dataclass(frozen=True)
class TetrahedralVolume:
    nodes: np.ndarray
    elements: np.ndarray
    boundary_nodes: np.ndarray
    block_shape: tuple[int, int, int]


@dataclass(frozen=True)
class SurfaceObservationMap:
    volume_node_ids: np.ndarray
    gaussian_ids: np.ndarray
    weights: np.ndarray
    nearest_distance: np.ndarray
    coverage_threshold: float

    def transfer(self, gaussian_displacement: np.ndarray) -> np.ndarray:
        displacement = np.asarray(gaussian_displacement, dtype=np.float64)
        selected = displacement[self.gaussian_ids]
        return (selected * self.weights[..., None]).sum(axis=1)


_TET_PATTERN = np.asarray(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int64,
)


def _block_reduce_mask(mask: np.ndarray, block: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    bz, by, bx = block
    pad = tuple(
        (0, (size - dim % size) % size)
        for dim, size in zip(mask.shape, block)
    )
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    z, y, x = padded.shape
    reduced = padded.reshape(
        z // bz,
        bz,
        y // by,
        by,
        x // bx,
        bx,
    ).mean(axis=(1, 3, 5))
    return reduced >= 0.5


def _boundary_nodes(elements: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        (
            elements[:, [0, 1, 2]],
            elements[:, [0, 1, 3]],
            elements[:, [0, 2, 3]],
            elements[:, [1, 2, 3]],
        ),
        axis=0,
    )
    faces = np.sort(faces, axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])


def voxel_mask_to_tetrahedra(
    mask: np.ndarray,
    *,
    block: tuple[int, int, int] = (2, 8, 8),
    spacing: tuple[float, float, float] | None = None,
) -> TetrahedralVolume:
    """Convert a binary ``(z,y,x)`` mask to a shared-node tetrahedral mesh."""
    occupancy = _block_reduce_mask(mask, block)
    active = np.argwhere(occupancy)
    if not len(active):
        raise ValueError("mask contains no occupied coarse voxels")

    spacing_zyx = np.asarray(spacing or (1.0, 1.0, 1.0), dtype=np.float64)
    block_zyx = np.asarray(block, dtype=np.float64)
    corner_offsets = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )

    node_lookup: dict[tuple[int, int, int], int] = {}
    node_grid: list[tuple[int, int, int]] = []
    elements: list[list[int]] = []
    for cell in active:
        corner_ids = []
        for offset in corner_offsets:
            key = tuple((cell + offset).tolist())
            node_id = node_lookup.get(key)
            if node_id is None:
                node_id = len(node_grid)
                node_lookup[key] = node_id
                node_grid.append(key)
            corner_ids.append(node_id)
        corner_ids_array = np.asarray(corner_ids, dtype=np.int64)
        elements.extend(corner_ids_array[_TET_PATTERN].tolist())

    grid = np.asarray(node_grid, dtype=np.float64)
    nodes_zyx = grid * block_zyx * spacing_zyx
    nodes_xyz = nodes_zyx[:, ::-1]
    elements_array = np.asarray(elements, dtype=np.int64)
    return TetrahedralVolume(
        nodes=nodes_xyz,
        elements=elements_array,
        boundary_nodes=_boundary_nodes(elements_array),
        block_shape=tuple(int(value) for value in occupancy.shape),
    )


def _umeyama(source: np.ndarray, target: np.ndarray, allow_scale: bool) -> SimilarityTransform:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    if allow_scale:
        source_variance = np.square(source_centered).sum() / len(source)
        scale = float((singular * np.diag(correction)).sum() / max(source_variance, 1e-12))
    else:
        scale = 1.0
    translation = target_mean - scale * (source_mean @ rotation.T)
    return SimilarityTransform(scale, rotation, translation)


def compose(
    left: SimilarityTransform,
    right: SimilarityTransform,
) -> SimilarityTransform:
    """Return ``left(right(points))``."""
    return SimilarityTransform(
        scale=left.scale * right.scale,
        rotation=left.rotation @ right.rotation,
        translation=left.scale * (right.translation @ left.rotation.T)
        + left.translation,
    )


def _pca_initial_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_scale: bool,
) -> SimilarityTransform:
    """Choose the best proper signed PCA-axis alignment by nearest distance."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    _, source_axes = np.linalg.eigh(np.cov(source_centered, rowvar=False))
    _, target_axes = np.linalg.eigh(np.cov(target_centered, rowvar=False))
    source_axes = source_axes[:, ::-1]
    target_axes = target_axes[:, ::-1]
    source_radius = np.sqrt(np.square(source_centered).sum(axis=1).mean())
    target_radius = np.sqrt(np.square(target_centered).sum(axis=1).mean())
    scale = float(target_radius / max(source_radius, 1e-12)) if allow_scale else 1.0
    tree = cKDTree(target)
    best: SimilarityTransform | None = None
    best_score = np.inf
    for permutation in permutations(range(3)):
        permutation_matrix = np.zeros((3, 3))
        permutation_matrix[np.arange(3), permutation] = 1.0
        for signs in product((-1.0, 1.0), repeat=3):
            signed = permutation_matrix @ np.diag(signs)
            rotation = target_axes @ signed @ source_axes.T
            if np.linalg.det(rotation) <= 0:
                continue
            translation = target_mean - scale * (source_mean @ rotation.T)
            candidate = SimilarityTransform(scale, rotation, translation)
            distance, _ = tree.query(candidate.apply(source), k=1)
            score = float(np.median(distance))
            if score < best_score:
                best_score = score
                best = candidate
    if best is None:
        raise RuntimeError("failed to construct a proper PCA alignment")
    return best


def similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 40,
    trim_fraction: float = 0.7,
    allow_scale: bool = True,
) -> tuple[SimilarityTransform, dict]:
    """Trimmed similarity ICP for a partial Gaussian surface."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source and target must have shape (N,3)")
    if len(source) < 4 or len(target) < 4:
        raise ValueError("at least four source and target points are required")
    if not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0,1]")

    target_extent = np.linalg.norm(np.ptp(target, axis=0))
    transform = _pca_initial_transform(
        source,
        target,
        allow_scale=allow_scale,
    )
    tree = cKDTree(target)
    previous = np.inf
    history = []
    for _ in range(iterations):
        moved = transform.apply(source)
        distance, nearest = tree.query(moved, k=1)
        keep_count = max(4, int(np.ceil(trim_fraction * len(source))))
        keep = np.argpartition(distance, keep_count - 1)[:keep_count]
        incremental = _umeyama(
            moved[keep],
            target[nearest[keep]],
            allow_scale=allow_scale,
        )
        transform = compose(incremental, transform)
        rmse = float(np.sqrt(np.mean(np.square(distance[keep]))))
        history.append(rmse)
        if abs(previous - rmse) <= 1e-7 * max(previous, 1.0):
            break
        previous = rmse

    moved = transform.apply(source)
    distance, _ = tree.query(moved, k=1)
    return transform, {
        "iterations": len(history),
        "trim_fraction": trim_fraction,
        "trimmed_rmse": history[-1] if history else None,
        "median_distance": float(np.median(distance)),
        "p95_distance": float(np.quantile(distance, 0.95)),
        "target_diagonal": float(target_extent),
        "normalized_median_distance": float(np.median(distance) / max(target_extent, 1e-12)),
    }


def build_surface_observation_map(
    transformed_gaussians: np.ndarray,
    volume_nodes: np.ndarray,
    boundary_nodes: np.ndarray,
    *,
    neighbors: int = 4,
    coverage_fraction: float = 0.05,
) -> SurfaceObservationMap:
    """Map covered volume-boundary nodes to nearby tracked Gaussians."""
    gaussians = np.asarray(transformed_gaussians, dtype=np.float64)
    nodes = np.asarray(volume_nodes, dtype=np.float64)
    boundary = np.asarray(boundary_nodes, dtype=np.int64)
    if neighbors < 1 or neighbors > len(gaussians):
        raise ValueError("invalid neighbor count")
    boundary_positions = nodes[boundary]
    tree = cKDTree(gaussians)
    distance, indices = tree.query(boundary_positions, k=neighbors)
    if neighbors == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    diagonal = np.linalg.norm(np.ptp(nodes, axis=0))
    threshold = coverage_fraction * diagonal
    covered = distance[:, 0] <= threshold
    distance = distance[covered]
    indices = indices[covered]
    weights = 1.0 / np.maximum(distance, 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)
    return SurfaceObservationMap(
        volume_node_ids=boundary[covered],
        gaussian_ids=indices.astype(np.int64),
        weights=weights,
        nearest_distance=distance[:, 0],
        coverage_threshold=float(threshold),
    )
