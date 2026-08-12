"""Explicit camera/world coordinate conversions for external datasets."""
from __future__ import annotations

from enum import Enum

import torch


class CameraConvention(str, Enum):
    """Supported camera bases for camera-to-world matrices."""

    OPENCV = "opencv"  # +x right, +y down, +z forward
    OPENGL = "opengl"  # +x right, +y up, -z forward


_CV_TO_GL = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def convert_camera_pose(
    camera_to_world: torch.Tensor,
    source: CameraConvention | str,
    target: CameraConvention | str,
) -> torch.Tensor:
    """Convert a (...,4,4) camera-to-world pose between camera bases."""
    source = CameraConvention(source)
    target = CameraConvention(target)
    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must end with shape (4,4)")
    if source == target:
        return camera_to_world.clone()
    basis = _CV_TO_GL.to(camera_to_world)
    return camera_to_world @ basis


def world_to_camera_from_c2w(camera_to_world: torch.Tensor) -> torch.Tensor:
    """Invert a rigid camera pose while preserving batch dimensions."""
    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must end with shape (4,4)")
    rotation = camera_to_world[..., :3, :3]
    translation = camera_to_world[..., :3, 3:4]
    result = torch.zeros_like(camera_to_world)
    result[..., :3, :3] = rotation.transpose(-1, -2)
    result[..., :3, 3:4] = -rotation.transpose(-1, -2) @ translation
    result[..., 3, 3] = 1
    return result


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous (...,4,4) transform to (...,N,3) points."""
    if transform.shape[-2:] != (4, 4) or points.shape[-1] != 3:
        raise ValueError("expected transform (...,4,4) and points (...,N,3)")
    return points @ transform[..., :3, :3].transpose(-1, -2) + transform[
        ..., None, :3, 3
    ]


def transform_normals(transform: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Apply inverse-transpose transformation and renormalize normals."""
    if transform.shape[-2:] != (4, 4) or normals.shape[-1] != 3:
        raise ValueError("expected transform (...,4,4) and normals (...,N,3)")
    normal_matrix = torch.linalg.inv(transform[..., :3, :3]).transpose(-1, -2)
    transformed = normals @ normal_matrix.transpose(-1, -2)
    return torch.nn.functional.normalize(transformed, dim=-1, eps=1e-8)


def validate_rigid_pose(pose: torch.Tensor, atol: float = 1e-4) -> None:
    """Raise when a pose is not a finite right-handed rigid transform."""
    if pose.shape[-2:] != (4, 4) or not torch.isfinite(pose).all():
        raise ValueError("pose must be a finite (...,4,4) tensor")
    rotation = pose[..., :3, :3]
    identity = torch.eye(3, device=pose.device, dtype=pose.dtype)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=atol):
        raise ValueError("pose rotation is not orthonormal")
    if torch.any(torch.linalg.det(rotation) < 1.0 - atol):
        raise ValueError("pose rotation must be right-handed")
    expected_bottom = torch.tensor([0, 0, 0, 1], device=pose.device, dtype=pose.dtype)
    if not torch.allclose(pose[..., 3, :], expected_bottom, atol=atol):
        raise ValueError("pose has an invalid homogeneous bottom row")
