"""Camera geometry for the world-space MVBRDF pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PerspectiveCamera:
    """Pinhole camera using a camera-to-world pose.

    Coordinates follow OpenCV convention: +x right, +y down, +z forward.
    """

    K: torch.Tensor
    c2w: torch.Tensor
    width: int
    height: int
    frame_id: int = 0
    view_id: int | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None):
        dtype = dtype or self.K.dtype
        return PerspectiveCamera(
            K=self.K.to(device=device, dtype=dtype),
            c2w=self.c2w.to(device=device, dtype=dtype),
            width=self.width,
            height=self.height,
            frame_id=self.frame_id,
            view_id=self.view_id,
        )

    @property
    def w2c(self) -> torch.Tensor:
        return torch.linalg.inv(self.c2w)

    @property
    def center(self) -> torch.Tensor:
        return self.c2w[:3, 3]

    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(points[..., :1])
        hom = torch.cat([points, ones], dim=-1)
        return (hom @ self.w2c.T)[..., :3]

    def camera_to_world(self, points: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(points[..., :1])
        hom = torch.cat([points, ones], dim=-1)
        return (hom @ self.c2w.T)[..., :3]

    def project(
        self, points: torch.Tensor, near: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project world points to pixel coordinates.

        Returns:
            uv: (...,2), depth: (...,), valid-front mask: (...,)
        """
        pc = self.world_to_camera(points)
        z = pc[..., 2]
        safe_z = z.clamp(min=near)
        u = self.K[0, 0] * pc[..., 0] / safe_z + self.K[0, 2]
        v = self.K[1, 1] * pc[..., 1] / safe_z + self.K[1, 2]
        uv = torch.stack([u, v], dim=-1)
        valid = (
            (z > near)
            & (u > -1)
            & (u < self.width)
            & (v > -1)
            & (v < self.height)
        )
        return uv, z, valid

    def projection_jacobian(self, points_camera: torch.Tensor) -> torch.Tensor:
        """Jacobian d(u,v)/d(x,y,z), shape (...,2,3)."""
        x, y = points_camera[..., 0], points_camera[..., 1]
        z = points_camera[..., 2].clamp(min=1e-4)
        fx, fy = self.K[0, 0], self.K[1, 1]
        zeros = torch.zeros_like(z)
        row_u = torch.stack([fx / z, zeros, -fx * x / (z * z)], dim=-1)
        row_v = torch.stack([zeros, fy / z, -fy * y / (z * z)], dim=-1)
        return torch.stack([row_u, row_v], dim=-2)

    def pixel_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world-space ray origins/directions for every pixel."""
        ys, xs = torch.meshgrid(
            torch.arange(self.height, device=self.K.device, dtype=self.K.dtype),
            torch.arange(self.width, device=self.K.device, dtype=self.K.dtype),
            indexing="ij",
        )
        pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
        dirs_cam = pix @ torch.linalg.inv(self.K).T
        dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)
        dirs_world = dirs_cam @ self.c2w[:3, :3].T
        dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
        origins = self.center.view(1, 1, 3).expand_as(dirs_world)
        return origins, dirs_world

