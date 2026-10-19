"""Persistent 2D correspondence observations for fixed-identity tracking.

Every primitive that lies on the observed surface of the reference frame
receives a 2D track. Frame-to-frame RAFT chaining moves the track; every
``refresh`` frames the track is also refreshed by flowing from the reference
frame to the current one, so long sequences do not accumulate chain drift.
A track that is lost is re-anchored at the current projection of its
primitive only if that projection lies within ``reanchor_jump`` pixels of
the last accepted track, so a lost identity is not snapped onto a distant
surface. For each
frame the observer returns the 2D target, a confidence weight, and a depth
target (stereo depth at the target plus the primitive's reference offset
from the surface), which enter the tracking objective as observation terms.
Depth targets are used only where the depth map is locally smooth and when
their frame-to-frame change agrees with the robust consensus of the frame
and with the targets of neighbouring primitives.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mvbrdf_shr.world.flow_init import _knn


def _project(points: torch.Tensor, camera) -> tuple[torch.Tensor, torch.Tensor]:
    K = camera.K.to(points).float()
    c2w = camera.c2w.to(points).float()
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = local[:, 2]
    uv = torch.stack([K[0, 0] * local[:, 0] / z + K[0, 2], K[1, 1] * local[:, 1] / z + K[1, 2]], dim=-1)
    return uv, z


def _backproject(uv: torch.Tensor, z: torch.Tensor, camera) -> torch.Tensor:
    K = camera.K.to(uv).float()
    c2w = camera.c2w.to(uv).float()
    local = torch.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z], dim=-1)
    return local @ c2w[:3, :3].T + c2w[:3, 3]


def _sample(image: torch.Tensor, uv: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample a (C,H,W) or (H,W) image at pixel positions uv (N,2)."""
    squeeze = image.ndim == 2
    image = image[None] if squeeze else image
    h, w = image.shape[-2:]
    grid = torch.stack([2 * uv[:, 0] / (w - 1) - 1, 2 * uv[:, 1] / (h - 1) - 1], dim=-1).view(1, -1, 1, 2)
    values = F.grid_sample(image[None].float(), grid, mode=mode, align_corners=True)[0, :, :, 0]
    return values[0] if squeeze else values.T


def _smooth_depth(depth: torch.Tensor, window: int, relative: float) -> torch.Tensor:
    """Pixels whose depth neighbourhood is fully valid and varies by less than
    ``relative`` of the depth (excludes depth discontinuities and holes)."""
    valid = (depth > 0.01).float()[None, None]
    big = torch.where(depth > 0.01, depth, torch.full_like(depth, 1e9))[None, None]
    small = torch.where(depth > 0.01, depth, torch.full_like(depth, -1e9))[None, None]
    pad = window // 2
    upper = F.max_pool2d(small, window, 1, pad)[0, 0]
    lower = -F.max_pool2d(-big, window, 1, pad)[0, 0]
    complete = -F.max_pool2d(-valid, window, 1, pad)[0, 0] > 0.5
    return complete & ((upper - lower) <= relative * depth.clamp_min(1e-6))


def _inside(uv: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)


def _tissue(frame, uv: torch.Tensor) -> torch.Tensor:
    depth = frame.depth_gt.to(uv)
    valid = _sample(depth, uv, mode="nearest") > 0.01
    if frame.mask_gt is not None:
        valid = valid & (_sample(frame.mask_gt.to(uv), uv, mode="nearest") < 0.5)
    return valid


@dataclass
class Observation:
    target_uv: torch.Tensor
    target_z: torch.Tensor
    weight: torch.Tensor
    depth_valid: torch.Tensor


class CorrespondenceObserver:
    """RAFT tracks for the primitives of a canonical field, with reference refresh."""

    def __init__(self, reference_frame, device: torch.device, *, surface_tolerance: float = 0.03,
                 fb_abs: float = 1.0, fb_rel: float = 0.05, knn_k: int = 8, reanchor: bool = True,
                 reanchor_jump: float = 12.0,
                 depth_window: int = 5, depth_range: float = 0.05, depth_gate: float = 5.0,
                 refresh: int = 5):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.reference = reference_frame
        self.device = device
        self.tolerance = surface_tolerance
        self.fb_abs, self.fb_rel = fb_abs, fb_rel
        self.knn_k = knn_k
        self.reanchor = reanchor
        self.reanchor_jump = float(reanchor_jump)
        self.depth_window, self.depth_range, self.depth_gate = depth_window, depth_range, depth_gate
        self.refresh = max(1, int(refresh))
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        self.canonical = None
        self.frame_index = 0

    def _on_surface(self, frame, uv: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth = _sample(frame.depth_gt.to(uv), uv, mode="nearest")
        h, w = frame.depth_gt.shape[-2:]
        ok = _inside(uv, h, w) & _tissue(frame, uv) & (z > 0)
        ok = ok & ((z - depth).abs() <= self.tolerance * depth.clamp_min(1e-6))
        return ok, depth

    def bind(self, canonical: torch.Tensor) -> None:
        self.canonical = canonical.to(self.device)
        self.knn = _knn(self.canonical, self.knn_k)
        uv, z = _project(self.canonical, self.reference.camera)
        ok, depth = self._on_surface(self.reference, uv, z)
        self.uv0 = uv.clone()
        self.uv = uv
        self.valid = ok
        self.offset = torch.where(ok, z - depth, torch.zeros_like(z))
        self.previous = self.reference
        self.frame_index = 0

    def _flow(self, a, b) -> torch.Tensor:
        pair = [f.image.to(self.device).float().unsqueeze(0) * 2 - 1 for f in (a, b)]
        return self.model(pair[0], pair[1])[-1][0]

    def _advect(self, source, target, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move ``uv`` from ``source`` to ``target`` with forward-backward check."""
        forward = self._flow(source, target)
        backward = self._flow(target, source)
        h, w = forward.shape[-2:]
        motion = _sample(forward, uv)
        uv_new = uv + motion
        clamp = uv_new.clamp(min=0).clamp_max(torch.tensor([w - 1, h - 1], device=uv_new.device))
        back = _sample(backward, clamp)
        fb_error = (motion + back).norm(dim=-1)
        consistent = fb_error <= self.fb_abs + self.fb_rel * motion.norm(dim=-1)
        inside = _inside(uv_new, h, w)
        return uv_new, fb_error, consistent & inside

    @torch.no_grad()
    def step(self, frame, disp_prev: torch.Tensor) -> Observation:
        self.frame_index += 1
        # Chained step from the previous frame (primary motion model).
        uv_chain, fb_chain, ok_chain = self._advect(self.previous, frame, self.uv)
        uv_new, fb_error, consistent = uv_chain, fb_chain, ok_chain
        # Periodic reference refresh cancels chain drift, but only when the
        # reference match is both FB-better and close to the chained position
        # in pixels — large jumps from long-range RAFT are usually mismatches.
        if self.frame_index % self.refresh == 0 or self.previous is self.reference:
            uv_ref, fb_ref, ok_ref = self._advect(self.reference, frame, self.uv0)
            jump = (uv_ref - uv_chain).norm(dim=-1)
            prefer_ref = ok_ref & ((~ok_chain) | ((fb_ref <= fb_chain) & (jump <= 12.0)))
            uv_new = torch.where(prefer_ref[:, None], uv_ref, uv_chain)
            fb_error = torch.where(prefer_ref, fb_ref, fb_chain)
            consistent = ok_chain | prefer_ref

        valid = self.valid & consistent & _tissue(frame, uv_new.clamp(min=0))

        if self.reanchor:
            uv_cur, z_cur = _project(self.canonical + disp_prev.to(self.device), frame.camera)
            surface, depth_cur = self._on_surface(frame, uv_cur, z_cur)
            jump = (uv_cur - self.uv).norm(dim=-1)
            renew = (~valid) & surface & (jump <= self.reanchor_jump)
            uv_new = torch.where(renew[:, None], uv_cur, uv_new)
            self.offset = torch.where(renew, z_cur - depth_cur, self.offset)
            valid = valid | renew

        depth_map = frame.depth_gt.to(self.device).float()
        depth = _sample(depth_map, uv_new.clamp(min=0), mode="nearest")
        stable = _sample(_smooth_depth(depth_map, self.depth_window, self.depth_range).float(),
                         uv_new.clamp(min=0), mode="nearest") > 0.5
        depth_valid = valid & (depth > 0.01) & stable
        target_z = depth + self.offset

        _, z_cur = _project(self.canonical + disp_prev.to(self.device), frame.camera)
        change = torch.where(depth_valid, target_z - z_cur, torch.full_like(z_cur, float("nan")))
        if depth_valid.sum() > 10:
            centre = change[depth_valid].median()
            spread = (change[depth_valid] - centre).abs().median().clamp_min(1e-6)
            depth_valid = depth_valid & ((change - centre).abs() <= self.depth_gate * spread)
            neighbour = torch.where(depth_valid, target_z, torch.full_like(target_z, float("nan")))[self.knn]
            local = neighbour.nanmedian(dim=1).values
            deviation = (target_z - local).abs()
            finite = depth_valid & torch.isfinite(deviation)
            if finite.any():
                depth_valid = finite & (deviation <= self.depth_gate * deviation[finite].median().clamp_min(1e-6))

        weight = valid.float() * torch.exp(-0.5 * (fb_error / (self.fb_abs + 1e-6)) ** 2).clamp_min(0.25)
        self.uv, self.valid, self.previous = uv_new, valid, frame
        return Observation(uv_new, target_z, weight, depth_valid)

    @torch.no_grad()
    def initialize(self, frame, disp_prev: torch.Tensor, observation: Observation,
                   fill_passes: int = 4) -> torch.Tensor:
        """Lift the 2D targets to a displacement. The lateral position comes from
        the track and the depth from the gated stereo target; without a valid
        depth target the primitive keeps its current depth. Untracked
        primitives take the mean displacement of assigned neighbours,
        propagated over a few passes, and otherwise keep their previous
        displacement."""
        disp_prev = disp_prev.to(self.device)
        current = self.canonical + disp_prev
        _, z_cur = _project(current, frame.camera)
        tracked = observation.weight > 0
        usable = observation.depth_valid & tracked
        z = torch.where(usable, observation.target_z, z_cur)
        lifted = _backproject(observation.target_uv, z, frame.camera)

        disp = torch.where(tracked[:, None], lifted - self.canonical, disp_prev)
        # Reject outlier lifts whose step from the previous pose is far above
        # the robust consensus — stops depth/UV mismatches from exploding Δ.
        step = (disp - disp_prev).norm(dim=-1)
        if tracked.sum() > 16:
            med = step[tracked].median().clamp_min(1e-4)
            mad = (step[tracked] - med).abs().median().clamp_min(1e-4)
            keep = (~tracked) | (step <= med + self.depth_gate * mad)
            disp = torch.where(keep[:, None], disp, disp_prev)
            tracked = tracked & keep
        assigned = tracked.clone()
        for _ in range(fill_passes):
            has = assigned[self.knn]
            count = has.sum(1)
            mean = (disp[self.knn] * has[..., None]).sum(1) / count.clamp_min(1)[:, None]
            grow = (~assigned) & (count > 0)
            if not grow.any():
                break
            disp = torch.where(grow[:, None], mean, disp)
            assigned = assigned | grow
        return disp


def correspondence_loss(points: torch.Tensor, camera, observation: Observation, depth_scale: float):
    """Robust reprojection (pixels) and depth (relative) residuals of tracked primitives."""
    uv, z = _project(points, camera)
    weight = observation.weight
    total = weight.sum().clamp_min(1.0)
    pixel = torch.sqrt(((uv - observation.target_uv) ** 2).sum(-1) + 1.0) - 1.0
    reprojection = (weight * pixel).sum() / total
    depth_weight = weight * observation.depth_valid.float()
    residual = (z - observation.target_z) / depth_scale
    depth = (depth_weight * (torch.sqrt(residual ** 2 + 1.0) - 1.0)).sum() / depth_weight.sum().clamp_min(1.0)
    return reprojection, depth


def arap_energy(canonical: torch.Tensor, disp: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
    """As-rigid-as-possible energy on the k-NN graph with local rotations
    from the current configuration (held fixed during the gradient step)."""
    rest = canonical[knn] - canonical[:, None]
    deformed_points = canonical + disp
    deformed = deformed_points[knn] - deformed_points[:, None]
    with torch.no_grad():
        covariance = rest.transpose(1, 2) @ deformed
        U, _, Vh = torch.linalg.svd(covariance)
        V = Vh.transpose(1, 2)
        sign = torch.sign(torch.det(V @ U.transpose(1, 2)))
        V = torch.cat([V[:, :, :2], V[:, :, 2:] * sign[:, None, None]], dim=2)
        rotation = V @ U.transpose(1, 2)
    residual = deformed - rest @ rotation.transpose(1, 2)
    return (residual ** 2).mean()
