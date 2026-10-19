"""Optical-flow initialization for fixed-identity tracking.

The displacement of every primitive in frame t is initialized by moving its
projection in frame t-1 along the RAFT flow between the two frames and
back-projecting the new pixel position at the primitive's current depth;
stereo depth changes are too noisy for initialization and are left to the
depth term of the tracker. Increments that deviate from their k-NN median
are replaced by that median. The photometric tracker then refines this
initialization.
"""
from __future__ import annotations

import torch


def _sample_nearest(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h, w = image.shape[-2:]
    col = u.round().long().clamp(0, w - 1)
    row = v.round().long().clamp(0, h - 1)
    return image[..., row, col]


def _knn(points: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    neighbours = []
    for start in range(0, points.shape[0], chunk):
        distance = torch.cdist(points[start : start + chunk], points)
        neighbours.append(distance.topk(k + 1, largest=False).indices[:, 1:])
    return torch.cat(neighbours)


def reject_outliers(increment: torch.Tensor, neighbours: torch.Tensor, factor: float = 5.0) -> torch.Tensor:
    """Replace increments that deviate from their k-NN median by more than
    ``factor`` times the median deviation with that median."""
    median = increment[neighbours].median(dim=1).values
    deviation = (increment - median).norm(dim=-1)
    outlier = deviation > factor * deviation.median().clamp_min(1e-6)
    return torch.where(outlier[:, None], median, increment)


def make_flow_init(first_frame, device: torch.device, knn_k: int = 8):
    """Lift RAFT flow between consecutive frames to a 3D displacement initialization."""
    import torch.nn.functional as F
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    cache: dict[str, torch.Tensor] = {}

    def init_fn(previous, frame, canonical, disp_prev):
        if "knn" not in cache:
            cache["knn"] = _knn(canonical.to(device), knn_k)
        previous = previous if previous is not None else first_frame
        with torch.no_grad():
            pair = [f.image.to(device).float().unsqueeze(0) * 2 - 1 for f in (previous, frame)]
            flow = model(pair[0], pair[1])[-1][0]
            K = frame.camera.K.to(device).float()
            c2w = frame.camera.c2w.to(device).float()
            camera = (canonical + disp_prev - c2w[:3, 3]) @ c2w[:3, :3]
            z = camera[:, 2]
            u = K[0, 0] * camera[:, 0] / z + K[0, 2]
            v = K[1, 1] * camera[:, 1] / z + K[1, 2]
            h, w = flow.shape[-2:]
            inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (z > 0)
            grid = torch.stack([2 * u / (w - 1) - 1, 2 * v / (h - 1) - 1], dim=-1).view(1, -1, 1, 2)
            motion = F.grid_sample(flow[None], grid, align_corners=True)[0, :, :, 0].T
            u_new, v_new = u + motion[:, 0], v + motion[:, 1]
            inside = inside & (u_new >= 0) & (u_new <= w - 1) & (v_new >= 0) & (v_new <= h - 1)
            depth_prev = _sample_nearest(previous.depth_gt.to(device), u, v)
            depth_new = _sample_nearest(frame.depth_gt.to(device), u_new, v_new)
            valid_depth = (depth_prev > 0) & (depth_new > 0)
            if previous.mask_gt is not None and frame.mask_gt is not None:
                tissue_prev = _sample_nearest(previous.mask_gt.to(device), u, v) < 0.5
                tissue_new = _sample_nearest(frame.mask_gt.to(device), u_new, v_new) < 0.5
                inside = inside & tissue_prev & tissue_new & valid_depth
            lifted = torch.stack(
                [(u_new - K[0, 2]) / K[0, 0] * z, (v_new - K[1, 2]) / K[1, 1] * z, z], dim=-1
            )
            world = lifted @ c2w[:3, :3].T + c2w[:3, 3]
            increment = torch.where(inside[:, None], world - canonical - disp_prev, torch.zeros_like(disp_prev))
            return disp_prev + reject_outliers(increment, cache["knn"])

    return init_fn
