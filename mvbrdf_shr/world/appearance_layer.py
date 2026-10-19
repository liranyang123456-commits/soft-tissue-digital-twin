"""Appearance Gaussians with an EndoGaussian-style 4D residual.

Paper numbers stay on the v7 identity appearance path. HGI/HexPlane copies of
EndoGaussian are experimental only and must not replace the Lagrangian field.

Identity nodes stay mechanical. Appearance is a denser canonical field whose
time dependence is a HexPlane queried at any t (hold-out never sees images).

  mu(t) = mu0 + F_mu(E(mu0, t))
  R,S,o likewise; colour / BRDF stay in the canonical field (no Delta-RGB).

Identity warp W(u_id) is only a teacher for F_mu, not the hold-out interpolator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mvbrdf_shr.world.deform import DeformedFieldView
from mvbrdf_shr.world.scene import quaternion_to_matrix


def _knn_weights(query: torch.Tensor, anchors: torch.Tensor, k: int, chunk: int = 4096):
    index, weight = [], []
    for start in range(0, query.shape[0], chunk):
        distance = torch.cdist(query[start:start + chunk], anchors)
        d, i = distance.topk(k, largest=False)
        sigma = d[:, -1:].clamp_min(1e-6)
        w = torch.exp(-(d / sigma) ** 2)
        index.append(i)
        weight.append(w / w.sum(1, keepdim=True))
    return torch.cat(index), torch.cat(weight)


def _backproject_frame(frame, device, max_points: int, generator: torch.Generator):
    depth = frame.depth_gt.to(device)
    valid = depth > 0.01
    if frame.mask_gt is not None:
        valid = valid & (frame.mask_gt.to(device) < 0.5)
    rows, cols = torch.nonzero(valid, as_tuple=True)
    if rows.numel() > max_points:
        pick = torch.randperm(rows.numel(), generator=generator, device="cpu")[:max_points].to(device)
        rows, cols = rows[pick], cols[pick]
    K = frame.camera.K.to(device).float()
    c2w = frame.camera.c2w.to(device).float()
    z = depth[rows, cols]
    local = torch.stack([(cols.float() - K[0, 2]) / K[0, 0] * z, (rows.float() - K[1, 2]) / K[1, 1] * z, z], -1)
    points = local @ c2w[:3, :3].T + c2w[:3, 3]
    colours = frame.image.to(device)[:, rows, cols].T
    return points, colours


@dataclass
class DeformDelta:
    dxyz: torch.Tensor
    dquat: torch.Tensor
    dlogscale: torch.Tensor
    dopacity: torch.Tensor


class HexPlaneDeform(torch.nn.Module):
    """HexPlane encoder + four tiny heads, matching EndoGaussian SGT."""

    def __init__(self, n_feat: int = 32, res: int = 64):
        super().__init__()
        self.planes = torch.nn.ParameterList(
            [torch.nn.Parameter(0.02 * torch.randn(1, n_feat, res, res)) for _ in range(6)]
        )
        hidden = 64
        in_dim = n_feat + 1

        def head(out_dim: int) -> torch.nn.Sequential:
            module = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, out_dim),
            )
            torch.nn.init.zeros_(module[-1].weight)
            torch.nn.init.zeros_(module[-1].bias)
            return module

        self.head_mu = head(3)
        self.head_r = head(4)
        self.head_s = head(3)
        self.head_o = head(1)
        self.register_buffer("center", torch.zeros(3))
        self.register_buffer("extent", torch.ones(3))
        self.t_max = 1.0

    def set_bounds(self, points: torch.Tensor, t_max: float) -> None:
        lo, hi = points.min(0).values, points.max(0).values
        self.center.copy_((lo + hi) * 0.5)
        self.extent.copy_((hi - lo).clamp_min(1e-3) * 0.5)
        self.t_max = float(max(t_max, 1.0))

    def _sample(self, plane: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        grid = torch.stack([u, v], dim=-1).view(1, 1, -1, 2)
        feat = F.grid_sample(plane, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return feat[0, :, 0, :].transpose(0, 1)

    def encode(self, points: torch.Tensor, t_value: float) -> torch.Tensor:
        x = ((points - self.center) / self.extent).clamp(-1, 1)
        t = torch.full((points.shape[0],), 2.0 * float(t_value) / self.t_max - 1.0, device=points.device)
        t = t.clamp(-1, 1)
        xy = self._sample(self.planes[0], x[:, 0], x[:, 1])
        xz = self._sample(self.planes[1], x[:, 0], x[:, 2])
        yz = self._sample(self.planes[2], x[:, 1], x[:, 2])
        xt = self._sample(self.planes[3], x[:, 0], t)
        yt = self._sample(self.planes[4], x[:, 1], t)
        zt = self._sample(self.planes[5], x[:, 2], t)
        feat = xy * xz * yz + xt * yt * zt
        return torch.cat([feat, t[:, None]], dim=-1)

    def forward(self, points: torch.Tensor, t_value: float) -> DeformDelta:
        feat = self.encode(points, t_value)
        dxyz = 0.08 * torch.tanh(self.head_mu(feat)) * self.extent
        dquat = 0.05 * torch.tanh(self.head_r(feat))
        dlogscale = 0.15 * torch.tanh(self.head_s(feat))
        dopacity = 2.0 * torch.tanh(self.head_o(feat))
        return DeformDelta(dxyz, dquat, dlogscale, dopacity)


class AppearanceView(DeformedFieldView):
    """Identity-free time view: canonical Gaussians plus HexPlane attribute deltas."""

    def __init__(self, base_field, displacement: torch.Tensor, delta: DeformDelta | None = None):
        super().__init__(base_field, displacement)
        object.__setattr__(self, "_delta", delta)

    @property
    def quaternions(self) -> torch.Tensor:
        q = self._base.quaternions
        if self._delta is None:
            return q
        return F.normalize(q + self._delta.dquat, dim=-1)

    @property
    def rotations(self) -> torch.Tensor:
        return quaternion_to_matrix(self.quaternions)

    @property
    def log_scales(self) -> torch.Tensor:
        s = self._base.log_scales
        if self._delta is None:
            return s
        return s + self._delta.dlogscale

    @property
    def scales(self) -> torch.Tensor:
        return self.log_scales.exp().clamp(1e-4, 1.0)

    @property
    def opacity_logits(self) -> torch.Tensor:
        o = self._base.opacity_logits
        if self._delta is None:
            return o
        extra = self._delta.dopacity
        if extra.ndim == 1:
            extra = extra[:, None]
        return o + extra

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)


@dataclass
class AppearanceLayer:
    field: object
    index: torch.Tensor
    weight: torch.Tensor
    birth: torch.Tensor
    hexplane: HexPlaneDeform | None = None
    t_max: float = 1.0

    def displacement(self, node_disp: torch.Tensor) -> torch.Tensor:
        return (node_disp[self.index] * self.weight[..., None]).sum(1)

    def appearance_motion(self, node_disp: torch.Tensor, t_value: float) -> tuple[torch.Tensor, DeformDelta | None]:
        del node_disp
        if self.hexplane is None:
            return torch.zeros_like(self.field.means), None
        delta = self.hexplane(self.field.means, t_value)
        return delta.dxyz, delta


def build_appearance_layer(frames, node_canonical: torch.Tensor, node_disp_by_frame: dict, device,
                           *, points_per_frame: int = 50000, keyframe_stride: int = 1, k: int = 6,
                           birth_radius_factor: float = 1.0, seed: int = 0) -> AppearanceLayer:
    from mvbrdf_shr.world.scene import GaussianBRDFField

    generator = torch.Generator().manual_seed(seed)
    frames = sorted(frames, key=lambda f: int(f.view_id))
    keyframes = [f for i, f in enumerate(frames) if i % keyframe_stride == 0]
    points, colours, births = [], [], []
    spacing = None
    max_gaussians = 70000
    for frame in keyframes:
        fid = int(frame.view_id)
        world, colour = _backproject_frame(frame, device, points_per_frame, generator)
        canonical = world
        if spacing is None:
            sample = canonical[:: max(1, canonical.shape[0] // 2000)]
            d = torch.cdist(sample, sample)
            d.fill_diagonal_(float("inf"))
            spacing = float(d.min(1).values.median())
        else:
            existing = torch.cat(points)
            nearest = torch.cat([
                torch.cdist(canonical[s:s + 2048], existing).min(1).values
                for s in range(0, canonical.shape[0], 2048)
            ])
            new = nearest > birth_radius_factor * spacing
            canonical, colour = canonical[new], colour[new]
        remain = max_gaussians - sum(p.shape[0] for p in points)
        if remain <= 0:
            break
        if canonical.shape[0] > remain:
            canonical, colour = canonical[:remain], colour[:remain]
        if canonical.shape[0] == 0:
            continue
        points.append(canonical)
        colours.append(colour)
        births.append(torch.full((canonical.shape[0],), fid, device=device))
        print(f"  appearance init frame {fid}: +{int(canonical.shape[0])} "
              f"(total {sum(p.shape[0] for p in points)})", flush=True)
    points, colours, births = torch.cat(points), torch.cat(colours), torch.cat(births)
    field = GaussianBRDFField(points, colours, initial_scale=max(0.45 * spacing, 1e-3)).to(device)
    cam = frames[0].camera.to(device)
    with torch.no_grad():
        field.normal_raw.copy_(F.normalize(cam.center.to(device).view(1, 3) - field.means, dim=-1))
        field.metalness_logits.fill_(-6.0)
        field.roughness_logits.fill_(3.0)
        field.material_budget_logits[:, 0] = 6.0
        field.material_budget_logits[:, 1] = -3.0
        field.material_budget_logits[:, 2] = -6.0
    index, weight = _knn_weights(points, node_canonical, k)
    return AppearanceLayer(field, index, weight, births)


def appearance_image(out, exposure: torch.Tensor) -> torch.Tensor:
    shaded = out.full[:3]
    albedo = out.albedo[:3] if out.albedo.shape[0] >= 3 else out.albedo.expand(3, -1, -1)
    return (0.5 * exposure.reshape(-1, 1, 1) * (shaded + albedo)).clamp(0, 1)


def _ssim_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = pred.permute(2, 0, 1).unsqueeze(0)
    target_n = target.permute(2, 0, 1).unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_p = F.avg_pool2d(pred_n, 11, 1, 5)
    mu_t = F.avg_pool2d(target_n, 11, 1, 5)
    sigma_p = F.avg_pool2d(pred_n * pred_n, 11, 1, 5) - mu_p * mu_p
    sigma_t = F.avg_pool2d(target_n * target_n, 11, 1, 5) - mu_t * mu_t
    cov = F.avg_pool2d(pred_n * target_n, 11, 1, 5) - mu_p * mu_t
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (sigma_p + sigma_t + c2) + 1e-8
    )
    return ssim[0].mean(0)


def _interp_identity(t_value: float, disps: dict, keys: list[int]) -> torch.Tensor:
    keys = sorted(keys)
    lower = max((k for k in keys if k < t_value), default=None)
    upper = min((k for k in keys if k > t_value), default=None)
    if lower is None:
        return disps[upper]
    if upper is None:
        return disps[lower]
    alpha = (t_value - lower) / (upper - lower)
    return (1.0 - alpha) * disps[lower] + alpha * disps[upper]


def _warp_chw(image: torch.Tensor, flow: torch.Tensor, alpha: float) -> torch.Tensor:
    _, h, w = image.shape
    ys = torch.linspace(0, h - 1, h, device=image.device, dtype=image.dtype)
    xs = torch.linspace(0, w - 1, w, device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([
        2 * (xx + alpha * flow[0]) / max(w - 1, 1) - 1,
        2 * (yy + alpha * flow[1]) / max(h - 1, 1) - 1,
    ], dim=-1)
    return F.grid_sample(image[None], grid[None], mode="bilinear", padding_mode="border", align_corners=True)[0]


def _precompute_pair_flow(frames, device):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    by_id = {int(f.view_id): f for f in frames}
    ids = sorted(by_id)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    pairs = []
    with torch.no_grad():
        for a, b in zip(ids[:-1], ids[1:]):
            fa, fb = by_id[a], by_id[b]
            ia = fa.image.to(device).float().unsqueeze(0) * 2 - 1
            ib = fb.image.to(device).float().unsqueeze(0) * 2 - 1
            fw = model(ia, ib)[-1][0]
            bw = model(ib, ia)[-1][0]
            warped_bw = _warp_chw(bw, fw, 1.0)
            fb_ok = ((fw + warped_bw).pow(2).sum(0).sqrt() < 2.5)
            tissue_a = (fa.depth_gt.to(device) > 0.01).float()
            if fa.mask_gt is not None:
                tissue_a = tissue_a * (fa.mask_gt.to(device) < 0.5).float()
            pairs.append({
                "a": a, "b": b, "fa": fa, "fb": fb,
                "flow": fw, "fb_ok": fb_ok, "tissue_a": tissue_a,
            })
    del model
    torch.cuda.empty_cache()
    print(f"  motion-aware pairs: {len(pairs)}", flush=True)
    return pairs


def fit_appearance_layer(layer: AppearanceLayer, frames, node_disp_by_frame: dict, renderer, light_fn,
                         image_fn, log_exposure: torch.nn.Parameter, device, *, steps: int = 15000,
                         log_every: int = 500, residual_weight: float = 1e-4,
                         densify_every: int = 0, node_canonical: torch.Tensor | None = None):
    del densify_every, node_canonical, residual_weight
    field = layer.field
    frames = list(frames)
    ids = [int(f.view_id) for f in frames]
    hexplane = HexPlaneDeform().to(device)
    t_max = float(max(ids) if ids else 1)
    hexplane.set_bounds(field.means.detach(), t_max)
    layer.hexplane, layer.t_max = hexplane, t_max

    optimizer = torch.optim.Adam([
        {"params": [field.means], "lr": 1.6e-4},
        {"params": [field.log_scales], "lr": 5e-3},
        {"params": [field.quaternions], "lr": 1e-3},
        {"params": [field.normal_raw], "lr": 8e-4},
        {"params": [field.opacity_logits], "lr": 5e-2},
        {"params": [field.base_color_logits], "lr": 2.5e-2},
        {"params": [field.material_budget_logits], "lr": 4e-3},
        {"params": [log_exposure], "lr": 1e-3},
        {"params": list(hexplane.parameters()), "lr": 1.6e-3},
    ])
    render_fn = appearance_image if image_fn is None else image_fn
    curve, started = [], time.time()
    generator = np.random.default_rng(0)
    for step in range(steps):
        frame = frames[generator.integers(len(frames))]
        fid = int(frame.view_id)
        cam = frame.camera.to(device)
        delta = hexplane(field.means, float(fid))
        view = AppearanceView(field, delta.dxyz, delta)
        out = renderer(view, cam, light=light_fn(frame))
        pred = render_fn(out, log_exposure.exp()).permute(1, 2, 0)
        target = frame.image.to(device).permute(1, 2, 0)
        depth = frame.depth_gt.to(device)
        tissue = (depth > 0.01).float()
        if frame.mask_gt is not None:
            tissue = tissue * (frame.mask_gt.to(device) < 0.5).float()
        m = tissue.unsqueeze(-1)
        denom = m.sum().clamp_min(1)
        l1 = ((pred - target).abs() * m).sum() / denom
        mse = (((pred - target) ** 2) * m).sum() / denom
        loss = l1
        rendered_depth = out.depth[0] if out.depth.ndim == 3 else out.depth
        valid = tissue > 0.5
        if valid.any():
            loss = loss + 0.02 * F.l1_loss(rendered_depth[valid], depth[valid])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.quaternions.data.copy_(F.normalize(field.quaternions.data, dim=-1))
        curve.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            psnr = float((10.0 * torch.log10(1.0 / mse.clamp_min(1e-8))).detach())
            print(f"  appearance step {step:5d}: loss {np.mean(curve[-log_every:]):.4f} "
                  f"tissue-PSNR {psnr:.2f} dB ({time.time() - started:.0f}s)", flush=True)
    zeros = np.zeros_like(next(iter(node_disp_by_frame.values())))
    residual_by_frame = {fid: zeros.copy() for fid in ids}
    gain_by_frame = {fid: np.zeros(3, dtype=np.float32) for fid in ids}
    bias_by_frame = {fid: np.zeros(3, dtype=np.float32) for fid in ids}
    return curve, residual_by_frame, gain_by_frame, bias_by_frame
