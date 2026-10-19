"""Tier 2: deformation tracking on the reconstructed Gaussian field.

Given a canonical :class:`GaussianBRDFField` (Tier 1, built from a reference
frame) and a sequence of later frames, estimate a per-frame displacement
for every Gaussian so the deformed field explains that frame's image and
depth. The output doubles as (a) replayable boundary conditions for a
simulator and (b) the observation for Tier 3 elasticity inversion.

Model: ``means_t = means_canonical + D_t`` with

- photometric L1 + rendered-depth supervision per frame;
- kNN-graph Laplacian smoothness on D_t (tissue deforms coherently);
- temporal inertia prior ||D_t - D_{t-1}|| (motion is smooth in time).

This is deliberately a *tracking* model (per-frame displacement on a shared
canonical topology), not a free per-frame rebuild — the twin needs
correspondences, which free rebuilding would not provide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Deformed view of a canonical field (renderer-compatible proxy)
# ---------------------------------------------------------------------------


class DeformedFieldView:
    """Proxy exposing ``means = base.means + displacement`` to the renderer.

    All other attributes/methods (``materials()``, ``normals``, ``scales``,
    ``opacity`` ...) are forwarded to the canonical field, so the existing
    :class:`GaussianBRDFRenderer` works unchanged.
    """

    def __init__(self, base_field, displacement: torch.Tensor):
        object.__setattr__(self, "_base", base_field)
        object.__setattr__(self, "_disp", displacement)

    @property
    def means(self) -> torch.Tensor:
        return self._base.means + self._disp

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)


# ---------------------------------------------------------------------------
# kNN graph (Laplacian smoothness)
# ---------------------------------------------------------------------------


def build_knn_graph(positions: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Return neighbor indices (N,k) for each position (excluding self)."""
    with torch.no_grad():
        d = torch.cdist(positions, positions)
        d.fill_diagonal_(float("inf"))
        knn = d.topk(min(k, positions.shape[0] - 1), largest=False).indices
    return knn


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DeformationResult:
    canonical_positions: np.ndarray  # (N,3)
    displacements: np.ndarray        # (T,N,3), one per tracked frame
    frame_ids: list[int]             # tracked frame ids (T)
    losses: dict[int, list[float]] = field(default_factory=dict)
    notes: str = ""

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            str(path),
            canonical_positions=self.canonical_positions,
            displacements=self.displacements,
            frame_ids=np.asarray(self.frame_ids),
        )

    @property
    def displacement_magnitudes(self) -> np.ndarray:
        return np.linalg.norm(self.displacements, axis=-1)  # (T,N)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def track_deformation(
    field,
    frames: list,
    renderer,
    light_for_frame,
    *,
    device: torch.device,
    steps_per_frame: int = 60,
    lr: float = 5e-3,
    knn_k: int = 8,
    w_photo: float = 1.0,
    w_depth: float = 0.5,
    w_smooth: float = 2.0,
    w_temporal: float = 1.0,
    w_accel: float = 0.5,
    max_step_ratio: float = 3.0,
    divergence_factor: float = 3.0,
    max_gaussians: int | None = None,
    log_prefix: str = "  ",
    image_fn=None,
    temporal_model: str = "constant_velocity",
    init_fn=None,
    observer=None,
    w_corr: float = 0.5,
    w_corr_depth: float = 0.15,
    regularizer: str = "laplacian",
) -> DeformationResult:
    """Track per-frame displacements with constant-velocity + divergence guards.

    Temporal model (fixes the occasional frame divergence seen with a plain
    inertia term):

    - **Constant-velocity init**: D_t starts at ``2·D_{t-1} - D_{t-2}``
      (linear extrapolation), so the optimizer begins near the solution
      instead of from rest.
    - **Acceleration prior**: ``||D_t - 2·D_{t-1} + D_{t-2}||²`` penalizes
      changes in velocity, not velocity itself — fast but smooth motion is
      allowed, jitter is not.
    - **Per-frame step cap**: the mean displacement may not exceed
      ``max_step_ratio ×`` the running median of past steps (robust to the
      first frame). Prevents runaway drift on ambiguous frames.
    - **Divergence rollback**: if the final *photometric* loss is worse than
      ``divergence_factor ×`` the previous accepted frame's photo loss, the
      frame is re-optimized with half the learning rate; if still diverged,
      the previous displacement is kept. Correspondence residuals are not
      used for this test (they live on a different scale).

    ``image_fn`` maps a renderer output to the predicted RGB image; it must be
    the same image-formation model that was used to fit the canonical field.
    ``temporal_model="inertia"`` starts from ``D_{t-1}`` and penalizes
    ``||D_t - D_{t-1}||^2`` without a step cap; divergence rollback applies to
    both temporal models.

    ``init_fn(previous_frame, frame, canonical, D_{t-1})`` optionally returns a
    correspondence-based initialization (e.g. lifted optical flow); with the
    inertia model, optimization then starts from it and the temporal prior is
    centred on it instead of on ``D_{t-1}``.

    ``observer`` (a :class:`~mvbrdf_shr.world.correspondence.CorrespondenceObserver`)
    adds persistent 2D correspondence observations: each frame starts from the
    lifted correspondences, and robust reprojection and depth residuals of the
    tracked primitives enter the objective with weights ``w_corr`` and
    ``w_corr_depth``. ``regularizer`` selects the k-NN Laplacian of the
    displacement or an as-rigid-as-possible energy.
    """
    if temporal_model not in {"constant_velocity", "inertia"}:
        raise ValueError(f"unknown temporal model {temporal_model!r}")
    if image_fn is None:
        image_fn = lambda out: out.full[:3]
    inertia = temporal_model == "inertia"
    field = field.to(device)
    canonical = field.means.detach().clone()
    n = canonical.shape[0]
    if max_gaussians is not None and n > max_gaussians:
        sel = torch.randperm(n, device=device)[:max_gaussians]
        canonical = canonical[sel]
        n = max_gaussians
    knn = build_knn_graph(canonical, k=knn_k)
    if regularizer not in {"laplacian", "arap"}:
        raise ValueError(f"unknown regularizer {regularizer!r}")
    if observer is not None:
        from mvbrdf_shr.world.correspondence import arap_energy, correspondence_loss

        observer.bind(canonical)
        reference_depth = observer.reference.depth_gt
        depth_scale = 0.01 * float(reference_depth[reference_depth > 0.01].median())
    elif regularizer == "arap":
        from mvbrdf_shr.world.correspondence import arap_energy

    disp_hist: list[torch.Tensor] = [torch.zeros(n, 3, device=device)]
    step_mags: list[float] = []
    all_disps: list[np.ndarray] = []
    frame_ids: list[int] = []
    losses: dict[int, list[float]] = {}
    photo_losses: dict[int, float] = {}
    flagged: list[int] = []

    def optimize_frame(frame, disp_init, lr_cur, disp_m1, disp_m2, step_cap, observation=None):
        cam = frame.camera.to(device)
        light = light_for_frame(frame)
        target = frame.image.to(device).permute(1, 2, 0)
        depth_gt = (
            frame.depth_gt.to(device) if frame.depth_gt is not None else None
        )
        mask_gt = (
            frame.mask_gt.to(device) if frame.mask_gt is not None else None
        )
        # When correspondence covers most primitives, photometric refine must not
        # rediscover motion: keep it as a residual appearance term, but never
        # below 0.25 so rollback still sees a meaningful photo signal.
        photo_scale = 1.0
        if observation is not None:
            coverage = float((observation.weight > 0).float().mean())
            photo_scale = max(0.25, 1.0 - 0.7 * coverage)
        disp = torch.nn.Parameter(disp_init.clone())
        opt = torch.optim.Adam([disp], lr=lr_cur)
        curve: list[float] = []
        photo_curve: list[float] = []
        for step in range(steps_per_frame):
            opt.zero_grad()
            view = DeformedFieldView(field, disp)
            out = renderer(view, cam, light=light)
            pred = image_fn(out).permute(1, 2, 0)
            if pred.shape[:2] != target.shape[:2]:
                pred = F.interpolate(
                    pred.permute(2, 0, 1).unsqueeze(0),
                    size=target.shape[:2], mode="bilinear", align_corners=False,
                )[0].permute(1, 2, 0)
            if mask_gt is not None:
                tissue = 1.0 - mask_gt.clamp(0, 1)
                if depth_gt is not None:
                    tissue = tissue * (depth_gt > 0.01).float()
                m = tissue.unsqueeze(-1)
                photo = ((pred - target).abs() * m).sum() / m.sum().clamp_min(1)
            elif depth_gt is not None:
                tissue = (depth_gt > 0.01).float()
                m = tissue.unsqueeze(-1)
                photo = ((pred - target).abs() * m).sum() / m.sum().clamp_min(1)
            else:
                photo = (pred - target).abs().mean()
            loss = (w_photo * photo_scale) * photo

            if depth_gt is not None and out.depth is not None:
                rd = out.depth
                if rd.ndim == 3:
                    rd = rd[0]
                if rd.shape != depth_gt.shape:
                    rd = F.interpolate(
                        rd.unsqueeze(0).unsqueeze(0), size=depth_gt.shape,
                        mode="bilinear", align_corners=False,
                    )[0, 0]
                valid = depth_gt > 0.01
                if valid.any():
                    loss = loss + (w_depth * photo_scale) * F.l1_loss(rd[valid], depth_gt[valid])

            if regularizer == "arap":
                loss = loss + w_smooth * arap_energy(canonical, disp, knn)
            else:
                neighbor_mean = disp[knn].mean(dim=1)
                loss = loss + w_smooth * ((disp - neighbor_mean) ** 2).mean()
            if observation is not None:
                reprojection, depth_residual = correspondence_loss(
                    canonical + disp, cam, observation, depth_scale
                )
                # Reprojection is in pixels (~1-20); map onto photo scale.
                loss = loss + w_corr * (reprojection / 8.0) + w_corr_depth * depth_residual
            if inertia:
                loss = loss + w_temporal * ((disp - disp_m1) ** 2).mean()
            else:
                accel_ref = 2 * disp_m1 - disp_m2
                loss = loss + w_accel * ((disp - accel_ref) ** 2).mean()
            # Step cap (soft hinge on mean displacement).
            if step_cap is not None:
                mean_step = disp.norm(dim=-1).mean()
                loss = loss + 10.0 * F.relu(mean_step - step_cap)

            loss.backward()
            opt.step()
            curve.append(float(loss.item()))
            photo_curve.append(float(photo.detach()))
        return disp.detach(), curve, photo_curve

    previous_frame = None
    for frame in frames:
        disp_m1 = disp_hist[-1]
        disp_m2 = disp_hist[-2] if len(disp_hist) > 1 else disp_m1
        # Constant-velocity initialization.
        disp_init = disp_m1.clone() if inertia else 2 * disp_m1 - disp_m2
        prior = disp_m1
        observation = None
        if observer is not None:
            observation = observer.step(frame, disp_m1)
            disp_init = observer.initialize(frame, disp_m1, observation).to(disp_m1)
            prior = disp_init
        elif init_fn is not None and inertia:
            disp_init = init_fn(previous_frame, frame, canonical, disp_m1).to(disp_m1)
            prior = disp_init
        previous_frame = frame
        # Robust step cap from running median of past step magnitudes.
        step_cap = None
        if step_mags and not inertia:
            step_cap = max_step_ratio * float(np.median(step_mags))

        disp, curve, photo_curve = optimize_frame(
            frame, disp_init, lr, prior, disp_m2, step_cap, observation
        )
        fid = int(frame.view_id) if frame.view_id is not None else len(frame_ids)

        # Divergence rollback: photometric blow-up or mean|D| explosion.
        accepted = [fid_ for fid_ in frame_ids if fid_ not in flagged]
        prev_photo = photo_losses[accepted[-1]] if accepted else float("inf")
        prev_mag = float(disp_m1.norm(dim=-1).mean())
        photo_final = photo_curve[-1]
        mag = float(disp.norm(dim=-1).mean())
        # Sudden mean|D| jumps (not gradual large motion) or photo blow-ups.
        jump = mag - prev_mag
        exploded = accepted and (
            photo_final > divergence_factor * prev_photo
            or (prev_mag > 1e-4 and jump > max(2.5, 0.75 * prev_mag))
        )
        if exploded:
            disp, curve, photo_curve = optimize_frame(
                frame, disp_init, lr * 0.5, prior, disp_m2, step_cap, observation
            )
            photo_final = photo_curve[-1]
            mag = float(disp.norm(dim=-1).mean())
            jump = mag - prev_mag
            still = (
                photo_final > divergence_factor * prev_photo
                or (prev_mag > 1e-4 and jump > max(2.5, 0.75 * prev_mag))
            )
            if still:
                # Keep the gated correspondence init when photo is stable; only
                # freeze when the photometric residual itself diverged.
                if photo_final <= divergence_factor * prev_photo and observation is not None:
                    disp = disp_init.detach()
                    mag = float(disp.norm(dim=-1).mean())
                    flagged.append(fid)
                else:
                    disp = disp_m1.clone()
                    flagged.append(fid)
                    photo_final = prev_photo
                    mag = prev_mag

        disp_hist.append(disp)
        step_mags.append(mag)
        frame_ids.append(fid)
        losses[fid] = curve
        photo_losses[fid] = photo_final
        all_disps.append(disp.cpu().numpy())
        tag = " [ROLLBACK]" if fid in flagged else ""
        print(
            f"{log_prefix}frame {fid}: loss {curve[-1]:.5f} photo {photo_final:.5f} "
            f"mean|D| {mag:.3f} (scene units){tag}"
        )

    return DeformationResult(
        canonical_positions=canonical.cpu().numpy(),
        displacements=np.stack(all_disps, axis=0),
        frame_ids=frame_ids,
        losses=losses,
        notes=(
            (
                "inertia prior + divergence rollback; kNN-Laplacian smoothness"
                if inertia
                else "constant-velocity init + acceleration prior + step cap + "
                "divergence rollback; kNN-Laplacian smoothness"
            )
            + (f"; flagged frames: {flagged}" if flagged else "")
        ),
    )


# ---------------------------------------------------------------------------
# Visualization (intermediate results for inspection / paper figures)
# ---------------------------------------------------------------------------


def visualize_deformation(
    result: DeformationResult,
    frame_image: np.ndarray,
    camera,
    out_path: str | Path,
    *,
    frame_index: int = -1,
    max_arrows: int = 400,
) -> None:
    """Overlay the tracked 3D displacement field on the frame image.

    Projects canonical and displaced positions with the frame camera and
    draws a quiver field colored by displacement magnitude.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = frame_index if frame_index >= 0 else result.displacements.shape[0] - 1
    canonical = torch.from_numpy(result.canonical_positions).float()
    disp = torch.from_numpy(result.displacements[t]).float()
    deformed = canonical + disp

    K = camera.K.cpu()
    c2w = camera.c2w.cpu()
    w2c = torch.linalg.inv(c2w)

    def project(pts):
        cam_pts = (pts - w2c[:3, 3]) @ w2c[:3, :3]  # world->camera
        z = cam_pts[:, 2].clamp_min(1e-6)
        uv = torch.stack(
            [K[0, 0] * cam_pts[:, 0] / z + K[0, 2],
             K[1, 1] * cam_pts[:, 1] / z + K[1, 2]], dim=-1)
        return uv, z

    uv0, z0 = project(canonical)
    uv1, _ = project(deformed)
    H, W = frame_image.shape[:2]
    inside = (uv0[:, 0] >= 0) & (uv0[:, 0] < W) & (uv0[:, 1] >= 0) & (uv0[:, 1] < H) & (z0 > 0)
    idx = torch.nonzero(inside).squeeze(-1)
    if idx.numel() > max_arrows:
        idx = idx[torch.randperm(idx.numel())[:max_arrows]]
    mag = disp[idx].norm(dim=-1).numpy()  # scene units

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
    ax.imshow(frame_image)
    q = ax.quiver(
        uv0[idx, 0].numpy(), uv0[idx, 1].numpy(),
        (uv1[idx, 0] - uv0[idx, 0]).numpy(), (uv1[idx, 1] - uv0[idx, 1]).numpy(),
        mag, cmap="turbo", angles="xy", scale_units="xy", scale=1.5, width=0.004,
    )
    fig.colorbar(q, ax=ax, label="displacement (scene units)")
    ax.set_title(f"Tier-2 tracked deformation (frame {result.frame_ids[t]})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def save_deformation_summary(result: DeformationResult, path: str | Path) -> None:
    """Write a compact JSON summary of the tracking run."""
    mags = result.displacement_magnitudes  # (T,N)
    summary = {
        "n_points": int(result.canonical_positions.shape[0]),
        "n_frames": len(result.frame_ids),
        "frame_ids": result.frame_ids,
        "mean_displacement_per_frame": mags.mean(axis=1).tolist(),
        "max_displacement_per_frame": mags.max(axis=1).tolist(),
        "units": "scene_units (EndoNeRF near/far scale, not metres)",
        "final_losses": {str(k): v[-1] for k, v in result.losses.items()},
        "notes": result.notes,
    }
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
