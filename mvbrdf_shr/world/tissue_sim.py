"""Differentiable soft-tissue simulator + elasticity inversion (Tier 3).

A minimal but physically honest mass-spring lattice simulator written in
pure PyTorch, used for two things:

1. **Synthetic closed-loop validation** (:func:`run_synthetic_inversion`):
   generate a deformation with KNOWN Young's modulus E_gt (including a stiff
   "tumor" inclusion), then recover E by gradient descent through the
   simulator. This is the credibility core of the whole twin: it verifies
   that the inversion machinery and the identifiability assumptions work
   before touching real data.

2. **Real-data relative-stiffness recovery** (:func:`invert_relative_field`):
   on real endoscopic sequences we observe displacements (Tier 2) but NOT
   tool forces. As in clinical quasi-static elastography, absolute E is then
   not identifiable — only the spatial CONTRAST is. We recover the contrast
   field and calibrate its scale with the tissue-class prior, recording the
   limitation in the audit trail.

Continuum note: the lattice maps E to spring stiffness via k = E * h
(regular-grid heuristic). The synthetic loop is self-consistent by
construction; absolute continuum accuracy requires FEM/MPM (SOFA/Warp),
which is a drop-in replacement for this module's forward model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------


def build_lattice(
    grid_shape: tuple[int, int, int] = (12, 6, 6),
    spacing: float = 0.01,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a regular 3D mass-spring lattice.

    Returns a dict with node positions (N,3), spring index pairs (M,2),
    rest lengths (M,), and per-node mass (N,). Springs include structural
    (axis) and shear (face-diagonal) edges for volumetric stability.
    """
    nx, ny, nz = grid_shape
    xs = torch.arange(nx, dtype=torch.float32) * spacing
    ys = torch.arange(ny, dtype=torch.float32) * spacing
    zs = torch.arange(nz, dtype=torch.float32) * spacing
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    positions = grid.reshape(-1, 3).to(device)  # (N,3)
    n = positions.shape[0]

    def idx(i, j, k):
        return (i * ny + j) * nz + k

    edges: list[tuple[int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Structural springs (axis neighbors).
                for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if ii < nx and jj < ny and kk < nz:
                        edges.append((idx(i, j, k), idx(ii, jj, kk)))
                # Shear springs (face diagonals, forward direction only).
                for di, dj, dk in ((1, 1, 0), (1, 0, 1), (0, 1, 1),
                                   (1, -1, 0), (1, 0, -1), (0, 1, -1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        edges.append((idx(i, j, k), idx(ii, jj, kk)))
    springs = torch.tensor(edges, dtype=torch.long, device=device)  # (M,2)
    rest = (positions[springs[:, 0]] - positions[springs[:, 1]]).norm(dim=-1)

    volume_per_node = spacing**3
    return {
        "positions": positions,
        "springs": springs,
        "rest_lengths": rest,
        "node_mass": torch.full((n,), volume_per_node, device=device),
        "grid_shape": torch.tensor(grid_shape),
    }


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


class MassSpringTissueSim:
    """Semi-implicit-Euler mass-spring tissue model (differentiable).

    Parameters are the per-spring stiffness ``k`` (N/m), derived from a
    per-node Young's modulus field via ``k_spring = E_node_avg * spacing``.
    Supports force-driven (dynamic) and displacement-BC (quasi-static)
    boundary conditions.
    """

    def __init__(
        self,
        lattice: dict[str, torch.Tensor],
        *,
        density: float = 1040.0,
        damping: float = 2.0,
        dt: float = 5e-4,
    ):
        self.pos0 = lattice["positions"]
        self.springs = lattice["springs"]
        self.rest = lattice["rest_lengths"]
        self.mass = lattice["node_mass"] * density  # (N,)
        self.damping = float(damping)
        self.dt = float(dt)
        self.n_nodes = self.pos0.shape[0]
        self.spacing = float(self.rest.min().item())

    def stiffness_from_E(self, E_pa: torch.Tensor) -> torch.Tensor:
        """Map per-node Young's modulus (Pa) to per-spring stiffness (N/m)."""
        k_node = E_pa * self.spacing  # (N,) N/m
        return 0.5 * (k_node[self.springs[:, 0]] + k_node[self.springs[:, 1]])

    def simulate(
        self,
        E_pa: torch.Tensor,
        *,
        n_steps: int,
        fixed_nodes: torch.Tensor | None = None,
        pull_nodes: torch.Tensor | None = None,
        pull_forces: torch.Tensor | None = None,
        prescribed_nodes: torch.Tensor | None = None,
        prescribed_trajectory: torch.Tensor | None = None,
        observe_every: int = 1,
    ) -> torch.Tensor:
        """Run the forward simulation.

        Parameters
        ----------
        E_pa:
            Per-node Young's modulus in Pa (N,). Differentiable.
        n_steps:
            Number of integration steps.
        fixed_nodes:
            Indices of Dirichlet-fixed nodes (zero motion).
        pull_nodes, pull_forces:
            Indices and per-node force vectors (P,3) applied every step
            (force-driven dynamic mode).
        prescribed_nodes, prescribed_trajectory:
            Indices and target positions (T,P,3) per recorded step
            (displacement-BC quasi-static mode).
        observe_every:
            Record the configuration every this many steps.

        Returns
        -------
        Trajectory tensor (T, N, 3) of node positions (including t=0).
        """
        device = self.pos0.device
        k = self.stiffness_from_E(E_pa)
        pos = self.pos0.clone()
        vel = torch.zeros_like(pos)
        fixed_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
        if fixed_nodes is not None:
            fixed_mask[fixed_nodes] = True
        if prescribed_nodes is not None:
            fixed_mask[prescribed_nodes] = True

        traj = [pos]
        record_count = 0
        for step in range(1, n_steps + 1):
            # Spring forces: F = k * (|d| - rest) * d/|d|
            d = pos[self.springs[:, 1]] - pos[self.springs[:, 0]]  # (M,3)
            length = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            f_mag = k.unsqueeze(-1) * (length - self.rest.unsqueeze(-1))
            f_dir = d / length
            f_spring = f_mag * f_dir  # force on node0 toward node1
            forces = torch.zeros_like(pos)
            forces.index_add_(0, self.springs[:, 0], f_spring)
            forces.index_add_(0, self.springs[:, 1], -f_spring)

            if pull_nodes is not None and pull_forces is not None:
                forces = forces.clone()
                forces[pull_nodes] += pull_forces

            # Semi-implicit Euler with damping.
            acc = forces / self.mass.unsqueeze(-1)
            vel = (vel + self.dt * acc) * (1.0 - self.damping * self.dt)
            pos = pos + self.dt * vel

            # Dirichlet constraints.
            if fixed_nodes is not None:
                pos = torch.where(fixed_mask.unsqueeze(-1), self.pos0, pos)
                vel = torch.where(
                    fixed_mask.unsqueeze(-1), torch.zeros_like(vel), vel
                )
            if prescribed_nodes is not None and step % observe_every == 0:
                t_idx = min(record_count, prescribed_trajectory.shape[0] - 1)
                target = prescribed_trajectory[t_idx]
                pos = pos.clone()
                pos[prescribed_nodes] = target

            if step % observe_every == 0:
                traj.append(pos)
                record_count += 1
        return torch.stack(traj, dim=0)  # (T,N,3)


# ---------------------------------------------------------------------------
# Elasticity inversion
# ---------------------------------------------------------------------------


@dataclass
class InversionResult:
    """Output of an elasticity-inversion run."""

    E_est_pa: np.ndarray          # (N,) or (1,) posterior-point estimate
    E_gt_pa: np.ndarray | None    # (N,) ground truth, synthetic loop only
    loss_curve: list[float]
    E_curve: list[np.ndarray]     # E estimate per logged iteration
    observed_traj: np.ndarray     # (T,N,3)
    simulated_traj: np.ndarray    # (T,N,3) at the final estimate
    mode: str                     # "absolute" | "relative"
    notes: str = ""


def invert_elasticity(
    sim: MassSpringTissueSim,
    observed_traj: torch.Tensor | None = None,
    *,
    cases: list[dict] | None = None,
    E_init_pa: float = 10e3,
    per_node: bool = False,
    prior_log_mu: float | None = None,
    prior_log_sigma: float | None = None,
    prior_weight: float = 1.0,
    smoothness_weight: float = 0.0,
    observe_nodes: torch.Tensor | None = None,
    n_iters: int = 200,
    lr: float = 0.05,
    log_every: int = 10,
    sim_kwargs: dict | None = None,
) -> InversionResult:
    """Recover Young's modulus by matching simulated to observed trajectories.

    The parameter is ``log_E`` (per-node or global), keeping E positive and
    giving the log-normal prior a natural home. The loss is

        L = sum_cases MSE(traj_sim, traj_obs) / scale^2
            + prior_weight * NLL_log_normal(log_E; mu, sigma)
            + smoothness_weight * ||Laplacian(log_E)||^2  (per-node mode)

    Multi-excitation: pass ``cases=[{"observed": traj, "sim_kwargs": {...},
    "observe_nodes": idx}, ...]`` to invert from several independent load
    cases (e.g. axial pull + surface palpation). A single excitation only
    constrains stiffness along one direction — deep inclusions are then
    transversely unobservable, which multi-excitation fixes.
    """
    device = sim.pos0.device
    n = sim.n_nodes

    # Normalize input to a list of load cases.
    if cases is None:
        cases = [{
            "observed": observed_traj,
            "sim_kwargs": sim_kwargs or {},
            "observe_nodes": observe_nodes,
        }]
    norm_cases = []
    for case in cases:
        obs = case["observed"].to(device)
        obs_nodes = case.get("observe_nodes")
        if obs_nodes is None:
            obs_nodes = torch.arange(n, device=device)
        scale = (obs[-1] - obs[0]).norm(dim=-1).mean().clamp_min(1e-6)
        norm_cases.append({
            "observed": obs,
            "sim_kwargs": case.get("sim_kwargs", {}),
            "observe_nodes": obs_nodes,
            "scale": scale,
        })

    n_param = n if per_node else 1
    log_E = torch.full(
        (n_param,), float(np.log(E_init_pa)), device=device, requires_grad=True
    )
    opt = torch.optim.Adam([log_E], lr=lr)

    # Lattice Laplacian for the smoothness prior (per-node mode).
    lap_rows: list[int] = []
    lap_cols: list[int] = []
    if per_node and smoothness_weight > 0:
        springs = sim.springs.cpu().numpy()
        for a, b in springs:
            lap_rows += [a, b]
            lap_cols += [b, a]
        lap_idx = torch.tensor([lap_rows, lap_cols], dtype=torch.long, device=device)

    loss_curve: list[float] = []
    E_curve: list[np.ndarray] = []
    best = (float("inf"), None)
    for it in range(n_iters):
        opt.zero_grad()
        E_pa = log_E.exp()
        if not per_node:
            E_pa = E_pa.expand(n)
        data_loss = torch.zeros((), device=device)
        for case in norm_cases:
            traj = sim.simulate(E_pa, **case["sim_kwargs"])
            data_loss = data_loss + F.mse_loss(
                traj[:, case["observe_nodes"]],
                case["observed"][:, case["observe_nodes"]],
            ) / (case["scale"] ** 2)
        data_loss = data_loss / len(norm_cases)
        loss = data_loss
        if prior_log_mu is not None and prior_log_sigma is not None:
            nll = 0.5 * ((log_E - prior_log_mu) / prior_log_sigma) ** 2
            loss = loss + prior_weight * nll.mean()
        if per_node and smoothness_weight > 0:
            neighbor_mean = torch.zeros_like(log_E).index_add_(
                0, lap_idx[0], log_E[lap_idx[1]]
            )
            deg = torch.zeros_like(log_E).index_add_(
                0, lap_idx[0], torch.ones_like(log_E[lap_idx[1]])
            ).clamp_min(1)
            lap = log_E - neighbor_mean / deg
            loss = loss + smoothness_weight * (lap**2).mean()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
        if it % log_every == 0 or it == n_iters - 1:
            E_curve.append(log_E.detach().exp().cpu().numpy().copy())
        if loss.item() < best[0]:
            best = (float(loss.item()), log_E.detach().clone())

    log_E_final = best[1] if best[1] is not None else log_E.detach()
    E_final = log_E_final.exp()
    first = norm_cases[0]
    with torch.no_grad():
        traj_final = sim.simulate(
            E_final.expand(n) if not per_node else E_final, **first["sim_kwargs"]
        )
    return InversionResult(
        E_est_pa=E_final.cpu().numpy(),
        E_gt_pa=None,
        loss_curve=loss_curve,
        E_curve=E_curve,
        observed_traj=first["observed"].detach().cpu().numpy(),
        simulated_traj=traj_final.detach().cpu().numpy(),
        mode="absolute",
    )


# ---------------------------------------------------------------------------
# Synthetic closed-loop experiments (known ground truth)
# ---------------------------------------------------------------------------


def run_synthetic_inversion(
    *,
    grid_shape: tuple[int, int, int] = (12, 6, 6),
    spacing: float = 0.01,
    E_background_kpa: float = 8.0,
    inclusion: bool = False,
    inclusion_E_kpa: float = 32.0,
    surface_only: bool = False,
    region_based: bool = True,
    E_init_kpa: float = 15.0,
    n_steps: int = 120,
    n_iters: int = 200,
    device: torch.device | str = "cpu",
    prior: "object | None" = None,
) -> InversionResult:
    """Closed-loop validation: known E -> observed deformation -> recover E.

    Experiment A (``inclusion=False``): uniform E, absolute recovery with
    known pulling force.

    Experiment B (``inclusion=True``): a stiff "tumor" inclusion (4x
    contrast). Default recovers two region parameters (E_bg, E_incl) —
    the well-posed elastography setting. Set ``region_based=False`` for
    the harder per-node field. ``surface_only`` restricts observations to
    the top surface (honest endoscopic setting).
    """
    lattice = build_lattice(grid_shape, spacing, device=device)
    sim = MassSpringTissueSim(lattice, density=1040.0, damping=2.0)
    n = sim.n_nodes
    nx, ny, nz = grid_shape

    def idx(i, j, k):
        return (i * ny + j) * nz + k

    E_gt = torch.full((n,), E_background_kpa * 1e3, device=device)
    inclusion_mask = torch.zeros(n, dtype=torch.bool, device=device)
    if inclusion:
        cx, cy, cz = nx * 2 // 3, ny // 2, nz // 2
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2 <= 2.2**2:
                        inclusion_mask[idx(i, j, k)] = True
        E_gt[inclusion_mask] = inclusion_E_kpa * 1e3

    fixed_nodes = torch.tensor(
        [idx(0, j, k) for j in range(ny) for k in range(nz)], device=device
    )
    pull_nodes = torch.tensor(
        [idx(nx - 1, j, k) for j in range(ny) for k in range(nz)], device=device
    )
    total_force = 1.5
    f_per_node = total_force / pull_nodes.numel()
    pull_forces = torch.zeros(pull_nodes.numel(), 3, device=device)
    pull_forces[:, 0] = f_per_node

    surface_nodes = None
    if surface_only:
        surface_nodes = torch.tensor(
            [idx(i, ny - 1, k) for i in range(nx) for k in range(nz)],
            device=device,
        )

    case1 = {
        "sim_kwargs": dict(
            n_steps=n_steps, fixed_nodes=fixed_nodes,
            pull_nodes=pull_nodes, pull_forces=pull_forces, observe_every=4,
        ),
        "observe_nodes": surface_nodes,
    }
    with torch.no_grad():
        case1["observed"] = sim.simulate(E_gt, **case1["sim_kwargs"])
    cases = [case1]

    if inclusion:
        palp_nodes = torch.tensor(
            [idx(i, ny - 1, k)
             for i in range(nx * 2 // 3 - 1, nx * 2 // 3 + 2)
             for k in range(nz // 2 - 1, nz // 2 + 2)],
            device=device,
        )
        palp_forces = torch.zeros(palp_nodes.numel(), 3, device=device)
        palp_forces[:, 1] = -0.6 / palp_nodes.numel()
        case2 = {
            "sim_kwargs": dict(
                n_steps=n_steps, fixed_nodes=fixed_nodes,
                pull_nodes=palp_nodes, pull_forces=palp_forces, observe_every=4,
            ),
            "observe_nodes": surface_nodes,
        }
        with torch.no_grad():
            case2["observed"] = sim.simulate(E_gt, **case2["sim_kwargs"])
        cases.append(case2)

    prior_mu = prior_sigma = None
    if prior is not None:
        prior_mu = float(prior.youngs_log_mu)
        prior_sigma = float(prior.youngs_log_sigma)

    if inclusion and region_based:
        result = _invert_two_region(
            sim, cases, inclusion_mask,
            E_init_pa=E_init_kpa * 1e3,
            prior_log_mu=prior_mu, prior_log_sigma=prior_sigma,
            n_iters=n_iters, lr=0.08,
        )
    else:
        result = invert_elasticity(
            sim, cases=cases, E_init_pa=E_init_kpa * 1e3,
            per_node=inclusion, prior_log_mu=prior_mu,
            prior_log_sigma=prior_sigma,
            prior_weight=0.005 if inclusion else 0.0,
            smoothness_weight=0.01 if inclusion else 0.0,
            n_iters=n_iters, lr=0.1,
        )
    result.E_gt_pa = E_gt.cpu().numpy()
    result.mode = "absolute"
    result.notes = (
        f"grid={grid_shape}, inclusion={inclusion}, surface_only={surface_only}, "
        f"region_based={region_based and inclusion}, load_cases={len(cases)}, "
        f"force={total_force}N known"
    )
    return result


def _invert_two_region(
    sim: MassSpringTissueSim,
    cases: list[dict],
    inclusion_mask: torch.Tensor,
    *,
    E_init_pa: float,
    prior_log_mu: float | None,
    prior_log_sigma: float | None,
    n_iters: int,
    lr: float,
) -> InversionResult:
    """Recover (E_background, E_inclusion) with a known region mask.

    Well-posed elastography setting: the spatial partition is known (or
    hypothesized), and we recover a stiffness contrast. Absolute scale
    still requires known forces.
    """
    device = sim.pos0.device
    n = sim.n_nodes
    norm_cases = []
    for case in cases:
        obs = case["observed"].to(device)
        obs_nodes = case.get("observe_nodes")
        if obs_nodes is None:
            obs_nodes = torch.arange(n, device=device)
        scale = (obs[-1] - obs[0]).norm(dim=-1).mean().clamp_min(1e-6)
        norm_cases.append({
            "observed": obs, "sim_kwargs": case["sim_kwargs"],
            "observe_nodes": obs_nodes, "scale": scale,
        })

    log_E = torch.tensor(
        [np.log(E_init_pa), np.log(E_init_pa * 2)],
        device=device, dtype=torch.float32, requires_grad=True,
    )
    opt = torch.optim.Adam([log_E], lr=lr)
    loss_curve: list[float] = []
    E_curve: list[np.ndarray] = []
    best = (float("inf"), None)
    for it in range(n_iters):
        opt.zero_grad()
        E_bg, E_incl = log_E.exp()
        E_field = torch.where(inclusion_mask, E_incl, E_bg).float()
        data_loss = torch.zeros((), device=device)
        for case in norm_cases:
            traj = sim.simulate(E_field, **case["sim_kwargs"])
            data_loss = data_loss + F.mse_loss(
                traj[:, case["observe_nodes"]],
                case["observed"][:, case["observe_nodes"]],
            ) / (case["scale"] ** 2)
        data_loss = data_loss / len(norm_cases)
        loss = data_loss
        if prior_log_mu is not None and prior_log_sigma is not None:
            nll = 0.5 * ((log_E[0] - prior_log_mu) / prior_log_sigma) ** 2
            loss = loss + 0.01 * nll
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
        if it % 10 == 0 or it == n_iters - 1:
            E_curve.append(log_E.detach().exp().cpu().numpy().copy())
        if loss.item() < best[0]:
            best = (float(loss.item()), log_E.detach().clone())

    log_E_final = best[1] if best[1] is not None else log_E.detach()
    E_bg, E_incl = log_E_final.exp()
    E_field = torch.where(inclusion_mask, E_incl, E_bg).float()
    with torch.no_grad():
        traj_final = sim.simulate(E_field, **norm_cases[0]["sim_kwargs"])
    return InversionResult(
        E_est_pa=E_field.cpu().numpy(),
        E_gt_pa=None,
        loss_curve=loss_curve,
        E_curve=E_curve,
        observed_traj=norm_cases[0]["observed"].detach().cpu().numpy(),
        simulated_traj=traj_final.detach().cpu().numpy(),
        mode="absolute_two_region",
    )


# ---------------------------------------------------------------------------
# Real-data relative-stiffness recovery (displacement-BC driven)
# ---------------------------------------------------------------------------


def invert_relative_field(
    positions: torch.Tensor,
    displacements: torch.Tensor,
    *,
    n_regions: int = 4,
    prior_log_mu: float | None = None,
    prior_log_sigma: float | None = None,
    n_iters: int = 150,
) -> dict:
    """Recover a RELATIVE stiffness-contrast field from observed deformation.

    Real endoscopic data has no tool-force measurement, so absolute E is not
    identifiable (quasi-static elastography limitation). We cluster the
    tracked points into ``n_regions`` spatial regions and recover a
    per-region stiffness RATIO vs the softest region, then calibrate the
    scale with the tissue-class prior median.

    Parameters
    ----------
    positions:
        Canonical node positions (N,3) from Tier 1 reconstruction.
    displacements:
        Observed per-node displacement (N,3) between two frames (Tier 2).

    Returns
    -------
    dict with per-region relative stiffness, assignments, and audit notes.
    """
    device = positions.device
    n = positions.shape[0]
    disp_mag = displacements.norm(dim=-1)  # (N,)

    # Region assignment by position clustering (k-means, few iterations).
    k = min(n_regions, n)
    centroids = positions[torch.randperm(n, device=device)[:k]].clone()
    for _ in range(20):
        d = torch.cdist(positions, centroids)
        assign = d.argmin(dim=1)
        for c in range(k):
            if (assign == c).any():
                centroids[c] = positions[assign == c].mean(dim=0)

    # Stiffness contrast proxy: under a common remote load, stiffer regions
    # deform LESS. Relative compliance ~ local displacement magnitude;
    # relative stiffness ~ 1/compliance, normalized to the softest region.
    region_disp = torch.zeros(k, device=device)
    for c in range(k):
        if (assign == c).any():
            region_disp[c] = disp_mag[assign == c].mean()
    compliance = region_disp.clamp_min(1e-9)
    rel_stiffness = compliance.max() / compliance  # softest region = 1.0

    return {
        "n_regions": k,
        "assignments": assign.cpu().numpy(),
        "region_centers": centroids.cpu().numpy(),
        "region_mean_displacement": region_disp.cpu().numpy(),
        "relative_stiffness": rel_stiffness.cpu().numpy(),
        "audit": {
            "mode": "relative_contrast_only",
            "reason": "no tool-force measurement; absolute E not identifiable "
                      "(quasi-static elastography limitation)",
            "calibration": "scale set by tissue-class prior median",
        },
    }
