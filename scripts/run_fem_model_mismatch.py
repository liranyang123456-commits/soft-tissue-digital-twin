"""Model-mismatch benchmark for the two-region Neo-Hookean inversion.

The original FEM60 campaign generates and inverts data with the same coarse
Neo-Hookean model. Here the observations of each scenario are regenerated
under three mismatched forward models and inverted with the unchanged coarse
Neo-Hookean model:

    matched     coarse Neo-Hookean (reference, regenerated)
    mooney      coarse compressible Mooney-Rivlin, C01 = C10, identical
                small-strain moduli
    fine        Neo-Hookean on a uniformly refined mesh (1:8 tetrahedra),
                loads transferred by virtual-work-consistent prolongation
    mooney_fine Mooney-Rivlin on the refined mesh

The force measurement error, the surface-motion noise realization, the
observed nodes, the loading frames, and the inversion protocol (Adam on
log-moduli, 20 iterations, learning rate 0.1) are those of FEM60.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

DIFF_ROOT = Path(r"E:\Diff_Rending_Re_3D")
if str(DIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFF_ROOT))

from physics import deformation_gradient  # noqa: E402
from physics.fem import make_heterogeneous_E_field  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = DIFF_ROOT / "dataset" / "ion_ct_synthetic_mechanics60"
OUT = ROOT / "outputs" / "fem_model_mismatch"
NU = 0.45
FRAMES = (1, 2, 4)
CONDITIONS = ("matched", "mooney", "fine", "mooney_fine")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Constitutive laws (per-element parameter tensors)
# ---------------------------------------------------------------------------


def lame(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))


def _invariants(F: torch.Tensor):
    C = F.transpose(-1, -2) @ F
    I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)
    I2 = 0.5 * (I1**2 - (C @ C).diagonal(dim1=-2, dim2=-1).sum(-1))
    logJ = torch.log(torch.linalg.det(F).clamp_min(1e-9))
    return I1, I2, logJ


def neo_hookean(F: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    mu, lam = lame(E)
    I1, _, logJ = _invariants(F)
    return 0.5 * mu * (I1 - 3) - mu * logJ + 0.5 * lam * logJ**2


def mooney_rivlin(F: torch.Tensor, E: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Compressible Mooney-Rivlin with the Neo-Hookean small-strain moduli.

    psi = C10 (I1-3) + C01 (I2-3) - 2(C10+2C01) ln J + lam'/2 (ln J)^2 with
    C10 = (1-ratio) mu/2, C01 = ratio mu/2 and lam' = lam - 4 C01, so that the
    linearized shear and Lame moduli equal those of the Neo-Hookean model.
    """
    mu, lam = lame(E)
    c10, c01 = (1 - ratio) * mu / 2, ratio * mu / 2
    I1, I2, logJ = _invariants(F)
    return (
        c10 * (I1 - 3)
        + c01 * (I2 - 3)
        - 2 * (c10 + 2 * c01) * logJ
        + 0.5 * (lam - 4 * c01) * logJ**2
    )


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------


class Mesh:
    def __init__(self, nodes: torch.Tensor, elems: torch.Tensor):
        self.X = nodes.to(DTYPE)
        self.elems = elems.long()
        self.Nn = self.X.shape[0]
        self.Ne = self.elems.shape[0]
        self.Xe = self.X[self.elems]
        cols = torch.stack([self.Xe[:, k + 1] - self.Xe[:, 0] for k in range(3)], dim=-1)
        self.vol = torch.linalg.det(cols).abs() / 6.0
        self.loc_dofs = torch.stack(
            [3 * self.elems[:, n] + a for n in range(4) for a in range(3)], dim=1
        )
        rows = self.loc_dofs.unsqueeze(-1).expand(-1, -1, 12).reshape(-1)
        cols_ = self.loc_dofs.unsqueeze(-2).expand(-1, 12, -1).reshape(-1)
        self.k_rows = rows.numpy()
        self.k_cols = cols_.numpy()

    def element_E(self, E_nodes: torch.Tensor) -> torch.Tensor:
        return E_nodes[self.elems].mean(dim=1)


def boundary_faces(elems: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        [elems[:, [0, 1, 2]], elems[:, [0, 1, 3]], elems[:, [0, 2, 3]], elems[:, [1, 2, 3]]]
    )
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def refine(nodes: np.ndarray, elems: np.ndarray):
    """Uniform 1:8 refinement; original nodes keep their indices."""
    edge_index: dict[tuple[int, int], int] = {}
    parents: list[tuple[int, int]] = []
    new_nodes = [nodes]

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in edge_index:
            edge_index[key] = len(nodes) + len(parents)
            parents.append(key)
        return edge_index[key]

    fine = []
    for a, b, c, d in elems:
        ab, ac, ad = midpoint(a, b), midpoint(a, c), midpoint(a, d)
        bc, bd, cd = midpoint(b, c), midpoint(b, d), midpoint(c, d)
        fine += [
            (a, ab, ac, ad), (b, ab, bc, bd), (c, ac, bc, cd), (d, ad, bd, cd),
            (ab, ac, ad, bd), (ab, ac, bc, bd), (ac, ad, bd, cd), (ac, bc, bd, cd),
        ]
    parents_arr = np.asarray(parents, dtype=np.int64)
    new_nodes.append(0.5 * (nodes[parents_arr[:, 0]] + nodes[parents_arr[:, 1]]))
    return np.concatenate(new_nodes), np.asarray(fine, dtype=np.int64), parents_arr


def transfer_forces(
    coarse_nodes: int, parents: np.ndarray, fine_elems: np.ndarray, forces: np.ndarray
) -> np.ndarray:
    """Fine surface loads f_f = P (P^T P)^-1 f_c, so that P^T f_f = f_c.

    P is the linear prolongation restricted to boundary nodes; the transferred
    loads therefore do the same virtual work as the coarse nodal loads for any
    displacement field representable on the coarse mesh.
    """
    n_fine = coarse_nodes + len(parents)
    fine_boundary = np.unique(boundary_faces(fine_elems))
    coarse_boundary = fine_boundary[fine_boundary < coarse_nodes]
    loaded = np.flatnonzero(np.abs(forces.reshape(-1, 3)).sum(1) > 0)
    if not np.isin(loaded, coarse_boundary).all():
        raise ValueError("loaded nodes must lie on the boundary")
    col = {node: k for k, node in enumerate(coarse_boundary)}
    rows, cols, vals = [], [], []
    for r, node in enumerate(fine_boundary):
        if node < coarse_nodes:
            rows.append(r); cols.append(col[node]); vals.append(1.0)
        else:
            a, b = parents[node - coarse_nodes]
            rows += [r, r]; cols += [col[a], col[b]]; vals += [0.5, 0.5]
    P = sp.csr_matrix((vals, (rows, cols)), shape=(len(fine_boundary), len(coarse_boundary)))
    gram = (P.T @ P).toarray()
    f_c = forces.reshape(-1, 3)[coarse_boundary]
    f_boundary = P @ np.linalg.solve(gram, f_c)
    out = np.zeros((n_fine, 3))
    out[fine_boundary] = f_boundary
    return out.reshape(-1)


def fine_fixed_dofs(coarse_fixed_nodes: np.ndarray, coarse_nodes: int, parents: np.ndarray) -> np.ndarray:
    fixed = set(int(n) for n in coarse_fixed_nodes)
    nodes = sorted(fixed) + [
        coarse_nodes + k for k, (a, b) in enumerate(parents) if a in fixed and b in fixed
    ]
    nodes = np.asarray(nodes, dtype=np.int64)
    return (3 * nodes[:, None] + np.arange(3)[None]).reshape(-1)


# ---------------------------------------------------------------------------
# Assembly and Newton solver
# ---------------------------------------------------------------------------


def internal_force(mesh: Mesh, u: torch.Tensor, psi, E_e: torch.Tensor, tangent: bool):
    uloc = u.view(-1, 3)[mesh.elems].reshape(mesh.Ne, 12).detach().clone().requires_grad_(True)
    with torch.enable_grad():
        F = deformation_gradient(mesh.Xe + uloc.view(mesh.Ne, 4, 3), mesh.Xe)
        energy = (mesh.vol * psi(F, E_e)).sum()
        fe = torch.autograd.grad(energy, uloc, create_graph=tangent)[0]
        K = None
        if tangent:
            blocks = []
            for c in range(12):
                g = torch.zeros_like(fe)
                g[:, c] = 1.0
                blocks.append(torch.autograd.grad(fe, uloc, grad_outputs=g, retain_graph=True)[0])
            K_loc = torch.stack(blocks, dim=-1).detach()
            n = 3 * mesh.Nn
            K = sp.csr_matrix(
                (K_loc.reshape(-1).numpy(), (mesh.k_rows, mesh.k_cols)), shape=(n, n)
            )
    fint = torch.zeros(3 * mesh.Nn, dtype=DTYPE)
    fint.index_add_(0, mesh.loc_dofs.reshape(-1), fe.detach().reshape(-1))
    return fint, K


def newton(mesh: Mesh, psi, E_e, f_ext, fixed_dofs, *, load_steps=8, max_iter=50, tol=1e-8):
    n = 3 * mesh.Nn
    free = np.setdiff1d(np.arange(n), np.asarray(fixed_dofs))
    u = torch.zeros(n, dtype=DTYPE)
    for step in list(range(1, load_steps + 1)) + [load_steps]:
        f_s = f_ext * (step / load_steps)
        for _ in range(max_iter):
            fint, K = internal_force(mesh, u, psi, E_e, tangent=True)
            r = (f_s - fint).numpy()[free]
            if np.linalg.norm(r) < tol:
                break
            Kff = K[free][:, free].tocsc()
            du = np.zeros(n)
            du[free] = spla.spsolve(Kff, r)
            u = u + torch.from_numpy(du)
    return u


class TwoRegionSolve(torch.autograd.Function):
    """Equilibrium u(E_bg, E_incl) with an adjoint gradient.

    The internal force is linear in (E_bg, E_incl) at fixed u, so
    d f_int / d E_bg and d f_int / d E_incl are the internal forces of the
    unit-modulus background and inclusion fields.
    """

    @staticmethod
    def forward(ctx, theta, mesh, weight, f_ext, fixed_dofs):
        E_e = mesh.element_E(theta[0] * (1 - weight) + theta[1] * weight)
        u = newton(mesh, neo_hookean, E_e, f_ext, fixed_dofs)
        ctx.mesh, ctx.weight, ctx.fixed = mesh, weight, fixed_dofs
        ctx.save_for_backward(theta, u)
        return u

    @staticmethod
    def backward(ctx, grad_u):
        theta, u = ctx.saved_tensors
        mesh, weight = ctx.mesh, ctx.weight
        n = 3 * mesh.Nn
        free = np.setdiff1d(np.arange(n), np.asarray(ctx.fixed))
        E_e = mesh.element_E(theta[0] * (1 - weight) + theta[1] * weight)
        _, K = internal_force(mesh, u, neo_hookean, E_e, tangent=True)
        adjoint = np.zeros(n)
        adjoint[free] = spla.spsolve(K[free][:, free].tocsc().T, grad_u.numpy()[free])
        adjoint = torch.from_numpy(adjoint)
        grads = []
        for field in (1 - weight, weight):
            dfint, _ = internal_force(mesh, u, neo_hookean, mesh.element_E(field), tangent=False)
            grads.append(-(adjoint * dfint).sum())
        return torch.stack(grads), None, None, None, None


def invert(mesh, center, radius, forces, fixed_dofs, surf, u_obs, *, iters=20, lr=0.1):
    weight = torch.sigmoid((radius - (mesh.X - center).norm(dim=1)) / 0.02)
    scale = u_obs.norm(dim=-1).mean().clamp_min(1e-9)
    log_E = torch.tensor([np.log(10e3), np.log(20e3)], dtype=DTYPE, requires_grad=True)
    optimizer = torch.optim.Adam([log_E], lr=lr)
    curve = []
    for _ in range(iters):
        optimizer.zero_grad()
        theta = log_E.exp()
        loss = torch.zeros((), dtype=DTYPE)
        for k, frame in enumerate(FRAMES):
            u = TwoRegionSolve.apply(theta, mesh, weight, forces[frame], fixed_dofs)
            loss = loss + torch.mean(((u.view(-1, 3)[surf] - u_obs[k]) / scale) ** 2)
        loss = loss / len(FRAMES)
        loss.backward()
        optimizer.step()
        curve.append(loss.item())
    E_bg, E_incl = log_E.detach().exp().tolist()
    return E_bg, E_incl, curve


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------


def run_scenario(scenario: Path, conditions: tuple[str, ...]) -> dict:
    gt = torch.load(scenario / "press_left" / "gt.pt", map_location="cpu", weights_only=False)
    nodes, elems = gt["nodes"].to(DTYPE), gt["elems"].long()
    fixed = gt["fixed"].long()
    surf = gt["surface_node_ids"].long()
    center = torch.as_tensor(gt["inclusion_center"], dtype=DTYPE)
    radius = float(gt["inclusion_radius"])
    E_bg, E_incl = float(gt["E_background"]), float(gt["E_inclusion"])
    noise = (gt["surface_motion_observed"] - gt["surface_motion_true"]).to(DTYPE)
    forces_true, forces_measured = gt["forces"].to(DTYPE), gt["forces_measured"].to(DTYPE)

    coarse = Mesh(nodes, elems)
    E_nodes = make_heterogeneous_E_field(nodes, center, radius, E_bg, E_incl)
    fine_nodes, fine_elems, parents = refine(nodes.numpy(), elems.numpy())
    fine = Mesh(torch.from_numpy(fine_nodes), torch.from_numpy(fine_elems))
    fine_E = make_heterogeneous_E_field(fine.X, center, radius, E_bg, E_incl)
    fine_fixed = torch.from_numpy(
        fine_fixed_dofs(torch.unique(fixed // 3).numpy(), coarse.Nn, parents)
    )

    def observe(condition: str, frame: int) -> torch.Tensor:
        if condition in ("matched", "mooney"):
            psi = neo_hookean if condition == "matched" else mooney_rivlin
            u = newton(coarse, psi, coarse.element_E(E_nodes), forces_true[frame], fixed)
        else:
            psi = neo_hookean if condition == "fine" else mooney_rivlin
            f_fine = torch.from_numpy(
                transfer_forces(coarse.Nn, parents, fine_elems, forces_true[frame].numpy())
            )
            u = newton(fine, psi, fine.element_E(fine_E), f_fine, fine_fixed)
        return u.view(-1, 3)[surf]

    record = {
        "scenario": scenario.name,
        "E_bg_gt": E_bg,
        "E_incl_gt": E_incl,
        "contrast_gt": E_incl / E_bg,
        "coarse_nodes": coarse.Nn,
        "fine_nodes": fine.Nn,
        "fine_elems": fine.Ne,
        "conditions": {},
    }
    reference = None
    for condition in conditions:
        started = time.time()
        clean = torch.stack([observe(condition, f) for f in FRAMES])
        if condition == "matched":
            reference = clean
            check = float(
                (clean - gt["surface_motion_true"][list(FRAMES)].to(DTYPE)).norm(dim=-1).max()
            )
            record["matched_reproduction_max_mm"] = 1e3 * check
        u_obs = clean + noise[list(FRAMES)]
        est_bg, est_incl, curve = invert(
            coarse, center, radius, forces_measured, fixed, surf, u_obs
        )
        entry = {
            "E_bg_est": est_bg,
            "E_incl_est": est_incl,
            "contrast_est": est_incl / est_bg,
            "E_bg_rel_err": abs(est_bg - E_bg) / E_bg,
            "E_incl_rel_err": abs(est_incl - E_incl) / E_incl,
            "contrast_rel_err": abs(est_incl / est_bg - E_incl / E_bg) / (E_incl / E_bg),
            "objective_final": curve[-1],
            "runtime_s": time.time() - started,
        }
        if reference is not None and condition != "matched":
            entry["forward_discrepancy"] = float(
                (clean - reference).norm(dim=-1).mean() / reference.norm(dim=-1).mean()
            )
        entry["surface_motion_mean_mm"] = 1e3 * float(clean.norm(dim=-1).mean())
        record["conditions"][condition] = entry
        print(
            f"  {condition:12s} E_bg {est_bg/1e3:6.2f} (gt {E_bg/1e3:5.2f}) "
            f"err {100*entry['E_bg_rel_err']:5.1f}%  contrast err "
            f"{100*entry['contrast_rel_err']:5.1f}%  {entry['runtime_s']:.0f}s",
            flush=True,
        )
    return record


def summarize(out: Path) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted((out / "scenarios").glob("*.json"))]
    summary = {"scenarios": len(rows), "conditions": {}}
    for condition in CONDITIONS:
        entries = [r["conditions"][condition] for r in rows if condition in r["conditions"]]
        if not entries:
            continue
        stats = {}
        for key in ("E_bg_rel_err", "E_incl_rel_err", "contrast_rel_err", "forward_discrepancy"):
            values = np.array([e[key] for e in entries if key in e])
            if len(values):
                stats[key] = {
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "mean": float(values.mean()),
                    "max": float(values.max()),
                }
        signed = np.array(
            [(e["E_bg_est"] - r["E_bg_gt"]) / r["E_bg_gt"] for r, e in zip(rows, entries)]
        )
        stats["E_bg_signed_median"] = float(np.median(signed))
        stats["n"] = len(entries)
        summary["conditions"][condition] = stats
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out = args.output
    (out / "scenarios").mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        scenarios = sorted(p for p in args.data.iterdir() if p.is_dir())
        scenarios = scenarios[args.shard :: args.num_shards][: args.limit]
        for k, scenario in enumerate(scenarios, 1):
            target = out / "scenarios" / f"{scenario.name}.json"
            if target.exists():
                continue
            print(f"[{k}/{len(scenarios)}] {scenario.name}", flush=True)
            record = run_scenario(scenario, tuple(args.conditions))
            tmp = target.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(target)
    print(json.dumps(summarize(out), indent=2))


if __name__ == "__main__":
    main()
