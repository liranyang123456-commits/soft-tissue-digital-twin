"""Hyperelastic registration of a volumetric organ model to an observed surface.

The observation is Eulerian: points on the intra-operative surface without
correspondence to the model. The unknowns are a rigid transform and the
nodal displacement of a tetrahedral model; the objective combines a trimmed,
robust point-to-plane distance from the observed points to the deformed
boundary with the compressible Neo-Hookean strain energy of the model,
relaxed from stiff to flexible. With displacement-type data only, the
absolute modulus scales out of the minimizer (the energy weight absorbs
it), consistent with the force-scale proposition; the shear modulus is
therefore fixed to one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree


def load_medit_mesh(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes (N,3), tetrahedra (M,4), boundary triangles (K,3) of a MEDIT .mesh file."""
    lines = open(path).read().split("\n")
    sections, i = {}, 0
    while i < len(lines):
        key = lines[i].strip()
        if key in ("Vertices", "Tetrahedra", "Triangles"):
            count = int(lines[i + 1].strip())
            rows = np.array([lines[i + 2 + k].split() for k in range(count)], dtype=float)
            sections[key] = rows
            i += 2 + count
        else:
            i += 1
    nodes = sections["Vertices"][:, :3]
    tets = sections["Tetrahedra"][:, :4].astype(np.int64) - 1
    tris = sections["Triangles"][:, :3].astype(np.int64) - 1 if "Triangles" in sections else boundary_triangles(tets)
    return nodes, tets, tris


def boundary_triangles(tets: np.ndarray) -> np.ndarray:
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    key = np.sort(faces, axis=1)
    _, index, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[index[counts == 1]]


def barycentric_embedding(points: np.ndarray, nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Containing (or nearest) tetrahedron and barycentric weights of each point."""
    centroids = nodes[tets].mean(1)
    tree = cKDTree(centroids)
    _, candidates = tree.query(points, k=min(64, len(tets)))
    owner, weights = np.zeros(len(points), dtype=np.int64), np.zeros((len(points), 4))
    for p, (point, cand) in enumerate(zip(points, candidates)):
        best, best_w, best_violation = cand[0], None, np.inf
        for t in cand:
            a, b, c, d = nodes[tets[t]]
            T = np.stack([a - d, b - d, c - d], axis=1)
            try:
                lam = np.linalg.solve(T, point - d)
            except np.linalg.LinAlgError:
                continue
            w = np.append(lam, 1 - lam.sum())
            violation = -min(0.0, w.min())
            if violation < best_violation:
                best, best_w, best_violation = t, w, violation
                if violation == 0:
                    break
        owner[p] = best
        weights[p] = np.clip(best_w, 0, None) / np.clip(best_w, 0, None).sum()
    return owner, weights


def _rotation(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = axis_angle.norm().clamp_min(1e-12)
    k = axis_angle / theta
    K = torch.zeros(3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


class NeoHookeanBody:
    def __init__(self, nodes: np.ndarray, tets: np.ndarray, nu: float = 0.45, device="cuda"):
        self.device = torch.device(device)
        self.X = torch.as_tensor(nodes, dtype=torch.float64, device=self.device)
        self.tets = torch.as_tensor(tets, dtype=torch.long, device=self.device)
        Dm = self._edges(self.X)
        self.Dm_inv = torch.linalg.inv(Dm)
        self.volume = torch.det(Dm).abs() / 6.0
        self.mu = 1.0
        self.lam = 2.0 * nu / (1.0 - 2.0 * nu)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        p = x[self.tets]
        return torch.stack([p[:, 0] - p[:, 3], p[:, 1] - p[:, 3], p[:, 2] - p[:, 3]], dim=2)

    def energy(self, u: torch.Tensor) -> torch.Tensor:
        F = self._edges(self.X + u) @ self.Dm_inv
        J = torch.det(F)
        logJ = torch.log(J.clamp_min(1e-6))
        I1 = (F * F).sum((1, 2))
        psi = 0.5 * self.mu * (I1 - 3) - self.mu * logJ + 0.5 * self.lam * logJ ** 2
        psi = psi + 1e3 * torch.relu(1e-3 - J) ** 2 / 1e-6
        return (psi * self.volume).sum() / self.volume.sum()


def vertex_normals(x: np.ndarray, tris: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(x)
    face = np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
    for k in range(3):
        np.add.at(normals, tris[:, k], face)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)


@dataclass
class RegistrationResult:
    R: np.ndarray
    t: np.ndarray
    u: np.ndarray
    history: list = field(default_factory=list)

    def transform(self, points_ref: np.ndarray, owner=None, weights=None, tets=None) -> np.ndarray:
        """Map reference points through the deformation (if embedded) and the rigid transform."""
        if owner is not None:
            moved = points_ref + (self.u[tets[owner]] * weights[..., None]).sum(1)
        else:
            moved = points_ref
        return moved @ self.R.T + self.t


def register(nodes: np.ndarray, tets: np.ndarray, tris: np.ndarray, observed: np.ndarray,
             R0: np.ndarray, t0: np.ndarray, *, deformable: bool = True,
             energy_weights=(10.0, 3.0, 1.0, 0.3, 0.1), iterations: int = 300, rigid_iterations: int = 300,
             trim: float = 0.8, max_distance: float = 20.0, device="cuda", log=print,
             lagrangian=None, lagrangian_weight: float = 1.0, eulerian_weight: float = 1.0,
             solver: str = "adam", outer_rounds: int = 3) -> RegistrationResult:
    """Rigid trimmed ICP from (R0,t0), followed by stiff-to-flexible hyperelastic registration.

    ``lagrangian = (owner, weights, targets)`` adds identified surface points:
    the reference points embedded in tetrahedra ``owner`` with barycentric
    ``weights`` must map to ``targets``. ``observed`` may be empty when only
    Lagrangian data are used.
    """
    dev = torch.device(device)
    body = NeoHookeanBody(nodes, tets, device=device)
    boundary = np.unique(tris)
    obs = torch.as_tensor(observed if observed is not None else np.zeros((0, 3)), dtype=torch.float64, device=dev)
    axis = torch.zeros(3, dtype=torch.float64, device=dev, requires_grad=True)
    shift = torch.zeros(3, dtype=torch.float64, device=dev, requires_grad=True)
    R0_t = torch.as_tensor(R0, dtype=torch.float64, device=dev)
    t0_t = torch.as_tensor(t0, dtype=torch.float64, device=dev)
    u = torch.zeros_like(body.X, requires_grad=True)
    history = []
    if lagrangian is not None:
        l_owner = torch.as_tensor(tets[lagrangian[0]], device=dev)
        l_weight = torch.as_tensor(lagrangian[1], dtype=torch.float64, device=dev)
        l_target = torch.as_tensor(lagrangian[2], dtype=torch.float64, device=dev)

    def lagrangian_term():
        R, t = pose()
        x = ((body.X[l_owner] + u[l_owner]) * l_weight[..., None]).sum(1) @ R.T + t
        return torch.sqrt(((x - l_target) ** 2).sum(-1) + 1.0).mean() - 1.0

    use_surface = observed is not None and len(observed) > 0

    def pose():
        R = _rotation(axis) @ R0_t if axis.abs().sum() > 0 else R0_t
        return R, t0_t + shift

    def correspondences(threshold: float):
        with torch.no_grad():
            R, t = pose()
            x = ((body.X + u) @ R.T + t).cpu().numpy()
        normals = vertex_normals(x, tris)[boundary]
        if not use_surface:
            empty = torch.zeros(0, dtype=torch.long, device=dev)
            return empty, empty, torch.zeros(0, 3, dtype=torch.float64, device=dev), 0.0
        tree = cKDTree(x[boundary])
        distance, index = tree.query(observed)
        keep = distance < threshold
        if keep.sum() > 10:
            cut = np.quantile(distance[keep], trim)
            keep = keep & (distance <= cut)
        node = torch.as_tensor(boundary[index[keep]], device=dev)
        normal = torch.as_tensor(normals[index[keep]], dtype=torch.float64, device=dev)
        return torch.as_tensor(np.flatnonzero(keep), device=dev), node, normal, float(np.median(distance[keep])) if keep.any() else np.inf

    def data_term(pairs):
        total = 0.0
        if lagrangian is not None:
            total = lagrangian_weight * lagrangian_term()
        if not use_surface:
            return total
        return total + eulerian_weight * surface_term(pairs)

    def surface_term(pairs):
        obs_index, node, normal = pairs
        R, t = pose()
        x = (body.X[node] + u[node]) @ R.T + t
        diff = x - obs[obs_index]
        plane = (diff * normal).sum(-1)
        point = diff.norm(dim=-1)
        residual = 0.7 * plane ** 2 + 0.3 * point ** 2
        return torch.sqrt(residual + 1.0).mean() - 1.0

    threshold = max_distance
    optimizer = torch.optim.Adam([axis, shift], lr=1e-2)
    for it in range(rigid_iterations):
        if it % 10 == 0:
            *pairs, median = correspondences(threshold)
            threshold = max(5.0, min(threshold, 3.0 * median))
        optimizer.zero_grad()
        loss = data_term(pairs)
        loss.backward()
        optimizer.step()
    history.append(("rigid", loss.item(), median))
    log(f"    rigid: data {loss.item():.3f}, median distance {median:.2f} mm")

    if deformable and solver == "lbfgs":
        for weight in energy_weights:
            *pairs, median = correspondences(threshold)
            optimizer = torch.optim.LBFGS([u, axis, shift], lr=1.0, max_iter=iterations,
                                          history_size=50, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                loss = data_term(pairs) + weight * body.energy(u) * 1e3
                loss.backward()
                return loss

            for _ in range(outer_rounds):
                optimizer.step(closure)
                *pairs, median = correspondences(threshold)
            with torch.no_grad():
                data, elastic = data_term(pairs), body.energy(u)
            history.append((weight, data.item(), elastic.item(), median))
            log(f"    L-BFGS energy weight {weight}: data {data.item():.3f}, elastic {elastic.item():.2e}, median {median:.2f} mm")
    elif deformable:
        for weight in energy_weights:
            optimizer = torch.optim.Adam([{"params": [u], "lr": 0.05}, {"params": [axis, shift], "lr": 2e-3}])
            for it in range(iterations):
                if it % 10 == 0:
                    *pairs, median = correspondences(threshold)
                optimizer.zero_grad()
                data = data_term(pairs)
                elastic = body.energy(u)
                loss = data + weight * elastic * 1e3
                loss.backward()
                optimizer.step()
            history.append((weight, data.item(), elastic.item(), median))
            log(f"    energy weight {weight}: data {data.item():.3f}, elastic {elastic.item():.2e}, median {median:.2f} mm")
    with torch.no_grad():
        R, t = pose()
    return RegistrationResult(R.cpu().numpy(), t.cpu().numpy(), u.detach().cpu().numpy(), history)
