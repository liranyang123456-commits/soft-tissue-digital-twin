"""FEM-dataset validation suite for the soft-tissue twin (A1/A3/A4/B1/B4).

Validates the Tier-3 mechanical inversion against the FEM-constructed
``ion_ct_synthetic_mechanics`` dataset (Neo-Hookean tetrahedra, nu=0.45,
known E_background / E_inclusion, calibrated force measurements, noisy
surface motion). This is the strongest credibility experiment available:
the data generator (3D FEM) is *different* from any model we invert with.

Experiments
-----------
A1  FEM-in-the-loop 2-region inversion (E_bg, E_incl) per scenario, using
    measured forces (5 % calibrated error) + noisy surface motion.
B1  Surrogate cross-validation: quasi-static mass-spring (tet-edge springs)
    vs FEM on identical mesh/forces/BCs -> forward error; plus
    surrogate-based inversion (model-mismatch robustness).
A3  Noise robustness: extra motion noise {0, 0.5, 1.0} mm x seeds.
A4  Held-out load case: invert on press_left, predict shear_x / shear_y /
    press_right, compare displacement fields with FEM GT.
B4  Uncertainty propagation: Laplace posterior over (log E_bg, log E_incl)
    -> Monte-Carlo forward -> prediction-interval coverage on held-out load.

Outputs: ``outputs/fem_validation/{*.json,*.png}``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- FEM solver from the sibling repository (read-only reuse) ---------------
DIFF_ROOT = Path(r"E:\Diff_Rending_Re_3D")
if str(DIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFF_ROOT))

from physics.fem import (  # noqa: E402
    newton_solve,
    node_E_to_per_elem_mu_lam,
    solve_nh_heterogeneous,
    make_heterogeneous_E_field,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = DIFF_ROOT / "dataset" / "ion_ct_synthetic_mechanics60"
OUT = ROOT / "outputs" / "fem_validation"
NU = 0.45  # dataset generation value (verified: reproduces GT to 0.00 mm)
_NU_T: torch.Tensor | None = None


def _nu_t() -> torch.Tensor:
    """Lazily-created float64 tensor form of NU (the FEM layer saves it)."""
    global _NU_T
    if _NU_T is None:
        _NU_T = torch.tensor(NU, dtype=torch.float64)
    return _NU_T
LOAD_CASES = ["press_left", "press_right", "shear_x", "shear_y"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_case(scenario_dir: Path, case: str) -> dict:
    gt = torch.load(scenario_dir / case / "gt.pt", map_location="cpu",
                    weights_only=False)
    return gt


def scenario_dirs(limit: int | None = None) -> list[Path]:
    dirs = sorted(p for p in DATA.iterdir() if p.is_dir())
    return dirs[:limit] if limit else dirs


def case_tensors(gt: dict, frames=(1, 2, 4)):
    """Return the tensors needed for inversion from one load case."""
    nodes = gt["nodes"]                      # (N,3) float64
    elems = gt["elems"]                      # (Ne,4)
    fixed = gt["fixed"]                      # (K,) node ids
    surf = gt["surface_node_ids"]            # (S,)
    forces_m = gt["forces_measured"]         # (T,3N)
    u_obs = gt["surface_motion_observed"]    # (T,S,3) noisy displacement
    fixed_nodes = torch.unique(fixed // 3)  # dataset stores DOF indices
    return {
        "nodes": nodes, "elems": elems, "fixed": fixed_nodes, "surf": surf,
        "fixed_dofs": fixed,
        "forces_m": forces_m, "u_obs": u_obs, "frames": list(frames),
        "E_bg": float(gt["E_background"]), "E_incl": float(gt["E_inclusion"]),
        "center": gt["inclusion_center"], "radius": float(gt["inclusion_radius"]),
    }


# ---------------------------------------------------------------------------
# A1: FEM-in-the-loop 2-region inversion
# ---------------------------------------------------------------------------


def invert_two_param_fem(
    data: dict,
    *,
    frames: list[int] | None = None,
    n_iters: int = 40,
    lr: float = 0.10,
    E_init_bg: float = 10e3,
    E_init_incl: float = 20e3,
    force_scale: float = 1.0,
    extra_noise_std: float = 0.0,
    noise_seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Recover (E_bg, E_incl) with known inclusion center/radius.

    Forward model = the same differentiable Neo-Hookean FEM used to generate
    the data (upper-bound setting); inputs are the *measured* forces (5 %
    calibrated error) and *noisy* surface motion, never the clean GT.
    """
    nodes, elems, surf = data["nodes"], data["elems"], data["surf"]
    frames = frames if frames is not None else data["frames"]
    fixed_dofs = data["fixed_dofs"]

    u_obs = data["u_obs"][frames].clone()
    if extra_noise_std > 0:
        g = torch.Generator().manual_seed(noise_seed)
        u_obs = u_obs + extra_noise_std * torch.randn(u_obs.shape, generator=g)
    scale = u_obs.norm(dim=-1).mean().clamp_min(1e-9)

    log_E = torch.tensor(
        [np.log(E_init_bg), np.log(E_init_incl)],
        dtype=torch.float64, requires_grad=True,
    )
    opt = torch.optim.Adam([log_E], lr=lr)
    curve: list[float] = []
    t0 = time.time()
    for it in range(n_iters):
        opt.zero_grad()
        E_bg, E_incl = log_E.exp()
        E_field = make_heterogeneous_E_field(
            nodes, data["center"], data["radius"], E_bg, E_incl, soft_width=0.02,
        )
        loss = torch.zeros((), dtype=torch.float64)
        for f_i in frames:
            u = solve_nh_heterogeneous(
                nodes, elems, E_field, _nu_t(),
                data["forces_m"][f_i] * force_scale, fixed_dofs, D=3,
            )
            u_surf = u.view(-1, 3)[surf]
            loss = loss + torch.mean(
                ((u_surf - u_obs[list(frames).index(f_i)]) / scale) ** 2
            )
        loss = loss / len(frames)
        loss.backward()
        opt.step()
        curve.append(float(loss.item()))
        if verbose and (it % 10 == 0 or it == n_iters - 1):
            print(f"    it {it:3d}: loss {loss.item():.5f} "
                  f"E_bg {E_bg.item()/1e3:.2f} E_incl {E_incl.item()/1e3:.2f} kPa")
    E_bg, E_incl = log_E.detach().exp()
    return {
        "E_bg_est": float(E_bg) , "E_incl_est": float(E_incl),
        "E_bg_gt": data["E_bg"], "E_incl_gt": data["E_incl"],
        "contrast_est": float(E_incl / E_bg),
        "contrast_gt": data["E_incl"] / data["E_bg"],
        "loss_curve": curve, "runtime_s": time.time() - t0,
        "log_E_est": [float(v) for v in log_E.detach()],
    }


# ---------------------------------------------------------------------------
# B1: quasi-static mass-spring surrogate + cross-validation
# ---------------------------------------------------------------------------


def build_springs_from_tets(nodes: torch.Tensor, elems: torch.Tensor):
    """Unique edge springs from a tet mesh. Returns (springs (M,2), rest (M,))."""
    edge_set: set[tuple[int, int]] = set()
    e = elems.numpy()
    for tet in e:
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = sorted((int(tet[a]), int(tet[b])))
                edge_set.add((i, j))
    springs = torch.tensor(sorted(edge_set), dtype=torch.long)
    rest = (nodes[springs[:, 0]] - nodes[springs[:, 1]]).norm(dim=-1)
    return springs, rest


def solve_quasistatic_ms(
    nodes: torch.Tensor,
    springs: torch.Tensor,
    rest: torch.Tensor,
    E_nodes: torch.Tensor,
    f_ext: torch.Tensor,
    fixed: torch.Tensor,
    *,
    n_iters: int = 400,
    lr: float | None = None,
    k_scale: float = 1.0,
) -> torch.Tensor:
    """Quasi-static mass-spring equilibrium via energy minimization.

    Minimizes  sum_e 0.5 k_e (|d_e|-L_e)^2 - f.u  over free DOFs, with
    k_e = E_avg(e) * L_e * k_scale (edge-length heuristic). The stiffness is
    DETACHED inside the loop — call this under torch.no_grad() and use
    finite differences for parameter gradients (2-parameter problems).
    """
    Nn = nodes.shape[0]
    k_edge = (
        0.5 * (E_nodes[springs[:, 0]] + E_nodes[springs[:, 1]]) * rest * k_scale
    ).detach()
    free = torch.ones(Nn, dtype=torch.bool)
    free[fixed] = False
    u_free = torch.zeros(int(free.sum()), 3, dtype=nodes.dtype,
                         device=nodes.device, requires_grad=True)
    if lr is None:
        # Scale lr to the problem: stiffer springs need smaller steps.
        lr = 0.02 / max(float(k_edge.max() / 1e3), 1.0)
    opt = torch.optim.Adam([u_free], lr=lr)
    f_vec = f_ext.view(Nn, 3)
    for _ in range(n_iters):
        opt.zero_grad()
        u = torch.zeros(Nn, 3, dtype=nodes.dtype, device=nodes.device)
        u[free] = u_free
        pos = nodes + u
        d = pos[springs[:, 1]] - pos[springs[:, 0]]
        length = d.norm(dim=-1).clamp_min(1e-12)
        energy = (0.5 * k_edge * (length - rest) ** 2).sum()
        work = (f_vec * u).sum()
        loss = energy - work
        loss.backward()
        opt.step()
    u = torch.zeros(Nn, 3, dtype=nodes.dtype, device=nodes.device)
    u[free] = u_free.detach()
    return u


def cross_validate_surrogate(data: dict, frame: int = 2) -> dict:
    """B1: FEM vs mass-spring forward on identical inputs (GT material).

    Reports the raw heuristic error AND a one-parameter scale-calibrated
    error: in the near-linear regime a single stiffness scale absorbs most
    of the surrogate bias, which is the honest way to state surrogate
    accuracy.
    """
    nodes, elems, fixed = data["nodes"], data["elems"], data["fixed"]
    E_gt = make_heterogeneous_E_field(
        nodes, data["center"], data["radius"], data["E_bg"], data["E_incl"]
    )
    fixed_dofs = data["fixed_dofs"]
    mu, lam, _ = node_E_to_per_elem_mu_lam(E_gt, elems, NU)
    f = data["forces_m"][frame]

    u_fem = newton_solve(nodes, elems, mu, lam, f, fixed_dofs, 3).view(-1, 3)
    springs, rest = build_springs_from_tets(nodes, elems)
    u_ms = solve_quasistatic_ms(nodes, springs, rest, E_gt, f, fixed)

    err = (u_ms - u_fem).norm(dim=-1)
    mag = u_fem.norm(dim=-1).mean()

    # One-parameter calibration: displacement is ~1/stiffness in this regime,
    # so the optimal k_scale ≈ u_ms/u_fem mean ratio.
    ratio = float(u_ms.norm(dim=-1).mean() / mag.clamp_min(1e-12))
    k_scale = max(ratio, 1e-3)
    u_ms_c = solve_quasistatic_ms(
        nodes, springs, rest, E_gt, f, fixed, k_scale=k_scale
    )
    err_c = (u_ms_c - u_fem).norm(dim=-1)

    return {
        "frame": frame,
        "n_springs": int(springs.shape[0]),
        "fem_mean_disp_mm": float(mag * 1e3),
        "ms_mean_disp_mm": float(u_ms.norm(dim=-1).mean().item() * 1e3),
        "err_mean_mm": float(err.mean() * 1e3),
        "err_max_mm": float(err.max() * 1e3),
        "err_relative": float(err.mean() / mag.clamp_min(1e-12)),
        "calibration_k_scale": k_scale,
        "err_relative_calibrated": float(err_c.mean() / mag.clamp_min(1e-12)),
        "err_mm_calibrated": float(err_c.mean() * 1e3),
        "u_fem": u_fem, "u_ms": u_ms, "springs": springs, "rest": rest,
    }


def invert_two_param_surrogate(
    data: dict, springs: torch.Tensor, rest: torch.Tensor,
    *, frames: list[int] | None = None, n_iters: int = 30, lr: float = 0.15,
    k_scale: float = 1.0, inner_iters: int = 200,
) -> dict:
    """B1b: 2-region inversion with the mass-spring surrogate (model mismatch).

    Outer gradient by central finite differences (2 parameters; the inner
    energy minimization is run under no_grad). This is exact enough for a
    smooth 2-D objective and avoids unrolling the inner solver.
    """
    nodes, fixed, surf = data["nodes"], data["fixed"], data["surf"]
    frames = frames if frames is not None else data["frames"]
    center, radius = data["center"], data["radius"]
    u_obs = data["u_obs"][frames]
    scale = u_obs.norm(dim=-1).mean().clamp_min(1e-9)

    def loss_at(log_E: np.ndarray) -> float:
        E_bg, E_incl = float(np.exp(log_E[0])), float(np.exp(log_E[1]))
        E_field = make_heterogeneous_E_field(
            nodes, center, radius, E_bg, E_incl
        )
        total = 0.0
        for k, f_i in enumerate(frames):
            u = solve_quasistatic_ms(
                nodes, springs, rest, E_field,
                data["forces_m"][f_i], fixed,
                n_iters=inner_iters, k_scale=k_scale,
            )
            total += float((((u[surf] - u_obs[k]) / scale) ** 2).mean())
        return total / len(frames)

    log_E = np.array([np.log(10e3), np.log(20e3)])
    curve: list[float] = []
    eps = 0.05
    for it in range(n_iters):
        f0 = loss_at(log_E)
        grad = np.zeros(2)
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            grad[j] = (loss_at(log_E + e) - loss_at(log_E - e)) / (2 * eps)
        log_E = log_E - lr * grad / max(np.abs(grad).max(), 1e-9) * 0.5
        curve.append(f0)
    E_bg, E_incl = float(np.exp(log_E[0])), float(np.exp(log_E[1]))
    return {
        "E_bg_est": E_bg, "E_incl_est": E_incl,
        "contrast_est": E_incl / E_bg,
        "contrast_gt": data["E_incl"] / data["E_bg"],
        "loss_curve": curve,
        "k_scale": k_scale,
    }


# ---------------------------------------------------------------------------
# A4: held-out load case
# ---------------------------------------------------------------------------


def predict_load_case(data: dict, E_bg: float, E_incl: float, frame: int = 2):
    """Simulate a load case with given material and compare with FEM GT."""
    nodes, elems, surf = data["nodes"], data["elems"], data["surf"]
    fixed_dofs = data["fixed_dofs"]
    E_field = make_heterogeneous_E_field(
        nodes, data["center"], data["radius"], E_bg, E_incl
    )
    mu, lam, _ = node_E_to_per_elem_mu_lam(E_field, elems, NU)
    u_pred = newton_solve(
        nodes, elems, mu, lam, data["forces_m"][frame], fixed_dofs, 3
    ).view(-1, 3)
    u_gt = data["u_obs"][frame]  # noisy observed surface displacement
    err = (u_pred[surf] - u_gt).norm(dim=-1)
    mag = u_gt.norm(dim=-1).mean().clamp_min(1e-9)
    return {
        "err_mean_mm": float(err.mean() * 1e3),
        "err_max_mm": float(err.max() * 1e3),
        "err_relative": float(err.mean() / mag),
        "u_pred_surf": u_pred[surf],
    }


# ---------------------------------------------------------------------------
# B4: Laplace posterior + Monte-Carlo prediction interval
# ---------------------------------------------------------------------------


def laplace_posterior(loss_fn, log_E_star: np.ndarray, eps: float = 1e-3):
    """Diagonal Laplace approximation of p(log_E | data) at the optimum."""
    n = len(log_E_star)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                continue
            e = np.zeros(n)
            e[i] = eps
            f_pp = loss_fn(log_E_star + e)
            f_mm = loss_fn(log_E_star - e)
            f_00 = loss_fn(log_E_star)
            H[i, i] = (f_pp - 2 * f_00 + f_mm) / eps**2
    H = np.maximum(H, 1e-9)
    cov = np.linalg.inv(H)
    return np.diag(cov)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--noise-iters", type=int, default=25)
    p.add_argument("--skip-surrogate", action="store_true")
    p.add_argument("--skip-noise", action="store_true")
    p.add_argument("--skip-holdout", action="store_true")
    p.add_argument("--skip-uq", action="store_true")
    p.add_argument("--skip-a1", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    scenarios = scenario_dirs(args.scenarios)
    print(f"FEM validation suite: {len(scenarios)} scenarios from {DATA}")

    # ---------------- A1: FEM-in-the-loop inversion ------------------------
    if not args.skip_a1:
        print("\n[A1] FEM-in-the-loop 2-region inversion (measured forces + noisy motion)")
        a1_rows = []
        for sd in scenarios:
            data = case_tensors(load_case(sd, "press_left"))
            r = invert_two_param_fem(data, n_iters=args.iters)
            r["scenario"] = sd.name
            a1_rows.append(r)
            print(f"  {sd.name}: E_bg {r['E_bg_gt']/1e3:.2f}->{r['E_bg_est']/1e3:.2f} kPa, "
                  f"E_incl {r['E_incl_gt']/1e3:.2f}->{r['E_incl_est']/1e3:.2f} kPa, "
                  f"contrast {r['contrast_gt']:.2f}x->{r['contrast_est']:.2f}x "
                  f"({r['runtime_s']:.0f}s)")
        bg_err = [abs(r["E_bg_est"] - r["E_bg_gt"]) / r["E_bg_gt"] for r in a1_rows]
        inc_err = [abs(r["E_incl_est"] - r["E_incl_est"] * 0 - r["E_incl_gt"]) / r["E_incl_gt"] for r in a1_rows]
        summary["A1_fem_inversion"] = {
            "per_scenario": [
                {k: v for k, v in r.items() if k != "loss_curve"}
                for r in a1_rows
            ],
            "E_bg_rel_err_mean": float(np.mean(bg_err)),
            "E_incl_rel_err_mean": float(np.mean(inc_err)),
            "contrast_est_mean": float(np.mean([r["contrast_est"] for r in a1_rows])),
            "contrast_gt_mean": float(np.mean([r["contrast_gt"] for r in a1_rows])),
        }
        print(f"  -> E_bg rel err {np.mean(bg_err)*100:.1f}%, "
              f"E_incl rel err {np.mean(inc_err)*100:.1f}%")

    # ---------------- B1: surrogate cross-validation -----------------------
    if not args.skip_surrogate:
        print("\n[B1] Mass-spring surrogate vs FEM (same mesh/forces/BCs)")
        data = case_tensors(load_case(scenarios[0], "press_left"))
        cv = cross_validate_surrogate(data)
        print(f"  forward: FEM mean disp {cv['fem_mean_disp_mm']:.3f} mm, "
              f"MS {cv['ms_mean_disp_mm']:.3f} mm, "
              f"err {cv['err_mean_mm']:.3f} mm ({cv['err_relative']*100:.1f}%)")
        print(f"  calibrated (k_scale {cv['calibration_k_scale']:.2f}): "
              f"err {cv['err_mm_calibrated']:.3f} mm "
              f"({cv['err_relative_calibrated']*100:.1f}%)")
        print("  [B1b] surrogate-based inversion (model mismatch)")
        r_ms = invert_two_param_surrogate(
            data, cv["springs"], cv["rest"], n_iters=args.iters,
            k_scale=cv["calibration_k_scale"],
        )
        print(f"  surrogate inversion: contrast {r_ms['contrast_gt']:.2f}x -> "
              f"{r_ms['contrast_est']:.2f}x")
        summary["B1_surrogate"] = {
            "forward": {k: v for k, v in cv.items()
                        if k not in {"u_fem", "u_ms", "springs", "rest"}},
            "inversion": r_ms,
        }

    # ---------------- A3: noise robustness + multi-seed --------------------
    if not args.skip_noise:
        print("\n[A3] Noise robustness (extra motion noise x seeds)")
        data = case_tensors(load_case(scenarios[0], "press_left"))
        noise_rows = []
        for noise_std in [0.0, 5e-4, 1e-3]:  # 0, 0.5, 1.0 (mesh units ~ m)
            for seed in range(3):
                r = invert_two_param_fem(
                    data, n_iters=args.noise_iters,
                    extra_noise_std=noise_std, noise_seed=seed,
                )
                err_bg = abs(r["E_bg_est"] - r["E_bg_gt"]) / r["E_bg_gt"]
                noise_rows.append({
                    "noise_std": noise_std, "seed": seed,
                    "E_bg_est": r["E_bg_est"], "E_bg_rel_err": err_bg,
                    "contrast_est": r["contrast_est"],
                })
                print(f"  noise {noise_std*1e3:.1f} mm seed {seed}: "
                      f"E_bg err {err_bg*100:.1f}%, contrast {r['contrast_est']:.2f}x")
        summary["A3_noise"] = {"runs": noise_rows}

    # ---------------- A4: held-out load case -------------------------------
    if not args.skip_holdout:
        print("\n[A4] Held-out load-case prediction (invert press_left -> predict others)")
        sd = scenarios[0]
        train = case_tensors(load_case(sd, "press_left"))
        r = invert_two_param_fem(train, n_iters=args.iters)
        hold_rows = []
        for case in ["press_right", "shear_x", "shear_y"]:
            test = case_tensors(load_case(sd, case))
            pred = predict_load_case(test, r["E_bg_est"], r["E_incl_est"])
            oracle = predict_load_case(test, test["E_bg"], test["E_incl"])
            hold_rows.append({
                "case": case,
                "err_relative_est": pred["err_relative"],
                "err_relative_oracle": oracle["err_relative"],
                "err_mm_est": pred["err_mean_mm"],
                "err_mm_oracle": oracle["err_mean_mm"],
            })
            print(f"  {case}: est err {pred['err_relative']*100:.1f}% "
                  f"(oracle {oracle['err_relative']*100:.1f}%)")
        summary["A4_holdout_load"] = {
            "train_case": "press_left",
            "E_bg_est": r["E_bg_est"], "E_incl_est": r["E_incl_est"],
            "held_out": hold_rows,
        }

    # ---------------- B4: uncertainty propagation --------------------------
    if not args.skip_uq:
        print("\n[B4] Uncertainty propagation (Laplace posterior -> MC interval)")
        sd = scenarios[0]
        train = case_tensors(load_case(sd, "press_left"))
        # Reuse a previous inversion from the merged summary when available.
        prev_log_E = None
        prev_path = OUT / "fem_validation_summary.json"
        if prev_path.exists():
            try:
                prev = json.loads(prev_path.read_text(encoding="utf-8"))
                prev_log_E = prev["A1_fem_inversion"]["per_scenario"][0]["log_E_est"]
            except Exception:
                prev_log_E = None
        if prev_log_E is not None:
            log_E_star = np.array(prev_log_E, dtype=float)
            print(f"  reusing A1 inversion: log_E = {log_E_star}")
        else:
            r = invert_two_param_fem(train, n_iters=args.iters)
            log_E_star = np.array(r["log_E_est"])

        nodes, elems, surf = train["nodes"], train["elems"], train["surf"]
        fixed_dofs = train["fixed_dofs"]
        frames = train["frames"]
        u_obs = train["u_obs"][frames]
        scale = u_obs.norm(dim=-1).mean()

        def loss_at(log_E):
            E_bg, E_incl = np.exp(log_E)
            E_field = make_heterogeneous_E_field(
                nodes, train["center"], train["radius"],
                float(E_bg), float(E_incl),
            )
            total = 0.0
            for k, f_i in enumerate(frames):
                u = solve_nh_heterogeneous(
                    nodes, elems, E_field, _nu_t(),
                    train["forces_m"][f_i], fixed_dofs, D=3,
                )
                total += float(
                    (((u.view(-1, 3)[surf] - u_obs[k]) / scale) ** 2).mean()
                )
            return total / len(frames)

        var = laplace_posterior(loss_at, log_E_star)
        std = np.sqrt(np.maximum(var, 1e-12))
        print(f"  posterior std(log E): bg {std[0]:.3f}, incl {std[1]:.3f}")

        # MC forward on held-out case.
        test = case_tensors(load_case(sd, "shear_x"))
        rng = np.random.default_rng(0)
        samples = rng.normal(log_E_star, std, size=(24, 2))
        u_samples = []
        for s in samples:
            pred = predict_load_case(test, float(np.exp(s[0])), float(np.exp(s[1])))
            u_samples.append(pred["u_pred_surf"])
        u_stack = torch.stack(u_samples)          # (M,S,3)
        u_mean = u_stack.mean(0)
        u_std = u_stack.std(0)
        gt_u = test["u_obs"][2]
        covered = ((gt_u - u_mean).norm(dim=-1)
                   < 1.96 * u_std.norm(dim=-1).clamp_min(1e-12)).float().mean()
        summary["B4_uq"] = {
            "posterior_std_logE": std.tolist(),
            "mc_samples": len(samples),
            "held_out_case": "shear_x",
            "interval_coverage_95": float(covered),
            "mean_pred_err_mm": float((u_mean - gt_u).norm(dim=-1).mean() * 1e3),
        }
        print(f"  95% prediction-interval coverage on shear_x: {covered*100:.0f}%")

    # ---------------- save (merge with any previous run) -------------------
    out_path = OUT / "fem_validation_summary.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev.update(summary)  # new sections win; old sections preserved
            summary = prev
        except Exception:
            pass
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
