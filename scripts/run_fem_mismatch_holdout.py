"""Unseen-load prediction under model mismatch for the representative scenario.

For each data-generating condition, the held-out loads (press-right, shear-x,
shear-y; frame 2) are regenerated with that condition's forward model and the
dataset's noise realization, and predicted with the coarse Neo-Hookean model
using the material inverted on press-left under the same condition. The
oracle prediction uses the true material in the coarse model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fem_model_mismatch import (  # noqa: E402
    DATA, DTYPE, Mesh, fine_fixed_dofs, make_heterogeneous_E_field, mooney_rivlin,
    neo_hookean, newton, refine, transfer_forces,
)

SCENARIO = "geom_58db559b24da6eb7cc0d_scenario_000"
CASES = ("press_right", "shear_x", "shear_y")
FRAME = 2
OUT = ROOT / "outputs" / "fem_model_mismatch"


def main() -> None:
    record = json.loads((OUT / "scenarios" / f"{SCENARIO}.json").read_text())
    results = {"scenario": SCENARIO, "frame": FRAME, "conditions": {}}
    for condition, entry in record["conditions"].items():
        rows = []
        for case in CASES:
            gt = torch.load(DATA / SCENARIO / case / "gt.pt", map_location="cpu", weights_only=False)
            nodes, elems = gt["nodes"].to(DTYPE), gt["elems"].long()
            fixed, surf = gt["fixed"].long(), gt["surface_node_ids"].long()
            center = torch.as_tensor(gt["inclusion_center"], dtype=DTYPE)
            radius = float(gt["inclusion_radius"])
            E_bg, E_incl = float(gt["E_background"]), float(gt["E_inclusion"])
            noise = (gt["surface_motion_observed"][FRAME] - gt["surface_motion_true"][FRAME]).to(DTYPE)
            coarse = Mesh(nodes, elems)
            truth_field = make_heterogeneous_E_field(nodes, center, radius, E_bg, E_incl)
            if condition in ("matched", "mooney"):
                psi = neo_hookean if condition == "matched" else mooney_rivlin
                u = newton(coarse, psi, coarse.element_E(truth_field), gt["forces"][FRAME].to(DTYPE), fixed)
            else:
                psi = neo_hookean if condition == "fine" else mooney_rivlin
                fine_nodes, fine_elems, parents = refine(nodes.numpy(), elems.numpy())
                fine = Mesh(torch.from_numpy(fine_nodes), torch.from_numpy(fine_elems))
                fine_field = make_heterogeneous_E_field(fine.X, center, radius, E_bg, E_incl)
                forces = torch.from_numpy(
                    transfer_forces(coarse.Nn, parents, fine_elems, gt["forces"][FRAME].numpy())
                )
                dofs = torch.from_numpy(fine_fixed_dofs(torch.unique(fixed // 3).numpy(), coarse.Nn, parents))
                u = newton(fine, psi, fine.element_E(fine_field), forces, dofs)
            observed = u.view(-1, 3)[surf] + noise
            magnitude = observed.norm(dim=-1).mean()

            def error(e_bg: float, e_incl: float) -> float:
                field = make_heterogeneous_E_field(nodes, center, radius, e_bg, e_incl)
                pred = newton(
                    coarse, neo_hookean, coarse.element_E(field), gt["forces_measured"][FRAME].to(DTYPE), fixed
                ).view(-1, 3)[surf]
                return float((pred - observed).norm(dim=-1).mean() / magnitude)

            rows.append({
                "case": case,
                "estimated_rel_err": error(entry["E_bg_est"], entry["E_incl_est"]),
                "oracle_rel_err": error(E_bg, E_incl),
            })
            print(condition, rows[-1], flush=True)
        results["conditions"][condition] = rows
    (OUT / "holdout_scenario000.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
