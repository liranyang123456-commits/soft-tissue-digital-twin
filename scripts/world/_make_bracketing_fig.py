"""Schematic: camera-coverage map (azimuth-elevation) showing why holdout
view 0 is a boundary camera while views 4/8 are well bracketed."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.hub._validate_not_a_forked_repo = lambda *a, **k: None
from mvbrdf_shr.world.train import load_dataset, split_scene_indices

CFG = "configs/world/benchmark_scene_0070_rgb_only_ominormal_smoke.yaml"
OUT = Path("submission/tmm/figures/fig_view0_bracketing")

cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
ds = load_dataset(cfg)
tr, ho = split_scene_indices(ds, cfg)


def sph(i):
    p = ds[i].camera.c2w[:3, 3].numpy().astype(np.float64)
    r = np.linalg.norm(p)
    az = np.degrees(np.arctan2(p[2], p[0]))
    el = np.degrees(np.arcsin(p[1] / r))
    return az, el, p / r


train = {int(ds[i].view_id): sph(i) for i in tr}
hold = {int(ds[i].view_id): sph(i) for i in ho}


def ang3d(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))


# nearest two training views per holdout (by 3D angle)
nearest = {}
for hv, (az, el, d) in hold.items():
    ds_ = sorted(((ang3d(d, t[2]), tv) for tv, t in train.items()))
    nearest[hv] = ds_[:2]

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif",
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(5.6, 3.4))

# light 30-degree reference rings around each holdout
theta = np.linspace(0, 2 * np.pi, 120)
for hv, (az, el, d) in hold.items():
    ax.plot(az + 30 * np.cos(theta), el + 30 * np.sin(theta),
            color="0.85", lw=0.7, zorder=1)

# training views
for tv, (az, el, d) in train.items():
    ax.scatter(az, el, s=46, c="#1f6fb2", marker="o", zorder=3,
               edgecolors="white", linewidths=0.6)
    ax.annotate(f"{tv}", (az, el), textcoords="offset points",
                xytext=(0, -13), ha="center", fontsize=7, color="#1f6fb2")

# holdout views + links to two nearest training views
hcol = {0: "#c0392b", 4: "#1e8449", 8: "#1e8449"}
for hv, (az, el, d) in hold.items():
    ax.scatter(az, el, s=190, c=hcol[hv], marker="*", zorder=4,
               edgecolors="black", linewidths=0.5)
    ax.annotate(f"view {hv}", (az, el), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8,
                fontweight="bold", color=hcol[hv])
    for dist, tv in nearest[hv]:
        taz, tel, _ = train[tv]
        ax.plot([az, taz], [el, tel], ls="--", lw=1.0, color=hcol[hv],
                alpha=0.75, zorder=2)
        mx, my = (az + taz) / 2, (el + tel) / 2
        ax.annotate(f"{dist:.0f}$^\\circ$", (mx, my), fontsize=6.8,
                    color=hcol[hv], ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.75))

ax.set_xlabel("azimuth (deg)")
ax.set_ylabel("elevation (deg)")
ax.set_xlim(-185, 185)
ax.set_ylim(-28, 40)
ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, ls=":", lw=0.4, color="0.9", zorder=0)

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f6fb2",
           markersize=7, label="training view"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#1e8449",
           markeredgecolor="k", markersize=12, label="held-out, bracketed (4, 8)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#c0392b",
           markeredgecolor="k", markersize=12, label="held-out, boundary (0)"),
]
ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True,
          framealpha=0.9, edgecolor="0.7", borderpad=0.4, labelspacing=0.3)

fig.tight_layout()
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight")
print("saved", OUT.with_suffix(".pdf"), "and .png")
for hv in sorted(nearest):
    print(f"view {hv}: nearest two -> " +
          ", ".join(f"v{tv} {dd:.1f}deg" for dd, tv in nearest[hv]))
