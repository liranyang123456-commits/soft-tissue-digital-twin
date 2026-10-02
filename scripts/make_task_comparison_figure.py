"""Fair task-comparison figure for the soft-tissue twin paper.

Rows are tasks. A published number is placed beside ours only when the
protocol is the same. Different protocols are marked not comparable, and
no row is scored as beating reconstruction SOTA.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTS = [
    ROOT / "submission" / "bme_online" / "figures" / "fig_task_comparison.png",
    ROOT / "submission" / "bme_online" / "figures" / "fig_task_comparison.pdf",
]

# verdict colors
C = {
    "same": "#1B7F4E",
    "own": "#1F4E79",
    "diff": "#8A6A12",
    "open": "#8C3A3A",
}

ROWS = [
    ["Reconstruction\n(EndoNeRF NVS)", "held-out-view PSNR",
     "EndoGaussian reports NVS PSNR\n(not re-run here)",
     "not measured\non this split", "not comparable", "diff"],
    ["Reconstruction\n(our canonical fit)", "tissue-masked PSNR",
     "no published number\non this mask",
     "36.79 / 34.09 dB\npulling / cutting", "reported only", "own"],
    ["SCARED portability", "tissue-masked PSNR,\nsingle keyframe",
     "no matched published number",
     "26.99 / 26.17 dB\nkf1 / kf2", "reported only", "own"],
    ["Tracking ablation", "worst-frame loss,\nsame sequence",
     "inertia-only 37.23\n(diverges)",
     "const-vel + rollback\n1.55", "same protocol,\nbetter", "same"],
    ["Mask ablation", "tissue-masked PSNR,\nsame frames",
     "tool-mask supervision\n11.13 dB",
     "depth>0 tissue mask\n36.79 dB", "same protocol,\nbetter", "same"],
    ["Uniform modulus", "relative error,\nclosed loop",
     "no external method\non this loop",
     "0.01%", "sanity check", "own"],
    ["FEM background E", "median relative error",
     "prior 60-scene line:\n21.3% median",
     "4.3% median\n(6 of 60 scenarios)", "not the same set", "diff"],
    ["Phantom contrast", "error vs compression test",
     "no published run\non this phantom split",
     "3.80x vs 3.56x\n(6.9%)", "reported only", "own"],
    ["Provenance audit", "measured / estimated\n/ assumed",
     "no prior method\nstates this tag",
     "explicit tag\non every parameter", "new statement,\nnot a metric win", "own"],
    ["Official NVS SOTA\nand in-vivo absolute E", "held-out PSNR;\nforce-free E",
     "EndoGaussian NVS;\nno force-free absolute E",
     "not done", "still open", "open"],
]


def main():
    fig_h = 0.62 * len(ROWS) + 1.6
    fig, ax = plt.subplots(figsize=(12.4, fig_h), dpi=160)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, len(ROWS) + 1.35)
    ax.axis("off")
    ax.set_title(
        "Task comparison. Green: same protocol. Blue: our measurement, no matched published number.\n"
        "Amber: protocols differ, do not rank. Red: not done. No row claims a reconstruction-SOTA win.",
        loc="left", fontsize=11, pad=8, color="#222",
    )
    headers = ["Task", "Metric", "Closest published or baseline", "This paper", "How to read it"]
    xs = [0.15, 2.35, 4.55, 7.55, 10.15]
    ws = [2.1, 2.1, 2.9, 2.5, 2.1]
    y0 = len(ROWS) + 0.15
    for x, w, h in zip(xs, ws, headers):
        ax.add_patch(FancyBboxPatch((x, y0), w, 0.72, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor="#243044", edgecolor="none"))
        ax.text(x + 0.08, y0 + 0.36, h, va="center", ha="left", color="white", fontsize=8.5, fontweight="bold")
    for i, row in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        bg = "#F7F4EC" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(plt.Rectangle((0.15, y), 11.95, 0.92, facecolor=bg, edgecolor="#E4E0D6", lw=0.4))
        vals = row[:5]
        for x, w, val in zip(xs, ws, vals):
            weight = "bold" if x == xs[-1] else "regular"
            color = C[row[5]] if x == xs[-1] else "#222"
            ax.text(x + 0.08, y + 0.46, val, va="center", ha="left", fontsize=7.6,
                    color=color, fontweight=weight)
    fig.tight_layout()
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
