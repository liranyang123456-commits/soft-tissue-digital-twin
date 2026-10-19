"""Build the Supporting Information (Supporting_Information.pdf): per-clip STIR errors."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESM = ROOT / "submission" / "ijcars" / "esm"
STIR = ROOT / "outputs" / "stir_tracking"
TITLE = ("Simulable soft-tissue digital twins from endoscopic video: fixed-identity "
         "Gaussian tracking and force-scale-aware finite-element inversion")


def fmt(value) -> str:
    return "--" if value in (None, "") else f"{float(value):.1f}"


def per_clip_rows() -> list[str]:
    def read(name):
        with (STIR / name).open(newline="") as stream:
            return {row["clip"]: row for row in csv.DictReader(stream)}
    base, fused, subset = read("per_clip_all60.csv"), read("per_clip_v7_2d.csv"), read("per_clip_subset20.csv")
    lines = []
    for clip, row in base.items():
        source = row["source"].replace("STIRChallenge_2024\\", "").replace("\\", "/")
        lines.append(
            f"{clip[4:]} & {source} & {row['video_frames']} & {row['points']} & {fmt(row['control_epe'])} & "
            f"{fmt(row.get('present_epe'))} & {fmt(row.get('present_flow_epe'))} & "
            f"{fmt(fused[clip]['present_fused_v7_epe'])} & {fmt(subset.get(clip, {}).get('endogaussian_epe'))} & "
            f"{fmt(row.get('raft_epe'))} \\\\")
    return lines


def main() -> None:
    tex = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{booktabs,longtable}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}
{\large\bfseries Supporting Information}\\[4pt]
""" + TITLE + r"""\\[4pt]
\emph{The International Journal of Medical Robotics and Computer Assisted Surgery}\\[4pt]
Ranyang Li, Zhuoyi Li, Zhipeng Lin, Nan Wei, Chao Fan, Junjun Pan\\[2pt]
Corresponding author: Junjun Pan, State Key Laboratory of Virtual Reality Technology and Systems,
Beihang University, Beijing 100191, China (pan\_junjun@buaa.edu.cn)
\end{center}

\footnotesize
\begin{longtable}{@{}rlrrrrrrrr@{}}
\caption{Median endpoint error (px, full resolution) per STIR clip. Control:
start points left in place; no flow, flow init., fused+refresh: the
present tracker without flow, with flow initialization, and with fused
reference-refresh correspondences; EndoGaussian: 20-clip subset.}\\
\toprule
Clip & Source & Frames & Points & Control & No flow & Flow init. & Fused+refresh & EndoGaussian & RAFT \\
\midrule
\endfirsthead
\toprule
Clip & Source & Frames & Points & Control & No flow & Flow init. & Fused+refresh & EndoGaussian & RAFT \\
\midrule
\endhead
""" + "\n".join(per_clip_rows()) + r"""
\bottomrule
\end{longtable}
\end{document}
"""
    ESM.mkdir(parents=True, exist_ok=True)
    (ESM / "Supporting_Information.tex").write_text(tex, encoding="utf-8", newline="\n")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "Supporting_Information.tex"], cwd=ESM,
                       stdout=subprocess.DEVNULL, check=False)
    for suffix in (".aux", ".log"):
        (ESM / f"Supporting_Information{suffix}").unlink(missing_ok=True)
    print(ESM / "Supporting_Information.pdf")


if __name__ == "__main__":
    main()