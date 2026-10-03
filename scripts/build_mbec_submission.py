"""Build a condensed MBEC manuscript from the full BME Online source."""
from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission" / "bme_online" / "main.tex"
OUTPUT = ROOT / "submission" / "mbec" / "main.tex"


ABSTRACT = r"""\abstract{Monocular endoscopic reconstruction provides tissue
appearance and surface motion but does not identify absolute mechanical
properties. We developed an audited soft-tissue digital twin based on a
canonical Gaussian surface and differentiable Neo-Hookean mechanics.
Fixed-identity Gaussians are reconstructed with instrument-excluded depth
supervision and tracked with constant-velocity initialization, graph
regularization, and rollback. Tracked displacement provides regional
stiffness contrast for uninstrumented video and surface observations for
known-force finite-element inversion. Parameter provenance is recorded as
measured, estimated, or assumed. Under a strict one-in-eight holdout,
tissue-masked peak signal-to-noise ratio was 16.7 dB across two EndoNeRF
sequences. A 60-scenario finite-element campaign and EndoGaussian baseline
evaluation are reported using the same frozen protocols. A cross-subject
mapping to a patient-specific liver volume produced 28,523 nodes and
149,028 tetrahedra, with 38.7\% boundary coverage. Phantom stiffness
contrast was $3.80\times$ versus $3.56\times$ by compression testing
(6.9\% error). The framework distinguishes mechanically supported
estimates from prior assumptions; absolute modulus on real tissue still
requires force sensing.}"""


REMOVE_LABELS = (
    "tab:position",
    "fig:twin_complete",
    "tab:recon_compare",
    "fig:tier2",
    "fig:tier3",
    "tab:perscenario",
    "fig:fem_perscenario",
    "fig:fem_dist",
    "fig:hospital_ct",
    "tab:mech_compare",
    "fig:material",
    "fig:taskcmp",
    "tab:taskcmp",
    "tab:notation",
    "tab:tissue",
    "fig:architecture",
    "fig:forcenet",
)


def remove_labeled_environment(text: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    index = text.find(marker)
    if index < 0:
        return text
    candidates = [
        (text.rfind(r"\begin{figure*}", 0, index), "figure*"),
        (text.rfind(r"\begin{figure}", 0, index), "figure"),
        (text.rfind(r"\begin{table*}", 0, index), "table*"),
        (text.rfind(r"\begin{table}", 0, index), "table"),
    ]
    start, environment = max(candidates, key=lambda item: item[0])
    if start < 0:
        raise ValueError(f"environment for {label} not found")
    end_marker = rf"\end{{{environment}}}"
    end = text.find(end_marker, index)
    if end < 0:
        raise ValueError(f"end of {environment} for {label} not found")
    return text[:start] + text[end + len(end_marker) :]


def remove_range(text: str, start: str, end: str) -> str:
    left = text.find(start)
    if left < 0:
        return text
    right = text.find(end, left)
    if right < 0:
        raise ValueError(f"range end not found: {end}")
    return text[:left] + text[right:]


def reorder_sections(text: str) -> str:
    declaration = text.index(r"\section*{Abbreviations}")
    end_document = text.index(r"\end{document}")
    tail = text[declaration:end_document]
    body = text[:declaration]
    matches = list(re.finditer(r"^\\section\{([^}]+)\}", body, flags=re.MULTILINE))
    preamble = body[: matches[0].start()]
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1)] = body[match.start() : end]
    introduction = sections["Background"].replace(
        r"\section{Background}", r"\section{Introduction}", 1
    )
    return (
        preamble
        + introduction
        + sections["Methods"]
        + sections["Results"]
        + sections["Discussion"]
        + sections["Conclusion"]
        + tail
        + "\n\\end{document}\n"
    )


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace(
        "pdflatex,lineno,referee,sn-vancouver,Numbered",
        "pdflatex,lineno,iicol,sn-mathphys,Numbered",
    )
    text = re.sub(
        r"\\abstract\{.*?\}\s*\n\s*\\keywords",
        lambda _: ABSTRACT + "\n\n\\keywords",
        text,
        count=1,
        flags=re.DOTALL,
    )
    for label in REMOVE_LABELS:
        text = remove_labeled_environment(text, label)

    text = remove_range(
        text,
        r"\subsection{Comparison with related approaches}",
        r"\subsection{Ablations}",
    )
    text = remove_range(
        text,
        r"\paragraph{Mechanics comparison}",
        r"\subsection{Posterior uncertainty}",
    )
    text = remove_range(
        text,
        r"\paragraph{The force estimator}",
        r"\subsection{Parameter provenance}",
    )
    text = re.sub(
        r"\\begin\{proposition\}.*?\\end\{remark\}",
        (
            "At fixed Poisson ratio, the Neo-Hookean internal force is linear "
            "in Young's modulus. Scaling all regional moduli and the applied "
            "force by the same factor therefore leaves equilibrium "
            "displacement unchanged, while regional stiffness contrast is "
            "invariant to that scale.\\n"
        ),
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = remove_range(
        text,
        r"\subsection*{Authors' information}",
        r"\bibliography{references}",
    )

    for label in REMOVE_LABELS:
        text = text.replace(rf"Fig.~\ref{{{label}}}", "")
        text = text.replace(rf"Figure~\ref{{{label}}}", "")
        text = text.replace(rf"Fig.~\ref{{{label}}},", "")
        text = text.replace(rf"Table~\ref{{{label}}}", "")
        text = text.replace(rf"(Table~\ref{{{label}}})", "")
    text = text.replace(
        r"Proposition~\ref{prop:forcescale}",
        "the force-scale relation",
    )
    text = text.replace(
        r"Remark~\ref{rem:nonlinear}",
        "the numerical sensitivity analysis",
    )
    text = text.replace("Fig.~\\ref{fig:fem_perscenario},\n", "")
    text = text.replace("and the robustness summary in\n", "")
    text = text.replace(
        r"""\begin{equation}
\sigma_{post}^{-2} = \sigma_0^{-2} + \sigma_{obs}^{-2}, \qquad
\mu_{post} = \sigma_{post}^{2}\left(\frac{\mu_0}{\sigma_0^{2}} + \frac{\mu_{obs}}{\sigma_{obs}^{2}}\right),
\label{eq:bayes}
\end{equation}""",
        r"""\begin{align}
\sigma_{post}^{-2} &= \sigma_0^{-2} + \sigma_{obs}^{-2}, \nonumber\\
\mu_{post} &= \sigma_{post}^{2}\left(
\frac{\mu_0}{\sigma_0^{2}} + \frac{\mu_{obs}}{\sigma_{obs}^{2}}
\right).
\label{eq:bayes}
\end{align}""",
    )
    text = text.replace(
        r"""\small
\begin{tabular}{lcc}
\toprule
Tracker & Worst-frame loss & Behaviour \\
\midrule
Inertia-only & 37.23 & frame 5 divergence \\
Const-vel + rollback & 1.55 & frame 5 rolled back \\""",
        r"""\footnotesize
\begin{tabular}{p{0.25\columnwidth}p{0.25\columnwidth}p{0.32\columnwidth}}
\toprule
Tracker & Worst-frame loss & Behaviour \\
\midrule
Inertia-only & 37.23 & frame 5 divergence \\
Constant velocity with rollback & 1.55 & frame 5 rolled back \\""",
    )
    text = text.replace(
        r"\begin{table}[htbp]" + "\n\\centering\n\\caption{Ablations.}",
        r"\begin{table*}[htbp]" + "\n\\centering\n\\caption{Ablations.}",
    )
    text = text.replace(
        "\\label{tab:ablation}\n\\footnotesize",
        "\\label{tab:ablation}\n\\small",
    )
    ablation_label = text.find(r"\label{tab:ablation}")
    if ablation_label >= 0:
        ablation_end = text.find(r"\end{table}", ablation_label)
        if ablation_end >= 0:
            text = (
                text[:ablation_end]
                + r"\end{table*}"
                + text[ablation_end + len(r"\end{table}") :]
            )
    text = text.replace(
        "\n shows the tracked deformation field; displacement\n"
        "concentrates at the tool contact region.",
        "\nTracked displacement concentrates at the tool contact region.",
    )
    text = reorder_sections(text)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    shutil.copy2(
        ROOT / "docs" / "experiments" / "gaussian_volume_mapping.png",
        OUTPUT.parent / "figures" / "gaussian_volume_mapping.png",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
