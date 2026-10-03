"""Build a condensed MBEC manuscript from the full BME Online source."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission" / "bme_online" / "main.tex"
OUTPUT = ROOT / "submission" / "mbec" / "main.tex"

MBEC_PREAMBLE = r"""\RequirePackage{fix-cm}
\documentclass[smallextended]{svjour3}
\smartqed

\usepackage{graphicx}
\usepackage{url}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\graphicspath{{figures/}}
\emergencystretch=2em
\def\UrlBreaks{\do\/\do-\do_}
\Urlmuskip=0mu plus 1mu\relax

\journalname{Medical \& Biological Engineering \& Computing}
"""

TITLE = (
    "From Endoscopic Image Sequences to a Simulable Soft-Tissue Digital Twin: "
    "Gaussian Surface Tracking and Force-Scale-Aware Mechanical Inversion"
)


ABSTRACT = r"""\abstract{Endoscopic reconstruction estimates
appearance and surface motion but does not by itself provide a mechanically
valid, simulable tissue model. We propose a soft-tissue digital-twin
framework that converts endoscopic images into a co-registered Gaussian
surface, optical material representation, displacement field, and
force-aware volumetric mechanics. A canonical anisotropic Gaussian field
is reconstructed with depth-valid, instrument-excluded supervision and
tracked by fixed primitive identities using graph regularization,
constant-velocity initialization, and divergence rollback. An explicit
Gaussian-to-tetrahedral interface transfers surface observations to a
differentiable Neo-Hookean solver. Known forces permit modulus inversion;
without force sensing, a derived force-scale relation limits inference to
stiffness contrast. Each parameter is labelled measured, estimated, or
assumed and accompanied by uncertainty. Evaluation combines two EndoNeRF
sequences, two SCARED keyframes, 60 finite-element scenarios, six phantom
acquisitions, and one clinical computed-tomography case. On a strict
28-frame holdout, __COMMON_HOLDOUT__ __FEM_ABSTRACT__ Phantom contrast was $3.80\times$ versus
$3.56\times$ by compression testing. The contribution is a traceable
observation-to-simulation interface; absolute tissue modulus still
requires measured force.}"""


REMOVE_LABELS = (
    "tab:position",
    "fig:pipeline",
    "fig:twin_complete",
    "tab:recon_compare",
    "fig:tier2",
    "fig:tier3",
    "fig:tier1",
    "tab:tier1",
    "tab:tier2",
    "tab:datasets",
    "tab:perscenario",
    "fig:fem_perscenario",
    "fig:fem_dist",
    "fig:hospital_ct",
    "tab:mech_compare",
    "fig:material",
    "fig:ets",
    "fig:robustness",
    "fig:taskcmp",
    "tab:taskcmp",
    "tab:notation",
    "tab:tissue",
    "fig:architecture",
    "fig:forcenet",
)

OVERALL_FRAMEWORK_FIGURE = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_overall_framework.png}
\caption{Overall soft-tissue digital-twin framework. (a) Endoscopic
observations are reconstructed, tracked, mechanically parameterized, and
driven forward for rehearsal. (b) The computational graph keeps Gaussian
surface identity fixed while transferring displacement to a separate
tetrahedral discretization. Force evidence determines whether absolute
modulus or only stiffness contrast is identifiable. The compact R3D-18
branch summarizes the auxiliary estimator used only to define a
force-error model; it is external to the core twin.}
\label{fig:overall_framework}
\end{figure}
"""

RECONSTRUCTION_FIGURE = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=0.94\textwidth]{fig_reconstruction_material_results.png}
\caption{Reconstruction and optical-material evidence. (a) Strict
one-in-eight holdout comparison using identical frames, tissue masks, and
metrics. (b) Input, fitted albedo, roughness, and specular components for
the two EndoNeRF scenes and two SCARED keyframes. EndoGaussian remains
substantially more accurate for novel-view synthesis; the material maps
are estimated outputs of the canonical field.}
\label{fig:reconstruction_material}
\end{figure}
"""

TRACKING_FIGURE = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=0.94\textwidth]{fig_tracking_results.png}
\caption{Fixed-identity deformation tracking on the EndoNeRF pulling
sequence. Columns show four times; rows show tracked displacement on the
observed frame, displacement magnitude, rendered deformed field, and
photometric error. Primitive identity is retained across time to provide
co-registered surface observations for mechanics.}
\label{fig:tracking_results}
\end{figure}
"""

MECHANICS_FIGURE = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=0.94\textwidth]{fig_mechanics_results.png}
\caption{Mechanical validation. (a) Complete 60-scenario FEM inversion,
including modulus agreement, error distributions, stiffness dependence,
and the background-modulus success curve. (b) Force-scale, motion-noise,
and held-out-load sensitivity. (c) Image-derived phantom stiffness
contrast against the independent compression-test reference.}
\label{fig:mechanics_results}
\end{figure}
"""


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


def latex_word_count(text: str) -> int:
    """Return a conservative word count for short LaTeX prose."""
    plain = re.sub(r"\\(?:abstract|textbf|emph)\b", " ", text)
    plain = re.sub(r"\\[A-Za-z]+", " ", plain)
    plain = re.sub(r"[{}$~]", " ", plain)
    return len(re.findall(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*", plain))


def official_front_matter(abstract: str) -> str:
    abstract_text = abstract.removeprefix(r"\abstract{").removesuffix("}")
    return rf"""{MBEC_PREAMBLE}

\begin{{document}}

\title{{{TITLE}}}
\titlerunning{{Simulable soft-tissue digital twin}}

\author{{Ranyang Li \and Haotian Ma \and Zhipeng Lin \and Nan Wei}}
\authorrunning{{R. Li et al.}}

\institute{{R. Li \and H. Ma \at
Henan University of Technology, Zhengzhou 450001, China\\
\email{{lry@haut.edu.cn}}
\and
Z. Lin \at
Beihang University, Beijing 100191, China
\and
N. Wei \at
Henan Provincial People's Hospital, Zhengzhou 450003, China}}

\date{{}}
\maketitle

\begin{{abstract}}
{abstract_text}
\keywords{{digital twin \and soft tissue \and endoscopic reconstruction
\and elasticity inversion \and parameter provenance \and biomechanics}}
\end{{abstract}}
"""


def condensed_introduction() -> str:
    return rf"""\section{{Introduction}}
\label{{sec:intro}}

Surgical rehearsal and patient-specific intervention planning require more
than a visual reconstruction: the virtual tissue must also respond to a
prescribed load through a mechanically interpretable forward model
\cite{{landig2025,laubenbacher2024}}. Endoscopic reconstruction can recover
appearance, surface geometry, and motion, but these observations do not by
themselves identify Young's modulus, Poisson's ratio, or stress. This work
therefore addresses the interface between endoscopic observation and a
simulable soft-tissue representation.

The central difficulty is evidential. Geometry and reflectance are
constrained by images and auxiliary depth, whereas elasticity is constrained
only through deformation under a known or otherwise calibrated load. Values
copied from a tissue table remain assumptions rather than patient-specific
measurements. The proposed twin consequently attaches a provenance state
(measured, estimated, or assumed) and uncertainty to each physical
parameter, so that simulated stress and strain can be interpreted according
to the evidence that supports them.

{OVERALL_FRAMEWORK_FIGURE}

\subsection{{Related work and methodological gap}}
EndoNeRF \cite{{endonerf}} and EndoSurf \cite{{endosurf}} reconstruct
deforming endoscopic scenes with neural radiance or signed-distance fields.
EndoGaussian \cite{{endogaussian}} uses anisotropic Gaussians and a learned
spatiotemporal deformation field to achieve high-quality, real-time
novel-view synthesis. Related inverse-rendering methods decompose geometry,
lighting, and bidirectional reflectance \cite{{gsir2024,relightable3dgs}},
while deformable SLAM and robust pose estimation address camera motion
\cite{{defslam,hayoz2023}}. These methods establish visual reconstruction
but do not produce force-conditioned tissue mechanics.

Physics-aware neural representations address complementary parts of the
problem. PAC-NeRF identifies material parameters from multi-view motion
under known loading \cite{{pacnerf}}; PhysDreamer estimates relative
stiffness without an absolute force scale \cite{{physdreamer}}; and
PhysGaussian and Spring-Gaus couple visual primitives to prescribed or
learned dynamics \cite{{physgaussian,springgaus}}. Quasi-static
elastography likewise shows that deformation can support stiffness contrast
under a shared-load approximation, while absolute modulus requires force
information \cite{{ophir1991,wells2011,sigrist2017}}. No compared method
provides the complete endoscopic observation-to-volumetric-simulation
interface together with explicit parameter provenance.

We propose a co-registered representation
$\mathcal{{T}}=(G,\{{\Delta_t\}},\Theta,\mathcal{{A}})$. A canonical
Gaussian field $G$ stores geometry and Cook--Torrance optical material;
fixed primitive identities carry displacement $\Delta_t$ through time; an
explicit alignment and interpolation map transfers surface observations to
a tetrahedral Neo-Hookean model with parameters $\Theta$; and
$\mathcal{{A}}$ records provenance and uncertainty. Constant-velocity
initialization, graph regularization, and divergence rollback stabilize
tracking. A force-scale analysis establishes that a common scaling of force
and regional moduli leaves equilibrium displacement and stiffness contrast
unchanged, which prevents unsupported absolute-modulus claims when force is
not measured.

The evaluation uses two EndoNeRF sequences, two SCARED keyframes, all 60
scenarios of a four-load finite-element benchmark, six ultrasound-phantom
acquisitions with compression-test references, and one de-identified
clinical CT geometry. Under the same 28-frame holdout, EndoGaussian reaches
36.6 dB tissue PSNR and 0.939 tissue SSIM, compared with 16.7 dB and 0.564
for the present representation; reconstruction superiority is therefore
not claimed. The complete FEM campaign gives median errors of 9.6\% for
background modulus, 32.0\% for inclusion modulus, and 32.4\% for contrast.
The phantom contrast is 3.80$\times$ versus 3.56$\times$ by compression
testing. These experiments test the integrated twin and its identifiability
limits rather than clinical effectiveness.

"""


def condensed_methods_prefix() -> str:
    return r"""\section{Methods}
\label{sec:method}

\subsection{Study design, data, and evaluation}
This computational methods study separates visual reconstruction,
surface tracking, known-force inversion, force-uncertain contrast
estimation, and geometry-to-volume coupling. EndoNeRF contributes two RGB
sequences (63 and 156 frames) with auxiliary depth and instrument masks;
SCARED contributes two stereo keyframes with metric depth
\cite{endonerf,scared}. The synthetic benchmark contains 60
Neo-Hookean scenarios and four loads with ground-truth regional moduli,
measured forces, and noisy surface motion. The physical dataset contains
six ultrasound indentation acquisitions and independent compression-test
moduli \cite{etsphantom}. A single de-identified hospital CT case is used
only to construct a patient-specific tetrahedral geometry.

The strict reconstruction split excludes frames satisfying
$(t-1)\bmod 8=0$ from both methods. PSNR and SSIM use identical source
images and masks; tissue SSIM averages the local 11-by-11 SSIM map only
over tissue pixels. Mechanical accuracy is reported as relative modulus
error, contrast error, and held-out-load displacement error. FEM scenarios
are processed in lexical order with a frozen 20-iteration inversion.
Medians, means, interquartile ranges, and per-case values are reported; the
phantom median uses a nonparametric bootstrap 95\% confidence interval.

\subsection{Digital-twin representation}
The twin is
$\mathcal{T}=(G,\{\Delta_t\},\Theta,\mathcal{A})$, where $G$ is the
canonical Gaussian geometry and optical material, $\Delta_t$ is
fixed-identity surface displacement, $\Theta$ contains the mechanical
parameters and forward operator, and $\mathcal{A}$ maps every parameter to
its provenance and uncertainty. Inputs are images $I_t$, depth $D_t$, and
tool masks $M_t$. A twin is considered simulable when the mechanical
operator maps a new load to volumetric displacement and derived stress or
strain; it is considered audited only when every parameter has a
provenance tag.

"""


def condensed_results(common: dict, fem: dict) -> str:
    present = common["present"]
    baseline = common["EndoGaussian"]
    bg = fem["E_bg_rel_err"]
    incl = fem["E_incl_rel_err"]
    contrast = fem["contrast_rel_err"]
    return rf"""\section{{Results}}
\label{{sec:experiments}}

\subsection{{Reconstruction and optical material}}
\begin{{table}}[htbp]
\centering
\caption{{Strict one-in-eight EndoNeRF holdout using the same 28 frames,
tissue masks, and evaluator.}}
\label{{tab:common_holdout}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Method & Full PSNR & Full SSIM & Tissue PSNR & Tissue SSIM \\
\midrule
Present method & {present['full_psnr']:.2f} & {present['full_ssim']:.3f} &
{present['tissue_psnr']:.2f} & {present['tissue_ssim']:.3f} \\
EndoGaussian & {baseline['full_psnr']:.2f} & {baseline['full_ssim']:.3f} &
{baseline['tissue_psnr']:.2f} & {baseline['tissue_ssim']:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

{RECONSTRUCTION_FIGURE}

EndoGaussian exceeds the present method by
{baseline['tissue_psnr'] - present['tissue_psnr']:.2f} dB tissue PSNR and
{baseline['tissue_ssim'] - present['tissue_ssim']:.3f} tissue SSIM
(Table~\ref{{tab:common_holdout}}). The gap is consistent across both
scenes. EndoGaussian uses a learned temporal deformation field optimized
for novel-view synthesis, whereas the present representation retains
approximately 6.5--6.8 thousand primitive identities and interpolates
held-out displacement to preserve mechanical correspondence. Optical
decomposition produces spatially varying albedo, roughness, and specular
maps on EndoNeRF and SCARED (Fig.~\ref{{fig:reconstruction_material}});
these are estimated image-based quantities, not mechanical measurements.

\subsection{{Deformation tracking}}
{TRACKING_FIGURE}

Fixed-identity tracking follows the contact-centred motion over the tested
sequence (Fig.~\ref{{fig:tracking_results}}). The inertia-only ablation
diverges at frame 5 with worst-frame loss 37.23. Constant-velocity
initialization, graph regularization, and rollback reduce the corresponding
loss to 1.55 and retain the preceding valid state when re-optimization does
not satisfy the acceptance criterion.

\subsection{{Mechanical inversion and physical validation}}
{MECHANICS_FIGURE}

\begin{{table}}[htbp]
\centering
\caption{{Quantitative mechanical and interface validation.}}
\label{{tab:mechanical_summary}}
\small
\begin{{tabular}}{{p{{0.28\textwidth}}p{{0.24\textwidth}}p{{0.35\textwidth}}}}
\toprule
Experiment & Evidence & Result \\
\midrule
Uniform synthetic loop & known force and $E$ & 0.01\% modulus error \\
FEM60 background & measured force, noisy motion &
median {100*bg['median']:.1f}\%; mean {100*bg['mean']:.1f}\% \\
FEM60 inclusion & measured force, surface motion &
median {100*incl['median']:.1f}\%; mean {100*incl['mean']:.1f}\% \\
FEM60 contrast & common force scale &
median {100*contrast['median']:.1f}\%; mean {100*contrast['mean']:.1f}\% \\
Held-out loads & material fitted on press-left &
11.1\% / 3.7\% / 14.9\% displacement error \\
Force-error propagation & auxiliary error model &
$E_{{bg}}$ error 4.1$\pm$4.0\%; contrast stable \\
Physical phantom & no absolute force scale &
3.80$\times$ vs 3.56$\times$; 6.9\% error \\
Gaussian--volume map & cross-subject interface &
2,755/7,111 boundary nodes; 38.7\% coverage \\
\bottomrule
\end{{tabular}}
\end{{table}}

All 60 predefined FEM scenarios complete under the frozen protocol
(Fig.~\ref{{fig:mechanics_results}}a). Background modulus is moderately
recovered, but inclusion modulus and contrast remain weak: their median
errors exceed 32\%, and contrast error reaches 89.3\% in the worst case.
The held-out-load experiment uses a representative six-scenario subset;
material fitted on press-left predicts press-right, shear-x, and shear-y
with 11.1\%, 3.7\%, and 14.9\% displacement error, respectively.

The force sweep follows the theoretical scale direction but is not exactly
one-to-one because inversion uses regularization, finite load steps, and
noisy partial-surface observations (Fig.~\ref{{fig:mechanics_results}}b).
The auxiliary R3D-18 error model is used only in this sensitivity test.
Without force sensing, the physical phantom supports contrast rather than
absolute modulus: the image-derived median is 3.80$\times$ (95\% CI
3.16--4.34) against the compression-test reference of 3.56$\times$
(Fig.~\ref{{fig:mechanics_results}}c).

\subsection{{Gaussian surface to patient-specific volume}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{{gaussian_volume_mapping.png}}
\caption{{Cross-subject computational mapping from the tracked Gaussian
surface to a CT-derived tetrahedral liver volume. Similarity alignment and
inverse-distance interpolation map 2,755 of 7,111 boundary nodes (38.7\%).
The case tests the software interface, not anatomical correspondence or
clinical validity.}}
\label{{fig:volume_mapping}}
\end{{figure}}

The CT mask yields 28,523 nodes and 149,028 tetrahedra. The normalized
median surface distance after alignment is 0.0237. Because CT spacing is
unavailable and the endoscopic and CT surfaces are from different
subjects, neither physical dimensions nor patient-specific material
parameters are inferred from this experiment.

\subsection{{Ablations and uncertainty}}
\begin{{table}}[htbp]
\centering
\caption{{Controlled component ablations.}}
\label{{tab:ablation}}
\small
\begin{{tabular}}{{lll}}
\toprule
Component & Comparison & Result \\
\midrule
Tissue supervision & tool / depth-valid mask & 11.13 / 36.79 dB \\
Gaussian count & 4,000 / 6,498 / 10,000 & 35.31 / 36.86 / 38.21 dB \\
Temporal model & inertia / proposed tracker & 37.23 / 1.55 loss \\
Parameterization & per-node / two-region & 1.69$\times$ / 3.19$\times$ \\
Forward model & mass--spring / FEM & 82\% forward-model discrepancy \\
\bottomrule
\end{{tabular}}
\end{{table}}

The Laplace approximation gives
$\mathrm{{std}}(\log E_{{bg}})=1.17$ and
$\mathrm{{std}}(\log E_{{incl}})=21.2$, marking the deep inclusion as
weakly identified. Thus, high prediction-interval coverage does not imply
that each modulus is accurately estimated; the interval is dominated by
the poorly constrained inclusion direction.

"""


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
    fem_path = ROOT / "outputs" / "fem_validation_60" / "fem60_summary.json"
    fem = json.loads(fem_path.read_text(encoding="utf-8")) if fem_path.exists() else {}
    common_path = ROOT / "docs" / "experiments" / "endonerf_common_summary.json"
    common_report = (
        json.loads(common_path.read_text(encoding="utf-8"))
        if common_path.exists()
        else {}
    )
    common = common_report.get("per_method", {})
    if "present" in common and "EndoGaussian" in common:
        present = common["present"]
        baseline = common["EndoGaussian"]
        common_abstract = (
            "tissue peak signal-to-noise ratio/structural similarity were "
            f"{present['tissue_psnr']:.1f} dB/{present['tissue_ssim']:.3f} "
            "for the present method and "
            f"{baseline['tissue_psnr']:.1f} dB/{baseline['tissue_ssim']:.3f} "
            "for EndoGaussian."
        )
    else:
        common_abstract = "reconstruction was evaluated with a common protocol."
    fem_complete = bool(fem.get("complete"))
    if fem_complete:
        bg = fem["E_bg_rel_err"]
        incl = fem["E_incl_rel_err"]
        contrast = fem["contrast_rel_err"]
        fem_abstract = (
            f"Across 60 finite-element scenarios, median errors were "
            f"{100*bg['median']:.1f}\\% for background modulus, "
            f"{100*incl['median']:.1f}\\% for inclusion modulus, and "
            f"{100*contrast['median']:.1f}\\% for contrast."
        )
    else:
        fem_abstract = "A 60-scenario finite-element campaign uses a frozen inversion protocol."
    abstract = (
        ABSTRACT.replace("__COMMON_HOLDOUT__", common_abstract)
        .replace("__FEM_ABSTRACT__", fem_abstract)
    )
    abstract_words = latex_word_count(abstract)
    if abstract_words > 200:
        raise ValueError(f"MBEC abstract has {abstract_words} words; limit is 200")
    source_body = text.index(r"\maketitle") + len(r"\maketitle")
    text = official_front_matter(abstract) + text[source_body:]
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

    figure_replacements = {
        "fig:pipeline": "fig:overall_framework",
        "fig:twin_complete": "fig:overall_framework",
        "fig:architecture": "fig:overall_framework",
        "fig:forcenet": "fig:overall_framework",
        "fig:tier1": "fig:reconstruction_material",
        "fig:material": "fig:reconstruction_material",
        "fig:tier2": "fig:tracking_results",
        "fig:tier3": "fig:mechanics_results",
        "fig:fem_perscenario": "fig:mechanics_results",
        "fig:fem_dist": "fig:mechanics_results",
        "fig:ets": "fig:mechanics_results",
        "fig:robustness": "fig:mechanics_results",
        "fig:hospital_ct": "fig:volume_mapping",
    }
    for old_label, new_label in figure_replacements.items():
        text = text.replace(
            rf"Fig.~\ref{{{old_label}}}",
            rf"Fig.~\ref{{{new_label}}}",
        )
        text = text.replace(
            rf"Figure~\ref{{{old_label}}}",
            rf"Figure~\ref{{{new_label}}}",
        )

    for label in REMOVE_LABELS:
        text = text.replace(rf"Fig.~\ref{{{label}}}", "")
        text = text.replace(rf"Figure~\ref{{{label}}}", "")
        text = text.replace(rf"Fig.~\ref{{{label}}},", "")
        text = text.replace(rf"Table~\ref{{{label}}}", "")
        text = text.replace(rf"(Table~\ref{{{label}}})", "")
    text = text.replace(
        "The six scenarios in  are retained only as\n"
        "representative cases for the separate held-out-load analysis. ",
        "",
    )
    text = text.replace(
        r"(Fig.~\ref{fig:pipeline}, ).",
        r"(Fig.~\ref{fig:pipeline}).",
    )
    text = text.replace("with\ntwo load cases ().", "with two load cases.")
    text = text.replace("Provincial People's Hospital ().", "Provincial People's Hospital.")
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
    text = text.replace(
        r"\subsection{Related work}",
        OVERALL_FRAMEWORK_FIGURE + "\n" + r"\subsection{Related work}",
        1,
    )
    results_index = text.index(r"\section{Results}")
    for subsection, figure_block in (
        (r"\subsection{Canonical reconstruction}", RECONSTRUCTION_FIGURE),
        (r"\subsection{Deformation tracking}", TRACKING_FIGURE),
        (r"\subsection{Mechanical inversion}", MECHANICS_FIGURE),
    ):
        marker_index = text.index(subsection, results_index)
        insert_at = marker_index + len(subsection)
        text = text[:insert_at] + "\n" + figure_block + text[insert_at:]

    mapping_marker = (
        "This CT case is not paired with an endoscopic sequence and therefore "
        "does\nnot validate optical classification or mechanical inversion.\n"
    )
    mapping_figure = r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gaussian_volume_mapping.png}
\caption{Cross-subject computational mapping from the tracked Gaussian
surface to the patient-specific CT-derived tetrahedral volume. Similarity
alignment is followed by local interpolation onto covered boundary nodes.
The mapping contains 28,523 nodes, 149,028 tetrahedra, and 2,755 mapped
boundary nodes (38.7\% coverage). Because the endoscopic and CT data are
from different subjects and CT spacing is unavailable, this demonstrates
the software interface rather than anatomical correspondence or clinical
validity.}
\label{fig:volume_mapping}
\end{figure}
"""
    if mapping_marker in text:
        text = text.replace(mapping_marker, mapping_marker + mapping_figure, 1)

    if common:
        if "present" in common and "EndoGaussian" in common:
            present = common["present"]
            baseline = common["EndoGaussian"]
            common_holdout_table = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Strict one-in-eight EndoNeRF holdout. Both methods use the same
28 held-out frames, tissue masks, and evaluator.}}
\label{{tab:common_holdout}}
\small
\begin{{tabular}}{{lccccc}}
\toprule
Method & Frames & Full PSNR & Full SSIM & Tissue PSNR & Tissue SSIM \\
\midrule
Present method & {present['frames']} & {present['full_psnr']:.2f} & {present['full_ssim']:.3f} & {present['tissue_psnr']:.2f} & {present['tissue_ssim']:.3f} \\
EndoGaussian & {baseline['frames']} & {baseline['full_psnr']:.2f} & {baseline['full_ssim']:.3f} & {baseline['tissue_psnr']:.2f} & {baseline['tissue_ssim']:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
            text = text.replace(
                r"\subsection{Canonical reconstruction}",
                r"\subsection{Canonical reconstruction}" + "\n" + common_holdout_table,
                1,
            )
    if fem_complete:
        bg = fem["E_bg_rel_err"]
        incl = fem["E_incl_rel_err"]
        contrast = fem["contrast_rel_err"]
        text = text.replace(
            r"Independent FEM benchmark & $E_{bg}$, measured force; noisy motion & \\textbf{median 4.3\\% error (6 scenarios)} \\\\",
            rf"Independent FEM benchmark & $E_{{bg}}$, measured force; noisy motion & \\textbf{{median {100*bg['median']:.1f}\\% error (60 scenarios)}} \\\\",
        )
        old_start = text.find("With measured forces (5\\% calibrated error) and noisy surface motion")
        old_end = text.find("The inclusion contrast is recovered", old_start)
        if old_start >= 0 and old_end >= 0:
            paragraph = (
                f"With measured forces (5\\% calibration error) and noisy surface "
                f"motion, all 60 scenarios were inverted using the frozen 20-iteration "
                f"protocol. Background-modulus relative error had median "
                f"{100*bg['median']:.1f}\\% and mean {100*bg['mean']:.1f}\\%; "
                f"inclusion-modulus error had median {100*incl['median']:.1f}\\%; "
                f"contrast error had median {100*contrast['median']:.1f}\\%. "
            )
            text = text[:old_start] + paragraph + text[old_end:]
        text = text.replace(
            "The principal FEM\nanalysis uses six selected scenarios from a synthetic 60-scenario\nbenchmark; the observed dependence on background stiffness is therefore\ndescriptive rather than a population-level estimate. ",
            "The complete FEM campaign covers all 60 predefined scenarios. ",
        )
        text = text.replace(
            "background\nmodulus has a median error of 4.3\\% when the force is measured",
            f"background modulus has a median error of {100*bg['median']:.1f}\\% "
            "when the force is measured",
        )
    text = reorder_sections(text)
    intro_start = text.index(r"\section{Introduction}")
    methods_start = text.index(r"\section{Methods}", intro_start)
    text = text[:intro_start] + condensed_introduction() + text[methods_start:]

    methods_start = text.index(r"\section{Methods}")
    canonical_start = text.index(
        r"\subsection{Canonical scene reconstruction}",
        methods_start,
    )
    text = text[:methods_start] + condensed_methods_prefix() + text[canonical_start:]
    text = remove_range(
        text,
        r"\subsection{Use of generative AI tools}",
        r"\section{Results}",
    )

    results_start = text.index(r"\section{Results}")
    discussion_start = text.index(r"\section{Discussion}", results_start)
    text = (
        text[:results_start]
        + condensed_results(common, fem)
        + text[discussion_start:]
    )
    text = text.replace(r"\ref{sec:force}", r"\ref{sec:forcescale}")
    availability_start = text.index(r"\subsection*{Availability of data and materials}")
    competing_start = text.index(r"\subsection*{Competing interests}", availability_start)
    availability = r"""\subsection*{Availability of data and materials}
Source code, frozen data splits, evaluation scripts, and machine-readable
result summaries are available at
\url{https://github.com/liranyang123456-commits/soft-tissue-digital-twin}.
EndoNeRF, SCARED, the FEM benchmark, and the \'ETS phantom are public
datasets cited in this article. The de-identified hospital CT images are
not publicly redistributed under the data-providing institution's
governance.

"""
    text = text[:availability_start] + availability + text[competing_start:]
    text = text.replace(r"\begin{figure*}", r"\begin{figure}")
    text = text.replace(r"\end{figure*}", r"\end{figure}")
    text = text.replace(r"\begin{table*}", r"\begin{table}")
    text = text.replace(r"\end{table*}", r"\end{table}")
    text = text.replace(
        r"\bibliography{references}",
        r"\bibliographystyle{spmpsci}" + "\n" + r"\bibliography{references}",
        1,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    shutil.copy2(
        SOURCE,
        OUTPUT.parent / "main_full.tex",
    )
    shutil.copy2(
        SOURCE.parent / "references.bib",
        OUTPUT.parent / "references.bib",
    )
    shutil.copy2(
        ROOT / "docs" / "experiments" / "gaussian_volume_mapping.png",
        OUTPUT.parent / "figures" / "gaussian_volume_mapping.png",
    )
    shutil.copy2(
        ROOT / "docs" / "experiments" / "endonerf_common_comparison.png",
        OUTPUT.parent / "figures" / "endonerf_common_comparison.png",
    )
    fem_figure = ROOT / "outputs" / "fem_validation_60" / "fem60_summary.png"
    if fem_figure.exists():
        shutil.copy2(fem_figure, OUTPUT.parent / "figures" / "fem60_summary.png")
    print(f"{OUTPUT} (abstract: {abstract_words} words)")


if __name__ == "__main__":
    main()
