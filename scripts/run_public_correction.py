"""Correct the full-model checkpoint selection and regenerate affected reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from run_public_campaign import ROOT, run_stage


def main() -> None:
    python = sys.executable
    campaign = ROOT / "outputs/public_campaign"
    train_cfg = ROOT / "configs/train/public_multidataset.yaml"
    adapt_cfg = ROOT / "configs/train/public_domain_adapt.yaml"
    eval_cfg = ROOT / "configs/eval/public_suite.yaml"
    initial = ROOT / "outputs/shiq_mse/best.pt"
    corrected = campaign / "ours_pretrained_mixed_corrected"

    run_stage(
        "correct_train_ours_mixed",
        [
            python,
            "scripts/train_public_multidataset.py",
            "--config",
            str(train_cfg),
            "--resume",
            str(initial),
            "--output",
            str(corrected),
        ],
        campaign,
        [corrected / "best.pt", corrected / "training_report.json"],
    )

    corrected_adapted: dict[str, Path] = {}
    for domain in ("sshr", "nsh", "psd"):
        output = campaign / "adapted_corrected" / domain / "ours"
        corrected_adapted[domain] = output
        run_stage(
            f"correct_adapt_{domain}_ours",
            [
                python,
                "scripts/train_public_multidataset.py",
                "--config",
                str(adapt_cfg),
                "--datasets",
                domain,
                "--resume",
                str(corrected / "best.pt"),
                "--output",
                str(output),
            ],
            campaign,
            [output / "best.pt", output / "training_report.json"],
        )

    image_mixed = campaign / "image_mixed"
    zero_output = campaign / "evaluation/corrected_zero_shot"
    run_stage(
        "correct_evaluate_zero_shot",
        [
            python,
            "scripts/evaluate_public_suite.py",
            "--config",
            str(eval_cfg),
            "--checkpoint",
            f"ours_pretrained={corrected / 'best.pt'}",
            "--checkpoint",
            f"image={image_mixed / 'best.pt'}",
            "--output",
            str(zero_output),
        ],
        campaign,
        [zero_output / "report.json"],
    )

    adapted_reports = {}
    for domain in ("sshr", "nsh", "psd"):
        split = "ft_test" if domain == "psd" else "test"
        output = campaign / "evaluation" / f"corrected_adapted_{domain}"
        adapted_reports[domain] = output / "report.json"
        run_stage(
            f"correct_evaluate_adapted_{domain}",
            [
                python,
                "scripts/evaluate_public_suite.py",
                "--config",
                str(eval_cfg),
                "--datasets",
                domain,
                "--split",
                f"{domain}={split}",
                "--checkpoint",
                f"ours_adapted={corrected_adapted[domain] / 'best.pt'}",
                "--checkpoint",
                f"ours_zero={corrected / 'best.pt'}",
                "--checkpoint",
                f"image_adapted={campaign / 'adapted' / domain / 'image/best.pt'}",
                "--checkpoint",
                f"image_zero={image_mixed / 'best.pt'}",
                "--output",
                str(output),
            ],
            campaign,
            [output / "report.json"],
        )

    ablation_output = campaign / "evaluation/ablations_corrected"
    ablation_names = (
        "no_synthetic",
        "no_uncertainty_fusion",
        "no_region_losses",
        "no_physics_supervision",
    )
    command = [
        python,
        "scripts/evaluate_public_suite.py",
        "--config",
        str(eval_cfg),
        "--checkpoint",
        f"full={corrected / 'best.pt'}",
    ]
    for name in ablation_names:
        command.extend(
            ["--checkpoint", f"{name}={campaign / 'ablations' / name / 'best.pt'}"]
        )
    run_stage(
        "correct_evaluate_ablations",
        command + ["--output", str(ablation_output)],
        campaign,
        [ablation_output / "report.json"],
    )

    manifest = {
        "reason": "full-model best checkpoint was reset during stabilized joint continuation",
        "corrected_full_checkpoint": str(corrected / "best.pt"),
        "zero_shot_report": str(zero_output / "report.json"),
        "adapted_reports": {
            name: str(path) for name, path in adapted_reports.items()
        },
        "ablation_report": str(ablation_output / "report.json"),
    }
    (campaign / "correction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("CORRECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
