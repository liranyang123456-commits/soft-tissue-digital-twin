"""Run the complete reproducible public-dataset training and comparison campaign."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run_stage(
    name: str,
    command: list[str],
    campaign_dir: Path,
    required: list[Path] | None = None,
    force: bool = False,
) -> None:
    required = required or []
    if required and not force and all(path.exists() for path in required):
        print(f"STAGE_SKIPPED {name}", flush=True)
        return
    print(f"STAGE_START {name}", flush=True)
    log_path = campaign_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"{name} completed without required outputs: {missing}")
    print(f"STAGE_DONE {name}", flush=True)


def write_ablation_configs(base_path: Path, campaign_dir: Path) -> dict[str, Path]:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    configs: dict[str, dict] = {}

    no_synthetic = deepcopy(base)
    no_synthetic["datasets"] = ["shiq", "sshr", "nsh", "psd"]
    configs["no_synthetic"] = no_synthetic

    no_fusion = deepcopy(base)
    no_fusion["use_uncertainty_fusion"] = False
    no_fusion["loss"]["uncertainty_nll"] = 0.0
    no_fusion["loss"]["uncertainty_calibration"] = 0.0
    no_fusion["loss"]["fusion_gate"] = 0.0
    configs["no_uncertainty_fusion"] = no_fusion

    no_region = deepcopy(base)
    no_region["loss"]["hl_weight"] = 0.0
    no_region["loss"]["boundary"] = 0.0
    no_region["loss"]["identity"] = 0.0
    no_region["loss"]["texture"] = 0.0
    configs["no_region_losses"] = no_region

    no_physics = deepcopy(base)
    for key in ("photo", "diffuse", "cons", "spec_ndf", "mask"):
        no_physics["loss"][key] = 0.0
    configs["no_physics_supervision"] = no_physics

    config_dir = campaign_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, config in configs.items():
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", default="outputs/public_campaign")
    parser.add_argument("--proposed-init", default="outputs/shiq_mse/best.pt")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Skip the four full-budget component ablations.",
    )
    args = parser.parse_args()

    campaign_dir = (ROOT / args.campaign_dir).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    train_cfg = ROOT / "configs/train/public_multidataset.yaml"
    adapt_cfg = ROOT / "configs/train/public_domain_adapt.yaml"
    eval_cfg = ROOT / "configs/eval/public_suite.yaml"
    synthetic_root = (ROOT / "../data/generated/domain_randomized_shr").resolve()
    proposed_init = (ROOT / args.proposed_init).resolve()
    if not proposed_init.exists():
        raise FileNotFoundError(f"missing proposed initialization: {proposed_init}")

    if not args.skip_tests:
        run_stage(
            "tests",
            [python, "-m", "pytest", "-q"],
            campaign_dir,
            force=args.force,
        )

    run_stage(
        "generate_synthetic",
        [
            python,
            "scripts/generate_domain_randomized_synth.py",
            "--out",
            str(synthetic_root),
        ],
        campaign_dir,
        [synthetic_root / "manifest.json"],
        args.force,
    )

    image_mixed = campaign_dir / "image_mixed"
    ours_mixed = campaign_dir / "ours_pretrained_mixed"
    run_stage(
        "train_image_mixed",
        [
            python,
            "scripts/train_public_image_baseline.py",
            "--config",
            str(train_cfg),
            "--output",
            str(image_mixed),
        ],
        campaign_dir,
        [image_mixed / "best.pt", image_mixed / "training_report.json"],
        args.force,
    )
    run_stage(
        "train_ours_mixed",
        [
            python,
            "scripts/train_public_multidataset.py",
            "--config",
            str(train_cfg),
            "--resume",
            str(proposed_init),
            "--output",
            str(ours_mixed),
        ],
        campaign_dir,
        [ours_mixed / "best.pt", ours_mixed / "training_report.json"],
        args.force,
    )

    adapted: dict[str, dict[str, Path]] = {}
    for domain in ("sshr", "nsh", "psd"):
        adapted[domain] = {}
        for family, script, source in (
            ("ours", "train_public_multidataset.py", ours_mixed / "best.pt"),
            ("image", "train_public_image_baseline.py", image_mixed / "best.pt"),
        ):
            output = campaign_dir / "adapted" / domain / family
            adapted[domain][family] = output
            run_stage(
                f"adapt_{domain}_{family}",
                [
                    python,
                    f"scripts/{script}",
                    "--config",
                    str(adapt_cfg),
                    "--datasets",
                    domain,
                    "--resume",
                    str(source),
                    "--output",
                    str(output),
                ],
                campaign_dir,
                [output / "best.pt", output / "training_report.json"],
                args.force,
            )

    zero_eval = campaign_dir / "evaluation" / "zero_shot"
    run_stage(
        "evaluate_zero_shot",
        [
            python,
            "scripts/evaluate_public_suite.py",
            "--config",
            str(eval_cfg),
            "--checkpoint",
            f"ours_pretrained={ours_mixed / 'best.pt'}",
            "--checkpoint",
            f"image={image_mixed / 'best.pt'}",
            "--output",
            str(zero_eval),
        ],
        campaign_dir,
        [zero_eval / "report.json"],
        args.force,
    )

    for domain in ("sshr", "nsh", "psd"):
        split = "ft_test" if domain == "psd" else "test"
        output = campaign_dir / "evaluation" / f"adapted_{domain}"
        run_stage(
            f"evaluate_adapted_{domain}",
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
                f"ours_adapted={adapted[domain]['ours'] / 'best.pt'}",
                "--checkpoint",
                f"ours_zero={ours_mixed / 'best.pt'}",
                "--checkpoint",
                f"image_adapted={adapted[domain]['image'] / 'best.pt'}",
                "--checkpoint",
                f"image_zero={image_mixed / 'best.pt'}",
                "--output",
                str(output),
            ],
            campaign_dir,
            [output / "report.json"],
            args.force,
        )

    ablation_outputs: dict[str, Path] = {}
    if not args.no_ablations:
        ablation_configs = write_ablation_configs(train_cfg, campaign_dir)
        for name, config in ablation_configs.items():
            output = campaign_dir / "ablations" / name
            ablation_outputs[name] = output
            run_stage(
                f"train_ablation_{name}",
                [
                    python,
                    "scripts/train_public_multidataset.py",
                    "--config",
                    str(config),
                    "--resume",
                    str(proposed_init),
                    "--output",
                    str(output),
                ],
                campaign_dir,
                [output / "best.pt", output / "training_report.json"],
                args.force,
            )
        ablation_eval = campaign_dir / "evaluation" / "ablations"
        command = [
            python,
            "scripts/evaluate_public_suite.py",
            "--config",
            str(eval_cfg),
            "--checkpoint",
            f"full={ours_mixed / 'best.pt'}",
        ]
        for name, output in ablation_outputs.items():
            command.extend(["--checkpoint", f"{name}={output / 'best.pt'}"])
        run_stage(
            "evaluate_ablations",
            command + ["--output", str(ablation_eval)],
            campaign_dir,
            [ablation_eval / "report.json"],
            args.force,
        )

    manifest = {
        "campaign_dir": str(campaign_dir),
        "proposed_initialization": str(proposed_init),
        "synthetic_root": str(synthetic_root),
        "zero_shot_report": str(zero_eval / "report.json"),
        "adapted_reports": {
            domain: str(
                campaign_dir / "evaluation" / f"adapted_{domain}" / "report.json"
            )
            for domain in adapted
        },
        "ablation_report": (
            str(campaign_dir / "evaluation" / "ablations" / "report.json")
            if ablation_outputs
            else None
        ),
    }
    (campaign_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("CAMPAIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
