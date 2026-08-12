"""Run resumable full-data training/evaluation on downloaded real datasets."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "outputs" / "full_external_campaign_nogt_v3"
GATE_REPORT = ROOT / "outputs" / "geometry_gated_v4" / "ablation_report.json"
DILIGENT_ROOT = (
    ROOT
    / "data"
    / "external"
    / "diligent_mv"
    / "DiLiGenT-MV"
    / "mvpmsData"
)
ORB_ROOT = ROOT / "data" / "external" / "stanford_orb" / "blender_HDR"
ORB_GT_ROOT = ROOT / "data" / "external" / "stanford_orb" / "ground_truth"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _scene_jobs(
    datasets: set[str],
    steps_diligent: int,
    steps_orb: int,
) -> list[tuple[str, str, dict[str, Any]]]:
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    if "diligent" in datasets:
        base = _load_yaml(ROOT / "configs" / "world" / "diligent_mv.yaml")
        for scene_root in sorted(DILIGENT_ROOT.glob("*PNG")):
            cfg = dict(base)
            cfg.update(
                protocol_label="Ours-NoGT",
                seed=2026 + sum(ord(character) for character in scene_root.name),
                data_root=str(scene_root.relative_to(ROOT)).replace("\\", "/"),
                steps=steps_diligent,
                image_scale=0.5,
                shadow_mode="none",
                raster_backend="gsplat",
                auto_resume=True,
                checkpoint_every=1000,
                use_mesh_points=False,
                freeze_geometry=False,
                freeze_normals=True,
                optimize_normals=False,
                visual_hull_init=False,
                multiview_mask_init=False,
                image_photometric_stereo_init=True,
                photometric_stereo_init=False,
                photometric_stereo_use_all_views=True,
                photometric_stereo_max_views=20,
                photometric_stereo_face_camera=False,
                view_locked_opacity=True,
                normal_mode="learned",
                freeze_lighting=True,
                world_scale=0.002,
                holdout_view_stride=5,
                holdout_view_offset=0,
                view_sampling="distinct_views",
                geometry_warmup_steps=0,
                freeze_geometry_after_warmup=False,
                densify_every=500,
                densify_start=500,
                densify_end=min(8000, steps_diligent),
                prune_opacity_threshold=0.05,
                scene_lr=0.001,
                output_dir=str(
                    (
                        CAMPAIGN_ROOT / "diligent_mv" / scene_root.name
                    ).relative_to(ROOT)
                ).replace("\\", "/"),
            )
            cfg["loss"].update(
                normal_gt=0.0,
                depth_gt=0.0,
                silhouette_bce=0.2,
                silhouette_dice=0.2,
                background_leak=0.1,
                depth_smooth=0.02,
                normal_smooth=0.05,
                normal_prior=1.0,
            )
            jobs.append(("diligent_mv", scene_root.name, cfg))
    if "stanford" in datasets:
        base = _load_yaml(ROOT / "configs" / "world" / "stanford_orb.yaml")
        for transforms in sorted(ORB_ROOT.glob("*/transforms_train.json")):
            scene_root = transforms.parent
            cfg = dict(base)
            cfg.update(
                protocol_label="Ours-NoGT",
                seed=2026 + sum(ord(character) for character in scene_root.name),
                data_root=str(scene_root.relative_to(ROOT)).replace("\\", "/"),
                ground_truth_root=str(ORB_GT_ROOT.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                steps=steps_orb,
                source_is_linear=True,
                linear_hdr=True,
                shadow_mode="none",
                raster_backend="gsplat",
                use_mesh_points=False,
                freeze_geometry=False,
                optimize_normals=False,
                visual_hull_init=True,
                visual_hull_resolution=64,
                normal_mode="covariance",
                geometry_warmup_steps=5000,
                freeze_geometry_after_warmup=True,
                densify_every=500,
                densify_start=500,
                densify_end=min(8000, steps_orb),
                prune_opacity_threshold=0.05,
                use_dataset_split=True,
                include_novel=True,
                auto_resume=True,
                checkpoint_every=1000,
                image_scale=0.25,
                output_dir=str(
                    (
                        CAMPAIGN_ROOT / "stanford_orb" / scene_root.name
                    ).relative_to(ROOT)
                ).replace("\\", "/"),
            )
            cfg["loss"].update(
                normal_gt=0.0,
                depth_gt=0.0,
                silhouette_bce=0.2,
                silhouette_dice=0.2,
                background_leak=0.1,
                depth_smooth=0.02,
                normal_smooth=0.05,
            )
            jobs.append(("stanford_orb", scene_root.name, cfg))
    return jobs


def _aggregate(jobs: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for dataset, scene, cfg in jobs:
        output = ROOT / cfg["output_dir"]
        metric_path = output / "eval_holdout" / "metrics.json"
        failure_path = output / "failure.json"
        if metric_path.exists():
            report = json.loads(metric_path.read_text(encoding="utf-8"))
            novel_path = output / "eval_novel" / "metrics.json"
            completed.append(
                {
                    "method": "Ours-NoGT",
                    "dataset": dataset,
                    "scene": scene,
                    "summary": report["summary"],
                    "novel_light_summary": (
                        json.loads(novel_path.read_text(encoding="utf-8"))["summary"]
                        if novel_path.exists()
                        else None
                    ),
                    "metrics_path": str(metric_path.relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                }
            )
        elif failure_path.exists():
            failed.append(json.loads(failure_path.read_text(encoding="utf-8")))
    by_dataset: dict[str, dict[str, float]] = {}
    for dataset in sorted({item["dataset"] for item in completed}):
        rows = [item["summary"] for item in completed if item["dataset"] == dataset]
        keys = sorted({key for row in rows for key in row})
        by_dataset[dataset] = {
            key: sum(float(row[key]) for row in rows if key in row)
            / sum(1 for row in rows if key in row)
            for key in keys
        }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "planned_scenes": len(jobs),
        "completed_scenes": len(completed),
        "failed_scenes": len(failed),
        "macro_average": by_dataset,
        "scenes": completed,
        "failures": failed,
    }
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN_ROOT / "campaign_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="diligent,stanford",
        help="Comma-separated: diligent,stanford",
    )
    parser.add_argument("--steps-diligent", type=int, default=30000)
    parser.add_argument("--steps-stanford", type=int, default=30000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-before-gate",
        action="store_true",
        help="Run despite a missing or failed six-scene geometry gate.",
    )
    args = parser.parse_args()
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    unknown = datasets - {"diligent", "stanford"}
    if unknown:
        raise ValueError(f"unsupported datasets: {sorted(unknown)}")
    if not args.dry_run and not args.force_before_gate:
        if not GATE_REPORT.exists():
            raise RuntimeError(f"geometry gate report is missing: {GATE_REPORT}")
        gate = json.loads(GATE_REPORT.read_text(encoding="utf-8")).get("gate", {})
        if not gate.get("geometry_gate_passed", False):
            raise RuntimeError(
                "Ours-NoGT geometry gate has not passed; refusing the 47-scene rerun"
            )

    all_jobs = _scene_jobs(datasets, args.steps_diligent, args.steps_stanford)
    selected = all_jobs[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(
        f"planned={len(all_jobs)} selected={len(selected)} "
        f"start={args.start} datasets={sorted(datasets)}"
    )
    if args.dry_run:
        for dataset, scene, _ in selected:
            print(f"{dataset}/{scene}")
        return

    for position, (dataset, scene, cfg) in enumerate(selected, start=args.start):
        output = ROOT / cfg["output_dir"]
        metrics = output / "eval_holdout" / "metrics.json"
        print(
            f"\n=== [{position + 1}/{len(all_jobs)}] {dataset}/{scene} ===",
            flush=True,
        )
        if metrics.exists():
            print(f"skip completed {metrics.relative_to(ROOT)}")
            continue
        config_path = CAMPAIGN_ROOT / "configs" / dataset / f"{scene}.yaml"
        _write_yaml(config_path, cfg)
        output.mkdir(parents=True, exist_ok=True)
        failure_path = output / "failure.json"
        train_code = _run(
            [
                sys.executable,
                "-m",
                "mvbrdf_shr.world.train",
                "--config",
                str(config_path),
            ],
            output / "campaign.log",
        )
        if train_code != 0:
            failure_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "stage": "train",
                        "exit_code": train_code,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _aggregate(all_jobs)
            continue
        export_dir = output / "export"
        export_code = _run(
            [
                sys.executable,
                "-m",
                "mvbrdf_shr.world.export",
                "--checkpoint",
                str(output / "last.pt"),
                "--output",
                str(export_dir),
            ],
            output / "campaign.log",
        )
        if export_code != 0:
            failure_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "stage": "export",
                        "exit_code": export_code,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _aggregate(all_jobs)
            continue
        eval_command = [
            sys.executable,
            "-m",
            "mvbrdf_shr.world.evaluate_ir",
            "--checkpoint",
            str(output / "last.pt"),
            "--split",
            "holdout",
            "--output",
            str(output / "eval_holdout"),
            "--lpips",
        ]
        if dataset == "stanford_orb":
            eval_command.extend(
                ["--shape-mesh", str(export_dir / "surface_mesh.ply")]
            )
        eval_code = _run(eval_command, output / "campaign.log")
        if eval_code == 0 and dataset == "stanford_orb":
            novel_code = _run(
                [
                    sys.executable,
                    "-m",
                    "mvbrdf_shr.world.evaluate_ir",
                    "--checkpoint",
                    str(output / "last.pt"),
                    "--split",
                    "novel",
                    "--output",
                    str(output / "eval_novel"),
                    "--lpips",
                ],
                output / "campaign.log",
            )
            if novel_code != 0:
                eval_code = novel_code
        if eval_code != 0:
            failure_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "stage": "evaluate",
                        "exit_code": eval_code,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif failure_path.exists():
            failure_path.unlink()
        _aggregate(all_jobs)

    report = _aggregate(all_jobs)
    print(
        f"campaign complete: {report['completed_scenes']}/{report['planned_scenes']} "
        f"finished, {report['failed_scenes']} failed"
    )


if __name__ == "__main__":
    main()
