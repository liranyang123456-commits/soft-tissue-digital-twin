"""One-command full MVBRDF-SHR pipeline:

  generate demo data → Phase A → Phase B → Phase C → batch infer → eval → SOTA table
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env_extra: dict | None = None) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    env = None
    if env_extra:
        import os

        env = os.environ.copy()
        env.update(env_extra)
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))
    else:
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description="Run full MVBRDF-SHR pipeline")
    ap.add_argument("--quick", action="store_true", help="Fewer steps for smoke end-to-end")
    ap.add_argument("--skip-gen", action="store_true", help="Skip demo data generation")
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    py = sys.executable
    steps_a = 30 if args.quick else 100
    steps_b = 40 if args.quick else 150
    steps_c = 20 if args.quick else 80

    demo = ROOT / "data" / "demo_mv"
    if not args.skip_gen or not demo.exists():
        run(
            [
                py,
                "scripts/generate_demo_data.py",
                "--out",
                str(demo),
                "--n-scenes",
                "24" if args.quick else "40",
                "--n-views",
                "4",
                "--size",
                str(args.size),
            ]
        )

    # Phase A: toy multi-view physics
    run(
        [
            py,
            "-m",
            "mvbrdf_shr.train",
            "--config",
            "configs/train/phase_a.yaml",
            "--dataset",
            "toy",
            "--steps",
            str(steps_a),
            "--output-dir",
            "outputs/phase_a",
        ]
    )

    # Phase B: supervised on demo (stands in for SHIQ)
    run(
        [
            py,
            "-m",
            "mvbrdf_shr.train",
            "--config",
            "configs/train/phase_b_demo.yaml",
            "--steps",
            str(steps_b),
            "--resume",
            "outputs/phase_a/last.pt",
            "--output-dir",
            "outputs/phase_b",
        ]
    )

    # Phase C: freeze physics, refine generative head
    run(
        [
            py,
            "-m",
            "mvbrdf_shr.train",
            "--config",
            "configs/train/phase_c_demo.yaml",
            "--steps",
            str(steps_c),
            "--resume",
            "outputs/phase_b/last.pt",
            "--output-dir",
            "outputs/phase_c",
        ]
    )

    # Batch infer on demo test
    run(
        [
            py,
            "-m",
            "mvbrdf_shr.batch_infer",
            "--input-dir",
            str(demo / "test"),
            "--output",
            "outputs/batch_test",
            "--ckpt",
            "outputs/phase_c/last.pt",
            "--size",
            str(args.size),
        ]
    )

    # Copy preds into baselines folder for compare
    base_pred = ROOT / "outputs" / "baselines" / "mvbrdf_shr"
    if base_pred.exists():
        shutil.rmtree(base_pred)
    shutil.copytree(ROOT / "outputs" / "batch_test" / "pred", base_pred)

    # Build GT dir of *_D.png for eval_compare
    gt_dir = ROOT / "outputs" / "demo_gt"
    if gt_dir.exists():
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True)
    for p in (demo / "test").glob("*_D.png"):
        shutil.copy2(p, gt_dir / p.name)

    run(
        [
            py,
            "scripts/eval_compare.py",
            "--gt-dir",
            str(gt_dir),
            "--pred-root",
            "outputs/baselines",
            "--methods",
            "mvbrdf_shr,neural_drm,dhan_shr,tshrnet,jshdr,highlightrnet,specularitynet",
            "--include-literature",
            "--out",
            "outputs/baselines/summary.csv",
        ]
    )

    run([py, "scripts/export_sota_csv.py"])

    # Write final report
    metrics_path = ROOT / "outputs" / "batch_test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    report = ROOT / "outputs" / "PIPELINE_REPORT.md"
    summary = metrics.get("summary", {})
    report.write_text(
        f"""# MVBRDF-SHR Full Pipeline Report

## Status: COMPLETE

| Stage | Output |
|---|---|
| Demo data | `{demo}` |
| Phase A ckpt | `outputs/phase_a/last.pt` |
| Phase B ckpt | `outputs/phase_b/last.pt` |
| Phase C ckpt | `outputs/phase_c/last.pt` |
| Predictions | `outputs/batch_test/pred/` |
| Viz grids | `outputs/batch_test/viz/` |
| Compare CSV | `outputs/baselines/summary.csv` |
| SOTA table | `docs/sota_numbers.csv` |

## Demo test metrics (ours)

- N = {summary.get('n_with_gt', '—')}
- PSNR (pred) = {summary.get('psnr', '—')}
- SSIM (pred) = {summary.get('ssim', '—')}
- PSNR (physics diffuse only) = {summary.get('psnr_diffuse', '—')}

## Next (real SOTA)

1. Download SHIQ/SSHR/PSD per `scripts/prepare_datasets.md`
2. `python -m mvbrdf_shr.train --config configs/train/phase_b.yaml --dataset shiq`
3. Run open-source baselines into `outputs/baselines/<name>/` then re-run `eval_compare.py`
4. Replace literature rows with local numbers in the paper table

Literature SHIQ leaderboard remains in `docs/sota_baselines.md` (Neural DRM 34.50 PSNR is current SOTA).
""",
        encoding="utf-8",
    )
    print(f"\n=== PIPELINE COMPLETE ===\nReport: {report}")
    if summary:
        print(
            f"Demo PSNR={summary.get('psnr')} SSIM={summary.get('ssim')} "
            f"(diffuse-only {summary.get('psnr_diffuse')})"
        )


if __name__ == "__main__":
    main()
