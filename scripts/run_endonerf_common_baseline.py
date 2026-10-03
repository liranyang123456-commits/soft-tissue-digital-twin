"""Train and evaluate official EndoNeRF with the strict common holdout."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}; see {log}")


def _write_config(
    path: Path,
    scene_root: Path,
    log_root: Path,
    experiment: str,
    iterations: int,
) -> None:
    path.write_text(
        f"""expname = {experiment}
basedir = {log_root.as_posix()}
datadir = {scene_root.as_posix()}
dataset_type = llff
factor = 1
llffhold = 8
llff_renderpath = fixidentity
N_rand = 2048
N_samples = 32
N_importance = 32
N_iter = {iterations}
use_viewdirs = True
raw_noise_std = 1e0
davinci_endoscopic = True
use_fgmask = True
use_depth = True
depth_sampling_sigma = 1.0
depth_refine_period = 4000
depth_refine_quantile = 0.1
nerf_type = direct_temporal
no_batching = True
not_zero_canonical = False
i_print = 200
i_testset = {iterations}
i_weights = 4000
i_video = {iterations + 1}
video_fps = 10
""",
        encoding="utf-8",
    )


def _render_directory(log_root: Path, experiment: str) -> Path:
    candidates = sorted((log_root / experiment).glob("renderonly_path_fixidentity_*/estim"))
    if not candidates:
        raise FileNotFoundError("EndoNeRF render-only output was not found")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(r"E:\third_party\EndoNeRF"))
    parser.add_argument("--conda-env", default="endogaussian-cu128")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_root = args.output / "logs"
    experiment = args.scene_root.name
    config = args.output / "endonerf_common.txt"
    log = args.output / "endonerf.log"
    _write_config(
        config,
        args.scene_root.resolve(),
        log_root.resolve(),
        experiment,
        args.iterations,
    )
    prefix = ["conda", "run", "-n", args.conda_env, "python", "run_endonerf.py"]
    if not args.skip_train:
        _run(prefix + ["--config", str(config.resolve())], args.repo, log)
    if not args.skip_render:
        _run(
            prefix + ["--config", str(config.resolve()), "--render_only"],
            args.repo,
            log,
        )
    prediction_dir = _render_directory(log_root, experiment)
    evaluation = args.output / "evaluation"
    _run(
        [
            sys.executable,
            str((ROOT / "scripts" / "evaluate_endonerf_common.py").resolve()),
            "--scene-root",
            str(args.scene_root.resolve()),
            "--predictions",
            str(prediction_dir.resolve()),
            "--layout",
            "sequential",
            "--method",
            "EndoNeRF-strict",
            "--output",
            str(evaluation.resolve()),
        ],
        ROOT,
        log,
    )
    metadata = {
        "method": "EndoNeRF",
        "repository": str(args.repo),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "scene": experiment,
        "iterations": args.iterations,
        "strict_holdout_patch": "test and validation frames excluded from i_train",
        "split": "np.arange(n_frames)[1:-1:8]",
        "evaluation": str(evaluation / "metrics.json"),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
