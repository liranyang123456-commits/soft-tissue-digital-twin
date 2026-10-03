"""Train, render, and evaluate official EndoGaussian on the common split."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = {
    "pulling_soft_tissues": "arguments/endonerf/pulling.py",
    "cutting_tissues_twice": "arguments/endonerf/cutting.py",
}


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
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


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(r"E:\third_party\EndoGaussian"))
    parser.add_argument("--conda-env", default="endogaussian-cu128")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=6017)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    scene_name = args.scene_root.name
    try:
        config = args.repo / SCENE_CONFIG[scene_name]
    except KeyError as exc:
        raise ValueError(f"unsupported EndoNeRF scene: {scene_name}") from exc
    model = args.output / "model"
    log = args.output / "endogaussian.log"
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        _run(
            [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "train.py",
                "-s",
                str(args.scene_root.resolve()),
                "--model_path",
                str(model.resolve()),
                "--port",
                str(args.port),
                "--configs",
                str(config.resolve()),
            ],
            cwd=args.repo,
            log=log,
        )
    if not args.skip_render:
        _run(
            [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "render.py",
                "--model_path",
                str(model.resolve()),
                "--skip_train",
                "--skip_video",
                "--configs",
                str(config.resolve()),
            ],
            cwd=args.repo,
            log=log,
        )

    evaluation = args.output / "evaluation"
    _run(
        [
            sys.executable,
            str((ROOT / "scripts" / "evaluate_endonerf_common.py").resolve()),
            "--scene-root",
            str(args.scene_root.resolve()),
            "--endogaussian-model",
            str(model.resolve()),
            "--method",
            "EndoGaussian",
            "--output",
            str(evaluation.resolve()),
        ],
        cwd=ROOT,
        log=log,
    )
    metadata = {
        "method": "EndoGaussian",
        "repository": str(args.repo),
        "commit": _git_commit(args.repo),
        "scene": scene_name,
        "config": str(config),
        "model": str(model),
        "evaluation": str(evaluation / "metrics.json"),
        "split": "(frame_id - 1) % 8 == 0",
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
