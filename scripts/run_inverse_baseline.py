"""Run a registered third-party inverse-rendering baseline."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from mvbrdf_shr.world.baselines import (
    BaselineEnvironmentError,
    CanonicalSceneInput,
    list_baselines,
    run_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=list_baselines())
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--camera-file", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--mesh-file", type=Path)
    parser.add_argument("--repo", type=Path, help="Third-party repository checkout")
    parser.add_argument("--checkpoint", type=Path, help="Third-party checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=3600.0, help="Seconds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when metrics.json already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = CanonicalSceneInput(
        scene_id=args.scene_id,
        scene_root=args.scene_root,
        images_dir=args.images_dir,
        camera_file=args.camera_file,
        mask_dir=args.mask_dir,
        mesh_file=args.mesh_file,
    )
    output_dir = args.output_dir or (
        Path("outputs") / "inverse_baselines" / args.method / args.scene_id
    )
    try:
        result = run_baseline(
            args.method,
            scene,
            output_dir,
            repository=args.repo,
            checkpoint=args.checkpoint,
            timeout=args.timeout,
            dry_run=args.dry_run,
            skip_completed=not args.force,
        )
    except (BaselineEnvironmentError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    payload = {
        "method": result.method,
        "status": result.status,
        "output_dir": str(result.output_dir),
    }
    if result.command:
        payload["command"] = subprocess.list2cmdline(result.command)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
