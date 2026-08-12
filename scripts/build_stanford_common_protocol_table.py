"""Build scene-matched Stanford-ORB relighting comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mvbrdf_shr.world.protocols import stanford_scene_group


METRICS = ("psnr_hdr", "psnr_ldr", "ssim", "lpips")
OURS_KEYS = {
    "psnr_hdr": "orb_psnr_h",
    "psnr_ldr": "orb_psnr_l",
    "ssim": "orb_ssim",
    "lpips": "orb_lpips",
}


def _macro(rows: dict[str, dict], scenes: list[str]) -> dict:
    selected = [rows[scene] for scene in scenes if scene in rows]
    return {
        **{
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in METRICS
        },
        "scene_count": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument(
        "--group",
        choices=("dev4", "pilot6", "calib10", "test32", "test38", "all42"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = list(stanford_scene_group(args.group))
    official = json.loads(args.official.read_text(encoding="utf-8"))
    ours_report = json.loads(args.ours.read_text(encoding="utf-8"))
    ours_rows = {
        row["scene"]: {
            metric: float(row["summary"][source_key])
            for metric, source_key in OURS_KEYS.items()
        }
        for row in ours_report["scenes"]
        if all(source_key in row["summary"] for source_key in OURS_KEYS.values())
    }
    common_scenes = [
        scene
        for scene in scenes
        if scene in ours_rows
        and all(
            scene in result["scenes"]
            for result in official["methods"].values()
        )
    ]
    if not common_scenes:
        raise ValueError("No common scenes across Ours and all requested baselines")
    methods = {}
    for method, result in official["methods"].items():
        methods[method] = {
            "supervision": result["supervision"],
            **_macro(result["scenes"], common_scenes),
        }
    methods["ours"] = {
        "supervision": "no geometry, material, or diffuse-image ground truth",
        **_macro(ours_rows, common_scenes),
    }
    report = {
        "protocol": {
            "task": "Stanford-ORB novel-scene relighting",
            "evaluator": "official scale-invariant HDR/LDR protocol",
            "scene_group": args.group,
            "requested_scene_count": len(scenes),
            "common_scene_count": len(common_scenes),
            "common_scenes": common_scenes,
            "comparison_rule": (
                "macro average over the intersection available for every method"
            ),
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
