"""Evaluate downloaded Stanford-ORB relighting outputs with the official code."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from mvbrdf_shr.world.protocols import stanford_scene_group


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "baselines_ext" / "Stanford-ORB"
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

from orb.utils.test import compute_metrics_image_similarity  # noqa: E402


_NAME = re.compile(
    r"^(?P<method>.+?)_scene(?P<scene>\d{3})_obj\d+_"
    r"(?P<object>[a-z]+)_(?P<frame>\d{3})\.(?:png|exr)$"
)
_OBJECT_ALIASES = {"cart": "car"}
_SUPERVISION = {
    "nvdiffrec": "estimated geometry and material",
    "nvdiffrecmc": "estimated geometry and material",
    "nerfactor": "estimated geometry with external initialization",
    "invrender": "estimated geometry and material",
    "physg": "estimated geometry and material",
    "nvdiffrec_pseudo_gt": "ground-truth mesh assisted",
    "nvdiffrecmc_pseudo_gt": "ground-truth mesh assisted",
}


def _source_scene(match: re.Match[str]) -> str:
    object_name = _OBJECT_ALIASES.get(
        match.group("object"), match.group("object")
    )
    return f"{object_name}_scene{match.group('scene')}"


def _target_path(
    hdr_root: Path, source_scene: str, frame_index: int
) -> Path:
    transforms = json.loads(
        (hdr_root / source_scene / "transforms_novel.json").read_text(
            encoding="utf-8"
        )
    )["frames"]
    if not 0 <= frame_index < len(transforms):
        raise IndexError((source_scene, frame_index, len(transforms)))
    frame = transforms[frame_index]
    relative = Path(frame["file_path"]).with_suffix(".exr")
    return hdr_root / str(frame["scene_name"]) / relative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data/external/stanford_orb/results",
    )
    parser.add_argument(
        "--hdr-root",
        type=Path,
        default=ROOT / "data/external/stanford_orb/blender_HDR",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["nvdiffrec", "nvdiffrecmc", "nerfactor", "invrender", "physg"],
    )
    parser.add_argument(
        "--stanford-group",
        choices=("dev4", "pilot6", "calib10", "test32", "test38", "all42"),
        default="all42",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/stanford_official_baselines.json",
    )
    args = parser.parse_args()

    allowed = set(stanford_scene_group(args.stanford_group))
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {
        method: {} for method in args.methods
    }
    for output_path in args.results.iterdir():
        match = _NAME.match(output_path.name)
        if match is None or match.group("method") not in grouped:
            continue
        source_scene = _source_scene(match)
        if source_scene not in allowed:
            continue
        target = _target_path(
            args.hdr_root, source_scene, int(match.group("frame"))
        )
        if not target.exists():
            raise FileNotFoundError(target)
        grouped[match.group("method")].setdefault(source_scene, []).append(
            {
                "output_image": str(output_path),
                "target_image": str(target),
            }
        )

    report = {
        "protocol": {
            "evaluator": "Stanford-ORB orb.utils.test.compute_metrics_image_similarity",
            "task": "novel scene relighting",
            "scale_invariant": True,
            "stanford_group": args.stanford_group,
            "scene_aggregation": "macro average",
            "target_mapping": (
                "Archive frame indices follow each source scene's ordered "
                "transforms_novel.json entries."
            ),
        },
        "methods": {},
    }
    for method, scenes in grouped.items():
        per_scene = {
            scene: compute_metrics_image_similarity(items, scale_invariant=True)
            for scene, items in sorted(scenes.items())
        }
        keys = sorted({key for values in per_scene.values() for key in values})
        report["methods"][method] = {
            "supervision": _SUPERVISION.get(method, "unknown"),
            "completed_scenes": len(per_scene),
            "completed_frames": sum(len(items) for items in scenes.values()),
            "macro_average": {
                key: float(np.mean([values[key] for values in per_scene.values()]))
                for key in keys
            },
            "scenes": per_scene,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
