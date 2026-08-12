"""Paired causal ablation for source-view opacity locking.

Both train-time variants start from the same checkpoint and differ only in the
explicit ``view_locked_opacity`` value.  A third condition disables locking
only during evaluation of the locked model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mvbrdf_shr.world.evaluate_ir import evaluate_inverse_rendering
from mvbrdf_shr.world.protocols import stanford_scene_group
from mvbrdf_shr.world.train import train


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(campaign: Path, dataset: str, scene: str, stage: str) -> Path:
    candidates = (
        campaign / dataset / scene / stage / "last.pt",
        campaign / dataset / scene / "last.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def _paired_config(
    source: Path,
    output: Path,
    *,
    locked: bool,
    steps: int,
) -> dict:
    state = torch.load(source, map_location="cpu", weights_only=False)
    cfg = dict(state["config"])
    cfg.update(
        init_from=str(source),
        output_dir=str(output),
        steps=int(steps),
        auto_resume=True,
        include_novel=False,
        use_dataset_split=True,
        view_locked_opacity=bool(locked),
        densify_every=0,
        checkpoint_every=max(1, min(1000, steps)),
    )
    return cfg


def _numeric_delta(left: dict, right: dict) -> dict[str, float]:
    keys = sorted(set(left) & set(right))
    return {
        key: float(left[key]) - float(right[key])
        for key in keys
        if isinstance(left[key], (int, float))
        and not isinstance(left[key], bool)
        and isinstance(right[key], (int, float))
        and not isinstance(right[key], bool)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="stanford_orb")
    parser.add_argument("--source-stage", default="densify")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument(
        "--stanford-group",
        choices=("dev4", "pilot6", "calib10", "test32", "test38", "all42"),
        default="dev4",
    )
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()

    campaign = args.campaign if args.campaign.is_absolute() else ROOT / args.campaign
    output = args.output if args.output.is_absolute() else ROOT / args.output
    scenes = (
        list(args.scenes)
        if args.scenes
        else list(stanford_scene_group(args.stanford_group))
    )

    rows = []
    for scene in scenes:
        source = _checkpoint(campaign, args.dataset, scene, args.source_stage)
        locked_dir = output / args.dataset / scene / "train_locked"
        unlocked_dir = output / args.dataset / scene / "train_unlocked"
        locked_checkpoint = train(
            _paired_config(source, locked_dir, locked=True, steps=args.steps)
        )
        unlocked_checkpoint = train(
            _paired_config(source, unlocked_dir, locked=False, steps=args.steps)
        )

        locked = evaluate_inverse_rendering(
            locked_checkpoint,
            locked_dir / f"eval_{args.split}",
            split=args.split,
            compute_lpips=args.lpips,
            view_lock_neighbors=0,
        )
        inference_unlocked = evaluate_inverse_rendering(
            locked_checkpoint,
            locked_dir / f"eval_{args.split}_no_lock",
            split=args.split,
            compute_lpips=args.lpips,
            view_lock_neighbors=-1,
        )
        train_unlocked = evaluate_inverse_rendering(
            unlocked_checkpoint,
            unlocked_dir / f"eval_{args.split}",
            split=args.split,
            compute_lpips=args.lpips,
            view_lock_neighbors=-1,
        )
        rows.append(
            {
                "scene": scene,
                "source_checkpoint": str(source),
                "locked_checkpoint": str(locked_checkpoint),
                "unlocked_checkpoint": str(unlocked_checkpoint),
                "locked": locked["summary"],
                "inference_unlocked": inference_unlocked["summary"],
                "train_unlocked": train_unlocked["summary"],
                "locked_minus_inference_unlocked": _numeric_delta(
                    locked["summary"], inference_unlocked["summary"]
                ),
                "locked_minus_train_unlocked": _numeric_delta(
                    locked["summary"], train_unlocked["summary"]
                ),
            }
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "view_lock_causal_report.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "stanford_group": args.stanford_group,
                    "single_factor": "view_locked_opacity",
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
