"""Re-evaluate completed checkpoints after metric or renderer corrections."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mvbrdf_shr.world.evaluate_ir import evaluate_inverse_rendering
from mvbrdf_shr.world.export import export_scene


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs" / "full_external_campaign"
OUTPUT = ROOT / "outputs" / "full_external_campaign_corrected_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("stanford_orb", "diligent_mv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--shape", action="store_true")
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else ["diligent_mv", "stanford_orb"]
    checkpoints = [
        checkpoint
        for dataset in datasets
        for checkpoint in sorted((CAMPAIGN / dataset).glob("*/last.pt"))
    ]
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]

    rows = []
    for position, checkpoint in enumerate(checkpoints, 1):
        dataset = checkpoint.parent.parent.name
        scene = checkpoint.parent.name
        output_dir = OUTPUT / dataset / scene / "eval_holdout"
        metrics_path = output_dir / "metrics.json"
        print(f"[{position}/{len(checkpoints)}] {dataset}/{scene}", flush=True)
        if metrics_path.exists() and not args.force:
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            shape_mesh = None
            if dataset == "stanford_orb" and args.shape:
                shape_mesh = export_scene(
                    checkpoint, OUTPUT / dataset / scene / "export"
                )
            report = evaluate_inverse_rendering(
                checkpoint,
                output_dir,
                split="holdout",
                compute_lpips=args.lpips,
                shape_mesh=shape_mesh,
            )
        rows.append(
            {
                "method": "Ours-NoGT",
                "dataset": dataset,
                "scene": scene,
                "summary": report["summary"],
            }
        )

        complete_rows = []
        for candidate in [
            path
            for name in ["diligent_mv", "stanford_orb"]
            for path in sorted((OUTPUT / name).glob("*/eval_holdout/metrics.json"))
        ]:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            complete_rows.append(
                {
                    "method": "Ours-NoGT",
                    "dataset": candidate.parents[2].name,
                    "scene": candidate.parents[1].name,
                    "summary": value["summary"],
                }
            )
        macro = {}
        for name in sorted({row["dataset"] for row in complete_rows}):
            summaries = [
                row["summary"] for row in complete_rows if row["dataset"] == name
            ]
            keys = sorted({key for summary in summaries for key in summary})
            macro[name] = {
                key: sum(
                    float(summary[key])
                    for summary in summaries
                    if summary.get(key) is not None
                )
                / sum(1 for summary in summaries if summary.get(key) is not None)
                for key in keys
                if any(summary.get(key) is not None for summary in summaries)
            }
        aggregate = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "completed_scenes": len(complete_rows),
            "macro_average": macro,
            "scenes": complete_rows,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "campaign_report.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
