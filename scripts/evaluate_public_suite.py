"""Fair, image-wise evaluation across SHIQ/SSHR/NSH/PSD with region metrics."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvbrdf_shr.data import (
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
    collate_batch,
)
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.models.restorer import PureUNetBaseline
from train_paired_pro import batch_metrics
from train_public_multidataset import load_compatible_state


def dataset_for(name: str, root: str | None, size: int, split: str):
    if name == "shiq":
        return SHIQAdapter(root, size, split)
    if name == "sshr":
        return SSHRAdapter(root, size, split, tone_corrected=True)
    if name == "nsh":
        return NSHAdapter(root, size, split)
    if name == "psd":
        return PSDAdapter(root, size, split)
    raise ValueError(name)


def region_psnr(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    squared = (prediction.float().clamp(0, 1) - target.float().clamp(0, 1)).square()
    denominator = weight.expand_as(squared).flatten(1).sum(1).clamp(min=1)
    mse = (squared * weight).flatten(1).sum(1) / denominator
    return -10 * torch.log10(mse.clamp(min=1e-12))


def boundary(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, 5, 1, 2)
    eroded = -F.max_pool2d(-mask, 5, 1, 2)
    return (dilated - eroded).clamp(0, 1)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(
        (rx * ry).sum()
        / np.sqrt((np.square(rx).sum() * np.square(ry).sum()) + 1e-12)
    )


def _group_id(dataset, index: int) -> str:
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        return str(index)
    name = Path(pairs[index][0]).name
    return name.rsplit("idx-", 1)[0].rstrip("-")


def paired_group_statistics(
    left_rows: list[dict],
    right_rows: list[dict],
    *,
    seed: int = 2026,
    bootstrap_samples: int = 10000,
) -> dict:
    metrics = (
        "psnr",
        "ssim",
        "highlight_psnr",
        "non_highlight_psnr",
        "boundary_psnr",
        "decomposition_l1",
    )
    lower_is_better = {"decomposition_l1"}
    left = {(row["dataset"], row["group"]): [] for row in left_rows}
    right = {(row["dataset"], row["group"]): [] for row in right_rows}
    for row in left_rows:
        left[(row["dataset"], row["group"])].append(row)
    for row in right_rows:
        right[(row["dataset"], row["group"])].append(row)
    rng = np.random.default_rng(seed)
    result = {}
    for dataset in sorted({key[0] for key in left} & {key[0] for key in right}):
        groups = sorted(
            key[1]
            for key in set(left) & set(right)
            if key[0] == dataset
        )
        dataset_stats = {}
        for metric in metrics:
            raw = np.asarray(
                [
                    np.mean([row[metric] for row in left[(dataset, group)]])
                    - np.mean([row[metric] for row in right[(dataset, group)]])
                    for group in groups
                ],
                dtype=np.float64,
            )
            improvement = -raw if metric in lower_is_better else raw
            sampled = improvement[
                rng.integers(0, len(improvement), (bootstrap_samples, len(improvement)))
            ].mean(axis=1)
            dataset_stats[metric] = {
                "higher_is_better": metric not in lower_is_better,
                "group_count": len(groups),
                "mean_improvement": float(improvement.mean()),
                "bootstrap_95_ci": [
                    float(np.quantile(sampled, 0.025)),
                    float(np.quantile(sampled, 0.975)),
                ],
                "improved_group_count": int((improvement > 0).sum()),
            }
        result[dataset] = dataset_stats
    return result


@torch.inference_mode()
def evaluate_checkpoint(checkpoint: str, cfg: dict, device: torch.device):
    size = int(cfg["size"])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_type = state.get("model_type", "mvbrdf_shr")
    model_cfg = state.get("config", state.get("cfg", {}))
    if model_type == "pure_unet":
        model = PureUNetBaseline(
            base=int(model_cfg.get("restorer_channels", cfg["restorer_channels"]))
        ).to(device)
    else:
        model = MVBRDFSHRPro(
            size,
            int(model_cfg.get("base_channels", cfg["base_channels"])),
            int(model_cfg.get("restorer_channels", cfg["restorer_channels"])),
            use_strong_restorer=bool(
                model_cfg.get("use_strong_restorer", True)
            ),
            restorer_kind=str(model_cfg.get("restorer_kind", "unet")),
        ).to(device)
    load_report = load_compatible_state(model, state)
    model.eval()
    rows = []
    for name in cfg["datasets"]:
        split = cfg["splits"][name]
        dataset = dataset_for(name, cfg.get("roots", {}).get(name), size, split)
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["workers"]),
            pin_memory=True,
            collate_fn=collate_batch,
        )
        offset = 0
        for batch in loader:
            image = batch.image.to(device)
            target = batch.image_clean.to(device)
            mask = batch.mask_gt.to(device)
            if model_type == "pure_unet":
                prediction = model(image)
                output = {
                    "pred": prediction,
                    "spec_pred": (image - prediction).clamp(min=0),
                }
            else:
                output = model(
                    image,
                    use_refine=True,
                    detach_physics=True,
                )
            psnr, ssim = batch_metrics(output["pred"], target)
            high = region_psnr(output["pred"], target, mask)
            non_high = region_psnr(output["pred"], target, 1 - mask)
            edge = region_psnr(output["pred"], target, boundary(mask))
            decomposition = (
                (output["pred"] + output["spec_pred"]).clamp(0, 1) - image
            ).abs().flatten(1).mean(1)
            uncertainty = output.get("uncertainty")
            actual_error = (output["pred"] - target).abs().mean(1, keepdim=True)
            for index in range(len(image)):
                row = {
                    "dataset": name,
                    "split": split,
                    "index": offset + index,
                    "group": _group_id(dataset, offset + index),
                    "psnr": float(psnr[index]),
                    "ssim": float(ssim[index]),
                    "highlight_psnr": float(high[index]),
                    "non_highlight_psnr": float(non_high[index]),
                    "boundary_psnr": float(edge[index]),
                    "decomposition_l1": float(decomposition[index]),
                    "mean_absolute_error": float(actual_error[index].mean()),
                    "mean_uncertainty": (
                        float(uncertainty[index].mean())
                        if uncertainty is not None
                        else float("nan")
                    ),
                }
                rows.append(row)
            offset += len(image)
    summary = {}
    for name in cfg["datasets"]:
        subset = [row for row in rows if row["dataset"] == name]
        summary[name] = {
            key: float(np.mean([row[key] for row in subset]))
            for key in (
                "psnr",
                "ssim",
                "highlight_psnr",
                "non_highlight_psnr",
                "boundary_psnr",
                "decomposition_l1",
            )
        }
        summary[name]["uncertainty_error_rank_correlation"] = rank_correlation(
            np.asarray([row["mean_uncertainty"] for row in subset]),
            np.asarray([row["mean_absolute_error"] for row in subset]),
        )
        summary[name]["n"] = len(subset)
    return rows, summary, load_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/public_suite.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--split", action="append", default=[], help="dataset=split")
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path")
    parser.add_argument("--output", default="outputs/public_suite")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["datasets"] = list(args.datasets or cfg["datasets"])
    for item in args.split:
        name, split = item.split("=", 1)
        cfg["splits"][name] = split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": cfg,
        "reference_integrity": {
            "psd": (
                "Measured paired diffuse reference from the fixed-polarization "
                "capture protocol (idx-01); groups are the independent unit."
            ),
            "shiq": (
                "Protocol reference generated by RPCA; not a measured "
                "diffuse-only ground truth."
            ),
        },
        "checkpoints": {},
    }
    rows_by_checkpoint = {}
    for item in args.checkpoint:
        name, checkpoint = item.split("=", 1)
        rows, summary, load_report = evaluate_checkpoint(checkpoint, cfg, device)
        rows_by_checkpoint[name] = rows
        with (output / f"{name}_per_image.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["checkpoints"][name] = {
            "path": checkpoint,
            "summary": summary,
            "load_report": load_report,
        }
    report["comparisons"] = {}
    for left, right in itertools.combinations(report["checkpoints"], 2):
        comparison = {}
        for dataset in cfg["datasets"]:
            left_metrics = report["checkpoints"][left]["summary"][dataset]
            right_metrics = report["checkpoints"][right]["summary"][dataset]
            comparison[dataset] = {
                key: left_metrics[key] - right_metrics[key]
                for key in (
                    "psnr",
                    "ssim",
                    "highlight_psnr",
                    "non_highlight_psnr",
                    "boundary_psnr",
                    "decomposition_l1",
                )
            }
        report["comparisons"][f"{left}_minus_{right}"] = comparison
        report["comparisons"][f"{left}_minus_{right}"][
            "paired_group_statistics"
        ] = paired_group_statistics(
            rows_by_checkpoint[left], rows_by_checkpoint[right]
        )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoints": report["checkpoints"], "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
