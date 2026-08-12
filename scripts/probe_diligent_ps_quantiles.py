"""Sweep robust image-PS quantiles on train-light-only DiLiGenT views."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mvbrdf_shr.metrics_ir import normal_angular_metrics
from mvbrdf_shr.world.geometry_init import _image_photometric_normal_map
from mvbrdf_shr.world.train import load_dataset, split_scene_indices
from scripts.run_geometry_gated_ablation import scene_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="readingPNG")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    base = yaml.safe_load(
        (ROOT / "configs/world/geometry_gated_clean_split.yaml").read_text(
            encoding="utf-8"
        )
    )
    base["diligent_holdout_light_stride"] = args.stride
    cfg = scene_config(base, "diligent_mv", args.scene, "coupled")
    dataset = load_dataset(cfg)
    train_ids, _ = split_scene_indices(dataset, cfg)
    allowed = set(train_ids)
    groups = [
        [index for index in values if index in allowed]
        for values in dataset.group_by_view().values()
    ]
    groups = [values for values in groups if len(values) >= 6]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [
        (0.00, 0.40),
        (0.00, 0.50),
        (0.05, 0.45),
        (0.05, 0.55),
        (0.10, 0.45),
        (0.10, 0.55),
        (0.15, 0.50),
        (0.15, 0.60),
    ]
    for lower, upper in candidates:
        errors = []
        for values in groups:
            normal, confidence, _ = _image_photometric_normal_map(
                dataset,
                values,
                device,
                face_camera=False,
                lower_quantile=lower,
                upper_quantile=upper,
            )
            frame = dataset[values[0]]
            mask = confidence[None]
            if frame.mask_gt is not None:
                mask = mask * frame.mask_gt.to(device)
            metric = normal_angular_metrics(
                normal,
                frame.normal_gt.to(device),
                mask,
                thresholds=(5.0, 10.0, 20.0),
            )
            errors.append(float(metric["mean"]))
        print(
            f"lower={lower:.2f} upper={upper:.2f} "
            f"mean={sum(errors) / len(errors):.4f}"
        )


if __name__ == "__main__":
    main()
