"""Evaluate scene_0070 inverse-rendering outputs with one common evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


TEST_IDS = (0, 4, 8)


def _read(path: Path, mode: str = "RGB") -> np.ndarray:
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def _psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    weights = (mask >= 0.99).astype(np.float32)[..., None]
    mse = float(np.sum((prediction - target) ** 2 * weights))
    mse /= max(float(np.sum(weights)) * prediction.shape[-1], 1.0)
    return -10.0 * math.log10(max(mse, 1e-12))


def _ssim(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    _, score_map = structural_similarity(
        target,
        prediction,
        data_range=1.0,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=-1)
    valid = mask >= 0.99
    return float(score_map[valid].mean()) if np.any(valid) else float("nan")


def _normal_error(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    pred = prediction * 2.0 - 1.0
    gt = target * 2.0 - 1.0
    pred /= np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-6)
    gt /= np.maximum(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-6)
    cosine = np.clip(np.sum(pred * gt, axis=-1), -1.0, 1.0)
    valid = mask >= 0.99
    return float(np.degrees(np.arccos(cosine[valid])).mean())


def _scale_align(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    valid = mask >= 0.99
    aligned = prediction.copy()
    if not np.any(valid):
        return aligned
    pred = prediction[valid]
    gt = target[valid]
    scale = np.sum(pred * gt, axis=0) / np.maximum(
        np.sum(pred * pred, axis=0), 1e-8
    )
    aligned[valid] = np.clip(pred * scale[None], 0.0, 1.0)
    return aligned


def _prediction(
    prediction_dir: Path,
    index: int,
    view_id: int,
    name: str,
    required: bool = True,
) -> Path | None:
    ours_name = {
        "opt": "full",
        "diffuse": "diffuse_lobe_off",
        "kd": "albedo",
        "normal": "normal",
    }[name]
    candidates = (
        prediction_dir / f"val_{index:06d}_{name}.png",
        prediction_dir / f"{index:03d}_{name}.png",
        prediction_dir / name / f"{index:03d}.png",
        prediction_dir / f"{view_id:04d}" / f"{ours_name}.png",
        prediction_dir / f"batch{index:09d}" / {
            "opt": "pred_rgb.png",
            "diffuse": "pred_diffuse.png",
            "kd": "pred_albedo.png",
            "normal": "pred_normal.png",
        }[name],
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Missing {name} prediction; tried {candidates}")
    return None


def evaluate(
    method: str,
    prediction_dir: Path,
    scene_root: Path,
    evaluation_root: Path,
) -> dict:
    rows = []
    for index, view_id in enumerate(TEST_IDS):
        mask = _read(evaluation_root / "mask" / f"{view_id:03d}.png", "L")
        targets = {
            "full": _read(scene_root / "images" / f"{view_id:03d}.png"),
            "diffuse": _read(
                evaluation_root / "diffuse" / f"{view_id:03d}.png"
            ),
            "albedo": _read(evaluation_root / "albedo" / f"{view_id:03d}.png"),
            "normal": _read(evaluation_root / "normal" / f"{view_id:03d}.png"),
        }
        predictions: dict[str, np.ndarray] = {
            "full": _read(_prediction(prediction_dir, index, view_id, "opt")),
            "albedo": _read(_prediction(prediction_dir, index, view_id, "kd")),
            "normal": _read(
                _prediction(prediction_dir, index, view_id, "normal")
            ),
        }
        diffuse_path = _prediction(
            prediction_dir, index, view_id, "diffuse", required=False
        )
        if diffuse_path is not None:
            predictions["diffuse"] = _read(diffuse_path)
        row = {"view_id": view_id}
        for name in ("full", "diffuse", "albedo"):
            if name not in predictions:
                continue
            row[f"{name}_psnr"] = _psnr(
                predictions[name], targets[name], mask
            )
            row[f"{name}_ssim"] = _ssim(
                predictions[name], targets[name], mask
            )
        aligned_albedo = _scale_align(
            predictions["albedo"], targets["albedo"], mask
        )
        row["albedo_scale_psnr"] = _psnr(
            aligned_albedo, targets["albedo"], mask
        )
        row["albedo_scale_ssim"] = _ssim(
            aligned_albedo, targets["albedo"], mask
        )
        row["normal_mae_deg"] = _normal_error(
            predictions["normal"], targets["normal"], mask
        )
        rows.append(row)

    metric_names = sorted(
        {key for row in rows for key in row if key != "view_id"}
    )
    macro = {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in metric_names
    }
    return {
        "protocol": "scene0070-rgb-mask-camera-only-v1",
        "method": method,
        "test_view_ids": list(TEST_IDS),
        "metric_domain": (
            "strict foreground interior (alpha >= 0.99); sRGB image metrics"
        ),
        "per_view": rows,
        "macro": macro,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.method,
        args.prediction_dir.resolve(),
        args.scene_root.resolve(),
        args.evaluation_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["macro"], indent=2))


if __name__ == "__main__":
    main()
