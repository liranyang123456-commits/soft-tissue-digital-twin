"""Generate camera-randomized paired SHR data without duplicating public data."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from generate_hard_synth import render_scene


def camera_pipeline(
    linear_rgb: np.ndarray,
    exposure: float,
    gamma: float,
    gains: np.ndarray,
    shoulder: float,
) -> np.ndarray:
    value = linear_rgb.astype(np.float32) / 255.0
    value = value * exposure * gains.reshape(1, 1, 3)
    value = value / (1.0 + shoulder * value)
    value = np.clip(value, 0, 1) ** (1.0 / gamma)
    return value


def degrade_observation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    value = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    if rng.random() < 0.35:
        value = value.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.1, 1.2))))
    array = np.asarray(value).astype(np.float32) / 255.0
    array += rng.normal(0, rng.uniform(0, 0.018), array.shape).astype(np.float32)
    value = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    if rng.random() < 0.5:
        buffer = io.BytesIO()
        value.save(buffer, format="JPEG", quality=int(rng.integers(70, 96)))
        buffer.seek(0)
        value = Image.open(buffer).convert("RGB")
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/generated/domain_randomized_shr")
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.out)
    manifest = {"seed": args.seed, "size": args.size, "splits": {}}
    offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split, count in counts.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for index in range(count):
            scene_seed = args.seed + offsets[split] + index
            rng = np.random.default_rng(scene_seed)
            azimuth = float(rng.uniform(0, 2 * math.pi))
            elevation = float(rng.uniform(0.1, 0.9))
            intensity = float(rng.uniform(0.5, 1.8))
            full, diffuse, _ = render_scene(
                args.size,
                args.size,
                scene_seed,
                azimuth,
                elevation,
                intensity,
            )
            exposure = float(2 ** rng.uniform(-1.0, 1.0))
            gamma = float(rng.uniform(1.8, 2.6))
            gains = rng.uniform(0.82, 1.18, 3).astype(np.float32)
            shoulder = float(rng.uniform(0.0, 0.8))
            full_camera = camera_pipeline(full, exposure, gamma, gains, shoulder)
            diffuse_camera = camera_pipeline(diffuse, exposure, gamma, gains, shoulder)
            observation = degrade_observation(full_camera, rng)
            target = (np.clip(diffuse_camera, 0, 1) * 255).astype(np.uint8)
            specular = np.clip(
                observation.astype(np.float32) - target.astype(np.float32), 0, 255
            ).astype(np.uint8)
            mask = (
                specular.astype(np.float32).mean(-1, keepdims=True) > 5.0
            ).astype(np.uint8) * 255
            stem = f"{split}_{index:06d}"
            Image.fromarray(observation).save(directory / f"{stem}_A.png")
            Image.fromarray(target).save(directory / f"{stem}_D.png")
            Image.fromarray(specular).save(directory / f"{stem}_S.png")
            Image.fromarray(mask[:, :, 0]).save(directory / f"{stem}_T.png")
            records.append(
                {
                    "stem": stem,
                    "scene_seed": scene_seed,
                    "exposure": exposure,
                    "gamma": gamma,
                    "white_balance": gains.tolist(),
                    "tone_shoulder": shoulder,
                }
            )
        manifest["splits"][split] = records
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
