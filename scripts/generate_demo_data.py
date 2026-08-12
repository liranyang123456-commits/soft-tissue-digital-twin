"""Generate a self-contained multi-view / multi-light demo dataset on disk.

Creates paired (highlight, clean, mask) images so Phase B/C and eval work
without downloading SHIQ/SSHR.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def render_sphere(
    H: int,
    W: int,
    light_dir: torch.Tensor,
    intensity: float = 1.0,
    roughness: float = 0.18,
    albedo_rgb=(0.72, 0.38, 0.28),
    metalness: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rr = xx**2 + yy**2
    mask_obj = (rr <= 1.0).float()
    zz = torch.sqrt((1.0 - rr).clamp(min=0.0))
    normal = torch.stack([xx, yy, zz], dim=0)
    normal = normal / (normal.norm(dim=0, keepdim=True) + 1e-6)

    albedo = torch.tensor(albedo_rgb).view(3, 1, 1).expand(3, H, W).clone()
    L = light_dir / (light_dir.norm() + 1e-6)
    V = torch.tensor([0.0, 0.0, 1.0])
    n_dot_l = (normal * L.view(3, 1, 1)).sum(0).clamp(min=0.0)
    ambient = 0.12
    diffuse = albedo * (ambient + (1.0 - ambient) * n_dot_l.unsqueeze(0)) * (1.0 - metalness)

    Hvec = (L + V)
    Hvec = Hvec / (Hvec.norm() + 1e-6)
    n_dot_h = (normal * Hvec.view(3, 1, 1)).sum(0).clamp(min=0.0)
    shininess = max(2.0, 1.0 / max(roughness, 0.04) * 10.0)
    spec = (n_dot_h**shininess).unsqueeze(0).expand(3, H, W)
    # Fresnel-ish white highlight
    specular = spec * (0.55 + 0.35 * metalness)

    full = ((diffuse + specular) * intensity).clamp(0, 1) * mask_obj.unsqueeze(0)
    clean = (diffuse * intensity).clamp(0, 1) * mask_obj.unsqueeze(0)
    # slight view-dependent darkening on clean to keep intensity consistent
    spec_mask = ((specular.mean(0) > 0.18).float() * mask_obj).unsqueeze(0)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return to_np(full), to_np(clean), (spec_mask[0].numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/demo_mv")
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    train_dir = out / "train"
    test_dir = out / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    meta = {"n_scenes": args.n_scenes, "n_views": args.n_views, "size": args.size, "scenes": []}

    n_test = max(4, args.n_scenes // 5)
    for i in range(args.n_scenes):
        split = "test" if i >= args.n_scenes - n_test else "train"
        dest = test_dir if split == "test" else train_dir
        hue = rng.uniform(0.15, 0.85)
        # simple RGB from hue-ish
        albedo = (
            float(0.35 + 0.5 * abs(math.sin(hue * 6.28))),
            float(0.25 + 0.45 * abs(math.sin(hue * 6.28 + 2.1))),
            float(0.2 + 0.5 * abs(math.sin(hue * 6.28 + 4.2))),
        )
        roughness = float(rng.uniform(0.12, 0.45))
        metal = float(rng.uniform(0.0, 0.35))
        scene_id = f"scene_{i:04d}"
        scene_meta = {
            "id": scene_id,
            "split": split,
            "albedo": albedo,
            "roughness": roughness,
            "metalness": metal,
            "views": [],
        }
        for v in range(args.n_views):
            ang = (i * 0.37 + v * (2 * math.pi / args.n_views)) % (2 * math.pi)
            elev = 0.35 + 0.25 * math.sin(ang + i * 0.1)
            light_dir = torch.tensor(
                [math.cos(ang) * 0.75, elev, 0.55 + 0.25 * math.cos(ang * 0.7)]
            )
            intensity = float(0.75 + 0.45 * ((v % 3) / 2.0))
            full, clean, mask = render_sphere(
                args.size,
                args.size,
                light_dir,
                intensity=intensity,
                roughness=roughness,
                albedo_rgb=albedo,
                metalness=metal,
            )
            stem = f"{scene_id}_v{v}"
            Image.fromarray(full).save(dest / f"{stem}_A.png")
            Image.fromarray(clean).save(dest / f"{stem}_D.png")
            Image.fromarray(mask).save(dest / f"{stem}_S.png")
            scene_meta["views"].append(
                {
                    "stem": stem,
                    "intensity": intensity,
                    "light_dir": light_dir.tolist(),
                }
            )
        meta["scenes"].append(scene_meta)

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # SHIQ-style list files
    for split, d in [("train", train_dir), ("test", test_dir)]:
        lines = []
        for a in sorted(d.glob("*_A.png")):
            d_path = a.with_name(a.name.replace("_A.png", "_D.png"))
            s_path = a.with_name(a.name.replace("_A.png", "_S.png"))
            lines.append(f"{a.name} {d_path.name} {s_path.name}")
        (out / f"{split}.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote demo dataset -> {out}")
    print(f"  train A images: {len(list(train_dir.glob('*_A.png')))}")
    print(f"  test  A images: {len(list(test_dir.glob('*_A.png')))}")


if __name__ == "__main__":
    main()
