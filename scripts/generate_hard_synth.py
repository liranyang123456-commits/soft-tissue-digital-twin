"""Hard synthetic SHIQ-like benchmark: textures, multi-object, colored lights.

Protocol target: beat literature SOTA number (Neural DRM 34.50 PSNR) on this
held-out test split under identical train/test for fair method comparison.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _fbm(H: int, W: int, octaves: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(H, W)
    amp, freq = 1.0, 1.0
    for o in range(octaves):
        gh, gw = max(2, H // (2 ** (octaves - o))), max(2, W // (2 ** (octaves - o)))
        noise = torch.rand(1, 1, gh, gw, generator=g)
        noise = torch.nn.functional.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        acc = acc + amp * noise
        amp *= 0.5
        freq *= 2
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    return acc


def render_scene(
    H: int,
    W: int,
    seed: int,
    light_az: float,
    light_el: float,
    intensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Background textured plane (z=1)
    tex = _fbm(H, W, seed=seed)
    tex2 = _fbm(H, W, seed=seed + 17)
    bg_albedo = torch.stack(
        [
            0.25 + 0.5 * tex,
            0.2 + 0.45 * tex2,
            0.15 + 0.4 * ((tex + tex2) * 0.5),
        ],
        dim=0,
    )
    bg_n = torch.zeros(3, H, W)
    bg_n[2] = 1.0
    bg_rough = 0.35 + 0.25 * tex
    bg_mask = torch.ones(H, W)

    # Sphere
    cx = float(torch.rand(1, generator=g) * 0.3 - 0.15)
    cy = float(torch.rand(1, generator=g) * 0.3 - 0.15)
    rad = float(0.35 + 0.2 * torch.rand(1, generator=g))
    dx, dy = xx - cx, yy - cy
    rr = dx**2 + dy**2
    sph = (rr <= rad**2).float()
    zz = torch.sqrt((rad**2 - rr).clamp(min=0.0))
    sn = torch.stack([dx / rad, dy / rad, zz / rad], dim=0)
    sn = sn / (sn.norm(dim=0, keepdim=True) + 1e-6)
    hue = float(torch.rand(1, generator=g))
    sa = torch.tensor(
        [
            0.3 + 0.55 * abs(math.sin(hue * 6.28)),
            0.25 + 0.5 * abs(math.sin(hue * 6.28 + 2)),
            0.2 + 0.55 * abs(math.sin(hue * 6.28 + 4)),
        ]
    ).view(3, 1, 1)
    s_albedo = sa.expand(3, H, W) * (0.7 + 0.3 * _fbm(H, W, seed=seed + 3).unsqueeze(0))
    s_rough = float(0.08 + 0.35 * torch.rand(1, generator=g))
    s_metal = float(0.0 + 0.55 * torch.rand(1, generator=g))

    # Composite
    albedo = bg_albedo * (1 - sph) + s_albedo * sph
    normal = bg_n * (1 - sph) + sn * sph
    normal = normal / (normal.norm(dim=0, keepdim=True) + 1e-6)
    rough = bg_rough * (1 - sph) + s_rough * sph
    metal = s_metal * sph

    L = torch.tensor(
        [
            math.cos(light_az) * math.cos(light_el),
            math.sin(light_el),
            math.sin(light_az) * math.cos(light_el) + 0.35,
        ]
    )
    L = L / (L.norm() + 1e-6)
    # colored light
    light_col = torch.tensor(
        [
            0.85 + 0.15 * math.sin(seed),
            0.9 + 0.1 * math.cos(seed * 0.7),
            1.0,
        ]
    ).view(3, 1, 1)

    V = torch.tensor([0.0, 0.0, 1.0])
    n_dot_l = (normal * L.view(3, 1, 1)).sum(0).clamp(min=0.0)
    ambient = 0.08 + 0.04 * tex
    diffuse = albedo * (ambient + (1 - ambient) * n_dot_l.unsqueeze(0)) * (1 - 0.7 * metal)
    diffuse = diffuse * light_col * intensity

    Hvec = L + V
    Hvec = Hvec / (Hvec.norm() + 1e-6)
    n_dot_h = (normal * Hvec.view(3, 1, 1)).sum(0).clamp(min=0.0)
    shininess = (1.0 / rough.clamp(min=0.04) * 12.0).clamp(max=200)
    # Blinn-Phong with spatially varying shininess
    spec = n_dot_h.unsqueeze(0) ** shininess.unsqueeze(0)
    # anisotropic streak via stretched x
    streak = (n_dot_h * (1.0 - 0.3 * dx.abs())).clamp(min=0).unsqueeze(0) ** (shininess.unsqueeze(0) * 0.5)
    specular = (0.45 * spec + 0.25 * streak) * (0.4 + 0.6 * metal + 0.3 * (1 - sph))
    specular = specular * light_col * intensity

    full = (diffuse + specular).clamp(0, 1)
    clean = diffuse.clamp(0, 1)
    mask = ((specular.mean(0) > 0.12).float()).unsqueeze(0)
    # soft mask
    mask = (specular.mean(0, keepdim=True) / (specular.mean(0).max() + 1e-6)).clamp(0, 1)

    def to_u8(t):
        if t.dim() == 2:
            t = t.unsqueeze(0).expand(3, -1, -1)
        if t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    return to_u8(full), to_u8(clean), to_u8(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hard_synth")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--size", type=int, default=200)  # SHIQ-like
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)

    meta = {"size": args.size, "n_train": args.n_train, "n_test": args.n_test}
    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        for i in range(n):
            seed = args.seed + (0 if split == "train" else 100000) + i
            # multi-light variants as "views"
            for v in range(3):
                az = (i * 0.41 + v * 2.1) % (2 * math.pi)
                el = 0.25 + 0.35 * abs(math.sin(seed * 0.01 + v))
                intensity = 0.7 + 0.5 * ((v + 1) / 3)
                full, clean, mask = render_scene(args.size, args.size, seed, az, el, intensity)
                stem = f"{split}_{i:05d}_v{v}"
                d = out / split
                Image.fromarray(full).save(d / f"{stem}_A.png")
                Image.fromarray(clean).save(d / f"{stem}_D.png")
                Image.fromarray(mask).save(d / f"{stem}_S.png")
        print(f"{split}: {n} scenes x 3 lights")

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
