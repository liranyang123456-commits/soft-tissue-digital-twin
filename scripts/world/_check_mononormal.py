"""Visualize the monocular normal prior (MiDaS depth -> world normals)."""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvbrdf_shr.world.train import load_dataset, split_scene_indices
from mvbrdf_shr.world.geometry_init import attach_monocular_normal_priors

CFG = "configs/world/benchmark_scene_0070_rgb_only_mononormal_smoke.yaml"
OUT = Path("outputs/scene0070_same_protocol/_mononormal_check")
OUT.mkdir(parents=True, exist_ok=True)

with open(CFG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset(cfg)
train_ids, _ = split_scene_indices(dataset, cfg)
n = attach_monocular_normal_priors(dataset, train_ids, device=device)
print(f"attached to {n} frames; train_ids={train_ids}")

fid = train_ids[0]
frame = dataset[fid]
normal = frame.normal_prior_image.numpy()  # (3,H,W) world, may be zero outside
conf = frame.normal_prior_confidence.numpy()[0]  # (H,W)
img = frame.image.numpy()
if img.shape[0] == 3:
    img = np.moveaxis(img, 0, -1)
img = np.clip(img, 0, 1)
if img.max() > 1.5:  # linear HDR -> rough display
    img = img / (1.0 + img)

# world normal -> display color
normal_vis = np.moveaxis(normal, 0, -1) * 0.5 + 0.5
normal_vis = np.clip(normal_vis, 0, 1) * conf[..., None]

mask = frame.mask_gt.numpy()[0] if frame.mask_gt is not None else conf

def save(arr, name):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(OUT / name)

save(img, "input.png")
save(normal_vis, "normal_prior.png")
save(np.repeat(conf[..., None], 3, axis=-1), "confidence.png")
save(np.repeat(mask[..., None], 3, axis=-1).astype(float), "mask.png")

# side-by-side
h, w = conf.shape
row = np.zeros((h, w * 3, 3), dtype=np.uint8)
row[:, :w] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
row[:, w:2*w] = (normal_vis * 255).astype(np.uint8)
row[:, 2*w:] = (np.repeat(conf[..., None], 3, axis=-1) * 255).astype(np.uint8)
Image.fromarray(row).save(OUT / "combined.png")
print(f"conf coverage={float((conf>0).mean()):.3f}  -> {OUT}")
