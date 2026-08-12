"""Decisive test: Omnidata vs MiDaS normal prior accuracy against GT normals."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Bypass the torch>=2.11 fork-repo validation network call (trusted hubs).
torch.hub._validate_not_a_forked_repo = lambda *a, **k: None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvbrdf_shr.world.train import load_dataset, split_scene_indices
from mvbrdf_shr.world.geometry_init import (
    attach_monocular_normal_priors,
    attach_omnidata_normal_priors,
    _load_omnidata_normal_model,
)

CFG = "configs/world/benchmark_scene_0070_rgb_only_ominormal_smoke.yaml"
GT_NORMAL = Path("data/baseline_protocols/scene0070_rgb_only/evaluation_only/normal")
OUT = Path("outputs/scene0070_same_protocol/_prior_vs_gt")
OUT.mkdir(parents=True, exist_ok=True)

with open(CFG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset(cfg)
train_ids, holdout_ids = split_scene_indices(dataset, cfg)

# Attach MiDaS priors to ALL frames (incl. holdout) for comparison
attach_monocular_normal_priors(dataset, list(train_ids) + list(holdout_ids), device=device)
omni = _load_omnidata_normal_model(torch.device(device))

def ang_err(pred_world, gt_world, mask):
    p = pred_world / np.maximum(np.linalg.norm(pred_world, axis=-1, keepdims=True), 1e-6)
    g = gt_world / np.maximum(np.linalg.norm(gt_world, axis=-1, keepdims=True), 1e-6)
    cos = np.clip((p * g).sum(-1), -1, 1)
    return float(np.degrees(np.arccos(cos[mask])).mean())

def omnidata_world(frame):
    image = frame.image.to(device).float()
    h, w = image.shape[-2:]
    img = image / (1.0 + image) if float(image.max()) > 1.5 else image
    img = img.clamp(0, 1)
    inp = F.interpolate(img.unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False)
    with torch.no_grad():
        out = omni(inp).clamp(0, 1)[0]
    nc = (out * 2 - 1)
    nc = F.interpolate(nc.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False)[0]
    nc = F.normalize(nc * torch.tensor([1., 1., -1.], device=device).view(3, 1, 1), dim=0, eps=1e-6)
    cam = frame.camera.to(device)
    rot = cam.c2w[:3, :3]
    nw = F.normalize(nc.permute(1, 2, 0) @ rot.T, dim=-1, eps=1e-6)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1)
    rays = F.normalize((pix @ torch.linalg.inv(cam.K).T) @ rot.T, dim=-1, eps=1e-6)
    s = torch.where((nw * rays).sum(-1, keepdim=True) > 0, -1., 1.)
    return F.normalize(nw * s, dim=-1, eps=1e-6).cpu().numpy()

for fid in holdout_ids:
    frame = dataset[fid]
    vid = frame.view_id
    gt_path = GT_NORMAL / f"{vid:03d}.png"
    if not gt_path.exists():
        continue
    gt = np.asarray(gt_path and Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    gt = gt * 2.0 - 1.0
    mask = frame.mask_gt[0].numpy() > 0.5
    h, w = mask.shape
    if gt.shape[:2] != (h, w):
        gt = np.asarray(Image.fromarray(((gt + 1) / 2 * 255).astype(np.uint8)).resize((w, h)), np.float32) / 255.0 * 2 - 1
    # MiDaS world normal
    midas = frame.normal_prior_image.numpy().transpose(1, 2, 0)
    omni_w = omnidata_world(frame)
    conf = frame.normal_prior_confidence[0].numpy() > 0.5
    conf_mask = mask & conf  # pixels MiDaS actually supervises
    e_midas_full = ang_err(midas, gt, mask)
    e_midas_conf = ang_err(midas, gt, conf_mask) if conf_mask.any() else float("nan")
    e_omni = ang_err(omni_w, gt, mask)
    e_omni_conf = ang_err(omni_w, gt, conf_mask) if conf_mask.any() else float("nan")
    print(f"view {vid}:  MiDaS full={e_midas_full:.2f}  MiDaS conf={e_midas_conf:.2f} (n={int(conf_mask.sum())})  "
          f"Omnidata full={e_omni:.2f}  Omnidata onMiDaSconf={e_omni_conf:.2f}")
