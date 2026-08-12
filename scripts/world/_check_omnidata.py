"""Diagnose Omnidata normal convention vs the known-correct MiDaS world normal."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvbrdf_shr.world.train import load_dataset, split_scene_indices
from mvbrdf_shr.world.geometry_init import attach_monocular_normal_priors

CFG = "configs/world/benchmark_scene_0070_rgb_only_mononormal_smoke.yaml"
OUT = Path("outputs/scene0070_same_protocol/_omnidata_check")
OUT.mkdir(parents=True, exist_ok=True)

torch.hub._validate_not_a_forked_repo = lambda *a, **k: None

with open(CFG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = load_dataset(cfg)
train_ids, _ = split_scene_indices(dataset, cfg)

# Reference: MiDaS-derived world normals (known correct)
attach_monocular_normal_priors(dataset, train_ids, device=device)

# Omnidata normal model
omni = torch.hub.load(
    "alexsax/omnidata_models:main", "surface_normal_dpt_hybrid_384",
    pretrained=True, trust_repo=True, verbose=False,
).eval().to(device)

fid = train_ids[0]
frame = dataset[fid]
image = frame.image.to(device).float()
h, w = image.shape[-2:]
inp = F.interpolate(image.unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False)
with torch.no_grad():
    out = omni(inp).clamp(0, 1)[0]  # (3,384,384) in [0,1]
normal_cam = (out * 2.0 - 1.0)  # (3,384,384) camera-space, to be convention-fixed
normal_cam = F.interpolate(normal_cam.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False)[0]
normal_cam = F.normalize(normal_cam, dim=0, eps=1e-6)  # (3,h,w)

camera = frame.camera.to(device)
rotation = camera.c2w[:3, :3]
cam_pos = camera.c2w[:3, 3]
mask = (frame.mask_gt[0].to(device) > 0.5)

# Reference world normal from MiDaS
ref_world = frame.normal_prior_image.to(device)  # (3,h,w)
ref_conf = frame.normal_prior_confidence[0].to(device) > 0.5
eval_mask = mask & ref_conf

# pixel world positions for view dir (use frame geometry via K and unit depth is
# unnecessary; view dir from camera center to pixel ray is enough for orientation)
ys, xs = torch.meshgrid(
    torch.arange(h, device=device, dtype=torch.float32),
    torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
dirs_cam = pix @ torch.linalg.inv(camera.K).T
dirs_world = F.normalize(dirs_cam @ rotation.T, dim=-1, eps=1e-6)  # viewing ray (away from cam)

def world_from(cam_normal_hw3, sign_flip):
    n = cam_normal_hw3 * sign_flip  # apply convention
    nw = F.normalize(n @ rotation.T, dim=-1, eps=1e-6)
    # orient toward camera: normal should oppose viewing ray
    s = torch.where((nw * dirs_world).sum(-1, keepdim=True) > 0, -1.0, 1.0)
    return F.normalize(nw * s, dim=-1, eps=1e-6)

nc = normal_cam.permute(1, 2, 0)  # (h,w,3)
cands = {
    "id":    torch.tensor([1.0, 1.0, 1.0], device=device),
    "negx":  torch.tensor([-1.0, 1.0, 1.0], device=device),
    "negy":  torch.tensor([1.0, -1.0, 1.0], device=device),
    "negz":  torch.tensor([1.0, 1.0, -1.0], device=device),
}
ref = ref_world.permute(1, 2, 0)
print(f"eval pixels={int(eval_mask.sum())}")
best = None
for name, sf in cands.items():
    wn = world_from(nc, sf)
    cos = (wn * ref).sum(-1)[eval_mask]
    mae = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
    print(f"  {name:6s}  agree_cos={cos.mean().item():+.3f}  ang_MAE_vs_MiDaS={mae:.2f}deg")
    if best is None or mae < best[1]:
        best = (name, mae, sf)
print("BEST convention:", best[0])

# Save visualization of best
name, _, sf = best
wn = world_from(nc, sf)
vis = (wn.permute(2, 0, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
vis = np.moveaxis(vis, 0, -1) * mask.cpu().numpy()[..., None]
img_np = np.moveaxis(frame.image.numpy(), 0, -1)
img_np = np.clip(img_np / (1.0 + img_np), 0, 1) if img_np.max() > 1.5 else np.clip(img_np, 0, 1)
refvis = np.moveaxis((ref_world.cpu().numpy() * 0.5 + 0.5), 0, -1) * mask.cpu().numpy()[..., None]
row = np.concatenate([img_np, np.clip(vis, 0, 1), np.clip(refvis, 0, 1)], axis=1)
Image.fromarray((np.clip(row, 0, 1) * 255).astype(np.uint8)).save(OUT / "compare.png")
print(f"saved {OUT/'compare.png'}  (input | omnidata[{name}] | midas-ref)")
