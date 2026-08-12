"""Visualize the MiDaS monocular depth prior on a train view for sanity check."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvbrdf_shr.world.geometry_init import _load_monocular_depth_model  # noqa: E402

root = Path("data/baseline_protocols/scene0070_rgb_only/ours")
img_path = root / "images" / "001.png"
mask_path = root / "mask" / "001.png"
img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
h, w = img.shape[:2]
print("image size", w, h, "mask_fg_frac", float((mask > 0.5).mean()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = _load_monocular_depth_model(device)
x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
inp = F.interpolate(x, size=(384, 384), mode="bicubic", align_corners=False)
inp = (inp - mean) / std
with torch.no_grad():
    disp = model(inp)
disp = F.interpolate(disp.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False)[0, 0]
disp = disp.cpu().numpy()
depth = 1.0 / np.clip(disp, 1e-3, None)

def norm01(a, m):
    a = a.copy()
    lo, hi = np.percentile(a[m > 0.5], 2), np.percentile(a[m > 0.5], 98)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return a

disp_v = norm01(disp, mask)
depth_v = norm01(depth, mask)
# Compose: [image, disparity, depth] each masked
def apply_m(a, m):
    out = np.stack([a] * 3, axis=-1)
    out[m < 0.5] = 0.0
    return (out * 255).astype(np.uint8)
row = np.concatenate([
    (img * 255).astype(np.uint8),
    apply_m(disp_v, mask),
    apply_m(depth_v, mask),
], axis=1)
out_path = Path("outputs/scene0070_same_protocol/monoprior_check.png")
Image.fromarray(row).save(out_path)
print("saved", out_path)
# Correlation between prior depth ordering and mask region stats
print("disp fg mean/std", float(disp[mask > 0.5].mean()), float(disp[mask > 0.5].std()))
