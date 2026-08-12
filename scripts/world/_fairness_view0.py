"""Evaluation-fairness analysis: is holdout view 0's high normal error a
coverage artifact (surface not observable from training cameras)?"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
torch.hub._validate_not_a_forked_repo = lambda *a, **k: None
from mvbrdf_shr.world.train import load_dataset, split_scene_indices

CFG = "configs/world/benchmark_scene_0070_rgb_only_ominormal_smoke.yaml"
EVAL = Path("data/baseline_protocols/scene0070_rgb_only/evaluation_only")
PRED = Path("outputs/scene0070_same_protocol/ours_constant_light_mononormal_w2/eval_holdout")
OUT = Path("outputs/scene0070_same_protocol/_fairness")
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
ds = load_dataset(cfg)
train_ids, holdout_ids = split_scene_indices(ds, cfg)

# --- camera positions and view dirs ---
def cam_pos(i):
    return ds[i].camera.c2w[:3, 3].numpy().astype(np.float64)

train_pos = {int(ds[i].view_id): cam_pos(i) for i in train_ids}
holdout_pos = {int(ds[i].view_id): cam_pos(i) for i in holdout_ids}

# object center: least-squares intersection of camera optical axes
# optical axis: p + t*d, d = -c2w[:3,2] (OpenCV forward). Solve for point
# minimizing sum of squared perpendicular distances.
A = np.zeros((3, 3))
b = np.zeros(3)
for i in list(train_ids) + list(holdout_ids):
    c = ds[i].camera.c2w[:3, 3].numpy().astype(np.float64)
    d = -ds[i].camera.c2w[:3, 2].numpy().astype(np.float64)
    d = d / np.linalg.norm(d)
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ c
obj_center = np.linalg.solve(A, b)
print(f"object center ~= [{obj_center[0]:.3f},{obj_center[1]:.3f},{obj_center[2]:.3f}]")

# --- 1) nearest-training-view angular distance (viewpoint extrapolation) ---
print("\n[1] viewpoint angular distance to nearest training camera:")
for hv, hp in holdout_pos.items():
    hd = hp - obj_center
    hd = hd / np.linalg.norm(hd)
    best = None
    for tv, tp in train_pos.items():
        td = tp - obj_center
        td = td / np.linalg.norm(td)
        ang = np.degrees(np.arccos(np.clip(np.dot(hd, td), -1, 1)))
        if best is None or ang < best[1]:
            best = (tv, ang)
    print(f"  holdout view {hv}: nearest train view {best[0]} at {best[1]:.1f} deg")

# --- 2) surface-normal observability from training cameras ---
print("\n[2] surface-normal observability (fraction of holdout-view surface):")
for hv in sorted(holdout_pos):
    npath = EVAL / "normal" / f"{hv:03d}.png"
    mpath = EVAL / "mask" / f"{hv:03d}.png"
    if not npath.exists():
        continue
    nrm = np.asarray(Image.open(npath).convert("RGB"), np.float64) / 255.0 * 2 - 1
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    mask = np.asarray(Image.open(mpath).convert("L")) > 127
    normals = nrm[mask]  # (N,3) world normals of surface visible in this view
    # viewing direction from surface (~obj center) to each training camera
    vdirs = {}
    for tv, tp in train_pos.items():
        vd = tp - obj_center
        vdirs[tv] = vd / np.linalg.norm(vd)
    facing = np.stack([normals @ vdirs[tv] for tv in train_pos], axis=1)  # (N,9)
    n_visible = (facing > 0.0).sum(axis=1)  # how many train cams this surface faces
    n_reliable = (facing > 0.5).sum(axis=1)  # faces with >60deg margin
    N = len(normals)
    print(f"  view {hv}: surface faces 0 train cams: {100*np.mean(n_visible==0):.1f}%  "
          f"<2 cams: {100*np.mean(n_visible<2):.1f}%  "
          f"reliable(>60deg)>=1: {100*np.mean(n_reliable>=1):.1f}%")

# --- 3) save view 0 input vs reconstruction ---
def save_img(arr, name):
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(OUT / name)

for hv in [0, 4, 8]:
    gt = np.asarray(Image.open(EVAL.parent.parent / "ours" / "images" / f"{hv:03d}.png").convert("RGB"), np.float64) / 255.0
    save_img(gt, f"view{hv}_input.png")
# reconstruction (full) from eval_holdout
for hv in [0, 4, 8]:
    for cand in [PRED / f"{hv:04d}" / "full.png", PRED / f"val_00000{hv}_opt.png"]:
        if cand.exists():
            rec = np.asarray(Image.open(cand).convert("RGB"), np.float64) / 255.0
            save_img(rec, f"view{hv}_recon.png")
            break
print(f"\nsaved view images -> {OUT}")
