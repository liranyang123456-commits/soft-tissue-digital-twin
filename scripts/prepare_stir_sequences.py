"""Convert STIR Challenge 2024 clips to the EndoNeRF layout used by both trackers.

For every clip with start and end tattoo labels, the left/right rectified
videos are decoded, at most ``--frames`` frames are sampled uniformly
(first and last frame included), depth is computed by semi-global block
matching on the half-resolution pair, and the images are translated so that
the principal point lies at the image centre, which is the intrinsic model
assumed by both EndoNeRF loaders. The tattoo centres of the first and last
frame (STIR protocol) are stored in full-resolution pixel coordinates together
with the applied translation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STIR = Path(r"E:\MIS_Datasets\STIRC2024")
OUT = Path(r"E:\MIS_Datasets\STIRC2024\endonerf_format")
W, H = 640, 512


def tattoo_centres(segmentation: Path) -> np.ndarray:
    """Bounding-box centres of the labelled tattoo blobs (STIRLoader rule)."""
    image = cv2.imread(str(segmentation), cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(
        (image > 127).astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    centres = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centres.append([x + w // 2, y + h // 2])
    return np.asarray(centres, dtype=np.float64).reshape(-1, 2)


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def clips(root: Path) -> list[Path]:
    found = []
    for left in sorted(root.glob("**/left/seq*")):
        if (left / "segmentation" / "icgstartseg.png").exists() and list(left.glob("frames/*.mp4")):
            found.append(left)
    return found


def disparity_matcher(min_disparity: int) -> cv2.StereoSGBM:
    block = 5
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=64,
        blockSize=block,
        P1=8 * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def convert(left_dir: Path, out_dir: Path, max_frames: int) -> dict:
    right_dir = left_dir.parent.parent / left_dir.parent.name.replace("left", "right", 1) / left_dir.name
    calib = json.loads((left_dir.parent.parent / "calib.json").read_text())
    K_full = np.asarray(calib["leftcameramat"], dtype=np.float64)
    pad_full = float(np.asarray(calib["rightcameramat"])[0][2] - K_full[0, 2])
    baseline_mm = abs(float(np.asarray(calib["translation"])[0]) * 1000.0)

    left_frames = read_video(next(left_dir.glob("frames/*.mp4")))
    right_frames = read_video(next(right_dir.glob("frames/*.mp4")))
    n = min(len(left_frames), len(right_frames))
    if n < 2:
        raise ValueError("clip has fewer than two frames")
    selected = np.unique(np.round(np.linspace(0, n - 1, min(n, max_frames))).astype(int))

    f_half = K_full[0, 0] / 2.0
    cx_half, cy_half = K_full[0, 2] / 2.0, K_full[1, 2] / 2.0
    shift = np.array([W / 2.0 - cx_half, H / 2.0 - cy_half])
    affine = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float64)
    pad_half = pad_full / 2.0
    matcher = disparity_matcher(int(np.floor(4 - pad_half)) - 4)

    for sub in ("images", "depth", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    depths = []
    for k, index in enumerate(selected):
        left = cv2.resize(left_frames[index], (W, H), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right_frames[index], (W, H), interpolation=cv2.INTER_AREA)
        raw = matcher.compute(left, right).astype(np.float32) / 16.0
        disparity = raw + pad_half
        valid = (raw > matcher.getMinDisparity()) & (disparity > 1.0)
        depth = np.zeros_like(disparity)
        depth[valid] = f_half * baseline_mm / disparity[valid]
        depth[(depth < 10) | (depth > 400)] = 0
        left = cv2.warpAffine(left, affine, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        depth = cv2.warpAffine(depth, affine, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        mask = np.where(depth > 0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "images" / f"frame-{k:06d}.color.png"), left)
        cv2.imwrite(str(out_dir / "depth" / f"frame-{k:06d}.depth.png"), np.round(depth).astype(np.uint16))
        cv2.imwrite(str(out_dir / "masks" / f"frame-{k:06d}.mask.png"), mask)
        depths.append(depth[depth > 0])

    all_depth = np.concatenate(depths)
    far = float(np.percentile(all_depth, 99.5))
    pose = np.concatenate([np.eye(3), np.zeros((3, 1)), np.array([[H], [W], [f_half]])], axis=1)
    poses_bounds = np.tile(np.concatenate([pose.reshape(-1), [0.0, far]]), (len(selected), 1))
    np.save(out_dir / "poses_bounds.npy", poses_bounds)

    start = tattoo_centres(left_dir / "segmentation" / "icgstartseg.png")
    end = tattoo_centres(left_dir / "segmentation" / "icgendseg.png")
    meta = {
        "clip": str(left_dir.relative_to(STIR)),
        "video_frames": int(n),
        "selected_frames": selected.tolist(),
        "shift_half_res": shift.tolist(),
        "focal_half_res": f_half,
        "baseline_mm": baseline_mm,
        "disparity_pad_full_res": pad_full,
        "far_mm": far,
        "valid_depth_fraction": float(np.mean([d.size for d in depths]) / (W * H)),
        "start_points_full_res": start.tolist(),
        "end_points_full_res": end.tolist(),
    }
    (out_dir / "stir_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STIR / "STIRChallenge_2024")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    found = clips(args.root)[: args.limit]
    print(f"{len(found)} labelled clips")
    index = []
    for k, left_dir in enumerate(found):
        name = f"clip{k:03d}"
        target = args.output / name
        if (target / "stir_meta.json").exists():
            meta = json.loads((target / "stir_meta.json").read_text())
        else:
            try:
                meta = convert(left_dir, target, args.frames)
            except Exception as exc:
                print(f"{name}: skipped ({type(exc).__name__}: {exc})")
                continue
        if len(meta["start_points_full_res"]) == 0 or len(meta["end_points_full_res"]) == 0:
            print(f"{name}: no tattoo labels, skipped")
            continue
        index.append({"name": name, **{k: meta[k] for k in ("clip", "video_frames", "valid_depth_fraction")},
                      "points": len(meta["start_points_full_res"])})
        print(f"{name}: {meta['clip']} frames {meta['video_frames']} -> {len(meta['selected_frames'])} "
              f"points {len(meta['start_points_full_res'])} depth {meta['valid_depth_fraction']:.2f}")
    (args.output / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
