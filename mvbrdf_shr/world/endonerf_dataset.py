"""EndoNeRF soft-tissue dataset loader for digital-twin reconstruction.

Loads EndoNeRF-style endoscopic video sequences (e.g. ``cutting_tissues_twice``,
``pulling_soft_tissues``) into the project's :class:`MultiViewScene` format so
they can be processed by the existing world-space Gaussian BRDF pipeline.

EndoNeRF layout::

    scene_name/
      images/          # 000000.png, 000001.png, ... (HxW RGB)
      images_right/    # stereo right (optional)
      depth/           # frame-000000.depth.png (uint16 mm or float)
      masks/           # frame-000000.mask.png (binary)
      gt_masks/        # 000000.png (tool masks)
      poses_bounds.npy # (N, 17): 3x5 pose+hwf flattened + [near, far]

The 3x5 matrix per frame is ``[R|t | H W f]`` where ``[R|t]`` is a 3x4
camera-to-world transform, and ``H W f`` are image height, width, focal
length (EndoNeRF uses a square-pixel pinhole assumption).

This loader converts to OpenCV convention (``+z forward, +y down``) used
by :class:`PerspectiveCamera`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .camera import PerspectiveCamera
from .data import MultiViewFrame, MultiViewScene


def load_endonerf_scene(
    scene_dir: str | Path,
    *,
    max_frames: int | None = None,
    frame_stride: int = 1,
    image_scale: float = 1.0,
) -> MultiViewScene:
    """Load an EndoNeRF soft-tissue scene into a MultiViewScene.

    Parameters
    ----------
    scene_dir:
        Path to the scene directory (e.g. ``.../cutting_tissues_twice``).
    max_frames:
        Optional cap on the number of frames to load.
    frame_stride:
        Load every N-th frame (downsample the sequence).
    image_scale:
        Scale factor for images (0.5 = half resolution). Affects the
        focal length and image dimensions consistently.

    Returns
    -------
    MultiViewScene with frames carrying image, depth, mask, and camera.
    """
    scene_dir = Path(scene_dir)
    images_dir = scene_dir / "images"
    depth_dir = scene_dir / "depth"
    mask_dir = scene_dir / "masks"
    poses_path = scene_dir / "poses_bounds.npy"

    if not poses_path.exists():
        raise FileNotFoundError(f"poses_bounds.npy not found in {scene_dir}")

    poses_bounds = np.load(str(poses_path))  # (N, 17)
    n_total = poses_bounds.shape[0]

    # Select frame indices.
    indices = list(range(0, n_total, frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]

    frames: list[MultiViewFrame] = []
    for frame_idx in indices:
        img_path = images_dir / f"{frame_idx:06d}.png"
        if not img_path.exists():
            img_path = images_dir / f"frame-{frame_idx:06d}.png"
        if not img_path.exists():
            img_path = images_dir / f"frame-{frame_idx:06d}.color.png"
        if not img_path.exists():
            continue

        row = poses_bounds[frame_idx]
        pose_hwf = row[:15].reshape(3, 5)  # 3x4 c2w + [H, W, f]
        c2w_endonerf = pose_hwf[:, :4]  # 3x4
        H, W, focal = pose_hwf[:, 4]  # scalar per-row
        H, W, focal = int(H), int(W), float(focal)
        near, far = float(row[15]), float(row[16])

        # Scale focal and dimensions if needed.
        new_H, new_W = int(H * image_scale), int(W * image_scale)
        new_focal = focal * image_scale

        # Load image.
        img = Image.open(str(img_path)).convert("RGB")
        if image_scale != 1.0:
            img = img.resize((new_W, new_H), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # EndoNeRF poses are typically in OpenCV-ish convention but may need
        # a convention flip. The standard EndoNeRF/NeRF-W convention is:
        # +x right, +y up, -z forward (OpenGL/NeRF). Convert to OpenCV
        # (+x right, +y down, +z forward) by flipping y and z axes of the
        # camera-to-world matrix.
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :4] = c2w_endonerf
        # Flip Y and Z columns to convert OpenGL → OpenCV.
        c2w[:3, 1] *= -1  # flip up → down
        c2w[:3, 2] *= -1  # flip -z → +z forward

        K = torch.tensor(
            [[new_focal, 0, new_W / 2], [0, new_focal, new_H / 2], [0, 0, 1.0]],
            dtype=torch.float32,
        )
        c2w_tensor = torch.from_numpy(c2w).float()
        camera = PerspectiveCamera(
            K=K, c2w=c2w_tensor, width=new_W, height=new_H,
            frame_id=frame_idx, view_id=frame_idx,
        )

        # Load depth (optional). EndoNeRF depth PNGs store values already in
        # the same units as poses_bounds near/far (typically uint8, max≈far).
        # Do NOT reinterpret as millimetres — that would shrink the scene by
        # 1000x relative to the camera intrinsics.
        depth_tensor = None
        depth_path = depth_dir / f"frame-{frame_idx:06d}.depth.png"
        if depth_path.exists():
            depth_img = Image.open(str(depth_path))
            if image_scale != 1.0:
                depth_img = depth_img.resize((new_W, new_H), Image.NEAREST)
            depth_np = np.array(depth_img, dtype=np.float32)
            # If someone stored metric mm as uint16 (>1000), convert to meters.
            if depth_np.dtype == np.float32 and depth_np.max() > 1000:
                depth_np = depth_np / 1000.0
            depth_tensor = torch.from_numpy(depth_np)

        # Load mask (optional).
        mask_tensor = None
        mask_path = mask_dir / f"frame-{frame_idx:06d}.mask.png"
        if mask_path.exists():
            mask_img = Image.open(str(mask_path))
            if image_scale != 1.0:
                mask_img = mask_img.resize((new_W, new_H), Image.NEAREST)
            mask_np = np.array(mask_img, dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_np / 255.0 if mask_np.max() > 1 else mask_np)

        frame = MultiViewFrame(
            image=img_tensor,
            camera=camera,
            image_path=img_path,
            depth_gt=depth_tensor,
            mask_gt=mask_tensor,
            view_id=frame_idx,
        )
        frames.append(frame)

    if not frames:
        raise ValueError(f"no valid frames loaded from {scene_dir}")

    # Build a sparse point cloud from depth-mapped pixels (first valid frame).
    points = None
    point_colors = None
    for frame in frames:
        if frame.depth_gt is not None:
            pts, cols = _depth_to_points(frame, max_points=2000)
            if pts is not None and len(pts) > 50:
                points = pts
                point_colors = cols
                break

    metadata = {
        "source": "endonerf",
        "scene_dir": str(scene_dir),
        "n_total_frames": n_total,
        "n_loaded_frames": len(frames),
        "near": near,
        "far": far,
        "focal": focal,
        "image_scale": image_scale,
    }

    return MultiViewScene(
        frames=frames,
        points=points,
        point_colors=point_colors,
        metadata=metadata,
    )


def _depth_to_points(
    frame: MultiViewFrame, *, max_points: int = 2000
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Back-project depth map to 3D points using the frame's camera."""
    depth = frame.depth_gt
    if depth is None:
        return None, None
    img = frame.image
    cam = frame.camera
    H, W = depth.shape[:2]
    if depth.ndim == 3:
        depth = depth[..., 0]

    # Subsample to max_points.
    step = max(1, int((H * W) ** 0.5 / (max_points ** 0.5)))
    ys, xs = torch.meshgrid(
        torch.arange(0, H, step, device=depth.device),
        torch.arange(0, W, step, device=depth.device),
        indexing="ij",
    )
    d = depth[ys, xs]
    valid = d > 0.01
    if valid.sum() < 10:
        return None, None

    xs_f = xs[valid].float()
    ys_f = ys[valid].float()
    d_f = d[valid].float()

    # Pixel → camera ray.
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    dirs_cam = torch.stack([
        (xs_f - cx) / fx,
        (ys_f - cy) / fy,
        torch.ones_like(xs_f),
    ], dim=-1) * d_f.unsqueeze(-1)

    # Camera → world.
    pts = (dirs_cam @ cam.c2w[:3, :3].T) + cam.c2w[:3, 3]

    # Colors.
    if img is not None:
        cols = img[:, ys[valid], xs[valid]].T  # (N, 3)
    else:
        cols = torch.ones_like(pts) * 0.5

    return pts, cols


def list_endonerf_scenes(datasets_dir: str | Path) -> list[str]:
    """List available EndoNeRF scenes under a datasets/endonerf directory."""
    base = Path(datasets_dir) / "endonerf"
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "poses_bounds.npy").exists()
    )
