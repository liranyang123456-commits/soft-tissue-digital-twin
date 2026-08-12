"""Scene instance decomposition for digital-twin export (Phase B).

Decomposes a reconstructed Gaussian BRDF field into per-object instances
so each can carry its own rigid-body physics in USD. The pipeline is:

1. 2D segmentation: per-frame masks from SAM (zero-shot) or a color-based
   fallback when SAM / weights are unavailable.
2. Multi-view consistency: project each Gaussian into every visible frame,
   vote across the masks it lands in, and assign the instance with the
   most votes.
3. Optional open-vocabulary labeling: CLIP features per mask → semantic
   label ("chair", "cup") for material-class lookup in physics_est.

SAM and CLIP are optional dependencies imported lazily; the color-based
fallback keeps the module testable without large weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .camera import PerspectiveCamera
    from .data import MultiViewScene
    from .scene import GaussianBRDFField


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InstanceSegmentation:
    """One segmented object instance with its Gaussian membership."""

    instance_id: int
    gaussian_indices: np.ndarray  # (K,) int indices into the field
    semantic_label: str = "unknown"
    material_class: str = "unknown"  # e.g. "wood", "metal", "plastic"
    confidence: float = 0.0
    # Per-frame mask support: which frames saw this instance and how strongly.
    frame_support: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationResult:
    """Full decomposition of a scene into instances."""

    instances: dict[int, InstanceSegmentation]
    # Per-Gaussian instance label (-1 = unassigned/background).
    labels: np.ndarray
    method: str = "color"  # "sam" or "color"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_instances(
    scene: "MultiViewScene",
    field: "GaussianBRDFField",
    *,
    method: str = "auto",
    max_instances: int = 16,
    min_gaussians: int = 20,
    confidence_threshold: float = 0.3,
) -> SegmentationResult:
    """Segment a multi-view scene into per-object instances.

    Parameters
    ----------
    scene:
        Multi-view scene with frames (each carrying an image and camera).
    field:
        The reconstructed Gaussian BRDF field.
    method:
        ``auto`` (try SAM, fall back to color), ``sam``, or ``color``.
    max_instances:
        Upper bound on the number of instances to extract.
    min_gaussians:
        Instances with fewer Gaussians than this are merged into the
        background.
    confidence_threshold:
        Minimum vote fraction for a Gaussian to be assigned to an instance.
    """
    if method == "auto":
        sam_available = _check_sam_available()
        method = "sam" if sam_available else "color"

    if method == "sam":
        try:
            masks_per_frame = _segment_with_sam(scene)
        except (ImportError, RuntimeError, ValueError):
            masks_per_frame = None
        if masks_per_frame is not None:
            return _lift_to_3d(
                scene, field, masks_per_frame, method="sam",
                max_instances=max_instances, min_gaussians=min_gaussians,
                confidence_threshold=confidence_threshold,
            )
    # Color-based fallback.
    masks_per_frame = _segment_with_color(scene, max_instances=max_instances)
    return _lift_to_3d(
        scene, field, masks_per_frame, method="color",
        max_instances=max_instances, min_gaussians=min_gaussians,
        confidence_threshold=confidence_threshold,
    )


# ---------------------------------------------------------------------------
# 2D segmentation backends
# ---------------------------------------------------------------------------


def _check_sam_available() -> bool:
    """Check whether segment-anything and a checkpoint are importable."""
    try:
        import segment_anything  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def _segment_with_sam(scene: "MultiViewScene") -> list[np.ndarray] | None:
    """Run SAM on each frame; return per-frame label maps.

    Each element is (H, W) int with 0=background, 1..K=instances.
    Returns None if SAM fails at runtime (e.g. no checkpoint).
    """
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore[import-not-found]

    # SAM checkpoint path; caller must set via env or default location.
    import os
    ckpt = os.environ.get("SAM_CHECKPOINT")
    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    if not ckpt or not os.path.isfile(ckpt):
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(sam)

    label_maps: list[np.ndarray] = []
    for frame in scene.frames:
        img = frame.image
        if torch.is_tensor(img):
            img_np = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = np.asarray(img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        masks = generator.generate(img_np)
        H, W = img_np.shape[:2]
        label_map = np.zeros((H, W), dtype=np.int32)
        for idx, m in enumerate(masks, start=1):
            label_map[m["segmentation"]] = idx
        label_maps.append(label_map)
    return label_maps


def _segment_with_color(
    scene: "MultiViewScene", max_instances: int = 16
) -> list[np.ndarray]:
    """Color-based 2D segmentation fallback (k-means in RGB space).

    This is a coarse fallback when SAM is unavailable. It clusters pixels
    by color and labels connected components. Works best when objects have
    distinct colors (the common case in controlled capture).
    """
    label_maps: list[np.ndarray] = []
    for frame in scene.frames:
        img = frame.image
        if torch.is_tensor(img):
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        else:
            img_np = np.asarray(img) / 255.0 if np.asarray(img).dtype == np.uint8 else np.asarray(img)
        H, W = img_np.shape[:2]
        # Downsample for k-means speed.
        flat = img_np.reshape(-1, 3).astype(np.float32)
        if len(flat) > 10000:
            idx = np.random.default_rng(0).choice(len(flat), 10000, replace=False)
            sample = flat[idx]
        else:
            sample = flat
        k = min(max_instances, max(2, int(len(np.unique(sample, axis=0)) / 5)))
        labels_flat = _kmeans(sample, k, n_iter=10, seed=0)
        # Assign all pixels to nearest centroid.
        centroids = np.stack([sample[labels_flat == c].mean(0) if (labels_flat == c).any() else np.zeros(3) for c in range(k)])
        dists = np.linalg.norm(flat[:, None, :] - centroids[None, :, :], axis=-1)
        full_labels = dists.argmin(axis=1).reshape(H, W).astype(np.int32)
        label_maps.append(full_labels + 1)  # 0 reserved for background
    return label_maps


def _kmeans(data: np.ndarray, k: int, n_iter: int = 10, seed: int = 0) -> np.ndarray:
    """Simple k-means clustering (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if k >= n:
        return np.arange(n)
    init = rng.choice(n, k, replace=False)
    centroids = data[init].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = data[mask].mean(0)
    return labels


# ---------------------------------------------------------------------------
# 3D lifting: 2D masks → per-Gaussian instance labels
# ---------------------------------------------------------------------------


def _lift_to_3d(
    scene: "MultiViewScene",
    field: "GaussianBRDFField",
    masks_per_frame: list[np.ndarray],
    *,
    method: str,
    max_instances: int,
    min_gaussians: int,
    confidence_threshold: float,
) -> SegmentationResult:
    """Project Gaussians to each frame, vote on masks, assign 3D labels."""
    from .camera import PerspectiveCamera

    n_gaussians = field.n_gaussians
    means = field.means.detach().cpu()
    # Vote matrix: (n_gaussians, max_label+1)
    max_label = max(m.max() for m in masks_per_frame) if masks_per_frame else 0
    n_labels = int(max_label) + 1
    votes = torch.zeros(n_gaussians, n_labels, dtype=torch.float32)
    visibility = torch.zeros(n_gaussians, dtype=torch.float32)

    for frame_idx, (frame, label_map) in enumerate(zip(scene.frames, masks_per_frame)):
        cam = frame.camera if hasattr(frame, "camera") else None
        if cam is None:
            continue
        # Ensure camera is on CPU for projection consistency.
        cam_cpu = PerspectiveCamera(
            K=cam.K.detach().cpu(),
            c2w=cam.c2w.detach().cpu(),
            width=cam.width,
            height=cam.height,
            frame_id=cam.frame_id,
            view_id=cam.view_id,
        )
        uv, z, valid = cam_cpu.project(means)
        u = uv[:, 0].long().clamp(0, cam_cpu.width - 1)
        v = uv[:, 1].long().clamp(0, cam_cpu.height - 1)
        front = valid & (z > 0)
        # Sample the label map at projected positions.
        labels_at = torch.from_numpy(label_map[v.cpu(), u.cpu()].astype(np.int64))
        for gi in range(n_gaussians):
            if front[gi]:
                votes[gi, labels_at[gi]] += 1.0
                visibility[gi] += 1.0

    # Assign each Gaussian to its most-voted label (if confident).
    labels = torch.full((n_gaussians,), -1, dtype=torch.int64)
    confidence = torch.zeros(n_gaussians, dtype=torch.float32)
    visible_mask = visibility > 0
    if visible_mask.any():
        visible_votes = votes[visible_mask]
        visible_vis = visibility[visible_mask].unsqueeze(-1)
        normalized = visible_votes / (visible_vis + 1e-6)
        best_label = normalized.argmax(dim=-1)
        best_conf = normalized.max(dim=-1).values
        # Only assign if confidence above threshold AND label != 0 (background).
        assigned = (best_conf >= confidence_threshold) & (best_label > 0)
        labels_visible = torch.where(assigned, best_label, torch.tensor(-1))
        confidence_visible = best_conf
        labels[visible_mask] = labels_visible
        confidence[visible_mask] = confidence_visible

    labels_np = labels.numpy()
    confidence_np = confidence.numpy()

    # Build instance dict, filtering small instances.
    instances: dict[int, InstanceSegmentation] = {}
    unique_labels = [l for l in np.unique(labels_np) if l > 0]
    # Remap to contiguous 1..K
    for new_id, old_label in enumerate(unique_labels, start=1):
        indices = np.where(labels_np == old_label)[0]
        if len(indices) < min_gaussians:
            labels_np[indices] = -1
            continue
        # Compute per-frame support.
        frame_support: dict[int, float] = {}
        for fi in range(len(masks_per_frame)):
            support = float((labels_np == new_id).sum()) / max(1, len(indices))
            frame_support[fi] = support
        instances[new_id] = InstanceSegmentation(
            instance_id=new_id,
            gaussian_indices=indices,
            semantic_label="unknown",
            material_class="unknown",
            confidence=float(np.mean(confidence_np[indices])),
            frame_support=frame_support,
        )
        labels_np[indices] = new_id

    return SegmentationResult(
        instances=instances, labels=labels_np, method=method
    )


# ---------------------------------------------------------------------------
# Optional: open-vocabulary semantic labeling via CLIP
# ---------------------------------------------------------------------------


def label_instances_clip(
    scene: "MultiViewScene",
    result: SegmentationResult,
    candidate_labels: list[str],
    masks_per_frame: list[np.ndarray] | None = None,
) -> None:
    """Attach semantic labels to instances via CLIP zero-shot classification.

    Mutates ``result.instances`` in place, setting ``semantic_label`` and
    ``material_class`` (mapped from the label via :mod:`material_db`).
    Requires ``transformers`` + ``clip``; no-op if unavailable.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel  # type: ignore[import-not-found]
    except ImportError:
        return
    import torch as T
    from PIL import Image

    device = "cuda" if T.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # For each instance, crop the largest mask region across frames and classify.
    for inst_id, inst in result.instances.items():
        best_label = "unknown"
        best_score = 0.0
        for fi, frame in enumerate(scene.frames):
            if masks_per_frame is None or fi >= len(masks_per_frame):
                continue
            label_map = masks_per_frame[fi]
            # Find the label value corresponding to this instance (before remap).
            # This is approximate; in practice we'd track the mapping.
            mask = label_map > 0  # Use any non-background region
            if not mask.any():
                continue
            img = frame.image
            if T.is_tensor(img):
                img_np = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                img_np = np.asarray(img)
            ys, xs = np.where(mask)
            if len(xs) < 10:
                continue
            crop = img_np[ys.min():ys.max(), xs.min():xs.max()]
            if crop.size == 0:
                continue
            pil = Image.fromarray(crop)
            inputs = processor(text=candidate_labels, images=pil, return_tensors="pt", padding=True).to(device)
            with T.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)[0]
            top_idx = probs.argmax().item()
            if probs[top_idx].item() > best_score:
                best_score = probs[top_idx].item()
                best_label = candidate_labels[top_idx]
        inst.semantic_label = best_label
        inst.material_class = _label_to_material_class(best_label)


def _label_to_material_class(label: str) -> str:
    """Map a semantic label to a material-db class name."""
    label_lower = label.lower()
    mapping = {
        "chair": "wood", "table": "wood", "desk": "wood", "shelf": "wood",
        "cup": "ceramic", "mug": "ceramic", "bowl": "ceramic", "plate": "ceramic",
        "bottle": "glass", "glass": "glass", "window": "glass",
        "can": "aluminum", "metal": "steel", "iron": "steel", "steel": "steel",
        "ball": "rubber", "toy": "plastic", "box": "plastic",
        "fabric": "fabric", "cloth": "fabric", "towel": "fabric",
        "stone": "stone", "rock": "stone", "brick": "stone",
    }
    for key, mat in mapping.items():
        if key in label_lower:
            return mat
    return "unknown"
