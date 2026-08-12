"""Visualization helpers for MVBRDF-SHR."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)
    x = x.clamp(0, 1).numpy()
    return (x * 255).astype(np.uint8)


def save_image(t: torch.Tensor, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8(t)).save(path)


def make_grid(tensors: list[torch.Tensor], nrow: int | None = None) -> np.ndarray:
    imgs = [tensor_to_uint8(t) for t in tensors]
    h, w = imgs[0].shape[:2]
    n = len(imgs)
    nrow = nrow or n
    ncol = int(np.ceil(n / nrow))
    canvas = np.zeros((ncol * h, nrow * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    return canvas


def save_pipeline_viz(out: dict, path: Path | str, input_img: torch.Tensor | None = None) -> None:
    """Save a comparison strip: input | full | diffuse | mask | pred [| GT]."""
    path = Path(path)
    tiles = []
    if input_img is not None:
        tiles.append(input_img)
    elif "image" in out:
        tiles.append(out["image"])
    render = out["render"]
    tiles.extend([render.full, render.diffuse, render.mask.expand_as(render.diffuse), out["pred"]])
    if "geom" in out:
        tiles.append((out["geom"].normal + 1) * 0.5)
    grid = make_grid(tiles, nrow=len(tiles))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
