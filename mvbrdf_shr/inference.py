"""Single-image / multi-view inference for MVBRDF-SHR."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mvbrdf_shr.models.pipeline import MVBRDFSHR


def load_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_image(t: torch.Tensor, path: Path) -> None:
    x = t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((x * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def run(input_path: str, output_dir: str, ckpt: str | None, size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = MVBRDFSHR(use_generative=True, resolution=size).to(device)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()

    img = load_image(Path(input_path), size).to(device)
    result = model(img, use_refine=True)
    save_image(result["pred"], out / "pred.png")
    save_image(result["render"].diffuse, out / "diffuse.png")
    save_image(result["render"].full, out / "full.png")
    save_image(result["render"].mask.expand(-1, 3, -1, -1), out / "mask.png")
    save_image((result["geom"].normal + 1) * 0.5, out / "normal.png")
    print(f"wrote results to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/infer")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    run(args.input, args.output, args.ckpt, args.size)


if __name__ == "__main__":
    main()
