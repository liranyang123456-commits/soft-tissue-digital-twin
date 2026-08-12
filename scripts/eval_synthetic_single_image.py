"""Run local single-image methods on the generated world benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mvbrdf_shr.metrics import psnr_torch, ssim_torch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.viz import save_image


def load(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/world_benchmark_v1")
    parser.add_argument("--checkpoint", default="outputs/shiq_mask/best.pt")
    parser.add_argument("--method", default="mvbrdf_shr")
    parser.add_argument("--flat", default="flat")
    args = parser.parse_args()
    root = Path(args.dataset)
    input_dir = root / args.flat / "test" / "input"
    gt_dir = root / args.flat / "test" / "gt"
    output_dir = root / "predictions" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVBRDFSHRPro(128, 48, 96).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    psnr, ssim = [], []
    for index, path in enumerate(sorted(input_dir.glob("*.png"))):
        image = load(path, device)
        target = load(gt_dir / path.name, device)
        prediction = model(image, use_refine=True, detach_physics=True)["pred"]
        save_image(prediction, output_dir / path.name)
        psnr.append(psnr_torch(prediction, target))
        ssim.append(ssim_torch(prediction, target))
        if index % 50 == 0:
            print(f"{index}/{len(list(input_dir.glob('*.png')))}", flush=True)
    result = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "n": len(psnr),
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "prediction_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

