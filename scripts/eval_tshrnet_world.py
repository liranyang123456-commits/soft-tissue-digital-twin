"""Run the official TSHRNet four-stage model on the flat world benchmark."""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TSHR_ROOT = ROOT / "baselines_ext" / "TSHRNet"
sys.path.insert(0, str(TSHR_ROOT))

from models.UNet import UNet  # noqa: E402


def load_model(channels: int, checkpoint: Path, device: torch.device) -> UNet:
    model = UNet(input_channels=channels, output_channels=3)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def save_image(tensor: torch.Tensor, path: Path, output_size: tuple[int, int]) -> None:
    tensor = F.interpolate(
        tensor, size=output_size[::-1], mode="bicubic", align_corners=False
    )
    array = (
        tensor.squeeze(0).permute(1, 2, 0).add(1).mul(127.5).clamp(0, 255)
        .byte().cpu().numpy()
    )
    Image.fromarray(array).save(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/world_benchmark_v2")
    parser.add_argument(
        "--checkpoints",
        default="baselines_ext/TSHRNet/checkpoints_mix_SSHR_SHIQ_PSD",
    )
    parser.add_argument("--output", default="predictions/tshrnet")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    checkpoints = ROOT / args.checkpoints
    output = dataset / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    networks = [
        load_model(3, checkpoints / "UNet1_60.pth", device),
        load_model(3, checkpoints / "UNet2_60.pth", device),
        load_model(6, checkpoints / "UNet3_60.pth", device),
        load_model(9, checkpoints / "UNet4_60.pth", device),
    ]
    inputs = sorted((dataset / "flat" / "test" / "input").glob("*.png"))
    for index, path in enumerate(inputs):
        original_size = Image.open(path).size
        image = load_image(path, args.size, device)
        diffuse = networks[0](image)
        specular = networks[1](image)
        refined = networks[2](torch.cat([diffuse, image], dim=1))
        prediction = networks[3](torch.cat([refined, specular, image], dim=1))
        save_image(prediction, output / path.name, original_size)
        if index % 50 == 0:
            print(f"{index}/{len(inputs)}", flush=True)


if __name__ == "__main__":
    main()
