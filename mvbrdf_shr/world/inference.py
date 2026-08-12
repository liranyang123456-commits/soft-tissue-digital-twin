"""Render highlight-free target views from a trained world-space scene."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..viz import save_image
from .pipeline import WorldGaussianBRDFPipeline
from .scene import GaussianBRDFField
from .train import load_dataset


@torch.no_grad()
def run(checkpoint: str, frame_id: int, output_dir: str, refine: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    dataset = load_dataset(cfg)
    colors = state.get("colors")
    scene = GaussianBRDFField.from_point_cloud(
        state["points"].to(device), colors.to(device) if colors is not None else None
    )
    pipeline = WorldGaussianBRDFPipeline(
        scene,
        len(dataset),
        lighting_mode=str(cfg.get("lighting_mode", "sh")),
        use_refiner=bool(cfg.get("use_refiner", True)),
        refiner_base=int(cfg.get("refiner_base", 96)),
        chunk_size=int(cfg.get("chunk_size", 32)),
        raster_backend=str(cfg.get("raster_backend", "auto")),
        shared_lighting=bool(cfg.get("shared_lighting", False)),
        background_color=cfg.get("background_color"),
        linear_hdr=bool(cfg.get("linear_hdr", False)),
        shadow_mode=str(cfg.get("shadow_mode", "none")),
        shadow_strength=float(cfg.get("shadow_strength", 1.0)),
        light_ids=[frame.light_id for frame in dataset.frames],
    ).to(device)
    pipeline.load_state_dict(state["model"], strict=False)
    pipeline.eval()

    frame = dataset[frame_id]
    image = frame.image.to(device)
    output = pipeline(
        frame.camera.to(device), frame_id, image=image, refine=refine
    )
    render = output["render"]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    save_image(image, target / "input.png")
    save_image(render.full, target / "full.png")
    save_image(render.diffuse, target / "diffuse.png")
    save_image(render.specular, target / "specular.png")
    save_image(render.mask.expand(3, -1, -1), target / "mask.png")
    save_image((render.normal + 1) * 0.5, target / "normal.png")
    save_image(render.projected_texture, target / "projected_texture.png")
    save_image(output["pred"], target / "pred.png")
    print(f"wrote world-space rendering to {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", default="outputs/world_inference")
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.frame, args.output, not args.no_refine)


if __name__ == "__main__":
    main()

