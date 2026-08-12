"""PlotNeuralNet diagram of the 3D-ResNet (R3D-18) force estimator.

Architecture (from models/force_regressor.py, prior work):
  input video clip (3 x T x H x W)
  -> conv1 (3D conv, 64) -> bn -> relu -> maxpool3d
  -> layer1: 2x BasicBlock3d(64)     [frozen]
  -> layer2: 2x BasicBlock3d(128)    [frozen]
  -> layer3: 2x BasicBlock3d(256)    [fine-tuned]
  -> layer4: 2x BasicBlock3d(512)    [fine-tuned]
  -> adaptive avgpool3d -> 512
  -> regressor head: Linear(512,512) -> ReLU -> Dropout -> Linear(512,1)
  -> scalar force
"""
from __future__ import annotations

import sys
from pathlib import Path

PNN = Path(r"E:\PlotNeuralNet-master")
sys.path.insert(0, str(PNN))

from pycore.tikzeng import *  # noqa: E402,F401,F403

OUT_DIR = Path(r"e:\Digital twin of soft tissue\mvbrdf_shr_next\submission\cmpb\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

arch = [
    to_head(str(PNN).replace("\\", "/")),
    to_cor(),
    to_begin(),

    # input: actual endoscopic video frame
    to_input("forcenet_input_frame.png", to="(0,0,0)", width=7, height=7,
             name="input"),

    # stem
    to_Conv("conv1", s_filer=16, n_filer=64, offset="(2.8,0,0)", to="(input-east)",
            width=3, height=40, depth=40, caption="conv3d\\\\$7{\\times}7{\\times}7$\\\\64"),
    to_connection("input", "conv1"),

    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
            width=2, height=32, depth=32, caption="maxpool3d"),
    to_connection("conv1", "pool1"),

    # residual layers
    to_ConvRes("layer1", s_filer=16, n_filer=64, offset="(2.2,0,0)", to="(pool1-east)",
               width=6, height=30, depth=30, caption="layer1\\\\$2{\\times}$BasicBlock\\\\64"),
    to_connection("pool1", "layer1"),

    to_ConvRes("layer2", s_filer=8, n_filer=128, offset="(2.4,0,0)", to="(layer1-east)",
               width=5.5, height=24, depth=24, caption="layer2\\\\$2{\\times}$BasicBlock\\\\128"),
    to_connection("layer1", "layer2"),

    to_ConvRes("layer3", s_filer=4, n_filer=256, offset="(2.4,0,0)", to="(layer2-east)",
               width=5, height=18, depth=18, caption="layer3\\\\$2{\\times}$BasicBlock\\\\256 (tuned)"),
    to_connection("layer2", "layer3"),

    to_ConvRes("layer4", s_filer=2, n_filer=512, offset="(2.4,0,0)", to="(layer3-east)",
               width=4.5, height=13, depth=13, caption="layer4\\\\$2{\\times}$BasicBlock\\\\512 (tuned)"),
    to_connection("layer3", "layer4"),

    # global average pool
    to_Pool("avgpool", offset="(2.0,0,0)", to="(layer4-east)",
            width=2, height=7, depth=7, caption="avgpool3d\\\\512"),
    to_connection("layer4", "avgpool"),

    # regressor head
    to_Conv("fc1", s_filer=1, n_filer=512, offset="(2.4,0,0)", to="(avgpool-east)",
            width=2.5, height=4, depth=24, caption="fc 512\\\\ReLU+Drop"),
    to_connection("avgpool", "fc1"),

    # output: scalar force
    to_SoftMax("fc2", s_filer=1, offset="(2.4,0,0)", to="(fc1-east)",
               width=2.5, height=4, depth=8, caption="output\\\\force $\\hat{f}$ (N)"),
    to_connection("fc1", "fc2"),

    to_end(),
]


def main():
    tex_path = OUT_DIR / "fig_forcenet.tex"
    to_generate(arch, str(tex_path))
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
