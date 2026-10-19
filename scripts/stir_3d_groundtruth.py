"""Official STIR 3D tattoo positions for the converted clips.

Uses ``STIRStereoClip.get3DSegmentationPositions`` from the STIR loader, which
matches left and right tattoo contours by normalized cross-correlation and
triangulates them with the rectified stereo geometry (millimetres, left
camera frame at full resolution). The left-image centres of the matched
start points are the 3D query points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STIR = Path(r"E:\MIS_Datasets\STIRC2024")
DATA = STIR / "endonerf_format"
sys.path.insert(0, str(STIR / "STIRLoader"))

from STIRLoader.STIRLoader import STIRStereoClip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stir_tracking" / "gt3d.json"


def main() -> None:
    index = json.loads((DATA / "index.json").read_text())
    result = {}
    for entry in index:
        clip = STIRStereoClip(STIR / entry["clip"])
        try:
            start_2d, _, start_3d = clip.get3DSegmentationPositions(start=True)
            end_2d, _, end_3d = clip.get3DSegmentationPositions(start=False)
        except (IndexError, ValueError) as error:
            print(f"{entry['name']}: no stereo match ({error})")
            continue
        start_2d, end_2d = np.asarray(start_2d, float), np.asarray(end_2d, float)
        start_3d, end_3d = np.asarray(start_3d, float), np.asarray(end_3d, float)
        if start_3d.size == 0 or end_3d.size == 0:
            print(f"{entry['name']}: no stereo match")
            continue
        K = np.asarray(clip.leftcameramat, float)
        result[entry["name"]] = {
            "start_2d": start_2d.tolist(),
            "start_3d": start_3d.tolist(),
            "end_2d": end_2d.tolist(),
            "end_3d": end_3d.tolist(),
            "K_full": K.tolist(),
        }
        print(f"{entry['name']}: {len(start_3d)} start / {len(end_3d)} end points, "
              f"depth {np.median(start_3d[:, 2]):.0f} mm")
    OUT.write_text(json.dumps(result, indent=1))
    print(f"{len(result)} clips -> {OUT}")


if __name__ == "__main__":
    main()
