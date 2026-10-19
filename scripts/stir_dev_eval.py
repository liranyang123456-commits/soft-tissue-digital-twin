"""Evaluate tracking variants on the clips they have completed (2D and 3D)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stir_tracking"


def main() -> None:
    methods = sys.argv[1:]
    gt3d = json.loads((OUT / "gt3d.json").read_text())
    done = None
    for method in methods:
        clips = {p.parent.name for p in (OUT / method).glob("*/tracks.npz")}
        done = clips if done is None else done & clips
    done = sorted(done & set(gt3d))
    clips_file = OUT / "dev_done.json"
    clips_file.write_text(json.dumps(done))
    print(f"{len(done)} clips: {' '.join(done)}")
    base = ["present_flow"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_stir_tracking.py"), "--methods", *base, *methods,
                    "raft", "--clips-file", str(clips_file), "--tag", "dev"], check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((OUT / "summary_dev.json").read_text())
    for name, values in summary["methods"].items():
        print(f"2D {name:16s} avg {values['acc_avg']:.3f}  median {values['epe_median']:.1f} px")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_stir_tracking_3d.py"), "--methods", *base, *methods,
                    "raft_sgm", "raft_stereo", "--clips-file", str(clips_file), "--tag", "dev"], check=True)


if __name__ == "__main__":
    main()
