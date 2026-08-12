"""Prepare / unpack datasets into data/raw (does not touch physgen_shr)."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def unpack_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"extracted {zip_path} -> {out_dir}")
    # quick stats
    pngs = list(out_dir.rglob("*.png"))
    print(f"  png files: {len(pngs)}")


def main():
    ap = argparse.ArgumentParser(description="Unpack SHIQ/SSHR zip into data/raw")
    ap.add_argument("--dataset", choices=["shiq", "sshr", "psd"], required=True)
    ap.add_argument("--zip", type=str, required=True, help="Local zip path")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir (default: ../data/raw/<DATASET>_extracted)",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]  # SpecularReflectionRemoving
    default_out = {
        "shiq": root / "data" / "raw" / "SHIQ_extracted",
        "sshr": root / "data" / "raw" / "SSHR_extracted",
        "psd": root / "data" / "raw" / "PSD_extracted",
    }[args.dataset]
    out = Path(args.out) if args.out else default_out
    unpack_zip(Path(args.zip), out)
    print("Next: point configs/train/phase_b.yaml data_root to", out)


if __name__ == "__main__":
    main()
