"""Extract video frames and estimate K/P_i with COLMAP.

Requires `ffmpeg` and `colmap` executables on PATH.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--camera-model", default="PINHOLE")
    parser.add_argument("--max-size", type=int, default=1600)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP is not on PATH")

    root = Path(args.out)
    images = root / "images"
    sparse = root / "sparse"
    sparse_txt = root / "sparse_txt"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    database = root / "database.db"

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            args.video,
            "-vf",
            f"fps={args.fps},scale='min({args.max_size},iw)':-2",
            str(images / "%06d.png"),
        ]
    )
    run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            args.camera_model,
        ]
    )
    run(
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            "10",
        ]
    )
    run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ]
    )
    sparse_txt.mkdir(exist_ok=True)
    run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(sparse / "0"),
            "--output_path",
            str(sparse_txt),
            "--output_type",
            "TXT",
        ]
    )
    print(
        "\nPrepared multiview scene. Configure:\n"
        f"  data_root: {root}\n"
        "  data_type: colmap\n"
        "  sparse_dir: sparse_txt"
    )


if __name__ == "__main__":
    main()

