"""Tests for literature baseline table / eval helper."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eval_compare_literature():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "eval_compare.py"
    out = root / "outputs" / "baselines" / "lit_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--gt-dir",
            str(root),
            "--pred-root",
            str(root / "outputs" / "baselines"),
            "--methods",
            "neural_drm,dhan_shr,tshrnet,mvbrdf_shr",
            "--include-literature",
            "--out",
            str(out),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "34.501" in proc.stdout  # Neural DRM SHIQ PSNR
    assert out.exists()


def test_phase_b_config_loads():
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "train" / "phase_b.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["dataset"] == "shiq"
