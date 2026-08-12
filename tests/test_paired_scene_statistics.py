import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "paired_scene_statistics.py"
)
_SPEC = importlib.util.spec_from_file_location("paired_scene_statistics", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_paired_statistics_respects_metric_direction():
    baseline = {
        "a": {"psnr": 20.0, "lpips": 0.20},
        "b": {"psnr": 21.0, "lpips": 0.18},
    }
    candidate = {
        "a": {"psnr": 21.0, "lpips": 0.15},
        "b": {"psnr": 22.0, "lpips": 0.16},
    }

    psnr = _MODULE.paired_statistics(
        baseline,
        candidate,
        "psnr",
        higher_is_better=True,
        samples=1000,
        seed=1,
    )
    lpips = _MODULE.paired_statistics(
        baseline,
        candidate,
        "lpips",
        higher_is_better=False,
        samples=1000,
        seed=1,
    )

    assert psnr["mean_improvement"] == 1.0
    assert lpips["mean_improvement"] > 0
    assert psnr["improved_scene_count"] == lpips["improved_scene_count"] == 2
