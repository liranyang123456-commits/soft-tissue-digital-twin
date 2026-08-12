"""Registry of external inverse-rendering methods.

The templates describe subprocess argv only; they do not imply that an
upstream repository, checkpoint, or its Python environment is installed.
"""
from __future__ import annotations

from .schema import BaselineSpec


_COMMON_OUTPUT_ARGS = (
    "--output-dir",
    "{output_dir}",
)


def _spec(
    name: str,
    entrypoints: tuple[str, ...],
    command: tuple[str, ...],
) -> BaselineSpec:
    prefix = name.upper().replace("-", "_")
    return BaselineSpec(
        name=name,
        repository_env=f"{prefix}_REPO",
        checkpoint_env=f"{prefix}_CHECKPOINT",
        entrypoint_candidates=entrypoints,
        command_template=command,
    )


_REGISTRY: dict[str, BaselineSpec] = {
    "nerfactor": _spec(
        "nerfactor",
        ("nerfactor/trainvali.py", "trainvali.py"),
        (
            "{python}",
            "{entrypoint}",
            "--mode",
            "test",
            "--scene-dir",
            "{scene_root}",
            "--checkpoint",
            "{checkpoint}",
            *_COMMON_OUTPUT_ARGS,
        ),
    ),
    "physg": _spec(
        "physg",
        ("code/evaluation/eval.py", "evaluation/eval.py", "eval.py"),
        (
            "{python}",
            "{entrypoint}",
            "--data-dir",
            "{scene_root}",
            "--checkpoint",
            "{checkpoint}",
            *_COMMON_OUTPUT_ARGS,
        ),
    ),
    "nvdiffrecmc": _spec(
        "nvdiffrecmc",
        ("train.py",),
        (
            "{python}",
            "{entrypoint}",
            "--validate",
            "--data",
            "{scene_root}",
            "--resume",
            "{checkpoint}",
            *_COMMON_OUTPUT_ARGS,
        ),
    ),
    "tensoir": _spec(
        "tensoir",
        ("scripts/relight.py", "relight.py"),
        (
            "{python}",
            "{entrypoint}",
            "--datadir",
            "{scene_root}",
            "--ckpt",
            "{checkpoint}",
            *_COMMON_OUTPUT_ARGS,
        ),
    ),
    "gs_ir": _spec(
        "gs_ir",
        ("render.py",),
        (
            "{python}",
            "{entrypoint}",
            "--source_path",
            "{scene_root}",
            "--checkpoint",
            "{checkpoint}",
            *_COMMON_OUTPUT_ARGS,
        ),
    ),
    "gaussian_shader": _spec(
        "gaussian_shader",
        ("render.py",),
        (
            "{python}",
            "{entrypoint}",
            "--source_path",
            "{scene_root}",
            "--checkpoint",
            "{checkpoint}",
            *_COMMON_OUTPUT_ARGS,
        ),
    ),
}


def list_baselines() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_baseline(name: str) -> BaselineSpec:
    try:
        return _REGISTRY[name.lower()]
    except KeyError as exc:
        choices = ", ".join(list_baselines())
        raise KeyError(f"Unknown baseline {name!r}; available methods: {choices}") from exc


def register_baseline(spec: BaselineSpec, *, replace: bool = False) -> None:
    key = spec.name.lower()
    if key in _REGISTRY and not replace:
        raise ValueError(f"Baseline {spec.name!r} is already registered")
    _REGISTRY[key] = spec
