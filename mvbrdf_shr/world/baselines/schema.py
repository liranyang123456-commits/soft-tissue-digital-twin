"""Dependency-light schemas for external inverse-rendering baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


OUTPUT_DIRECTORIES = (
    "full",
    "albedo",
    "normal",
    "depth",
    "diffuse",
    "specular",
    "relighting",
)


@dataclass(frozen=True, slots=True)
class CanonicalSceneInput:
    """Filesystem contract passed to every baseline adapter."""

    scene_id: str
    scene_root: Path
    images_dir: Path
    camera_file: Path
    mask_dir: Path | None = None
    mesh_file: Path | None = None

    def validate(self) -> None:
        required = {
            "scene root": self.scene_root,
            "images directory": self.images_dir,
            "camera file": self.camera_file,
        }
        for label, path in required.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Canonical {label} does not exist: {path}")
        for label, path in (("mask directory", self.mask_dir), ("mesh file", self.mesh_file)):
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"Canonical {label} does not exist: {path}")

    def template_values(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "scene_root": str(self.scene_root.resolve()),
            "images_dir": str(self.images_dir.resolve()),
            "camera_file": str(self.camera_file.resolve()),
            "mask_dir": str(self.mask_dir.resolve()) if self.mask_dir else "",
            "mesh_file": str(self.mesh_file.resolve()) if self.mesh_file else "",
        }


@dataclass(frozen=True, slots=True)
class StandardOutputLayout:
    """Canonical output locations, independent of upstream repository layout."""

    root: Path

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def log(self) -> Path:
        return self.root / "baseline.log"

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_DIRECTORIES:
            (self.root / name).mkdir(exist_ok=True)

    def template_values(self) -> dict[str, str]:
        values = {
            name: str((self.root / name).resolve()) for name in OUTPUT_DIRECTORIES
        }
        values.update(
            output_dir=str(self.root.resolve()),
            metrics_file=str(self.metrics.resolve()),
        )
        return values


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Registration metadata and an argv template for one external method."""

    name: str
    repository_env: str
    checkpoint_env: str
    entrypoint_candidates: tuple[str, ...]
    command_template: tuple[str, ...]
    required_executables: tuple[str, ...] = ()
    extra_environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BaselineResources:
    repository: Path
    checkpoint: Path
    entrypoint: Path


@dataclass(frozen=True, slots=True)
class RunResult:
    method: str
    command: tuple[str, ...]
    output_dir: Path
    status: str
    returncode: int | None = None
