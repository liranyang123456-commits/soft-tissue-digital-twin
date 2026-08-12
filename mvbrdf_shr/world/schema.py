"""Serializable canonical schema for multi-view inverse-rendering scenes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any

from .coords import CameraConvention


class ColorSpace(str, Enum):
    SRGB = "srgb"
    LINEAR = "linear"
    ACESCG = "acescg"


class GroundTruthStatus(str, Enum):
    NONE = "none"
    PSEUDO = "pseudo"
    MEASURED = "measured"
    SYNTHETIC = "synthetic"


@dataclass(slots=True)
class AssetPaths:
    """Relative paths for one capture and its optional supervision."""

    full: str
    diffuse: str | None = None
    specular: str | None = None
    albedo: str | None = None
    normal: str | None = None
    depth: str | None = None
    mask: str | None = None
    roughness: str | None = None
    metallic: str | None = None
    envmap: str | None = None


@dataclass(slots=True)
class CameraRecord:
    """Pinhole camera parameters in the declared camera convention."""

    intrinsic: list[list[float]]
    camera_to_world: list[list[float]]
    width: int
    height: int
    convention: CameraConvention = CameraConvention.OPENCV

    def validate(self) -> None:
        if len(self.intrinsic) != 3 or any(len(row) != 3 for row in self.intrinsic):
            raise ValueError("camera intrinsic must be 3x3")
        if len(self.camera_to_world) != 4 or any(
            len(row) != 4 for row in self.camera_to_world
        ):
            raise ValueError("camera_to_world must be 4x4")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")


@dataclass(slots=True)
class LightRecord:
    """Known or initialized capture illumination."""

    light_id: int | None = None
    position: list[float] | None = None
    direction: list[float] | None = None
    color: list[float] | None = None
    intensity: float | None = None
    env_sh: list[list[float]] | None = None
    envmap: str | None = None


@dataclass(slots=True)
class FrameRecord:
    """One view/light capture in a scene-level dataset."""

    frame_id: str
    split: str
    view_id: int
    assets: AssetPaths
    camera: CameraRecord
    light: LightRecord = field(default_factory=LightRecord)
    exposure: float = 1.0
    white_balance: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.camera.validate()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.exposure <= 0:
            raise ValueError("exposure must be positive")
        if len(self.white_balance) != 3 or min(self.white_balance) <= 0:
            raise ValueError("white_balance must contain three positive gains")


@dataclass(slots=True)
class SceneManifest:
    """Dataset-independent scene description used by loaders and baselines."""

    scene_id: str
    frames: list[FrameRecord]
    color_space: ColorSpace = ColorSpace.LINEAR
    gt_status: dict[str, GroundTruthStatus] = field(default_factory=dict)
    points: str | None = None
    mesh: str | None = None
    unit_scale_meters: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self, root: str | Path | None = None) -> None:
        if not self.scene_id or not self.frames:
            raise ValueError("manifest requires a scene_id and at least one frame")
        if self.unit_scale_meters <= 0:
            raise ValueError("unit_scale_meters must be positive")
        ids: set[str] = set()
        for frame in self.frames:
            frame.validate()
            if frame.frame_id in ids:
                raise ValueError(f"duplicate frame_id: {frame.frame_id}")
            ids.add(frame.frame_id)
        if root is not None:
            root = Path(root)
            for frame in self.frames:
                full = root / frame.assets.full
                if not full.exists():
                    raise FileNotFoundError(full)

    def to_json(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["color_space"] = self.color_space.value
        payload["gt_status"] = {
            key: value.value for key, value in self.gt_status.items()
        }
        for frame in payload["frames"]:
            frame["camera"]["convention"] = frame["camera"]["convention"].value
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SceneManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        frames = []
        for item in payload.pop("frames"):
            assets = AssetPaths(**item.pop("assets"))
            camera_data = item.pop("camera")
            camera_data["convention"] = CameraConvention(
                camera_data.get("convention", "opencv")
            )
            camera = CameraRecord(**camera_data)
            light = LightRecord(**item.pop("light", {}))
            frames.append(
                FrameRecord(assets=assets, camera=camera, light=light, **item)
            )
        payload["color_space"] = ColorSpace(payload.get("color_space", "linear"))
        payload["gt_status"] = {
            key: GroundTruthStatus(value)
            for key, value in payload.get("gt_status", {}).items()
        }
        manifest = cls(frames=frames, **payload)
        manifest.validate()
        return manifest
