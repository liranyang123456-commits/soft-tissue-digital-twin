"""COLMAP and Blender multiview scene loaders."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera import PerspectiveCamera
from .lighting import (
    latlong_dominant_light,
    latlong_residual_to_sh,
)


@dataclass(slots=True)
class MultiViewFrame:
    image: torch.Tensor
    camera: PerspectiveCamera
    image_path: Path
    diffuse_gt: torch.Tensor | None = None
    specular_gt: torch.Tensor | None = None
    albedo_gt: torch.Tensor | None = None
    normal_gt: torch.Tensor | None = None
    depth_gt: torch.Tensor | None = None
    # Soft depth from visual hull / carving (Ours-NoGT); never GT mesh depth.
    depth_prior: torch.Tensor | None = None
    # Train-image photometric-stereo observation; never benchmark normal GT.
    normal_prior_image: torch.Tensor | None = None
    normal_prior_confidence: torch.Tensor | None = None
    mask_gt: torch.Tensor | None = None
    roughness_gt: torch.Tensor | None = None
    metallic_gt: torch.Tensor | None = None
    envmap_gt: torch.Tensor | None = None
    light_position: torch.Tensor | None = None
    light_direction: torch.Tensor | None = None
    light_color: torch.Tensor | None = None
    point_light_weight: float | None = None
    light_intensity: float | None = None
    env_sh: torch.Tensor | None = None
    view_id: int | None = None
    light_id: int | None = None
    exposure: float = 1.0
    white_balance: torch.Tensor | None = None
    color_space: str = "srgb"
    metadata: dict[str, object] | None = None


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def _read_image_array(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image).copy()
    except (OSError, ValueError):
        try:
            import imageio.v3 as iio
        except ImportError as error:
            raise RuntimeError(
                f"{path.suffix} image support requires optional dependency imageio"
            ) from error
        return np.asarray(iio.imread(path))


def _image_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".npy":
        shape = np.load(path, mmap_mode="r").shape
        return int(shape[1]), int(shape[0])
    array = _read_image_array(path)
    return int(array.shape[1]), int(array.shape[0])


def _load_rgb(
    path: Path,
    size: tuple[int, int] | None = None,
    linearize: bool = False,
) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[0] in {1, 3} and array.ndim == 3:
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        if size is not None and tensor.shape[-2:] != (size[1], size[0]):
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(size[1], size[0]), mode="bilinear",
                align_corners=False,
            )[0]
        return tensor.float()
    array = _read_image_array(path)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    dtype = array.dtype
    array = array[..., :3].astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        array /= np.iinfo(dtype).max
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if size is not None and tensor.shape[-2:] != (size[1], size[0]):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(size[1], size[0]),
            mode="bilinear",
            align_corners=False,
        )[0]
    return _srgb_to_linear(tensor) if linearize else tensor


def _load_scalar(
    path: Path,
    size: tuple[int, int] | None = None,
    nearest: bool = False,
) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        tensor = torch.from_numpy(array).unsqueeze(0)
        if size is not None and tensor.shape[-2:] != (size[1], size[0]):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(size[1], size[0]),
                mode="nearest" if nearest else "bilinear",
                align_corners=None if nearest else False,
            )[0]
        return tensor.float()
    array = _read_image_array(path)
    if array.ndim == 3:
        array = array[..., 0]
    dtype = array.dtype
    array = array.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        array /= np.iinfo(dtype).max
    tensor = torch.from_numpy(array).unsqueeze(0)
    if size is not None and tensor.shape[-2:] != (size[1], size[0]):
        if nearest:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(size[1], size[0]), mode="nearest"
            )[0]
        else:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(size[1], size[0]),
                mode="bilinear",
                align_corners=False,
            )[0]
    return tensor


def _optional_path(root: Path, item: dict, key: str) -> Path | None:
    name = item.get(key)
    return _resolve_image_path(root, str(name)) if name else None


def _resolve_image_path(root: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".png")
    options = [root / candidate, root / "images" / candidate]
    for option in options:
        if option.exists():
            return option
    raise FileNotFoundError(f"cannot resolve image {name} under {root}")


def _read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        color = file.readline().decode("ascii").strip() == "PF"
        width, height = map(int, file.readline().decode("ascii").split())
        scale = float(file.readline().decode("ascii"))
        dtype = "<f4" if scale < 0 else ">f4"
        channels = 3 if color else 1
        data = np.fromfile(file, dtype=dtype, count=width * height * channels)
    shape = (height, width, channels) if color else (height, width)
    return np.flipud(data.reshape(shape)).copy()


def _read_mvsnet_camera(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    extrinsic_index = lines.index("extrinsic")
    intrinsic_index = lines.index("intrinsic")
    extrinsic = torch.tensor(
        [[float(value) for value in lines[extrinsic_index + row + 1].split()] for row in range(4)]
    )
    intrinsic = torch.tensor(
        [[float(value) for value in lines[intrinsic_index + row + 1].split()] for row in range(3)]
    )
    return intrinsic.float(), torch.linalg.inv(extrinsic.float())


def _decompose_projection(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    projection = np.loadtxt(path).reshape(3, 4)
    # RQ via NumPy QR avoids adding a second OpenMP runtime on Windows.
    q, r = np.linalg.qr(np.flipud(projection[:, :3]).T)
    intrinsic = np.fliplr(np.flipud(r.T))
    rotation = np.flipud(q.T)
    signs = np.sign(np.diag(intrinsic))
    signs[signs == 0] = 1
    correction = np.diag(signs)
    intrinsic, rotation = intrinsic @ correction, correction @ rotation
    if np.linalg.det(rotation) < 0:
        intrinsic[:, 0] *= -1
        rotation[0] *= -1
    intrinsic /= intrinsic[2, 2]
    translation = np.linalg.solve(intrinsic, projection[:, 3])
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return (
        torch.from_numpy(intrinsic).float(),
        torch.linalg.inv(torch.from_numpy(world_to_camera)),
    )


def _read_binary_ply(
    path: Path, max_points: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    with path.open("rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise NotImplementedError("only binary little-endian DTU PLY is supported")
        vertex_line = next(line for line in header_lines if line.startswith("element vertex"))
        vertex_count = int(vertex_line.split()[-1])
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.fromfile(file, dtype=dtype, count=vertex_count)
    indices = np.linspace(
        0, vertex_count - 1, min(max_points, vertex_count), dtype=np.int64
    )
    selected = vertices[indices]
    points = np.stack([selected["x"], selected["y"], selected["z"]], axis=-1)
    colors = np.stack(
        [selected["red"], selected["green"], selected["blue"]], axis=-1
    ).astype(np.float32) / 255
    return torch.from_numpy(points).float(), torch.from_numpy(colors).float()


class MultiViewScene:
    def __init__(
        self,
        frames: list[MultiViewFrame],
        points: torch.Tensor | None = None,
        point_colors: torch.Tensor | None = None,
        point_normals: torch.Tensor | None = None,
        point_scales: torch.Tensor | None = None,
        metadata: dict[str, object] | None = None,
    ):
        if not frames:
            raise ValueError("multiview scene has no frames")
        self.frames = frames
        self.points = points
        self.point_colors = point_colors
        self.point_normals = point_normals
        self.point_scales = point_scales
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> MultiViewFrame:
        return self.frames[index]

    def group_by_view(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for index, frame in enumerate(self.frames):
            view_id = frame.view_id if frame.view_id is not None else index
            groups.setdefault(view_id, []).append(index)
        return groups

    def group_by_light(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for index, frame in enumerate(self.frames):
            if frame.light_id is not None:
                groups.setdefault(frame.light_id, []).append(index)
        return groups

    @classmethod
    def from_canonical(
        cls,
        dataset,
        share_light_ids: bool = True,
        scale: float = 1.0,
        world_scale: float = 1.0,
        use_mesh_points: bool = False,
        max_mesh_points: int = 50000,
        mesh_sampling: str = "surface",
        mesh_normal_scale_ratio: float = 0.15,
    ) -> "MultiViewScene":
        """Bridge an external canonical adapter into the train-time scene API."""
        if scale <= 0 or world_scale <= 0:
            raise ValueError("scale and world_scale must be positive")
        if mesh_sampling not in {"surface", "vertices"}:
            raise ValueError("mesh_sampling must be surface or vertices")
        view_values = list(dict.fromkeys(frame.view_id for frame in dataset.frames))
        source_light_values = list(
            dict.fromkeys(
                frame.light_id
                for frame in dataset.frames
                if frame.light_id is not None
            )
        )
        view_lookup = {value: index for index, value in enumerate(view_values)}
        light_lookup = {
            value: index for index, value in enumerate(source_light_values)
        }
        frames: list[MultiViewFrame] = []
        for index, source in enumerate(dataset.frames):
            width = max(1, round(source.camera.width * scale))
            height = max(1, round(source.camera.height * scale))

            def resize(
                value: torch.Tensor | None,
                mode: str = "bilinear",
            ) -> torch.Tensor | None:
                if value is None or value.shape[-2:] == (height, width):
                    return value
                kwargs = {} if mode == "nearest" else {"align_corners": False}
                return F.interpolate(
                    value.unsqueeze(0),
                    size=(height, width),
                    mode=mode,
                    **kwargs,
                )[0]

            intensity = source.light_intensity
            light_direction = source.light_direction
            light_color = None
            light_intensity = None
            if source.envmap is not None:
                light_direction, light_color, light_intensity = (
                    latlong_dominant_light(source.envmap.float())
                )
            if intensity is not None:
                intensity = torch.as_tensor(intensity, dtype=torch.float32).reshape(-1)
                if intensity.numel() == 3:
                    light_intensity = float(intensity.max())
                    light_color = intensity / intensity.max().clamp_min(1e-8)
                else:
                    light_intensity = float(intensity.mean())
            frames.append(
                MultiViewFrame(
                    image=resize(source.image.float()),
                    camera=PerspectiveCamera(
                        source.camera.K.float()
                        * torch.tensor(
                            [[scale, scale, scale], [scale, scale, scale], [1, 1, 1]],
                            dtype=torch.float32,
                        ),
                        source.camera.c2w.float().clone(),
                        width,
                        height,
                        index,
                        view_id=view_lookup[source.view_id],
                    ),
                    image_path=source.image_path,
                    albedo_gt=resize(source.albedo),
                    normal_gt=(
                        F.normalize(resize(source.normal), dim=0, eps=1e-6)
                        if source.normal is not None
                        else None
                    ),
                    depth_gt=resize(source.depth),
                    mask_gt=resize(source.mask, "nearest"),
                    envmap_gt=source.envmap,
                    env_sh=(
                        # The brightest 0.5% is represented separately by the
                        # dominant GGX light; remove it from SH to conserve
                        # environment energy.
                        latlong_residual_to_sh(source.envmap.float())
                        if source.envmap is not None
                        else None
                    ),
                    light_direction=light_direction,
                    light_color=light_color,
                    light_intensity=light_intensity,
                    view_id=view_lookup[source.view_id],
                    light_id=(
                        (
                            light_lookup[source.light_id]
                            if share_light_ids
                            else index
                        )
                        if source.light_id is not None
                        else None
                    ),
                    color_space="linear",
                    metadata={
                        **source.metadata,
                        "source_view_id": source.view_id,
                        "source_light_id": source.light_id,
                        "albedo_is_pseudo": source.albedo_is_pseudo,
                    },
                )
            )
            frames[-1].camera.c2w[:3, 3].mul_(world_scale)
        points = None
        point_colors = None
        point_normals = None
        point_scales = None
        if use_mesh_points and dataset.mesh_path is not None:
            try:
                import trimesh
            except ImportError as error:
                raise RuntimeError(
                    "mesh initialization requires optional dependency trimesh"
                ) from error
            mesh = trimesh.load(dataset.mesh_path, force="mesh", process=False)
            if mesh_sampling == "surface" and len(mesh.faces) > 0:
                vertices, face_ids = trimesh.sample.sample_surface(
                    mesh, max_mesh_points, seed=0
                )
                vertices = np.asarray(vertices, dtype=np.float32)
                triangles = np.asarray(mesh.triangles, dtype=np.float32)[face_ids]
                barycentric = trimesh.triangles.points_to_barycentric(
                    triangles, vertices
                ).astype(np.float32)
                face_vertices = np.asarray(mesh.faces, dtype=np.int64)[face_ids]
                vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                normals = (
                    vertex_normals[face_vertices] * barycentric[..., None]
                ).sum(axis=1)
                point_normals = F.normalize(
                    torch.from_numpy(normals).float(), dim=-1, eps=1e-6
                )
                area = max(float(mesh.area), 1e-12) * world_scale**2
                tangent_scale = max((area / max_mesh_points) ** 0.5 * 1.5, 1e-4)
                point_scales = torch.tensor(
                    [tangent_scale, tangent_scale, tangent_scale * mesh_normal_scale_ratio]
                ).expand(max_mesh_points, -1).clone()
                colors = getattr(mesh.visual, "vertex_colors", None)
                if colors is not None and len(colors) >= len(mesh.vertices):
                    sampled_colors = (
                        np.asarray(colors, dtype=np.float32)[face_vertices, :3]
                        * barycentric[..., None]
                    ).sum(axis=1)
                    point_colors = torch.from_numpy(
                        sampled_colors.astype(np.float32) / 255.0
                    )
            else:
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                if len(vertices) > max_mesh_points:
                    ids = np.linspace(
                        0, len(vertices) - 1, max_mesh_points, dtype=np.int64
                    )
                    vertices = vertices[ids]
                else:
                    ids = np.arange(len(vertices))
                normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                if len(normals) >= len(ids):
                    point_normals = F.normalize(
                        torch.from_numpy(normals[ids]).float(), dim=-1, eps=1e-6
                    )
                colors = getattr(mesh.visual, "vertex_colors", None)
                if colors is not None and len(colors) >= len(ids):
                    point_colors = torch.from_numpy(
                        np.asarray(colors)[ids, :3].astype(np.float32) / 255.0
                    )
            points = torch.from_numpy(vertices).float() * world_scale
        metadata = dict(dataset.metadata)
        metadata.update(
            adapter=dataset.name,
            root=str(dataset.root),
            mesh_path=str(dataset.mesh_path) if dataset.mesh_path else None,
        )
        return cls(
            frames,
            points=points,
            point_colors=point_colors,
            point_normals=point_normals,
            point_scales=point_scales,
            metadata=metadata,
        )

    @classmethod
    def from_manifest(
        cls, root: str | Path, manifest_name: str = "scene.json"
    ) -> "MultiViewScene":
        """Load the serializable canonical scene schema."""
        from .coords import CameraConvention, convert_camera_pose
        from .schema import ColorSpace, SceneManifest

        root = Path(root)
        manifest = SceneManifest.from_json(root / manifest_name)
        frames: list[MultiViewFrame] = []

        def asset(name: str | None) -> Path | None:
            return root / name if name else None

        for index, record in enumerate(manifest.frames):
            camera_record = record.camera
            c2w = torch.tensor(camera_record.camera_to_world, dtype=torch.float32)
            c2w = convert_camera_pose(
                c2w, camera_record.convention, CameraConvention.OPENCV
            )
            camera = PerspectiveCamera(
                torch.tensor(camera_record.intrinsic, dtype=torch.float32),
                c2w,
                camera_record.width,
                camera_record.height,
                index,
            )
            size = (camera.width, camera.height)
            source_linear = manifest.color_space != ColorSpace.SRGB
            full_path = root / record.assets.full
            normal_path = asset(record.assets.normal)
            normal = _load_rgb(normal_path, size) if normal_path else None
            if normal is not None and normal.min() >= 0:
                normal = normal * 2 - 1
            if normal is not None:
                normal = F.normalize(normal, dim=0, eps=1e-6)
            light = record.light
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(full_path, size, linearize=not source_linear),
                    camera=camera,
                    image_path=full_path,
                    diffuse_gt=(
                        _load_rgb(asset(record.assets.diffuse), size, not source_linear)
                        if record.assets.diffuse
                        else None
                    ),
                    specular_gt=(
                        _load_rgb(asset(record.assets.specular), size, not source_linear)
                        if record.assets.specular
                        else None
                    ),
                    albedo_gt=(
                        _load_rgb(asset(record.assets.albedo), size, not source_linear)
                        if record.assets.albedo
                        else None
                    ),
                    normal_gt=normal,
                    depth_gt=(
                        _load_scalar(asset(record.assets.depth), size)
                        if record.assets.depth
                        else None
                    ),
                    mask_gt=(
                        _load_scalar(asset(record.assets.mask), size, nearest=True)
                        if record.assets.mask
                        else None
                    ),
                    roughness_gt=(
                        _load_scalar(asset(record.assets.roughness), size)
                        if record.assets.roughness
                        else None
                    ),
                    metallic_gt=(
                        _load_scalar(asset(record.assets.metallic), size)
                        if record.assets.metallic
                        else None
                    ),
                    envmap_gt=(
                        _load_rgb(asset(record.assets.envmap), linearize=not source_linear)
                        if record.assets.envmap
                        else None
                    ),
                    light_position=(
                        torch.tensor(light.position, dtype=torch.float32)
                        if light.position is not None
                        else None
                    ),
                    light_direction=(
                        torch.tensor(light.direction, dtype=torch.float32)
                        if light.direction is not None
                        else None
                    ),
                    light_color=(
                        torch.tensor(light.color, dtype=torch.float32)
                        if light.color is not None
                        else None
                    ),
                    light_intensity=light.intensity,
                    env_sh=(
                        torch.tensor(light.env_sh, dtype=torch.float32)
                        if light.env_sh is not None
                        else None
                    ),
                    view_id=record.view_id,
                    light_id=light.light_id,
                    exposure=record.exposure,
                    white_balance=torch.tensor(
                        record.white_balance, dtype=torch.float32
                    ),
                    color_space="linear" if source_linear else "srgb",
                    metadata={"split": record.split, **record.metadata},
                )
            )
        points = (
            torch.from_numpy(np.load(root / manifest.points)).float()
            if manifest.points
            else None
        )
        return cls(frames, points=points, metadata={"manifest": manifest})

    @classmethod
    def from_blender(
        cls,
        root: str | Path,
        transforms: str = "transforms.json",
        scale: float = 1.0,
    ) -> "MultiViewScene":
        root = Path(root)
        meta = json.loads((root / transforms).read_text(encoding="utf-8"))
        frames: list[MultiViewFrame] = []
        for index, item in enumerate(meta["frames"]):
            path = _resolve_image_path(root, item["file_path"])
            source_color_space = str(
                item.get("color_space", meta.get("color_space", "srgb"))
            ).lower()
            linearize = bool(
                item.get("linearize_srgb", meta.get("linearize_srgb", False))
            ) and source_color_space == "srgb"
            width, height = _image_size(path)
            width_s, height_s = round(width * scale), round(height * scale)
            if "fl_x" in item or "fl_x" in meta:
                fx = float(item.get("fl_x", meta["fl_x"])) * scale
                fy = float(item.get("fl_y", meta.get("fl_y", fx / scale))) * scale
            else:
                angle_x = float(meta["camera_angle_x"])
                fx = 0.5 * width_s / np.tan(0.5 * angle_x)
                fy = fx
            cx = float(item.get("cx", meta.get("cx", width / 2))) * scale
            cy = float(item.get("cy", meta.get("cy", height / 2))) * scale
            K = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
            )
            c2w_gl = torch.tensor(item["transform_matrix"], dtype=torch.float32)
            # Blender/OpenGL camera (-z forward, +y up) -> OpenCV.
            convert = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
            c2w = c2w_gl @ convert
            camera = PerspectiveCamera(K, c2w, width_s, height_s, index)
            diffuse_path = item.get("diffuse_path")
            diffuse = (
                _load_rgb(
                    _resolve_image_path(root, diffuse_path),
                    (width_s, height_s),
                    linearize=linearize,
                )
                if diffuse_path
                else None
            )
            specular_path = _optional_path(root, item, "specular_path")
            albedo_path = _optional_path(root, item, "albedo_path")
            normal_path = _optional_path(root, item, "normal_path")
            depth_path = _optional_path(root, item, "depth_path")
            mask_path = _optional_path(root, item, "mask_path")
            roughness_path = _optional_path(root, item, "roughness_path")
            metallic_path = _optional_path(root, item, "metallic_path")
            envmap_path = _optional_path(root, item, "envmap_path")
            normal_gt = (
                _load_rgb(normal_path, (width_s, height_s)) if normal_path else None
            )
            if normal_gt is not None and str(
                item.get("normal_encoding", meta.get("normal_encoding", "rgb01"))
            ).lower() == "rgb01":
                normal_gt = normal_gt * 2.0 - 1.0
            if normal_gt is not None:
                normal_gt = F.normalize(normal_gt, dim=0, eps=1e-6)
            env_sh = item.get("env_sh")
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width_s, height_s), linearize=linearize),
                    camera=camera,
                    image_path=path,
                    diffuse_gt=diffuse,
                    specular_gt=(
                        _load_rgb(
                            specular_path,
                            (width_s, height_s),
                            linearize=linearize,
                        )
                        if specular_path
                        else None
                    ),
                    albedo_gt=(
                        _load_rgb(albedo_path, (width_s, height_s), linearize=linearize)
                        if albedo_path
                        else None
                    ),
                    normal_gt=normal_gt,
                    depth_gt=(
                        _load_scalar(depth_path, (width_s, height_s))
                        if depth_path
                        else None
                    ),
                    mask_gt=(
                        _load_scalar(mask_path, (width_s, height_s), nearest=True)
                        if mask_path
                        else None
                    ),
                    roughness_gt=(
                        _load_scalar(roughness_path, (width_s, height_s))
                        if roughness_path
                        else None
                    ),
                    metallic_gt=(
                        _load_scalar(metallic_path, (width_s, height_s))
                        if metallic_path
                        else None
                    ),
                    envmap_gt=(
                        _load_rgb(envmap_path, linearize=linearize)
                        if envmap_path
                        else None
                    ),
                    light_position=(
                        torch.tensor(item["light_position"], dtype=torch.float32)
                        if "light_position" in item
                        else None
                    ),
                    light_direction=(
                        torch.tensor(item["light_direction"], dtype=torch.float32)
                        if "light_direction" in item
                        else None
                    ),
                    light_color=(
                        torch.tensor(item["light_color"], dtype=torch.float32)
                        if "light_color" in item
                        else None
                    ),
                    point_light_weight=(
                        float(item["point_light_weight"])
                        if "point_light_weight" in item
                        else None
                    ),
                    light_intensity=(
                        float(item["light_intensity"])
                        if "light_intensity" in item
                        else None
                    ),
                    env_sh=(
                        torch.tensor(env_sh, dtype=torch.float32).reshape(9, 3)
                        if env_sh is not None
                        else None
                    ),
                    view_id=int(item.get("view_id", index)),
                    light_id=(
                        int(item["light_id"]) if item.get("light_id") is not None else None
                    ),
                    exposure=float(item.get("exposure", 1.0)),
                    white_balance=(
                        torch.tensor(item["white_balance"], dtype=torch.float32)
                        if item.get("white_balance") is not None
                        else None
                    ),
                    color_space=(
                        "linear" if linearize or source_color_space == "linear"
                        else source_color_space
                    ),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "transform_matrix",
                            "file_path",
                            "diffuse_path",
                            "specular_path",
                            "albedo_path",
                            "normal_path",
                            "depth_path",
                            "mask_path",
                            "roughness_path",
                            "metallic_path",
                            "envmap_path",
                        }
                    },
                )
            )
        points_path = root / meta.get("points_path", "points.npy")
        colors_path = root / meta.get("point_colors_path", "point_colors.npy")
        points = (
            torch.from_numpy(np.load(points_path)).float()
            if points_path.exists()
            else None
        )
        colors = (
            torch.from_numpy(np.load(colors_path)).float()
            if colors_path.exists()
            else None
        )
        if colors is not None and colors.max() > 1:
            colors = colors / 255.0
        return cls(frames, points, colors, metadata=meta)

    @classmethod
    def from_mvsnet(
        cls,
        root: str | Path,
        images_dir: str = "blended_images",
        cams_dir: str = "cams",
        depth_dir: str = "rendered_depth_maps",
        scale: float = 0.25,
        max_points: int = 50000,
        max_point_views: int = 16,
    ) -> "MultiViewScene":
        """Load an MVSNet/BlendedMVS scene and unproject its rendered depths."""
        root = Path(root)
        image_root, camera_root = root / images_dir, root / cams_dir
        image_paths = sorted(
            path
            for path in image_root.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and "_masked" not in path.stem
        )
        frames: list[MultiViewFrame] = []
        original_sizes: dict[int, tuple[int, int]] = {}
        for path in image_paths:
            camera_path = camera_root / f"{path.stem}_cam.txt"
            if not camera_path.exists():
                continue
            K, c2w = _read_mvsnet_camera(camera_path)
            with Image.open(path) as image:
                width, height = image.size
            width_s, height_s = round(width * scale), round(height * scale)
            K[0] *= width_s / width
            K[1] *= height_s / height
            frame_id = len(frames)
            original_sizes[frame_id] = (width, height)
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width_s, height_s)),
                    camera=PerspectiveCamera(K, c2w, width_s, height_s, frame_id),
                    image_path=path,
                )
            )
        if not frames:
            raise FileNotFoundError(f"no MVSNet image/camera pairs under {root}")

        points, colors = [], []
        depth_root = root / depth_dir
        view_ids = torch.linspace(
            0, len(frames) - 1, min(max_point_views, len(frames))
        ).long().unique().tolist()
        per_view = max(1, max_points // max(len(view_ids), 1))
        for frame_id in view_ids:
            frame = frames[frame_id]
            depth_path = depth_root / f"{frame.image_path.stem}.pfm"
            if not depth_path.exists():
                continue
            depth = torch.from_numpy(_read_pfm(depth_path)).float()
            if depth.ndim == 3:
                depth = depth[..., 0]
            valid = torch.isfinite(depth) & (depth > 1e-5)
            valid_ids = valid.flatten().nonzero().flatten()
            if not len(valid_ids):
                continue
            selected = valid_ids[
                torch.linspace(0, len(valid_ids) - 1, min(per_view, len(valid_ids))).long()
            ]
            height_d, width_d = depth.shape
            pixel_y = selected // width_d
            pixel_x = selected % width_d
            z = depth.flatten()[selected]
            K = frame.camera.K.clone()
            original_width, original_height = original_sizes[frame_id]
            K[0] *= width_d / original_width
            K[1] *= height_d / original_height
            x = (pixel_x.float() - K[0, 2]) * z / K[0, 0]
            y = (pixel_y.float() - K[1, 2]) * z / K[1, 1]
            camera_points = torch.stack([x, y, z], dim=-1)
            rotation, translation = (
                frame.camera.c2w[:3, :3],
                frame.camera.c2w[:3, 3],
            )
            points.append(camera_points @ rotation.T + translation)
            grid = torch.stack(
                [
                    2 * pixel_x.float() / max(width_d - 1, 1) - 1,
                    2 * pixel_y.float() / max(height_d - 1, 1) - 1,
                ],
                dim=-1,
            ).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                frame.image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, :, :, 0].T
            colors.append(sampled)
        return cls(
            frames,
            torch.cat(points) if points else None,
            torch.cat(colors) if colors else None,
        )

    @classmethod
    def from_dtu(
        cls,
        root: str | Path,
        scan: int = 1,
        light: int = 3,
        scale: float = 0.125,
        max_points: int = 50000,
    ) -> "MultiViewScene":
        """Load a DTU SampleSet scan with official projection matrices."""
        root = Path(root)
        rectified = root / "Rectified" / f"scan{scan}"
        calibration = root / "Calibration" / "cal18"
        paths = sorted(rectified.glob(f"rect_*_{light}_r5000.png"))
        frames = []
        for path in paths:
            camera_number = int(path.stem.split("_")[1])
            projection_path = calibration / f"pos_{camera_number:03d}.txt"
            if not projection_path.exists():
                continue
            K, c2w = _decompose_projection(projection_path)
            with Image.open(path) as image:
                width, height = image.size
            width_s, height_s = round(width * scale), round(height * scale)
            K[0] *= width_s / width
            K[1] *= height_s / height
            frame_id = len(frames)
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width_s, height_s)),
                    camera=PerspectiveCamera(K, c2w, width_s, height_s, frame_id),
                    image_path=path,
                )
            )
        if not frames:
            raise FileNotFoundError(f"no DTU scan{scan} light{light} views under {root}")
        ply_path = root / "Points" / "stl" / f"stl{scan:03d}_total.ply"
        points, colors = (
            _read_binary_ply(ply_path, max_points)
            if ply_path.exists()
            else (None, None)
        )
        return cls(frames, points, colors)

    def fuse_projected_texture(
        self,
        points: torch.Tensor,
        frame_indices: list[int] | None = None,
        source_images: dict[int, torch.Tensor] | None = None,
        confidence_maps: dict[int, torch.Tensor] | None = None,
        occlusion_tolerance: float | None = None,
    ) -> torch.Tensor:
        """Project every 3D point into selected views and average RGB.

        The resulting persistent per-point texture is rendered into any target
        view and is the cross-view projected-texture condition for refinement.
        Restricting ``frame_indices`` prevents holdout inputs from leaking into
        train-time scene initialization.
        """
        points = points.cpu()
        color_sum = torch.zeros(len(points), 3)
        weight_sum = torch.zeros(len(points), 1)
        indices = (
            frame_indices if frame_indices is not None else list(range(len(self.frames)))
        )
        if not indices:
            raise ValueError("frame_indices must contain at least one view")
        for index in indices:
            frame = self.frames[index]
            uv, depth, valid = frame.camera.project(points)
            if occlusion_tolerance is not None:
                pixel_x = uv[:, 0].round().long().clamp(0, frame.camera.width - 1)
                pixel_y = uv[:, 1].round().long().clamp(0, frame.camera.height - 1)
                linear_index = pixel_y * frame.camera.width + pixel_x
                depth_buffer = torch.full(
                    (frame.camera.width * frame.camera.height,),
                    float("inf"),
                    dtype=depth.dtype,
                )
                depth_buffer.scatter_reduce_(
                    0,
                    linear_index[valid],
                    depth[valid],
                    reduce="amin",
                    include_self=True,
                )
                visible = depth <= (
                    depth_buffer[linear_index]
                    + occlusion_tolerance
                    * depth_buffer[linear_index].clamp(min=1)
                )
                valid = valid & visible
            gx = 2 * uv[:, 0] / max(frame.camera.width - 1, 1) - 1
            gy = 2 * uv[:, 1] / max(frame.camera.height - 1, 1) - 1
            grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
            source = (
                source_images[index]
                if source_images is not None and index in source_images
                else frame.image
            )
            sampled = F.grid_sample(
                source.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, :, :, 0].T
            weight = valid.float().unsqueeze(-1)
            if frame.mask_gt is not None:
                sampled_mask = F.grid_sample(
                    frame.mask_gt.unsqueeze(0),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0, :, :, 0].T
                weight = weight * sampled_mask.clamp(0, 1)
            if confidence_maps is not None and index in confidence_maps:
                sampled_confidence = F.grid_sample(
                    confidence_maps[index].unsqueeze(0),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0, :, :, 0].T
                weight = weight * sampled_confidence.clamp(0, 1)
            color_sum += sampled * weight
            weight_sum += weight
        return color_sum / weight_sum.clamp(min=1)

    @classmethod
    def from_colmap(
        cls,
        root: str | Path,
        sparse_dir: str = "sparse/0",
        images_dir: str = "images",
        scale: float = 1.0,
    ) -> "MultiViewScene":
        root = Path(root)
        sparse = root / sparse_dir
        cameras = _read_colmap_cameras(sparse / "cameras.txt", scale)
        image_records = _read_colmap_images(sparse / "images.txt")
        frames: list[MultiViewFrame] = []
        for frame_id, record in enumerate(image_records):
            camera_id, qvec, tvec, name = record
            K, width, height = cameras[camera_id]
            rotation = _qvec_to_rotation(qvec)
            w2c = torch.eye(4)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = tvec
            c2w = torch.linalg.inv(w2c)
            path = root / images_dir / name
            camera = PerspectiveCamera(K, c2w, width, height, frame_id)
            frames.append(
                MultiViewFrame(
                    image=_load_rgb(path, (width, height)),
                    camera=camera,
                    image_path=path,
                )
            )
        points, colors = _read_colmap_points(sparse / "points3D.txt")
        return cls(frames, points, colors)


def _read_colmap_cameras(
    path: Path, scale: float
) -> dict[int, tuple[torch.Tensor, int, int]]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        camera_id, model = int(values[0]), values[1]
        width, height = round(int(values[2]) * scale), round(int(values[3]) * scale)
        params = [float(v) for v in values[4:]]
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = params[0] * scale
            cx, cy = params[1] * scale, params[2] * scale
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
            fx, fy = params[0] * scale, params[1] * scale
            cx, cy = params[2] * scale, params[3] * scale
        else:
            raise NotImplementedError(f"COLMAP camera model {model}")
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        cameras[camera_id] = (K, width, height)
    return cameras


def _read_colmap_images(
    path: Path,
) -> list[tuple[int, torch.Tensor, torch.Tensor, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        values = line.split()
        if len(values) >= 10:
            qvec = torch.tensor([float(v) for v in values[1:5]])
            tvec = torch.tensor([float(v) for v in values[5:8]])
            records.append((int(values[8]), qvec, tvec, values[9]))
            index += 2  # skip POINTS2D line
        else:
            index += 1
    return records


def _qvec_to_rotation(qvec: torch.Tensor) -> torch.Tensor:
    qvec = qvec / qvec.norm()
    w, x, y, z = qvec
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y]
            ),
            torch.stack(
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x]
            ),
            torch.stack(
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
            ),
        ],
    )


def _read_colmap_points(path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not path.exists():
        return None, None
    points, colors = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        points.append([float(v) for v in values[1:4]])
        colors.append([int(v) / 255.0 for v in values[4:7]])
    if not points:
        return None, None
    return torch.tensor(points, dtype=torch.float32), torch.tensor(colors)

