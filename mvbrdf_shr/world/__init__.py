"""True world-space multi-view 3D Gaussian BRDF subsystem."""

from .camera import PerspectiveCamera
from .coords import CameraConvention, convert_camera_pose
from .data import MultiViewFrame, MultiViewScene
from .fusion import HybridMVBRDFSHR
from .lighting import NeuralIncidentLightField, PerFrameLighting, WorldLight
from .pipeline import WorldGaussianBRDFPipeline
from .renderer import GaussianBRDFRenderer, WorldRenderOutputs
from .scene import GaussianBRDFField, GaussianMaterials
from .schema import FrameRecord, SceneManifest
# Digital-twin export (Phase A+B). These import optional deps lazily.
from .material_db import MATERIAL_DATABASE, MaterialPrior, classify_from_pbr, get_prior
from .segment import InstanceSegmentation, SegmentationResult, segment_instances
from .usd_export import InstancePhysics, to_usd, to_usd_with_physics

__all__ = [
    "PerspectiveCamera",
    "CameraConvention",
    "convert_camera_pose",
    "MultiViewFrame",
    "MultiViewScene",
    "HybridMVBRDFSHR",
    "NeuralIncidentLightField",
    "PerFrameLighting",
    "WorldLight",
    "WorldGaussianBRDFPipeline",
    "GaussianBRDFRenderer",
    "WorldRenderOutputs",
    "GaussianBRDFField",
    "GaussianMaterials",
    "FrameRecord",
    "SceneManifest",
    # Digital-twin exports
    "MATERIAL_DATABASE",
    "MaterialPrior",
    "get_prior",
    "classify_from_pbr",
    "InstanceSegmentation",
    "SegmentationResult",
    "segment_instances",
    "InstancePhysics",
    "to_usd",
    "to_usd_with_physics",
]

