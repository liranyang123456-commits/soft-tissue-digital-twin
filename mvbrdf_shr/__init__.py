"""Package init for mvbrdf_shr."""
from .models.pipeline import MVBRDFSHR
from .types import Batch, MaterialParams, RenderOutputs

__all__ = ["MVBRDFSHR", "Batch", "MaterialParams", "RenderOutputs"]
__version__ = "0.1.0"
