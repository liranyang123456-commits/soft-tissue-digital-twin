"""External inverse-rendering baseline integration framework."""

from .registry import get_baseline, list_baselines, register_baseline
from .runner import (
    BaselineEnvironmentError,
    build_command,
    check_environment,
    run_baseline,
)
from .schema import (
    OUTPUT_DIRECTORIES,
    BaselineResources,
    BaselineSpec,
    CanonicalSceneInput,
    RunResult,
    StandardOutputLayout,
)

__all__ = [
    "OUTPUT_DIRECTORIES",
    "BaselineEnvironmentError",
    "BaselineResources",
    "BaselineSpec",
    "CanonicalSceneInput",
    "RunResult",
    "StandardOutputLayout",
    "build_command",
    "check_environment",
    "get_baseline",
    "list_baselines",
    "register_baseline",
    "run_baseline",
]
