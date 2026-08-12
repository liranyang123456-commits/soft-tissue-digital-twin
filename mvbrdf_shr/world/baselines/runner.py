"""Safe subprocess runner for third-party inverse-rendering baselines."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from .registry import get_baseline
from .schema import (
    BaselineResources,
    BaselineSpec,
    CanonicalSceneInput,
    RunResult,
    StandardOutputLayout,
)


class BaselineEnvironmentError(RuntimeError):
    """Raised when an explicitly external baseline dependency is unavailable."""


def _configured_path(
    explicit: str | Path | None,
    env_name: str,
    description: str,
    method: str,
    environment: Mapping[str, str],
) -> Path:
    raw = explicit if explicit is not None else environment.get(env_name)
    if not raw:
        raise BaselineEnvironmentError(
            f"{method}: missing {description}; pass it explicitly or set {env_name}. "
            "This framework does not download third-party code or checkpoints."
        )
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise BaselineEnvironmentError(
            f"{method}: configured {description} does not exist: {path}"
        )
    return path


def check_environment(
    spec: BaselineSpec,
    *,
    repository: str | Path | None = None,
    checkpoint: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> BaselineResources:
    """Resolve and verify external resources without importing third-party code."""
    env = os.environ if environment is None else environment
    repo = _configured_path(
        repository, spec.repository_env, "repository", spec.name, env
    )
    ckpt = _configured_path(
        checkpoint, spec.checkpoint_env, "checkpoint", spec.name, env
    )
    entrypoint = next(
        (repo / relative for relative in spec.entrypoint_candidates if (repo / relative).is_file()),
        None,
    )
    if entrypoint is None:
        expected = ", ".join(spec.entrypoint_candidates)
        raise BaselineEnvironmentError(
            f"{spec.name}: no supported entrypoint found under {repo}; expected one of: "
            f"{expected}. Check the repository revision or register a custom spec."
        )
    missing = [name for name in spec.required_executables if shutil.which(name) is None]
    if missing:
        raise BaselineEnvironmentError(
            f"{spec.name}: required executables not found on PATH: {', '.join(missing)}"
        )
    return BaselineResources(repo, ckpt, entrypoint)


def build_command(
    spec: BaselineSpec,
    scene: CanonicalSceneInput,
    outputs: StandardOutputLayout,
    resources: BaselineResources,
) -> tuple[str, ...]:
    """Expand a registered argv template without shell interpolation."""
    values = {
        **scene.template_values(),
        **outputs.template_values(),
        "python": sys.executable,
        "repo": str(resources.repository),
        "checkpoint": str(resources.checkpoint),
        "entrypoint": str(resources.entrypoint),
    }
    try:
        return tuple(part.format_map(values) for part in spec.command_template)
    except KeyError as exc:
        raise ValueError(
            f"{spec.name}: unknown command-template placeholder {exc.args[0]!r}"
        ) from exc


def run_baseline(
    method: str,
    scene: CanonicalSceneInput,
    output_dir: str | Path,
    *,
    repository: str | Path | None = None,
    checkpoint: str | Path | None = None,
    timeout: float = 3600.0,
    dry_run: bool = False,
    skip_completed: bool = True,
    environment: Mapping[str, str] | None = None,
) -> RunResult:
    """Run one registered baseline and enforce the canonical output contract."""
    spec = get_baseline(method)
    outputs = StandardOutputLayout(Path(output_dir))
    if skip_completed and outputs.metrics.is_file():
        return RunResult(spec.name, (), outputs.root, "skipped")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    scene.validate()
    resources = check_environment(
        spec,
        repository=repository,
        checkpoint=checkpoint,
        environment=environment,
    )
    command = build_command(spec, scene, outputs, resources)
    if dry_run:
        return RunResult(spec.name, command, outputs.root, "dry-run")

    outputs.prepare()
    process_env = dict(os.environ if environment is None else environment)
    process_env.update(spec.extra_environment)
    with outputs.log.open("a", encoding="utf-8") as log:
        log.write(f"$ {subprocess.list2cmdline(command)}\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=resources.repository,
                env=process_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            log.write(f"\nERROR: timed out after {timeout:g} seconds\n")
            raise RuntimeError(
                f"{spec.name} timed out after {timeout:g} seconds; see {outputs.log}"
            ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"{spec.name} exited with code {completed.returncode}; see {outputs.log}"
        )
    if not outputs.metrics.is_file():
        raise RuntimeError(
            f"{spec.name} completed but did not produce required {outputs.metrics}. "
            "The upstream command may need an adapter for the canonical output contract."
        )
    return RunResult(spec.name, command, outputs.root, "completed", completed.returncode)
