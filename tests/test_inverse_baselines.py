"""Tests for the isolated external-baseline runner."""
from __future__ import annotations

from pathlib import Path

import pytest

from mvbrdf_shr.world.baselines import (
    BaselineEnvironmentError,
    CanonicalSceneInput,
    StandardOutputLayout,
    build_command,
    check_environment,
    get_baseline,
    list_baselines,
    run_baseline,
)


EXPECTED = {
    "nerfactor",
    "physg",
    "nvdiffrecmc",
    "tensoir",
    "gs_ir",
    "gaussian_shader",
}


def _scene(tmp_path: Path) -> CanonicalSceneInput:
    root = tmp_path / "scene"
    images = root / "images"
    images.mkdir(parents=True)
    cameras = root / "transforms.json"
    cameras.write_text("{}", encoding="utf-8")
    return CanonicalSceneInput("chair", root, images, cameras)


def _resources(tmp_path: Path, method: str):
    spec = get_baseline(method)
    repo = tmp_path / f"{method}_repo"
    entrypoint = repo / spec.entrypoint_candidates[0]
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# test entrypoint\n", encoding="utf-8")
    checkpoint = tmp_path / f"{method}.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    return spec, repo, checkpoint


def test_registry_contains_supported_methods():
    assert set(list_baselines()) == EXPECTED
    for method in EXPECTED:
        spec = get_baseline(method)
        assert spec.command_template
        assert spec.repository_env
        assert spec.checkpoint_env


@pytest.mark.parametrize("method", sorted(EXPECTED))
def test_registered_commands_expand_without_shell(tmp_path: Path, method: str):
    scene = _scene(tmp_path)
    spec, repo, checkpoint = _resources(tmp_path, method)
    resources = check_environment(spec, repository=repo, checkpoint=checkpoint)
    output = StandardOutputLayout(tmp_path / "out")

    command = build_command(spec, scene, output, resources)

    assert command[0]
    assert str(resources.entrypoint) in command
    assert str(checkpoint.resolve()) in command
    assert str(output.root.resolve()) in command
    assert all("{" not in argument for argument in command)


def test_dry_run_has_no_output_side_effects(tmp_path: Path):
    scene = _scene(tmp_path)
    _, repo, checkpoint = _resources(tmp_path, "nerfactor")
    output = tmp_path / "dry-output"

    result = run_baseline(
        "nerfactor",
        scene,
        output,
        repository=repo,
        checkpoint=checkpoint,
        dry_run=True,
    )

    assert result.status == "dry-run"
    assert result.command
    assert not output.exists()


def test_missing_external_repository_is_explicit(tmp_path: Path):
    spec = get_baseline("physg")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()

    with pytest.raises(BaselineEnvironmentError, match="repository.*does not exist"):
        check_environment(
            spec,
            repository=tmp_path / "not-installed",
            checkpoint=checkpoint,
        )
