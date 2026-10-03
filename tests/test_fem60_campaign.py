from __future__ import annotations

import json

import pytest

from scripts.run_fem60_campaign import aggregate, load_completed, select_scenarios


def test_select_scenarios_is_deterministic_and_sharded(tmp_path):
    for name in ("005", "001", "003", "002", "004", "000"):
        (tmp_path / name).mkdir()
    assert [p.name for p in select_scenarios(tmp_path)] == [
        "000",
        "001",
        "002",
        "003",
        "004",
        "005",
    ]
    assert [
        p.name
        for p in select_scenarios(tmp_path, num_shards=2, shard_index=1)
    ] == ["001", "003", "005"]


def test_select_scenarios_rejects_invalid_shard(tmp_path):
    with pytest.raises(ValueError):
        select_scenarios(tmp_path, num_shards=2, shard_index=2)


def test_aggregate_reports_median_and_completion():
    rows = [
        {
            "scenario": "000",
            "E_bg_rel_err": 0.10,
            "E_incl_rel_err": 0.20,
            "contrast_rel_err": 0.30,
            "runtime_s": 2.0,
        },
        {
            "scenario": "001",
            "E_bg_rel_err": 0.30,
            "E_incl_rel_err": 0.40,
            "contrast_rel_err": 0.50,
            "runtime_s": 4.0,
        },
    ]
    summary = aggregate(rows, expected_scenarios=2)
    assert summary["complete"] is True
    assert summary["E_bg_rel_err"]["median"] == pytest.approx(0.20)
    assert summary["runtime_s"]["mean"] == pytest.approx(3.0)


def test_load_completed_ignores_failed_and_invalid_files(tmp_path):
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "000.json").write_text(
        json.dumps({"scenario": "000", "status": "completed"}),
        encoding="utf-8",
    )
    (scenario_dir / "001.json").write_text(
        json.dumps({"scenario": "001", "status": "failed"}),
        encoding="utf-8",
    )
    (scenario_dir / "bad.json").write_text("{", encoding="utf-8")
    assert [row["scenario"] for row in load_completed(scenario_dir)] == ["000"]
