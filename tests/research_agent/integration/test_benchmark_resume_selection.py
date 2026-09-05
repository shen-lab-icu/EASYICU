"""Benchmark resume selection requires a locked, unambiguous checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_research_agent_bench import _resolve_resume_run_id, _run_ehrflowbench_jsonl


def _write_bench_resume_checkpoint(
    run_dir: Path,
    *,
    run_status_claims_complete: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis_plan.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps({"per_step_records": []}), encoding="utf-8"
    )
    if run_status_claims_complete:
        (run_dir / "run_status.json").write_text(
            json.dumps({"gates": {"execution_complete": True}}),
            encoding="utf-8",
        )


def test_bench_runner_explicit_resume_id_wins_over_auto_discovery(tmp_path: Path):
    selected = tmp_path / "run_20260701T000000_selected"
    auto_latest = tmp_path / "run_20260701T999999_auto"
    _write_bench_resume_checkpoint(selected)
    _write_bench_resume_checkpoint(auto_latest)

    assert (
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=True,
            resume_run_id=selected.name,
        )
        == selected.name
    )


def test_bench_runner_auto_resume_does_not_trust_run_status_only_completion(
    tmp_path: Path,
):
    interrupted = tmp_path / "run_20260701T010000_interrupted"
    unverified_latest = tmp_path / "run_20260701T999999_unverified"
    _write_bench_resume_checkpoint(interrupted)
    _write_bench_resume_checkpoint(
        unverified_latest,
        run_status_claims_complete=True,
    )

    assert (
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=True,
            resume_run_id=None,
        )
        == unverified_latest.name
    )


def test_bench_runner_auto_resume_ignores_authoritatively_complete_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    interrupted = tmp_path / "run_20260701T010000_interrupted"
    complete_latest = tmp_path / "run_20260701T999999_complete"
    _write_bench_resume_checkpoint(interrupted)
    _write_bench_resume_checkpoint(complete_latest)
    (complete_latest / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.run_research_agent_bench._run_reached_execution_complete",
        lambda run_dir: run_dir == complete_latest,
    )

    assert (
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=True,
            resume_run_id=None,
        )
        == interrupted.name
    )


def test_bench_runner_explicit_resume_requires_locked_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run_20260701T000000_missing"
    run_dir.mkdir()

    with pytest.raises(SystemExit, match="analysis_plan.json"):
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=False,
            resume_run_id=run_dir.name,
        )


def test_bench_runner_resume_id_rejects_paths(tmp_path: Path):
    with pytest.raises(SystemExit, match="not a path"):
        _resolve_resume_run_id(
            workdir=tmp_path,
            reuse_existing=False,
            resume_run_id="../run_20260701T000000_bad",
        )


def test_bench_runner_ehrflow_resume_requires_single_row(tmp_path: Path):
    jsonl_path = tmp_path / "items.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"key": "E1"}),
                json.dumps({"key": "E2"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="one-row EHRFlowBench JSONL"):
        _run_ehrflowbench_jsonl(
            jsonl_path=jsonl_path,
            out_root=tmp_path / "out",
            seed=7,
            arms=["naive"],
            resume_run_id="run_20260701T000000_selected",
        )
