"""Safe runner selection and discovery trajectory handoff regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_safe_auto_runner_prefers_available_docker_image(monkeypatch):
    import easyicu.research_agent.runner as runner_mod

    seen = []

    monkeypatch.setattr(
        runner_mod.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return SimpleNamespace(
            returncode=0,
            stdout="sha256:" + "a" * 64 + "\n",
            stderr="",
        )

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    assert runner_mod.select_safe_runner_kind(image="easyicu:test") == "docker"
    assert seen == [
        [
            "/usr/bin/docker",
            "image",
            "inspect",
            "easyicu:test",
            "--format={{.Id}}",
        ]
    ]


def test_safe_auto_runner_uses_macos_sandbox_without_docker(monkeypatch):
    import easyicu.research_agent.runner as runner_mod

    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        runner_mod.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    assert runner_mod.select_safe_runner_kind() == "subprocess"


def test_safe_auto_runner_fails_before_execution_without_safe_backend(monkeypatch):
    import easyicu.research_agent.runner as runner_mod

    monkeypatch.setattr(runner_mod.sys, "platform", "win32")
    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: None)

    with pytest.raises(runner_mod.SafeRunnerUnavailableError, match="No safe"):
        runner_mod.select_safe_runner_kind()


def test_pipeline_default_auto_selects_probed_docker(ra, tmp_path, monkeypatch):
    import easyicu.research_agent.pipeline as pipeline_mod
    import easyicu.research_agent.runner as runner_mod

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort, index=False)
    monkeypatch.setattr(
        pipeline_mod,
        "select_safe_runner_kind",
        lambda **_kwargs: "docker",
    )
    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: "/usr/bin/docker")

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path / "work")
    built = pipeline._build_runner(run_dir=tmp_path / "run", cohort_path=cohort)

    assert pipeline._runner_kind == "auto"
    assert isinstance(built, ra.DockerRunner)


def test_discovery_jsonl_declares_trajectory_path(tmp_path):
    import tools.run_discovery_to_manuscript as launcher

    cohort = tmp_path / "universe.parquet"
    trajectory = tmp_path / "universe_trajectory.parquet"
    cohort.write_bytes(b"cohort")
    trajectory.write_bytes(b"trajectory")
    handoff = SimpleNamespace(
        literature_idea_id="idea-1",
        candidate_topic="Trajectory discovery",
        research_question="Do trajectories differ?",
        target_outcome="death",
        resolved_predictor_concept="sofa2",
        inclusion_criteria=[],
    )

    jsonl = launcher._write_ehrflowbench_row(
        out_root=tmp_path,
        handoff=handoff,
        cohort_path=cohort,
        trajectory_path=trajectory,
    )
    row = json.loads(jsonl.read_text(encoding="utf-8"))

    assert row["cohort_path"] == str(cohort.resolve())
    assert row["trajectory_path"] == str(trajectory.resolve())


def test_ehrflowbench_preserves_path_and_whitelists_trajectory(tmp_path, monkeypatch):
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "universe.parquet"
    trajectory = tmp_path / "universe_trajectory.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "concept": ["sofa2"],
            "value_num": [3.0],
        }
    ).to_parquet(trajectory, index=False)
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "trajectory-probe",
                "question": "Do trajectories differ?",
                "cohort_path": str(cohort),
                "trajectory_path": str(trajectory),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_one(**kwargs):
        seen.update(kwargs)
        return {"item_key": "trajectory-probe"}

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run_one)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "ok")

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == 0
    )

    assert isinstance(seen["cohort"], pd.DataFrame)
    pd.testing.assert_frame_equal(
        seen["cohort"].reset_index(drop=True),
        pd.read_parquet(cohort).reset_index(drop=True),
    )
    assert seen["item"].cohort_size == 2
    assert seen["item"].cohort_columns == ["stay_id", "death"]
    extra_env = seen["pipeline_options"]["runner_kwargs"]["extra_env"]
    assert extra_env == {
        "TRAJECTORY_PARQUET": str(trajectory.resolve()),
        "EASYICU_TRAJECTORY_PARQUET": str(trajectory.resolve()),
        "COHORT_TRAJECTORY_PARQUET": str(trajectory.resolve()),
    }


def test_declared_legacy_trajectory_options_reach_runner(ra, tmp_path):
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "universe.parquet"
    trajectory = tmp_path / "universe_trajectory.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort, index=False)
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "concept": ["sofa2"],
            "value_num": [3.0],
        }
    ).to_parquet(trajectory, index=False)
    options = bench._pipeline_options_with_trajectory(
        {"runner_kind": "subprocess"},
        trajectory_path=trajectory,
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        llm=ra.MockLLMClient(),
        enable_memory=False,
        **options,
    )

    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort,
        universe_path=cohort,
    )

    assert {runner.extra_env[key] for key in bench._TRAJECTORY_ENV_KEYS} == {
        str(trajectory.resolve())
    }


def test_ehrflowbench_rejects_missing_declared_trajectory(tmp_path, monkeypatch):
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort, index=False)
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "missing-trajectory",
                "question": "Do trajectories differ?",
                "cohort_path": str(cohort),
                "trajectory_path": str(tmp_path / "missing.parquet"),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "_run_one_item_from_cohort",
        lambda **_kwargs: pytest.fail("missing trajectory must not reach the runner"),
    )

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == 0
    )
    payload = json.loads(
        (tmp_path / "out" / "ehrflowbench_results.json").read_text(encoding="utf-8")
    )
    assert payload["pending"] == [
        {
            "key": "missing-trajectory",
            "status": "pending_missing_trajectory",
            "trajectory_path": str((tmp_path / "missing.parquet").resolve()),
        }
    ]
