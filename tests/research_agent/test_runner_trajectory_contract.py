"""Safe runner selection and discovery trajectory handoff regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryError,
)
from easyicu.research_agent.contracts.runtime import RunResult
from easyicu.research_agent.authority.step_capsule import (
    StepAuthorityCapsuleRef,
    load_verified_step_authority_capsule,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from tests.research_agent.test_materialized_trajectory_authority import _bundle


def _trajectory_authority_plan_llm() -> PatternScriptedMockLLMClient:
    plan = json.dumps(
        {
            "research_question": "Summarize the locked ICU cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce the declared cohort summary.",
                    "inputs": [],
                    "expected_outputs": ["table:cohort_summary"],
                    "method": "descriptive_summary",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "trajectory authority mutation regression",
        }
    )
    code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
pd.DataFrame({"n": [len(df)]}).to_csv(
    os.path.join(out, "cohort_summary.csv"), index=False
)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(
        {
            "n": len(df),
            "output_files": {"table:cohort_summary": "cohort_summary.csv"},
        },
        handle,
    )
"""
    return PatternScriptedMockLLMClient(
        [
            ("Produce an ICU-AWARE RESEARCH PLAN as JSON", [plan] * 8),
            ("REVISE THE ICU-AWARE RESEARCH PLAN", [plan] * 8),
            ("WRITE THE PYTHON CODE FOR STEP", [code] * 8),
            (
                "INTERPRET THE RESULTS OF STEP",
                ["The locked cohort summary is available."] * 8,
            ),
        ],
        contextual_default=True,
    )


def _typed_trajectory_run_kwargs(tmp_path):
    paths, source_cohort, source_trajectory = _bundle(tmp_path)
    return paths, {
        "question": "Summarize the locked ICU cohort.",
        "cohort": paths["parquet"],
        "cohort_authority_path": (
            paths["parquet"].parent / source_cohort.reference.file
        ),
        "cohort_authority_ref": source_cohort.reference,
        "trajectory_path": paths["trajectory"],
        "trajectory_authority_path": (
            paths["trajectory"].parent / source_trajectory.reference.file
        ),
        "trajectory_authority_ref": source_trajectory.reference,
        "cohort_name": "trajectory_authority_mutation",
        "database": "miiv",
        "target_outcome": "death",
        "stop_after_step_id": "01_summary",
        "stop_after_analysis": True,
    }


def _trajectory_test_pipeline(ra, tmp_path, monkeypatch, *, runner_factory):
    import easyicu.research_agent.agents.core as agent_core
    import easyicu.research_agent.pipeline as pipeline_module
    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_contract(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_contract)
    monkeypatch.setattr(
        agent_core,
        "_validate_required_primary_result",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_enforce_advanced_plan_contract",
        lambda *, plan, context: (plan, []),
    )
    return ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        llm=_trajectory_authority_plan_llm(),
        runner_factory=runner_factory,
        enable_probe_step=False,
        enable_replanning=False,
        enable_publication_figure_skill=False,
        enable_visual_qa=False,
        enable_literature=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=0,
    )


def test_safe_auto_runner_prefers_available_docker_image(monkeypatch):
    import easyicu.research_agent.execution.runner as runner_mod

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
    import easyicu.research_agent.execution.runner as runner_mod

    monkeypatch.setattr(runner_mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        runner_mod.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    assert runner_mod.select_safe_runner_kind() == "subprocess"


def test_safe_auto_runner_fails_before_execution_without_safe_backend(monkeypatch):
    import easyicu.research_agent.execution.runner as runner_mod

    monkeypatch.setattr(runner_mod.sys, "platform", "win32")
    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: None)

    with pytest.raises(runner_mod.SafeRunnerUnavailableError, match="No safe"):
        runner_mod.select_safe_runner_kind()


def test_pipeline_default_auto_selects_probed_docker(ra, tmp_path, monkeypatch):
    import easyicu.research_agent.pipeline as pipeline_mod
    import easyicu.research_agent.execution.runner as runner_mod

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


def test_discovery_jsonl_preserves_outcome_free_trajectory_task_shape(tmp_path):
    import tools.run_discovery_to_manuscript as launcher

    cohort = tmp_path / "universe.parquet"
    trajectory = tmp_path / "universe_trajectory.parquet"
    cohort.write_bytes(b"cohort")
    trajectory.write_bytes(b"trajectory")
    handoff = SimpleNamespace(
        literature_idea_id="sofa2-transportability",
        candidate_topic="SOFA-2 trajectory transportability",
        research_question="Are SOFA-2 trajectories reproducible across databases?",
        target_outcome=None,
        resolved_predictor_concept=None,
        analysis_family="trajectory_clustering",
        resolved_analysis_concepts=["sofa2"],
        inclusion_criteria=[],
    )

    jsonl = launcher._write_ehrflowbench_row(
        out_root=tmp_path,
        handoff=handoff,
        cohort_path=cohort,
        trajectory_path=trajectory,
    )
    row = json.loads(jsonl.read_text(encoding="utf-8"))

    assert row["kind"] == "longitudinal_trajectory_analysis"
    assert row["analysis_family"] == "trajectory_clustering"
    assert row["analysis_concepts"] == ["sofa2"]
    assert row["candidate_variables"] == ["sofa2"]
    assert "target_outcome" not in row
    assert "primary_predictor" not in row


def test_discovery_jsonl_declares_complete_typed_trajectory_authority(tmp_path):
    import tools.run_discovery_to_manuscript as launcher

    paths, source_cohort, source_trajectory = _bundle(tmp_path)
    handoff = SimpleNamespace(
        literature_idea_id="typed-idea",
        candidate_topic="Typed trajectory discovery",
        research_question="Do trajectories differ?",
        target_outcome="death",
        resolved_predictor_concept="lact_max",
        inclusion_criteria=[],
    )

    jsonl = launcher._write_ehrflowbench_row(
        out_root=tmp_path,
        handoff=handoff,
        cohort_path=paths["parquet"],
        cohort_authority_path=(paths["parquet"].parent / source_cohort.reference.file),
        cohort_authority_ref=source_cohort.reference.to_dict(),
        trajectory_path=paths["trajectory"],
        trajectory_authority_path=(
            paths["trajectory"].parent / source_trajectory.reference.file
        ),
        trajectory_authority_ref=source_trajectory.reference.to_dict(),
    )
    row = json.loads(jsonl.read_text(encoding="utf-8"))

    assert row["cohort_authority_required"] is True
    assert row["cohort_authority_ref"] == source_cohort.reference.to_dict()
    assert row["trajectory_authority_required"] is True
    assert row["trajectory_authority_ref"] == source_trajectory.reference.to_dict()
    assert row["trajectory_path"] == str(paths["trajectory"].resolve())


def test_ehrflowbench_preserves_trajectory_as_typed_pipeline_input(
    tmp_path, monkeypatch
):
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

    assert seen["cohort"] == cohort.resolve()
    assert seen["item"].cohort_size == 2
    assert seen["item"].cohort_columns == ["stay_id", "death"]
    assert seen["item"].trajectory_path == trajectory.resolve()
    assert "runner_kwargs" not in seen["pipeline_options"]


def test_ehrflowbench_preserves_typed_cohort_and_trajectory_authority(
    tmp_path,
    monkeypatch,
):
    import tools.run_research_agent_bench as bench

    paths, source_cohort, source_trajectory = _bundle(tmp_path)
    jsonl = tmp_path / "typed-items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "typed-trajectory-probe",
                "question": "Do trajectories differ?",
                "cohort_path": str(paths["parquet"]),
                "cohort_authority_required": True,
                "cohort_authority_path": str(
                    paths["parquet"].parent / source_cohort.reference.file
                ),
                "cohort_authority_ref": source_cohort.reference.to_dict(),
                "trajectory_path": str(paths["trajectory"]),
                "trajectory_authority_required": True,
                "trajectory_authority_path": str(
                    paths["trajectory"].parent / source_trajectory.reference.file
                ),
                "trajectory_authority_ref": source_trajectory.reference.to_dict(),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen = {}

    def fake_run_one(**kwargs):
        seen.update(kwargs)
        return {"item_key": "typed-trajectory-probe"}

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_run_one)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "ok")

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "typed-out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        == 0
    )
    assert seen["cohort"] == paths["parquet"]
    assert seen["item"].cohort_authority_ref == source_cohort.reference.to_dict()
    assert seen["item"].trajectory_path == paths["trajectory"].resolve()
    assert seen["item"].trajectory_authority_ref == (
        source_trajectory.reference.to_dict()
    )
    assert seen["item"].trajectory_authority_path == (
        paths["trajectory"].parent / source_trajectory.reference.file
    )


def test_ehrflowbench_rejects_path_only_trajectory_for_typed_cohort(
    tmp_path,
    monkeypatch,
):
    import tools.run_research_agent_bench as bench

    paths, source_cohort, _source_trajectory = _bundle(tmp_path)
    raw_trajectory = tmp_path / "raw-trajectory.parquet"
    raw_trajectory.write_bytes(paths["trajectory"].read_bytes())
    jsonl = tmp_path / "typed-path-only-items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "typed-path-only-trajectory",
                "question": "Do trajectories differ?",
                "cohort_path": str(paths["parquet"]),
                "cohort_authority_required": True,
                "cohort_authority_path": str(
                    paths["parquet"].parent / source_cohort.reference.file
                ),
                "cohort_authority_ref": source_cohort.reference.to_dict(),
                "trajectory_path": str(raw_trajectory),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "_run_one_item_from_cohort",
        lambda **_kwargs: pytest.fail("path-only typed trajectory must not execute"),
    )

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl,
            out_root=tmp_path / "typed-path-only-out",
            seed=7,
            arms=["aware"],
            provider="openai",
            model="model",
        )
        # The item was rejected at intake and never ran, which is what the
        # rest of this test proves. Exit 0 said that was a passing benchmark.
        == bench._PENDING_ITEMS_EXIT_CODE
    )
    payload = json.loads(
        (tmp_path / "typed-path-only-out" / "ehrflowbench_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["pending"] == [
        {
            "key": "typed-path-only-trajectory",
            "status": "typed_trajectory_authority_required",
            "trajectory_path": str(raw_trajectory.resolve()),
        }
    ]


def test_host_owned_legacy_trajectory_binding_reaches_runner(ra, tmp_path):
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
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "work",
        llm=ra.MockLLMClient(),
        enable_memory=False,
        runner_kind="subprocess",
    )

    runner = pipeline._build_runner(
        run_dir=tmp_path / "run",
        cohort_path=cohort,
        universe_path=cohort,
        trajectory_path=trajectory,
    )

    aliases = (
        "TRAJECTORY_PARQUET",
        "EASYICU_TRAJECTORY_PARQUET",
        "COHORT_TRAJECTORY_PARQUET",
    )
    assert {runner.extra_env[key] for key in aliases} == {str(trajectory.resolve())}


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
        # The item was rejected at intake and never ran, which is what the
        # rest of this test proves. Exit 0 said that was a passing benchmark.
        == bench._PENDING_ITEMS_EXIT_CODE
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


def test_ehrflowbench_rejects_trajectory_symlink_before_resolution(
    tmp_path,
    monkeypatch,
):
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "universe.parquet"
    target = tmp_path / "trajectory_target.parquet"
    trajectory = tmp_path / "trajectory_link.parquet"
    pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort, index=False)
    pd.DataFrame({"stay_id": [1], "charttime": [0.0]}).to_parquet(target, index=False)
    trajectory.symlink_to(target)
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "key": "symlink-trajectory",
                "question": "Do trajectories differ?",
                "cohort_path": str(cohort),
                "trajectory_path": str(trajectory),
                "target_outcome": "death",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "_run_one_item_from_cohort",
        lambda **_kwargs: pytest.fail("trajectory symlink must not reach the runner"),
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
        # The item was rejected at intake and never ran, which is what the
        # rest of this test proves. Exit 0 said that was a passing benchmark.
        == bench._PENDING_ITEMS_EXIT_CODE
    )
    payload = json.loads(
        (tmp_path / "out" / "ehrflowbench_results.json").read_text(encoding="utf-8")
    )
    assert payload["pending"] == [
        {
            "key": "symlink-trajectory",
            "status": "pending_missing_trajectory",
            "trajectory_path": str(trajectory.absolute()),
        }
    ]


def test_preexecution_trajectory_mutation_blocks_before_runner_call(
    ra, tmp_path, monkeypatch
):
    _paths, run_kwargs = _typed_trajectory_run_kwargs(tmp_path)
    runner_calls: list[str] = []

    class NeverCalledRunner:
        network_policy = "none"
        authority_identity_sha256 = "1" * 64

        def validate_runtime_capabilities(self):
            return ("pandas",)

        def run(self, **_kwargs):
            runner_calls.append("run")
            raise AssertionError("mutated trajectory must be rejected before execution")

    def runner_factory(*, extra_env, **_kwargs):
        if "TRAJECTORY_PARQUET" not in extra_env:
            return NeverCalledRunner()
        staged_path = Path(extra_env["TRAJECTORY_PARQUET"])
        assert staged_path.name == "cohort_trajectory.parquet"
        assert staged_path != run_kwargs["trajectory_path"]
        staged_path.write_bytes(staged_path.read_bytes() + b"tamper-before-run")
        return NeverCalledRunner()

    pipeline = _trajectory_test_pipeline(
        ra,
        tmp_path,
        monkeypatch,
        runner_factory=runner_factory,
    )

    with pytest.raises(MaterializedTrajectoryError, match="authoritative trajectory"):
        pipeline.run(**run_kwargs)
    assert runner_calls == []


def test_runner_trajectory_mutation_rejects_outputs_before_capsule_seal(
    ra,
    tmp_path,
    monkeypatch,
):
    _paths, run_kwargs = _typed_trajectory_run_kwargs(tmp_path)

    class MutatingRunner:
        network_policy = "none"
        authority_identity_sha256 = "2" * 64

        def __init__(self, *, workdir: Path, trajectory_path: Path) -> None:
            self.workdir = workdir
            self.trajectory_path = trajectory_path

        def validate_runtime_capabilities(self):
            return ("pandas",)

        def run(self, *, step_id, code, resolved_inputs_path=None):
            del resolved_inputs_path
            step_dir = self.workdir / "steps" / step_id
            out_dir = step_dir / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            script_path = step_dir / "analysis.py"
            script_path.write_text(code, encoding="utf-8")
            table_path = out_dir / "cohort_summary.csv"
            pd.DataFrame({"n": [2]}).to_csv(table_path, index=False)
            summary_path = out_dir / "step_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "n": 2,
                        "output_files": [
                            {
                                "kind": "table",
                                "name": "cohort_summary",
                                "path": "cohort_summary.csv",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            log_path = step_dir / "run.log"
            log_path.write_text("controlled mutating runner\n", encoding="utf-8")
            self.trajectory_path.write_bytes(
                self.trajectory_path.read_bytes() + b"tamper-during-run"
            )
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=step_dir,
                out_dir=out_dir,
                stdout="",
                stderr="",
                returncode=0,
                duration_seconds=0.01,
                artefacts=[table_path, summary_path],
                requested_network_policy="none",
                effective_isolation="controlled_test",
                runtime_provenance={"runner": "controlled_test"},
                outputs_safe_to_collect=True,
                runner_log_path=log_path,
            )

    def runner_factory(*, workdir, extra_env, **_kwargs):
        if "TRAJECTORY_PARQUET" not in extra_env:
            return MutatingRunner(
                workdir=Path(workdir),
                trajectory_path=run_kwargs["trajectory_path"],
            )
        return MutatingRunner(
            workdir=Path(workdir),
            trajectory_path=Path(extra_env["TRAJECTORY_PARQUET"]),
        )

    pipeline = _trajectory_test_pipeline(
        ra,
        tmp_path,
        monkeypatch,
        runner_factory=runner_factory,
    )
    result = pipeline.run(**run_kwargs)
    run_dir = Path(result.workdir)
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    ]
    assert records
    record = records[-1]
    assert record["status"] == "blocked_input_authority_mutation"
    assert record["input_authority_findings"][0]["validator"] == (
        "execution_input_authority_integrity"
    )
    detail = record["input_authority_findings"][0]["detail"]
    assert detail["expected_trajectory_sha256"] != detail["observed_trajectory_sha256"]
    output_dir = run_dir / "steps" / "01_summary" / "outputs"
    assert not list(output_dir.iterdir())
    assert all(
        item.get("produced_by_step") != "01_summary"
        for item in partial.get("evidence", [])
    )
    assert "cohort_summary" not in EvidenceStore(run_dir).aliases()
    capsule_ref = StepAuthorityCapsuleRef.model_validate(
        record["step_authority_capsule_ref"]
    )
    capsule = load_verified_step_authority_capsule(run_dir, ref=capsule_ref)
    assert capsule.capsule.execution is None
