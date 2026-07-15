"""Resume-from-partial-run + manifest streaming (T2.4).

These tests pin two contracts:

1. **Manifest streaming.** The pipeline must flush
   ``manifest_partial.json`` after every step so a crash mid-loop
   leaves a usable resume sentinel.
2. **Resume from partial.** Running the pipeline a second time with
   ``resume_run_id=<the previous run_id>`` must:

   - reuse the same ``run_dir``,
   - skip steps whose prior status is ``"ok"``,
   - re-execute steps that are missing from the partial manifest,
   - end with a final ``manifest.json`` that the rest of the
     pipeline (manuscript, latex, report) treats normally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.run_research_agent_bench import (
    _resolve_resume_run_id,
    _run_ehrflowbench_jsonl,
)
from easyicu.research_agent.pipeline import (
    _load_compatible_resume_plan,
    _load_resume_state,
)
from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.plan_utils import _render_only_figure_step_intent
from easyicu.research_agent.context import build_research_context
from easyicu.research_agent.run_input_capsule import (
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    build_environment_identity,
    build_scientific_identity,
    invalidate_unverified_successful_steps,
    load_verified_run_input_capsule,
    prepare_existing_resume_input,
    seal_run_input_capsule,
)
from easyicu.research_agent.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, TimeWindow


def _run_full(ra, synthetic_cohort, workdir: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=workdir, llm=ra.MockLLMClient())
    return pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
    )


def _write_bench_resume_checkpoint(run_dir: Path, *, complete: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis_plan.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps({"per_step_records": []}), encoding="utf-8"
    )
    if complete:
        (run_dir / "run_status.json").write_text(
            json.dumps({"gates": {"execution_complete": True}}),
            encoding="utf-8",
        )


def _write_capsule_resume_fixture(ra, tmp_path: Path):
    """Create a pre-execution run whose input identity is already sealed."""

    run_id = "run_capsule_identity"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure_a": [0, 1, 0, 1],
            "outcome": [0, 1, 0, 1],
        }
    )
    cohort_path = run_dir / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    window = TimeWindow(
        name="first_24h",
        anchor="icu_admission",
        start_hours=0,
        end_hours=24,
        rationale="Prespecified baseline window.",
    )
    run_kwargs = {
        "question": "Is exposure A associated with the outcome?",
        "cohort": cohort,
        "cohort_name": "capsule_cohort",
        "database": "synthetic",
        "target_outcome": "outcome",
        "primary_exposure": "exposure_a",
        "inclusion_criteria": ["adult ICU stays"],
        "exclusion_criteria": ["missing outcome"],
        "id_columns": ["stay_id"],
        "outcome_columns": ["outcome"],
        "time_windows": [window],
        "concept_descriptions": {"exposure_a": "Prespecified binary exposure."},
        "notes": "Prespecified context note.",
        "resume_run_id": run_id,
        "stop_after_analysis": True,
    }
    context = build_research_context(
        research_question=run_kwargs["question"],
        cohort=cohort_path,
        cohort_name=run_kwargs["cohort_name"],
        database=run_kwargs["database"],
        target_outcome=run_kwargs["target_outcome"],
        primary_exposure=run_kwargs["primary_exposure"],
        inclusion_criteria=run_kwargs["inclusion_criteria"],
        exclusion_criteria=run_kwargs["exclusion_criteria"],
        id_columns=run_kwargs["id_columns"],
        outcome_columns=run_kwargs["outcome_columns"],
        time_windows=run_kwargs["time_windows"],
        concept_descriptions=run_kwargs["concept_descriptions"],
        notes=run_kwargs["notes"],
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    plan = AnalysisPlan(
        research_question=run_kwargs["question"],
        steps=[
            AnalysisStep(
                step_id="01_summary",
                intent="Summarize the prespecified cohort.",
                inputs=["exposure_a", "outcome"],
                expected_outputs=["table:summary"],
                method="descriptive_statistics",
            )
        ],
    )
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Frozen research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    evidence.register_file(
        kind="log",
        description="Frozen analysis plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )
    identity = build_scientific_identity(
        cohort=cohort,
        question=run_kwargs["question"],
        cohort_name=run_kwargs["cohort_name"],
        database=run_kwargs["database"],
        target_outcome=run_kwargs["target_outcome"],
        primary_exposure=run_kwargs["primary_exposure"],
        cross_database_validation=None,
        inclusion_criteria=run_kwargs["inclusion_criteria"],
        exclusion_criteria=run_kwargs["exclusion_criteria"],
        id_columns=run_kwargs["id_columns"],
        time_columns=None,
        outcome_columns=run_kwargs["outcome_columns"],
        time_windows=run_kwargs["time_windows"],
        concept_descriptions=run_kwargs["concept_descriptions"],
        user_preferences=None,
        notes=run_kwargs["notes"],
        skill_key=None,
        experiment_spec=None,
        source_files=None,
        disable_icu_context=False,
    )
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=identity,
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.research_manifest_partial/1",
                "run_id": run_id,
                "context_path": context_path.name,
                "plan_path": plan_path.name,
                "per_step_records": [],
                "evidence": [
                    record.model_dump(mode="json") for record in evidence.records()
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pipeline, run_dir, run_kwargs


@pytest.mark.parametrize(
    "changed_field",
    [
        "cohort",
        "question",
        "primary_exposure",
        "target_outcome",
        "time_windows",
        "inclusion_criteria",
        "exclusion_criteria",
        "concept_descriptions",
        "notes",
    ],
)
def test_resume_rejects_scientific_identity_drift_before_any_write(
    ra,
    tmp_path: Path,
    changed_field: str,
):
    pipeline, run_dir, base_kwargs = _write_capsule_resume_fixture(ra, tmp_path)
    before = {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    kwargs = dict(base_kwargs)
    if changed_field == "cohort":
        changed = kwargs["cohort"].copy()
        changed.loc[0, "outcome"] = 1
        kwargs["cohort"] = changed
    elif changed_field == "question":
        kwargs[changed_field] = "Is a different exposure associated with outcome?"
    elif changed_field == "primary_exposure":
        kwargs[changed_field] = "exposure_b"
    elif changed_field == "target_outcome":
        kwargs[changed_field] = "different_outcome"
    elif changed_field == "time_windows":
        kwargs[changed_field] = [
            TimeWindow(
                name="first_48h",
                anchor="icu_admission",
                start_hours=0,
                end_hours=48,
            )
        ]
    elif changed_field == "inclusion_criteria":
        kwargs[changed_field] = ["paediatric ICU stays"]
    elif changed_field == "exclusion_criteria":
        kwargs[changed_field] = ["exclude exposed stays"]
    elif changed_field == "concept_descriptions":
        kwargs[changed_field] = {"exposure_a": "A different scientific definition."}
    else:
        kwargs[changed_field] = "A different study context."

    with pytest.raises(RunInputIdentityError, match="different scientific input"):
        pipeline.run(**kwargs)

    after = {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_resume_environment_drift_is_receipted_without_overwriting_inputs(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    from easyicu.research_agent import pipeline as pipeline_module

    pipeline, run_dir, run_kwargs = _write_capsule_resume_fixture(ra, tmp_path)
    context_before = (run_dir / "research_context.json").read_bytes()
    cohort_before = (run_dir / "cohort.parquet").read_bytes()
    original_environment = pipeline_module.build_environment_identity(
        llm_signature="model-drift"
    )
    changed_environment = {
        **original_environment,
        "engine_code_sha256": "f" * 64,
        "validator_code_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        pipeline_module,
        "build_environment_identity",
        lambda *, llm_signature: changed_environment,
    )

    class DifferentModelLLM:
        name = "different-model"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return "{}"

    pipeline._llm = DifferentModelLLM()

    def stop_after_receipt(**_kwargs):
        raise RuntimeError("stop after resume receipt")

    monkeypatch.setattr(pipeline, "_run_plan_phase", stop_after_receipt)
    with pytest.raises(RuntimeError, match="stop after resume receipt"):
        pipeline.run(**run_kwargs)

    receipts = sorted(run_dir.glob("resume_environment_receipt_*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["environment_drift"] is True
    assert {
        "llm_signature",
        "llm_signature_sha256",
        "engine_code_sha256",
        "validator_code_sha256",
    } <= set(payload["changed_fields"])
    assert (run_dir / "research_context.json").read_bytes() == context_before
    assert (run_dir / "cohort.parquet").read_bytes() == cohort_before
    assert (run_dir / RUN_INPUT_CAPSULE_FILENAME).is_file()


def test_legacy_failed_attempt_without_capsule_cannot_mix_new_inputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy_failed_attempt"
    run_dir.mkdir()

    with pytest.raises(RunInputIdentityError, match="after any step attempt"):
        prepare_existing_resume_input(
            run_dir=run_dir,
            resume_state={
                "per_step_records": [
                    {"step_id": "01_model", "status": "execution_failed"}
                ]
            },
            scientific_identity={"question": "new study"},
            current_environment={"llm_signature": "mock"},
            cohort=pd.DataFrame({"x": [1]}),
            question="new study",
            resume_from_step_id=None,
            enforcement_mode=None,
            load_compatible_plan=lambda **_kwargs: pytest.fail(
                "an attempted legacy run must fail before plan reuse"
            ),
        )


def test_resume_invalidates_only_successes_with_unverified_evidence(tmp_path: Path):
    run_dir = tmp_path / "run_evidence_invalidation"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    source_a = run_dir / "a.json"
    source_b = run_dir / "b.json"
    source_a.write_text('{"n": 1}', encoding="utf-8")
    source_b.write_text('{"n": 2}', encoding="utf-8")
    record_a = evidence.register_file(
        kind="statistic",
        description="Step A summary.",
        source_path=source_a,
        evidence_id="step_a_summary",
        produced_by_step="01_a",
    )
    record_b = evidence.register_file(
        kind="statistic",
        description="Step B summary.",
        source_path=source_b,
        evidence_id="step_b_summary",
        produced_by_step="02_b",
    )
    state = {
        "per_step_records": [
            {"step_id": "01_a", "status": "ok", "evidence_ids": [record_a.evidence_id]},
            {"step_id": "02_b", "status": "ok", "evidence_ids": [record_b.evidence_id]},
        ],
        "findings": [],
    }
    Path(verified_run_evidence_path(run_dir, record_a)).unlink()
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=state,
        records=records,
    )

    latest = {
        record["step_id"]: record
        for record in updated["per_step_records"]
    }
    assert invalidated == {
        "01_a": "evidence step_a_summary failed path/digest verification"
    }
    assert latest["01_a"]["status"] == "resume_evidence_invalid"
    assert latest["02_b"]["status"] == "ok"


def test_resume_invalidates_downstream_success_when_upstream_evidence_is_bad(
    tmp_path: Path,
):
    run_dir = tmp_path / "run_upstream_evidence_invalidation"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    upstream_path = run_dir / "upstream.json"
    downstream_path = run_dir / "downstream.json"
    upstream_path.write_text('{"n": 1}', encoding="utf-8")
    downstream_path.write_text('{"estimate": 2}', encoding="utf-8")
    upstream = evidence.register_file(
        kind="statistic",
        description="Upstream authority.",
        source_path=upstream_path,
        evidence_id="upstream_summary",
        produced_by_step="01_upstream",
    )
    downstream = evidence.register_file(
        kind="statistic",
        description="Downstream result derived from upstream authority.",
        source_path=downstream_path,
        evidence_id="downstream_summary",
        produced_by_step="02_downstream",
        inputs=[upstream.evidence_id],
    )
    state = {
        "per_step_records": [
            {
                "step_id": "01_upstream",
                "status": "ok",
                "evidence_ids": [upstream.evidence_id],
            },
            {
                "step_id": "02_downstream",
                "status": "ok",
                "evidence_ids": [downstream.evidence_id],
            },
        ],
        "findings": [],
    }
    Path(verified_run_evidence_path(run_dir, upstream)).unlink()
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(
                encoding="utf-8"
            )
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=state,
        records=records,
    )

    assert set(invalidated) == {"01_upstream", "02_downstream"}
    assert "upstream_summary failed path/digest verification" in invalidated[
        "02_downstream"
    ]
    latest = {
        record["step_id"]: record for record in updated["per_step_records"]
    }
    assert latest["01_upstream"]["status"] == "resume_evidence_invalid"
    assert latest["02_downstream"]["status"] == "resume_evidence_invalid"


def test_pipeline_resume_passes_invalidated_evidence_step_to_execution(
    ra,
    tmp_path: Path,
    monkeypatch,
):
    pipeline, run_dir, run_kwargs = _write_capsule_resume_fixture(ra, tmp_path)
    output = run_dir / "summary.json"
    output.write_text('{"n": 4}', encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    record = evidence.register_file(
        kind="statistic",
        description="Prior summary.",
        source_path=output,
        evidence_id="prior_summary",
        produced_by_step="01_summary",
    )
    partial_path = run_dir / "manifest_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    partial["per_step_records"] = [
        {
            "step_id": "01_summary",
            "status": "ok",
            "evidence_ids": [record.evidence_id],
        }
    ]
    partial["evidence"] = [
        item.model_dump(mode="json") for item in evidence.records()
    ]
    partial_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
    Path(verified_run_evidence_path(run_dir, record)).unlink()

    observed = {}

    def capture_resume_state(**kwargs):
        observed.update(kwargs["resume_state"])
        raise RuntimeError("captured invalidated resume state")

    monkeypatch.setattr(pipeline, "_run_plan_phase", capture_resume_state)
    with pytest.raises(RuntimeError, match="captured invalidated"):
        pipeline.run(**run_kwargs)

    current = {
        item["step_id"]: item
        for item in observed["per_step_records"]
    }
    assert current["01_summary"]["status"] == "resume_evidence_invalid"
    assert any(
        finding.get("validator") == "resume_evidence_integrity"
        for finding in observed["findings"]
    )


def test_legacy_completed_resume_is_adopted_only_from_verified_context_and_cohort(
    ra,
    tmp_path: Path,
):
    pipeline, run_dir, run_kwargs = _write_capsule_resume_fixture(ra, tmp_path)
    evidence = EvidenceStore(run_dir)
    result_path = run_dir / "legacy_result.json"
    result_path.write_text('{"n": 4}', encoding="utf-8")
    result_record = evidence.register_file(
        kind="statistic",
        description="Legacy completed result.",
        source_path=result_path,
        evidence_id="legacy_result",
        produced_by_step="01_summary",
    )
    provenance_path = run_dir / "provenance_sources.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.provenance_sources/1",
                "records": [
                    {
                        "relative_path": "cohort.parquet",
                        "role": "cohort",
                        "sha256": hashlib.sha256(
                            (run_dir / "cohort.parquet").read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence.register_file(
        kind="log",
        description="Legacy cohort provenance.",
        source_path=provenance_path,
        evidence_id="provenance_sources",
    )

    index_path = run_dir / "evidence" / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    capsule_record = next(
        record for record in index if record["evidence_id"] == "run_input_capsule"
    )
    Path(run_dir / capsule_record["relative_path"]).unlink()
    index_path.write_text(
        json.dumps(
            [
                record
                for record in index
                if record["evidence_id"] != "run_input_capsule"
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    aliases_path = run_dir / "evidence" / "evidence_aliases.json"
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    aliases_path.write_text(
        json.dumps(
            {
                alias: evidence_id
                for alias, evidence_id in aliases.items()
                if evidence_id != "run_input_capsule"
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / RUN_INPUT_CAPSULE_FILENAME).unlink()

    partial = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))
    partial.update(
        {
            "llm_signature": "mock",
            "prompt_pack_version": "legacy-prompts/v1",
            "prompt_pack_files": {},
            "per_step_records": [
                {
                    "step_id": "01_summary",
                    "status": "ok",
                    "evidence_ids": [result_record.evidence_id],
                }
            ],
        }
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(partial, indent=2), encoding="utf-8"
    )
    scientific_identity = build_scientific_identity(
        cohort=run_kwargs["cohort"],
        question=run_kwargs["question"],
        cohort_name=run_kwargs["cohort_name"],
        database=run_kwargs["database"],
        target_outcome=run_kwargs["target_outcome"],
        primary_exposure=run_kwargs["primary_exposure"],
        cross_database_validation=None,
        inclusion_criteria=run_kwargs["inclusion_criteria"],
        exclusion_criteria=run_kwargs["exclusion_criteria"],
        id_columns=run_kwargs["id_columns"],
        time_columns=None,
        outcome_columns=run_kwargs["outcome_columns"],
        time_windows=run_kwargs["time_windows"],
        concept_descriptions=run_kwargs["concept_descriptions"],
        user_preferences=None,
        notes=run_kwargs["notes"],
        skill_key=None,
        experiment_spec=None,
        source_files=None,
        disable_icu_context=False,
    )
    prepared = prepare_existing_resume_input(
        run_dir=run_dir,
        resume_state=partial,
        scientific_identity=scientific_identity,
        current_environment=build_environment_identity(llm_signature="mock"),
        cohort=run_kwargs["cohort"],
        question=run_kwargs["question"],
        resume_from_step_id=None,
        enforcement_mode="soft",
        load_compatible_plan=_load_compatible_resume_plan,
    )

    assert prepared.input_verified is True
    adopted = load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=scientific_identity,
    )
    assert adopted.capsule.legacy_adopted is True
    assert prepared.resume_state["resume_environment_drift"] is True


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


def test_bench_runner_auto_resume_ignores_complete_runs(tmp_path: Path):
    interrupted = tmp_path / "run_20260701T010000_interrupted"
    complete_latest = tmp_path / "run_20260701T999999_complete"
    _write_bench_resume_checkpoint(interrupted)
    _write_bench_resume_checkpoint(complete_latest, complete=True)

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


def test_load_resume_state_rejects_corrupt_partial_manifest(tmp_path: Path):
    run_dir = tmp_path / "run_corrupt"
    run_dir.mkdir()
    (run_dir / "manifest_partial.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt checkpoint"):
        _load_resume_state(run_dir)


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


def test_resume_prefers_latest_compatible_plan_revision(tmp_path: Path):
    run_dir = tmp_path / "run_20260701T000000_revision"
    run_dir.mkdir()
    original = AnalysisPlan(
        research_question="Resume E1.",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Define cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_table",
                intent="Completed table.",
                expected_outputs=["table:table"],
            ),
            AnalysisStep(
                step_id="05_sensitivity_figure",
                intent="Render the publication figure(s) declared by step '05_sensitivity'.",
                expected_outputs=["figure:sensitivity"],
            ),
        ],
    )
    revision = AnalysisPlan(
        research_question="Resume E1.",
        revision=2,
        steps=[
            AnalysisStep(
                step_id="00_probe",
                intent="Probe.",
                expected_outputs=["table:probe"],
            ),
            AnalysisStep(
                step_id="01_cohort",
                intent="Define cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_table",
                intent="Completed table.",
                expected_outputs=["table:table"],
            ),
            AnalysisStep(
                step_id="05_sensitivity",
                intent="Run sensitivity.",
                expected_outputs=["table:sensitivity"],
            ),
            AnalysisStep(
                step_id="05_sensitivity_figure",
                intent="Render the publication figure(s) declared by step '05_sensitivity'.",
                expected_outputs=["figure:sensitivity"],
            ),
        ],
    )
    original_path = run_dir / "analysis_plan.json"
    original_path.write_text(
        original.model_dump_json(indent=2), encoding="utf-8"
    )
    revision_path = run_dir / "analysis_plan_revision_2.json"
    revision_path.write_text(
        revision.model_dump_json(indent=2), encoding="utf-8"
    )
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Original plan.",
        source_path=original_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    evidence.register_file(
        kind="log",
        description="Revised plan.",
        source_path=revision_path,
        evidence_id="analysis_plan_revision_2",
        producer="replanner",
        generation_mode="llm",
    )
    resume_state = {
        "plan_path": "analysis_plan.json",
        "per_step_records": [
            {"step_id": "00_probe", "status": "ok"},
            {"step_id": "01_cohort", "status": "ok"},
            {"step_id": "02_table", "status": "ok"},
        ],
    }

    plan, path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state=resume_state,
    )

    assert path == verified_run_evidence_path(
        run_dir,
        evidence.get("analysis_plan_revision_2"),
    )
    assert [step.step_id for step in plan.steps][-2:] == [
        "05_sensitivity",
        "05_sensitivity_figure",
    ]


def test_resume_plan_compatibility_uses_latest_step_status(tmp_path: Path):
    run_dir = tmp_path / "run_latest_step_authority"
    run_dir.mkdir()
    plan = AnalysisPlan(
        research_question="Resume after a superseded success.",
        steps=[
            AnalysisStep(
                step_id="01_current_success",
                intent="Keep the current successful step.",
                expected_outputs=["table:current"],
            )
        ],
    )
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    record = evidence.register_file(
        kind="log",
        description="Resume plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    resume_state = {
        "per_step_records": [
            {"step_id": "01_current_success", "status": "ok"},
            {"step_id": "02_superseded", "status": "ok"},
            {"step_id": "02_superseded", "status": "contract_failed"},
        ]
    }

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state=resume_state,
    )

    assert selected == plan
    assert selected_path == verified_run_evidence_path(run_dir, record)


def test_partial_manifest_is_written_after_run(ra, synthetic_cohort, tmp_path: Path):
    result = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(result.workdir)
    partial = run_dir / "manifest_partial.json"
    assert partial.exists(), "manifest_partial.json must be written during the run"

    data = json.loads(partial.read_text(encoding="utf-8"))
    assert data["run_id"] == result.run_id
    assert data["schema_version"].startswith("easyicu.research_manifest_partial")
    # Every step in per_step_records should have status ok after a clean run.
    statuses = [r.get("status") for r in data.get("per_step_records", [])]
    assert statuses, "no step records persisted in partial manifest"
    assert all(s == "ok" for s in statuses), statuses


def test_partial_manifest_checkpoints_executed_step_before_interpretation(
    ra, synthetic_cohort, tmp_path: Path
):
    class InterruptingAnalyzerLLM(ra.MockLLMClient):
        name = "interrupting-analyzer"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if "INTERPRET THE RESULTS" in user.upper():
                raise KeyboardInterrupt("simulate interruption after runner outputs")
            return super().complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=InterruptingAnalyzerLLM(),
        enable_literature=False,
    )
    with pytest.raises(KeyboardInterrupt):
        pipeline.run(
            question="Is admission SOFA-2 associated with ICU mortality?",
            cohort=synthetic_cohort,
            cohort_name="resume_test",
            database="synthetic",
            target_outcome="death",
        )

    run_dirs = sorted(tmp_path.glob("run_*"))
    assert run_dirs
    partial = json.loads((run_dirs[-1] / "manifest_partial.json").read_text(encoding="utf-8"))
    records = partial.get("per_step_records") or []
    pending = [
        record
        for record in records
        if record.get("status") == "executed_pending_review"
    ]
    assert pending, records
    assert pending[-1].get("review_pending") is True
    assert pending[-1].get("step_summary"), pending[-1]
    assert pending[-1].get("evidence_ids"), pending[-1]


def test_resume_skips_completed_steps(ra, synthetic_cohort, tmp_path: Path):
    """A second invocation with ``resume_run_id`` should re-use the same
    workdir and add no new step records — every step is already ok."""
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    partial_before = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert all(r.get("status") == "ok" for r in partial_before["per_step_records"])
    n_records_before = len(partial_before["per_step_records"])
    n_evidence_before = len(partial_before["evidence"])

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
    )
    assert second.run_id == first.run_id, "resume must reuse the same run_id"
    assert second.workdir == first.workdir, "resume must reuse the same workdir"

    partial_after = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))
    statuses_after = [r.get("status") for r in partial_after["per_step_records"]]
    # No new step record should have been added, every step was already ok.
    assert len(partial_after["per_step_records"]) == n_records_before, statuses_after
    # Evidence count may grow by a constant (literature/manuscript/latex
    # are re-emitted on resume); the *step-bound* evidence does not grow.
    assert len(partial_after["evidence"]) >= n_evidence_before


def test_resume_reruns_missing_step(ra, synthetic_cohort, tmp_path: Path):
    """Doctor the partial manifest to drop the last step, then resume —
    the dropped step must be re-executed and ``per_step_records`` must
    grow by exactly one entry."""
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    partial_path = run_dir / "manifest_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))

    records = partial["per_step_records"]
    assert len(records) >= 2, "need ≥2 steps to test partial resume"
    drop_index = next(
        (
            i for i, record in enumerate(records)
            if record.get("step_id") == "04_primary_association"
        ),
        len(records) - 1,
    )
    dropped = records.pop(drop_index)
    partial_path.write_text(
        json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
    )
    assert second.run_id == first.run_id

    partial_after = json.loads(partial_path.read_text(encoding="utf-8"))
    new_step_ids = [r["step_id"] for r in partial_after["per_step_records"]]
    assert dropped["step_id"] in new_step_ids, (
        f"dropped step {dropped['step_id']!r} was not re-executed; new ids: {new_step_ids}"
    )


def test_resume_from_completed_step_can_stop_after_that_step(
    ra, synthetic_cohort, tmp_path: Path
):
    """A reviewed prior plan can be continued from an arbitrary completed step.

    This covers the interactive workflow where a user approves upstream plan
    work, then asks to rerun one downstream step without manually editing the
    checkpoint manifest.
    """
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    plan_order = {
        step["step_id"]: idx for idx, step in enumerate(plan.get("steps") or [])
    }
    assert "04_primary_association" in plan_order
    stop_index = plan_order["04_primary_association"]
    partial_before = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert any(
        plan_order.get(record.get("step_id"), -1) > stop_index
        for record in partial_before["per_step_records"]
    ), "full run should have completed steps after the selected resume point"

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="04_primary_association",
        stop_after_step_id="04_primary_association",
        stop_after_analysis=True,
    )
    assert second.run_id == first.run_id

    partial_after = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    step_ids_after = [record["step_id"] for record in partial_after["per_step_records"]]
    assert "04_primary_association" in step_ids_after
    assert all(
        plan_order.get(step_id, -1) <= stop_index for step_id in step_ids_after
    ), step_ids_after
    resume_findings = [
        finding
        for finding in partial_after["findings"]
        if finding.get("validator") == "resume"
    ]
    assert resume_findings
    dropped = resume_findings[-1]["detail"]["dropped_completed_step_ids"]
    assert "04_primary_association" in dropped
    assert not any(
        finding.get("validator") == "manuscript_gate"
        and finding.get("severity") == "error"
        for finding in partial_after["findings"]
    )


@pytest.mark.parametrize("reuse_step_code", [False, True])
def test_resume_from_step_reuses_prior_code(
    ra, tmp_path: Path, monkeypatch, reuse_step_code: bool
):
    """Resume reuses valid prior code only on failure or explicit opt-in."""

    class SingleStepLLM:
        name = "single-step-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps(
                    {
                        "research_question": "Is SOFA associated with ICU mortality?",
                        "steps": [
                            {
                                "step_id": "04_primary_association",
                                "intent": "Estimate SOFA and ICU mortality association.",
                                "inputs": ["sofa2", "death"],
                                "expected_outputs": ["table:cohort_summary"],
                                "method": "descriptive",
                                "icu_rule_refs": ["aggregation_rule_for"],
                            }
                        ],
                        "rationale": "single-step resume code reuse test",
                    }
                )
            if "WRITE THE PYTHON CODE" in upper:
                return """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "predictor": "sofa2",
    "n": int(len(df)),
    "sofa2_median": float(df["sofa2"].median()),
    "mortality_rate": float(df["death"].mean()),
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
summary["output_files"] = [
    {"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}
]
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
            if "INTERPRET THE RESULTS" in upper:
                return "The cohort table is available {evidence:cohort_summary}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe table is available {evidence:cohort_summary}."
            return "{}"

    class FailingCoderLLM(SingleStepLLM):
        name = "failing-coder-llm"

        def __init__(self):
            self.coder_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if "WRITE THE PYTHON CODE" in user.upper():
                self.coder_calls += 1
                raise RuntimeError("simulated coder outage")
            return super().complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "sofa2": [0, 1, 3, 6],
        "death": [1, 0, 0, 1],
    })
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=SingleStepLLM(),
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    first = first_pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="resume_code_reuse_test",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
    )
    run_dir = Path(first.workdir)
    partial_path = run_dir / "manifest_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    bad_code_path = run_dir / "evidence" / "code_bad__analysis.py"
    bad_code_path.write_text("{}", encoding="utf-8")
    partial["evidence"].append({
        "evidence_id": "code_bad",
        "kind": "code",
        "description": "Malformed code evidence that must not be reused.",
        "relative_path": "evidence/code_bad__analysis.py",
        "sha256": "bad",
        "produced_by_step": "04_primary_association",
        "inputs": [],
        "script_evidence_id": None,
        "producer": "coder",
        "generation_mode": "llm",
        "finding_severity": None,
        "finding_messages": [],
        "metadata": {},
        "created_at": "2026-07-01T00:00:00Z",
    })
    partial.setdefault("findings", []).extend([
        {
            "validator": "step_contract",
            "severity": "error",
            "message": (
                "stale pre-resume error for step "
                "04_primary_association should be cleared"
            ),
            "detail": {"step_id": "04_primary_association"},
        },
        {
            "validator": "manuscript_gate",
            "severity": "error",
            "message": "stale pre-resume gate error should be cleared",
            "detail": {
                "failed_steps": [
                    {"step_id": "04_primary_association", "status": "failed"}
                ]
            },
        },
    ])
    partial_path.write_text(
        json.dumps(partial, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.delenv("EASYICU_RESUME_REUSE_STEP_CODE", raising=False)
    if reuse_step_code:
        monkeypatch.setenv("EASYICU_RESUME_REUSE_STEP_CODE", "1")
    second_llm = FailingCoderLLM()
    second_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=second_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    second = second_pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="resume_code_reuse_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="04_primary_association",
        stop_after_step_id="04_primary_association",
        stop_after_analysis=True,
    )
    assert second.run_id == first.run_id
    assert second_llm.coder_calls == (0 if reuse_step_code else 1)

    partial = json.loads(
        (Path(second.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "04_primary_association"
    ]
    assert records[-1]["status"] == "ok"
    assert records[-1]["generation_mode"] == "resumed_code_reuse"
    assert records[-1]["resumed_code_evidence_id"]
    assert records[-1]["resumed_code_evidence_id"] != "code_bad"
    source_code_record = next(
        record
        for record in partial["evidence"]
        if record.get("evidence_id") == records[-1]["resumed_code_evidence_id"]
    )
    assert records[-1]["resumed_from_generation_mode"] == source_code_record[
        "generation_mode"
    ]
    final_code_records = [
        record
        for record in partial["evidence"]
        if record.get("kind") == "code"
        and record.get("produced_by_step") == "04_primary_association"
        and record.get("generation_mode") == "resumed_code_reuse"
    ]
    assert final_code_records
    assert final_code_records[-1]["evidence_id"].endswith(
        "_resumed_code_reuse"
    )
    assert final_code_records[-1]["description"].startswith(
        "Reused prior agent-generated analysis script"
    )
    assert final_code_records[-1]["metadata"]["resumed_code_evidence_id"]
    assert final_code_records[-1]["metadata"][
        "resumed_from_generation_mode"
    ] == source_code_record["generation_mode"]
    assert not any(
        "stale pre-resume" in finding.get("message", "")
        for finding in partial["findings"]
    )
    assert any(
        finding.get("validator") == "coder"
        and "reused prior agent-generated code" in finding.get("message", "")
        and "source mode:" in finding.get("message", "")
        for finding in partial["findings"]
    )
    coder_messages = [
        finding.get("message", "")
        for finding in partial["findings"]
        if finding.get("validator") == "coder"
    ]
    if reuse_step_code:
        assert any("before requesting a new coder script" in m for m in coder_messages)
    else:
        assert any("Coder agent failed" in m for m in coder_messages)


def test_concept_repair_failure_resumes_quarantined_draft_fail_closed(
    ra, tmp_path: Path, monkeypatch
):
    """A rejected draft is repaired on resume, never reused as executable code."""

    from easyicu.research_agent.audits.validators import LLMConceptAuditor
    from easyicu.research_agent.contracts import ValidationFinding

    audit_state = {"emit_error": True, "reject_marker": None}
    persisted_message = "Displayed percentage is not reconciled to its denominator."

    def fake_audit(self, *, context, script_text, step):
        del self, context
        reject_marker = audit_state["reject_marker"]
        if not audit_state["emit_error"] and not (
            reject_marker and reject_marker in script_text
        ):
            return []
        return [
            ValidationFinding(
                validator="llm_concept_auditor",
                severity="error",
                message=persisted_message,
                detail={"step_id": step.step_id},
            )
        ]

    monkeypatch.setattr(LLMConceptAuditor, "audit", fake_audit)

    draft_code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "n": int(len(df)),
    "draft_marker": True,
    "output_files": [
        {"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}
    ],
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
    repaired_code = draft_code.replace(
        '"draft_marker": True', '"status": "ok", "repaired_marker": True'
    )

    class QuarantineLLM:
        name = "quarantine-resume-llm"

        def __init__(self, *, repair_succeeds: bool, repair_code: str | None = None):
            self.repair_succeeds = repair_succeeds
            self.repair_code = repair_code
            self.write_calls = 0
            self.repair_calls = 0
            self.repair_prompts = []

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps(
                    {
                        "research_question": "Summarize the ICU cohort.",
                        "steps": [
                            {
                                "step_id": "01_summary",
                                "intent": "Produce a descriptive cohort summary.",
                                "inputs": ["stay_id"],
                                "expected_outputs": ["table:cohort_summary"],
                                "method": "descriptive_summary",
                                "icu_rule_refs": [],
                            }
                        ],
                        "rationale": "single-step quarantine resume test",
                    }
                )
            if "REPAIR THE PYTHON CODE" in upper:
                self.repair_calls += 1
                self.repair_prompts.append(user)
                if not self.repair_succeeds:
                    raise RuntimeError("simulated repair quota exhaustion")
                return self.repair_code or repaired_code
            if "WRITE THE PYTHON CODE" in upper:
                self.write_calls += 1
                return draft_code
            if "INTERPRET THE RESULTS" in upper:
                return "The cohort summary is available {evidence:cohort_summary}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nSummary {evidence:cohort_summary}."
            return "{}"

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = QuarantineLLM(repair_succeeds=False)
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=first_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
    )
    first = first_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="quarantine_resume_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    run_dir = Path(first.workdir)
    first_partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    first_record = next(
        record
        for record in first_partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    )
    assert first_record["status"] == "repair_failed"
    assert first_llm.write_calls == 1
    assert first_llm.repair_calls == 1
    assert not (run_dir / "steps" / "01_summary" / "analysis.py").exists()
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()
    assert not any(
        record.get("kind") == "code"
        and record.get("produced_by_step") == "01_summary"
        for record in first_partial["evidence"]
    )

    # Simulate a nondeterministic auditor forgetting its prior error. The saved
    # error must still force REPAIR, and another outage must still not execute.
    audit_state["emit_error"] = False
    second_llm = QuarantineLLM(repair_succeeds=False)
    second_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=second_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
    )
    second_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="quarantine_resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    assert second_llm.write_calls == 0
    assert second_llm.repair_calls == 1
    assert persisted_message in second_llm.repair_prompts[-1]
    assert not (run_dir / "steps" / "01_summary" / "analysis.py").exists()
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()

    # A hosted model can return the same program with only a comment/whitespace
    # change while claiming it repaired the error. That must remain quarantined
    # and must never reach the runner, even when the live auditor forgets the
    # original nondeterministic finding.
    noop_llm = QuarantineLLM(
        repair_succeeds=True,
        repair_code=(
            draft_code
            + "\n# claimed repair, no semantic change\npass\n'claimed repair'\n"
        ),
    )
    noop_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=noop_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
    )
    noop_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="quarantine_resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    noop_partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    noop_record = next(
        record
        for record in noop_partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    )
    assert noop_llm.write_calls == 0
    assert noop_llm.repair_calls == 2
    assert noop_record["status"] == "blocked_by_concept_audit"
    assert noop_record["quarantined_repair_succeeded"] is False
    assert noop_record["quarantined_repair_noop_count"] == 2
    assert noop_record["step_llm_repair_attempts"] == 2
    assert noop_record["step_llm_repair_budget"] == 2
    assert noop_record["step_llm_repair_budget_exhausted"] is True
    assert not (run_dir / "steps" / "01_summary" / "analysis.py").exists()
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()

    # A material but still-invalid partial repair is the next resume candidate,
    # not the older pre-repair draft. It remains quarantined and unexecuted.
    partial_code = draft_code.replace("draft_marker", "partial_marker")
    audit_state["reject_marker"] = "partial_marker"
    partial_llm = QuarantineLLM(
        repair_succeeds=True,
        repair_code=partial_code,
    )
    partial_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=partial_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
    )
    partial_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="quarantine_resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    from easyicu.research_agent.pipeline_resume import (
        load_quarantined_concept_draft,
    )

    latest_draft = load_quarantined_concept_draft(
        run_dir=run_dir, step_id="01_summary"
    )
    assert latest_draft is not None
    assert "partial_marker" in latest_draft.code
    assert "draft_marker" not in latest_draft.code
    assert not (run_dir / "steps" / "01_summary" / "analysis.py").exists()

    audit_state["reject_marker"] = None

    # Once a materially changed repair passes the complete concept loop, the
    # stale draft is retired before the runner is entered. Thus even a later
    # runtime/contract failure cannot make the old draft outrank newer evidence.
    from easyicu.research_agent.runner import CodeRunner

    original_run = CodeRunner.run
    quarantine_absent_at_runner = []

    def run_after_quarantine_retired(self, *, step_id, code, resolved_inputs_path=None):
        quarantine_absent_at_runner.append(
            not (run_dir / "steps" / step_id / ".quarantine").exists()
        )
        return original_run(
            self,
            step_id=step_id,
            code=code,
            resolved_inputs_path=resolved_inputs_path,
        )

    monkeypatch.setattr(CodeRunner, "run", run_after_quarantine_retired)
    final_llm = QuarantineLLM(repair_succeeds=True)
    final_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=final_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        runner_kind="subprocess",
    )
    final_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="quarantine_resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    final_partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    final_record = next(
        record
        for record in final_partial["per_step_records"]
        if record.get("step_id") == "01_summary"
    )
    assert final_llm.write_calls == 0
    assert final_llm.repair_calls == 1
    assert quarantine_absent_at_runner == [True]
    assert final_record["status"] == "ok"
    assert final_record["generation_mode"] == "repaired"
    assert final_record["resumed_quarantined_draft"] is True
    assert final_record["quarantined_repair_succeeded"] is True
    assert final_record["quarantined_requires_repair"] is False
    assert final_record["quarantine_retired"] is True
    assert not (run_dir / "steps" / "01_summary" / ".quarantine").exists()


@pytest.mark.parametrize(
    "after",
    [
        "import os\nvalue = 1\n",
        "import os\nvalue = 1\n# claimed repair\n",
        "import os\nvalue = 1\npass\n",
        "import os\nvalue = 1\n'claimed repair'\n",
    ],
)
def test_quarantined_repair_materiality_rejects_inert_edits(after: str) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _python_repair_is_materially_changed,
    )

    before = "import os\nvalue = 1\n"

    assert not _python_repair_is_materially_changed(before, after)
    assert _python_repair_is_materially_changed(
        before, "import os\nvalue = 2\n"
    )


def test_resume_retires_unchanged_draft_after_deterministic_policy_supersession(
    ra, tmp_path: Path, monkeypatch
) -> None:
    """A validator-policy fix may retire its own stale error without code churn."""

    from easyicu.research_agent.audits.validators import LLMConceptAuditor
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.runner import CodeRunner

    audit_state = {"old_policy": True}

    def policy_transition_audit(self, *, context, script_text, step):
        del self, context, script_text
        if not audit_state["old_policy"]:
            return []
        finding = _stored_horizon_error(ra)
        return [
            finding.model_copy(
                update={"detail": {**(finding.detail or {}), "step_id": step.step_id}}
            )
        ]

    monkeypatch.setattr(LLMConceptAuditor, "audit", policy_transition_audit)

    draft_code = """
import json
import os
import pandas as pd

OUTCOME_OVERRIDE = {
    "concept_id": "death",
    "time_window": {
        "anchor": "icu_admit",
        "start_offset_hours": 0.0,
        "end_offset_hours": 720.0,
    },
    "aggregation": "first",
    "op": "==",
    "value": 1,
}
df = pd.read_parquet(os.environ["COHORT_PARQUET"])
y = df["death"]
out = os.environ["STEP_OUT_DIR"]
summary = {
    "status": "ok",
    "n": int(len(y)),
    "output_files": [
        {"kind": "table", "name": "cohort_summary", "path": "cohort_summary.csv"}
    ],
}
pd.DataFrame([summary]).to_csv(
    os.path.join(out, "cohort_summary.csv"), index=False
)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""

    class PolicyTransitionLLM:
        name = "quarantine-policy-transition"

        def __init__(self):
            self.write_calls = 0
            self.repair_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            del max_tokens, temperature
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps(
                    {
                        "research_question": (
                            "Summarize the cohort for in-hospital mortality."
                        ),
                        "steps": [
                            {
                                "step_id": "01_summary",
                                "intent": "Produce a descriptive cohort summary.",
                                "inputs": ["death"],
                                "expected_outputs": ["table:cohort_summary"],
                                "method": "descriptive_summary",
                                "icu_rule_refs": [],
                            }
                        ],
                        "rationale": "single-step policy transition test",
                    }
                )
            if "REPAIR THE PYTHON CODE" in upper:
                self.repair_calls += 1
                raise RuntimeError("simulated old-policy repair outage")
            if "WRITE THE PYTHON CODE" in upper:
                self.write_calls += 1
                return draft_code
            if "INTERPRET THE RESULTS" in upper:
                return "The cohort summary is available {evidence:cohort_summary}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nSummary {evidence:cohort_summary}."
            return "{}"

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = PolicyTransitionLLM()
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=first_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
    )
    first = first_pipeline.run(
        question="Summarize the cohort for in-hospital mortality.",
        cohort=cohort,
        cohort_name="policy_supersession",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    run_dir = Path(first.workdir)
    assert first_llm.write_calls == 1
    assert first_llm.repair_calls == 1
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()

    audit_state["old_policy"] = False
    quarantine_absent_at_runner = []
    original_run = CodeRunner.run

    def run_after_policy_supersession(
        self, *, step_id, code, resolved_inputs_path=None
    ):
        quarantine_absent_at_runner.append(
            not (run_dir / "steps" / step_id / ".quarantine").exists()
        )
        return original_run(
            self,
            step_id=step_id,
            code=code,
            resolved_inputs_path=resolved_inputs_path,
        )

    monkeypatch.setattr(CodeRunner, "run", run_after_policy_supersession)
    resumed_llm = PolicyTransitionLLM()
    resumed_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=resumed_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        runner_kind="subprocess",
    )
    resumed_pipeline.run(
        question="Summarize the cohort for in-hospital mortality.",
        cohort=cohort,
        cohort_name="policy_supersession",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        item
        for item in partial["per_step_records"]
        if item.get("step_id") == "01_summary"
    )
    assert resumed_llm.write_calls == 0
    assert resumed_llm.repair_calls == 0
    assert quarantine_absent_at_runner == [True]
    assert record["status"] == "ok"
    assert record["resumed_quarantined_draft"] is True
    assert record["quarantined_repair_succeeded"] is False
    assert record["quarantine_policy_superseded"] is True
    assert record["quarantine_policy_superseded_findings"][0][
        "downgraded_reason"
    ]
    assert record["quarantine_retired_by"] == (
        "deterministic_validator_policy_supersession"
    )
    assert record["quarantine_retired"] is True
    assert record["quarantined_requires_repair"] is False
    assert not (run_dir / "steps" / "01_summary" / ".quarantine").exists()


def _policy_supersession_context_and_script(ra):
    context = ra.build_research_context(
        research_question=(
            "Is an early exposure associated with in-hospital mortality?"
        ),
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "exposure": [0.0, 1.0, 2.0],
                "death": [0, 1, 0],
            }
        ),
        cohort_name="policy_supersession",
        database="synthetic",
        target_outcome="death",
    )
    script = """
OUTCOME_OVERRIDE = {
    "concept_id": "death",
    "time_window": {
        "anchor": "icu_admit",
        "start_offset_hours": 0.0,
        "end_offset_hours": 720.0,
    },
    "aggregation": "first",
    "op": "==",
    "value": 1,
}
y = df["death"]
"""
    return context, script


def _stored_horizon_error(ra):
    del ra
    from easyicu.research_agent.contracts import ValidationFinding

    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message=(
            "The fixed-window death alternative is incompatible with the "
            "bound hospital-mortality outcome."
        ),
        detail={
            "context": (
                "The script copies a 0–720 hour window but consumes the "
                "hospital mortality flag without deriving 30-day mortality "
                "from event time."
            )
        },
    )


def test_quarantine_policy_supersession_reclassifies_the_stored_error(ra) -> None:
    from easyicu.research_agent.pipeline_execute import (
        _quarantined_errors_superseded_by_current_policy,
    )

    context, script = _policy_supersession_context_and_script(ra)
    result = _quarantined_errors_superseded_by_current_policy(
        prior_errors=[_stored_horizon_error(ra)],
        current_findings=[],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(
            script.encode("utf-8")
        ).hexdigest(),
    )

    assert result is not None
    reclassified, provenance = result
    assert reclassified[0].severity == "warning"
    assert reclassified[0].message == _stored_horizon_error(ra).message
    assert reclassified[0].detail["downgraded_reason"]
    assert provenance == [
        {
            "validator": "llm_concept_auditor",
            "message": _stored_horizon_error(ra).message,
            "prior_severity": "error",
            "reclassified_severity": "warning",
            "downgraded_reason": reclassified[0].detail["downgraded_reason"],
        }
    ]
    assert (
        _quarantined_errors_superseded_by_current_policy(
            prior_errors=[_stored_horizon_error(ra)],
            current_findings=[],
            context=context,
            script_text=script + "\n# changed after quarantine\n",
            quarantined_script_sha256=hashlib.sha256(
                script.encode("utf-8")
            ).hexdigest(),
        )
        is None
    )


def test_quarantine_policy_supersession_reclassifies_finalized_only_false_override(
    ra,
) -> None:
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _quarantined_errors_superseded_by_current_policy,
    )

    context = ra.build_research_context(
        research_question="Assess balance by treatment.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "treatment": [0, 1]}),
        cohort_name="c",
        database="synthetic",
        primary_exposure="treatment",
    )
    script = """
REQUESTED_INPUTS = ['artifact:primary_exposure_definition']
def resolve_exposure(definition, product_contract, frame):
    if not isinstance(definition, pd.DataFrame):
        raise RuntimeError('finalized table required')
    executable = product_contract['executable_column']
    finalized = pd.to_numeric(definition[executable], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    return finalized.astype(int)
treatment = resolve_exposure(exposure_definition, product_contract, frame)
"""
    stored = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Finalized exposure is overwritten.",
        detail={
            "context": (
                "The script replaces treatment with "
                "reconcile_binary_event_presence values."
            )
        },
    )

    result = _quarantined_errors_superseded_by_current_policy(
        prior_errors=[stored],
        current_findings=[],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(
            script.encode("utf-8")
        ).hexdigest(),
    )

    assert result is not None
    reclassified, provenance = result
    assert reclassified[0].severity == "warning"
    assert provenance[0]["reclassified_severity"] == "warning"


@pytest.mark.parametrize(
    "current_findings",
    [
        [],
        [
            {
                "validator": "llm_concept_auditor",
                "severity": "warning",
                "message": "A different warning.",
            }
        ],
        [
            {
                "validator": "llm_concept_auditor",
                "severity": "warning",
                "message": "The ordinary warning version of the old error.",
            }
        ],
    ],
    ids=["missing", "different-warning", "ordinary-warning"],
)
def test_quarantine_policy_supersession_does_not_trust_fresh_audit_absence_or_warnings(
    ra, current_findings
) -> None:
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _quarantined_errors_superseded_by_current_policy,
    )

    context, script = _policy_supersession_context_and_script(ra)
    ineligible_prior = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Displayed percentage is not reconciled to its denominator.",
    )
    result = _quarantined_errors_superseded_by_current_policy(
        prior_errors=[ineligible_prior],
        current_findings=[
            ValidationFinding.model_validate(finding)
            for finding in current_findings
        ],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(
            script.encode("utf-8")
        ).hexdigest(),
    )

    assert result is None


def test_quarantine_policy_supersession_requires_zero_current_errors(ra) -> None:
    from easyicu.research_agent.contracts import ValidationFinding
    from easyicu.research_agent.pipeline_execute import (
        _quarantined_errors_superseded_by_current_policy,
    )

    context, script = _policy_supersession_context_and_script(ra)
    result = _quarantined_errors_superseded_by_current_policy(
        prior_errors=[_stored_horizon_error(ra)],
        current_findings=[
            ValidationFinding(
                validator="concept_usage",
                severity="error",
                message="A current deterministic error remains.",
            )
        ],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(
            script.encode("utf-8")
        ).hexdigest(),
    )

    assert result is None


def test_resume_reuses_locked_plan_instead_of_replanning(
    ra, synthetic_cohort, tmp_path: Path
):
    """Resume must reuse the prior run's ``analysis_plan.json`` rather than
    re-running the planner.

    A non-deterministic hosted planner returns a *different* plan on resume,
    whose step_ids no longer match the completed-step skip set — so the
    "resume" would silently re-run the whole analysis under new names. Pin
    that the locked plan is reused: the resumed planner output is ignored and
    the saved step_ids are unchanged.
    """
    first = _run_full(ra, synthetic_cohort, tmp_path)
    run_dir = Path(first.workdir)
    plan_path = run_dir / "analysis_plan.json"
    plan_bytes_before = plan_path.read_bytes()
    step_ids_before = [s["step_id"] for s in json.loads(
        plan_path.read_text(encoding="utf-8"))["steps"]]
    assert step_ids_before, "first run produced no plan steps"

    class DifferentPlanLLM:
        """Planner here returns a plan with a step_id that must never appear
        if the locked plan is reused."""

        name = "different-plan-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next(
                (m.content for m in reversed(messages) if m.role == "user"), ""
            )
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is admission SOFA-2 associated with ICU mortality?",
                    "steps": [{
                        "step_id": "88_resume_should_ignore_this",
                        "intent": "This plan must be ignored on resume.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:ignored"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "resume must not use this plan",
                })
            if "INTERPRET THE RESULTS" in upper:
                return "Reused-plan interpretation {evidence:primary_association}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nReused {evidence:primary_association}."
            return "{}"

    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=DifferentPlanLLM())
    second = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
    )
    assert second.run_id == first.run_id

    step_ids_after = [s["step_id"] for s in json.loads(
        plan_path.read_text(encoding="utf-8"))["steps"]]
    assert step_ids_after == step_ids_before, (
        "resume re-planned instead of reusing the locked plan: "
        f"{step_ids_before} -> {step_ids_after}"
    )
    assert "88_resume_should_ignore_this" not in step_ids_after
    assert plan_path.read_bytes() == plan_bytes_before, (
        "ordinary resume must read immutable plan evidence without "
        "re-serializing the mutable analysis_plan.json"
    )

    manifest = json.loads(Path(second.manifest_path).read_text(encoding="utf-8"))
    assert any(
        (f.get("detail") or {}).get("generation_mode") == "resumed"
        for f in manifest["findings"]
    ), "no 'resumed' planner finding recorded"


def test_resume_adopts_legacy_figure_edge_migration_without_replanning(
    ra, synthetic_cohort, tmp_path: Path, monkeypatch
):
    """A reused legacy split plan is migrated and checkpointed as the active plan."""
    from easyicu.research_agent import pipeline as pipeline_module

    run_id = "run_legacy_figure_edge"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    parent = AnalysisStep(
        step_id="01_summary",
        intent="Summarize the analytic cohort.",
        inputs=["age", "death"],
        expected_outputs=["artifact:cohort_snapshot", "table:cohort_summary"],
        method="descriptive_statistics",
        icu_rule_refs=["admission_anchor"],
    )
    figure_outputs = ["figure:cohort_summary"]
    legacy_plan = AnalysisPlan(
        research_question="Summarize the analytic cohort.",
        revision=3,
        steps=[
            parent,
            AnalysisStep(
                step_id="01_summary_figure",
                intent=_render_only_figure_step_intent(
                    source_step_id=parent.step_id,
                    figure_outputs=figure_outputs,
                ),
                inputs=list(parent.inputs),
                expected_outputs=figure_outputs,
                method=parent.method,
                icu_rule_refs=[*parent.icu_rule_refs, "visualization_rule"],
            ),
        ],
    )
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(legacy_plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Legacy framework-split analysis plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.research_manifest_partial/1",
                "run_id": run_id,
                "plan_path": plan_path.name,
                "per_step_records": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    migration_calls = 0
    original_migration = pipeline_module._migrate_legacy_resume_figure_render_edges

    def observed_migration(**kwargs):
        nonlocal migration_calls
        migration_calls += 1
        return original_migration(**kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "_migrate_legacy_resume_figure_render_edges",
        observed_migration,
    )

    class PlannerCountingLLM(ra.MockLLMClient):
        def __init__(self):
            super().__init__()
            self.planner_calls = 0

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next(
                (message.content for message in reversed(messages) if message.role == "user"),
                "",
            )
            upper = user.upper()
            if upper.startswith(
                (
                    "PRODUCE AN ICU-AWARE RESEARCH PLAN",
                    "REVISE THE ICU-AWARE RESEARCH PLAN",
                )
            ):
                self.planner_calls += 1
            return super().complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    llm = PlannerCountingLLM()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_publication_figure_skill=False,
        enable_latex=False,
        enable_memory=False,
        enable_probe_step=False,
        enable_replanning=False,
        enable_reviewer_round=False,
        enable_fairness_subgroups=False,
        enable_causal_audit=False,
        enable_reporting_checklist=False,
        runner_kind="subprocess",
    )
    result = pipeline.run(
        question=legacy_plan.research_question,
        cohort=synthetic_cohort,
        cohort_name="legacy_figure_edge",
        database="synthetic",
        target_outcome="death",
        resume_run_id=run_id,
        resume_from_step_id=parent.step_id,
        stop_after_step_id=parent.step_id,
        stop_after_analysis=True,
    )

    assert result.run_id == run_id
    assert migration_calls == 1
    assert llm.planner_calls == 0
    revision_path = run_dir / "analysis_plan_revision_4.json"
    assert revision_path.is_file()
    migrated = AnalysisPlan.model_validate_json(
        revision_path.read_text(encoding="utf-8")
    )
    assert migrated.revision == 4
    assert migrated.steps[1].inputs == ["table:cohort_summary"]
    assert migrated.steps[1].method == "visualization"

    revision_record = EvidenceStore(run_dir).get("analysis_plan_revision_4")
    assert revision_record is not None
    assert revision_record.metadata["reason"] == "resume_legacy_figure_render_edges"
    assert revision_record.metadata["target_step_ids"] == ["01_summary_figure"]
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    assert partial["plan_path"] == revision_path.name
    assert any(
        finding.get("validator") == "planner_schema_migration"
        and (finding.get("detail") or {}).get("kind")
        == "legacy_figure_render_edge"
        for finding in partial["findings"]
    )


def test_resume_to_nonexistent_run_id_starts_fresh_directory(ra, synthetic_cohort,
                                                             tmp_path: Path):
    """Passing a resume_run_id that has no prior run_dir should still
    work — the pipeline creates the directory and runs everything from
    scratch (the partial manifest is just absent)."""
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id="run_does_not_exist_yet",
    )
    assert result.run_id == "run_does_not_exist_yet"
    assert (Path(result.workdir) / "manifest.json").exists()
    assert (Path(result.workdir) / "manifest_partial.json").exists()


def test_final_manifest_keeps_step_records_for_metered_hosted_stub(ra, tmp_path: Path):
    """The final manifest must keep per-step resume records outside Mock paths.

    The real hosted path is wrapped in ``MeteredClient`` when cost
    tracking is enabled. This stub exercises that routing without a
    network call and pins the final ``manifest.json`` contract that
    paper/provenance tooling reads.
    """

    class HostedStubLLM:
        name = "hosted-stub"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps({
                    "research_question": "Is SOFA associated with ICU mortality?",
                    "steps": [{
                        "step_id": "04_primary_association",
                        "intent": "Estimate SOFA and ICU mortality association.",
                        "inputs": ["sofa2", "death"],
                        "expected_outputs": ["table:primary_association"],
                        "method": "descriptive",
                        "icu_rule_refs": ["aggregation_rule_for"],
                    }],
                    "rationale": "single-step hosted-stub resume test",
                })
            if "WRITE THE PYTHON CODE" in upper:
                return """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "predictor": "sofa2",
    "n": int(len(df)),
    "sofa2_median": float(df["sofa2"].median()),
    "mortality_rate": float(df["death"].mean()),
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "primary_association.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
            if "INTERPRET THE RESULTS" in upper:
                return "The primary association table is available {evidence:primary_association}."
            if "MANUSCRIPT SCAFFOLD" in upper:
                return "# Title\n\n## Results\n\nThe table is available {evidence:primary_association}."
            return "{}"

    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "sofa2": [0, 1, 3, 6],
        "death": [1, 0, 0, 1],
    })
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=HostedStubLLM(),
        enable_cost_tracking=True,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
    )
    result = pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="hosted_stub_resume_test",
        database="synthetic",
        target_outcome="death",
    )

    run_dir = Path(result.workdir)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    partial = json.loads((run_dir / "manifest_partial.json").read_text(encoding="utf-8"))

    assert manifest["per_step_records"], "final manifest dropped per-step records"
    assert manifest["per_step_records"] == partial["per_step_records"]
    assert manifest["per_step_records"][0]["status"] == "ok"
    assert manifest["cost_records"], "hosted-stub path should be metered"
