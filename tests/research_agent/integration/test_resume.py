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
    _apply_resume_plan_migrations,
    _load_compatible_resume_plan,
    _load_resume_state,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.plan_utils import _render_only_figure_step_intent
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    build_environment_identity,
    build_scientific_identity,
    invalidate_unverified_successful_steps,
    load_verified_run_input_capsule,
    prepare_existing_resume_input,
    seal_run_input_capsule,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    verified_run_evidence_path,
)
from easyicu.research_agent.authority.provider_budget import (
    StepProviderCallBudget,
    load_provider_call_budget_state,
    provider_call_budget_receipt_path,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, TimeWindow
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient


def _pattern_llm(
    *,
    plan: dict,
    code: str | BaseException,
    repairs: list[str | BaseException] | None = None,
    interpretation: str = "{}",
    manuscript: str = "{}",
) -> PatternScriptedMockLLMClient:
    """Build an exact trusted prompt router for resume integration tests."""

    plan_response = json.dumps(plan)
    repair_responses = list(repairs or [])
    return PatternScriptedMockLLMClient(
        [
            ("PRODUCE AN ICU-AWARE RESEARCH PLAN", [plan_response] * 8),
            ("WRITE THE PYTHON CODE FOR STEP", [code] * 8),
            ("REPAIR THE PYTHON CODE FOR STEP", repair_responses * 8),
            ("INTERPRET THE RESULTS OF STEP", [interpretation] * 8),
            ("WRITE A MANUSCRIPT SCAFFOLD", [manuscript] * 8),
        ]
    )


def _prompt_calls(
    client: PatternScriptedMockLLMClient,
    marker: str,
    *,
    full: bool = False,
) -> list[str]:
    folded_marker = marker.casefold()
    matched: list[str] = []
    for messages, _kwargs in client.calls:
        user = next(
            (
                str(message.content or "")
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        if folded_marker not in user.casefold():
            continue
        matched.append(
            "\n".join(str(message.content or "") for message in messages)
            if full
            else user
        )
    return matched


def _disable_article_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep resume fixtures focused on lifecycle semantics, not paper roles."""

    from easyicu.research_agent.agents.core import PlannerAgent

    original_run = PlannerAgent.run

    def run_without_article_contract(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_contract)


def _run_full(ra, synthetic_cohort, workdir: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=workdir, llm=ra.MockLLMClient())
    return pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="resume_test",
        database="synthetic",
        target_outcome="death",
    )


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
        "metadata_projection_sha256": "d" * 64,
        "metadata_sidecar_sha256": "c" * 64,
        "icu_rules_sha256": "b" * 64,
        "metadata_implementation_bundle_sha256": "a" * 64,
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
        "metadata_projection_sha256",
        "metadata_sidecar_sha256",
        "icu_rules_sha256",
        "metadata_implementation_bundle_sha256",
    } <= set(payload["changed_fields"])
    assert (run_dir / "research_context.json").read_bytes() == context_before
    assert (run_dir / "cohort.parquet").read_bytes() == cohort_before
    assert (run_dir / RUN_INPUT_CAPSULE_FILENAME).is_file()


def test_resume_rechecks_input_authority_immediately_before_receipt(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.authority import run_input as capsule_module

    pipeline, run_dir, run_kwargs = _write_capsule_resume_fixture(ra, tmp_path)
    real_invalidate = capsule_module.invalidate_unverified_successful_steps

    def corrupt_after_initial_verification(**kwargs):
        result = real_invalidate(**kwargs)
        (run_dir / "cohort.parquet").write_bytes(b"changed-after-verification")
        return result

    monkeypatch.setattr(
        capsule_module,
        "invalidate_unverified_successful_steps",
        corrupt_after_initial_verification,
    )

    with pytest.raises(RunInputIdentityError, match="staged cohort bytes"):
        pipeline.run(**run_kwargs)

    assert not list(run_dir.glob("resume_environment_receipt_*.json"))


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


def _register_resume_step_authorities(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    step_id: str,
    prefix: str,
    summary_inputs: tuple[str, ...] = (),
):
    """Create the three host-owned authorities of a completed step."""

    script_path = run_dir / f"{prefix}_analysis.py"
    summary_path = run_dir / f"{prefix}_step_summary.json"
    interpretation_path = run_dir / f"{prefix}_interpretation.md"
    script_path.write_text("print('analysis')\n", encoding="utf-8")
    summary_path.write_text('{"estimate": 1}', encoding="utf-8")
    interpretation_path.write_text("Evidence-bound interpretation.\n", encoding="utf-8")
    script = evidence.register_file(
        kind="code",
        description="Step script.",
        source_path=script_path,
        evidence_id=f"{prefix}_script",
        produced_by_step=step_id,
        producer="coder",
    )
    summary = evidence.register_file(
        kind="statistic",
        description="Step summary.",
        source_path=summary_path,
        evidence_id=f"{prefix}_summary",
        produced_by_step=step_id,
        inputs=list(summary_inputs),
        script_evidence_id=script.evidence_id,
        producer="runner",
    )
    interpretation = evidence.register_file(
        kind="log",
        description="Step interpretation.",
        source_path=interpretation_path,
        evidence_id=f"{prefix}_interpretation",
        produced_by_step=step_id,
        script_evidence_id=script.evidence_id,
        producer="analyzer",
    )
    checkpoint = {
        "step_id": step_id,
        "status": "ok",
        "evidence_ids": [
            script.evidence_id,
            summary.evidence_id,
            interpretation.evidence_id,
        ],
        "step_summary_evidence_id": summary.evidence_id,
        "script_evidence_id": script.evidence_id,
        "interpretation_evidence_id": interpretation.evidence_id,
    }
    return checkpoint, script, summary, interpretation


def _register_legacy_host_probe_authorities(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
):
    summary_path = run_dir / "probe_summary.json"
    table_path = run_dir / "probe_variable_profile.csv"
    summary_path.write_text('{"n": 4}', encoding="utf-8")
    table_path.write_text("variable,non_missing_n\nexposure,4\n", encoding="utf-8")
    summary = evidence.register_file(
        kind="statistic",
        description="Deterministic probe summary.",
        source_path=summary_path,
        evidence_id="statistic_probe_summary_fixture",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
    )
    table = evidence.register_file(
        kind="table",
        description="Deterministic probe variable profile.",
        source_path=table_path,
        evidence_id="table_probe_variable_profile_fixture",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
    )
    checkpoint = {
        "step_id": "00_probe",
        "status": "ok",
        "evidence_ids": [table.evidence_id, summary.evidence_id],
    }
    return checkpoint, summary, table


def _register_legacy_host_cohort_materializer_authority(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    step_id: str = "01_cohort_definition",
    evidence_id: str = "analysis_cohort_execute_repair",
    source_name: str = "cohort_analysis.parquet",
    produced_by_step: str | None = None,
):
    cohort_path = run_dir / source_name
    pd.DataFrame(
        {
            "stay_id": [1, 2],
            "exposure": [0, 1],
            "outcome": [0, 1],
        }
    ).to_parquet(cohort_path, index=False)
    cohort = evidence.register_file(
        kind="table",
        description="Host-materialized analysis cohort.",
        source_path=cohort_path,
        evidence_id=evidence_id,
        produced_by_step=produced_by_step or step_id,
        producer="cohort_repair",
        generation_mode="llm",
        metadata={"reason": "probe_summary"},
    )
    checkpoint = {
        "step_id": step_id,
        "status": "ok",
        "generation_mode": "deterministic_cohort_materializer",
        "step_summary": {
            "output_files": {"table:analysis_cohort": source_name},
            "n_universe": 4,
            "n_analysis_cohort": 2,
        },
        "evidence_ids": [cohort.evidence_id],
    }
    return checkpoint, cohort


def test_resume_migrates_missing_script_only_from_explicit_summary_authority(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy_script_authority"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, script, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="legacy",
    )
    checkpoint.pop("script_evidence_id")
    original = dict(checkpoint)
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {}
    assert checkpoint == original
    assert len(updated["per_step_records"]) == 2
    migrated = updated["per_step_records"][-1]
    assert migrated["status"] == "ok"
    assert migrated["script_evidence_id"] == script.evidence_id
    assert migrated["resume_authority_migrated_fields"] == ["script_evidence_id"]


def test_resume_does_not_migrate_missing_script_from_unlisted_or_ambiguous_code(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ambiguous_legacy_script_authority"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="legacy",
    )
    checkpoint.pop("script_evidence_id")
    decoy_path = run_dir / "decoy_analysis.py"
    decoy_path.write_text("print('decoy')\n", encoding="utf-8")
    decoy = evidence.register_file(
        kind="code",
        description="A second active code record must make migration ambiguous.",
        source_path=decoy_path,
        evidence_id="legacy_decoy_script",
        produced_by_step="01_model",
        producer="coder",
    )
    checkpoint["evidence_ids"].append(decoy.evidence_id)
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_model": "successful checkpoint is missing required script_evidence_id"
    }
    assert len(updated["per_step_records"]) == 2
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_migrates_exact_host_owned_probe_without_script_or_analyzer(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy_host_probe"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, summary, table = _register_legacy_host_probe_authorities(
        run_dir=run_dir,
        evidence=evidence,
    )
    original = dict(checkpoint)
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {}
    assert checkpoint == original
    migrated = updated["per_step_records"][-1]
    assert migrated["step_authority_kind"] == "host_deterministic_probe"
    assert migrated["probe_summary_evidence_id"] == summary.evidence_id
    assert migrated["probe_table_evidence_id"] == table.evidence_id
    assert "script_evidence_id" not in migrated
    assert "interpretation_evidence_id" not in migrated


def test_resume_probe_rejects_arbitrary_table_as_host_authority(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "invalid_host_probe"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, table = _register_legacy_host_probe_authorities(
        run_dir=run_dir,
        evidence=evidence,
    )
    wrong_path = run_dir / "unrelated_probe_table.csv"
    wrong_path.write_text("variable,n\nexposure,4\n", encoding="utf-8")
    wrong = evidence.register_file(
        kind="table",
        description="Unrelated deterministic table.",
        source_path=wrong_path,
        evidence_id="table_unrelated_probe_fixture",
        produced_by_step="00_probe",
        producer="pipeline",
        generation_mode="deterministic_probe",
    )
    checkpoint["evidence_ids"] = [
        wrong.evidence_id if value == table.evidence_id else value
        for value in checkpoint["evidence_ids"]
    ]
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "00_probe": ("successful host probe checkpoint lacks migrated probe authority")
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_migrates_exact_host_owned_cohort_materializer_without_script(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy_host_cohort_materializer"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, cohort = _register_legacy_host_cohort_materializer_authority(
        run_dir=run_dir,
        evidence=evidence,
    )
    original = dict(checkpoint)
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {}
    assert checkpoint == original
    assert len(updated["per_step_records"]) == 2
    migrated = updated["per_step_records"][-1]
    assert migrated["step_authority_kind"] == "host_deterministic_cohort_materializer"
    assert migrated["cohort_table_evidence_id"] == cohort.evidence_id
    assert "step_summary_evidence_id" not in migrated
    assert "script_evidence_id" not in migrated
    assert "interpretation_evidence_id" not in migrated
    replayed, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=updated,
        records=records,
    )
    assert invalidated == {}
    assert len(replayed["per_step_records"]) == 2


@pytest.mark.parametrize(
    ("variant", "fixture_overrides"),
    [
        (
            "arbitrary_table",
            {
                "evidence_id": "unrelated_deterministic_table",
                "source_name": "unrelated.parquet",
            },
        ),
        ("cross_step", {"produced_by_step": "99_other_step"}),
    ],
)
def test_resume_cohort_materializer_rejects_non_authoritative_table(
    tmp_path: Path,
    variant: str,
    fixture_overrides: dict[str, str],
) -> None:
    run_dir = tmp_path / variant
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _ = _register_legacy_host_cohort_materializer_authority(
        run_dir=run_dir,
        evidence=evidence,
        **fixture_overrides,
    )
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_cohort_definition": (
            "successful host cohort materializer checkpoint lacks migrated "
            "cohort authority"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_does_not_generalize_script_free_authority_to_other_deterministic_modes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "unrecognized_deterministic_mode"
    run_dir.mkdir()
    source_path = run_dir / "other_deterministic_table.csv"
    source_path.write_text("group,n\na,2\n", encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    table = evidence.register_file(
        kind="table",
        description="Unrecognized deterministic output.",
        source_path=source_path,
        evidence_id="other_deterministic_table",
        produced_by_step="01_other",
        producer="pipeline",
        generation_mode="deterministic_other",
    )
    checkpoint = {
        "step_id": "01_other",
        "status": "ok",
        "generation_mode": "deterministic_other",
        "evidence_ids": [table.evidence_id],
    }
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_other": (
            "successful checkpoint is missing required step_summary_evidence_id"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_cohort_materializer_rejects_tampered_authority(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "tampered_host_cohort_materializer"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, cohort = _register_legacy_host_cohort_materializer_authority(
        run_dir=run_dir,
        evidence=evidence,
    )
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }
    migrated_state, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )
    assert invalidated == {}
    Path(verified_run_evidence_path(run_dir, cohort)).write_bytes(b"tampered")

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=migrated_state,
        records=records,
    )

    assert invalidated == {
        "01_cohort_definition": (
            "evidence analysis_cohort_execute_repair failed path/digest verification"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_cohort_materializer_rejects_checkpoint_count_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "count_mismatch_host_cohort_materializer"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _ = _register_legacy_host_cohort_materializer_authority(
        run_dir=run_dir,
        evidence=evidence,
    )
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }
    migrated_state, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )
    assert invalidated == {}
    migrated = json.loads(json.dumps(migrated_state["per_step_records"][-1]))
    migrated["step_summary"]["n_analysis_cohort"] = 1

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [migrated], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_cohort_definition": (
            "successful host cohort materializer canonical cohort row count 2 "
            "does not match checkpoint 1"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_cohort_materializer_rejects_canonical_cohort_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "canonical_drift_host_cohort_materializer"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _ = _register_legacy_host_cohort_materializer_authority(
        run_dir=run_dir,
        evidence=evidence,
    )
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }
    migrated_state, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )
    assert invalidated == {}
    (run_dir / "cohort_analysis.parquet").write_bytes(b"tampered canonical")

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state=migrated_state,
        records=records,
    )

    assert invalidated == {
        "01_cohort_definition": (
            "successful host cohort materializer canonical cohort differs "
            "from sealed evidence"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_invalidates_only_successes_with_unverified_evidence(tmp_path: Path):
    run_dir = tmp_path / "run_evidence_invalidation"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint_a, _, record_a, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_a",
        prefix="step_a",
    )
    checkpoint_b, _, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="02_b",
        prefix="step_b",
    )
    state = {
        "per_step_records": [checkpoint_a, checkpoint_b],
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

    latest = {record["step_id"]: record for record in updated["per_step_records"]}
    assert invalidated == {
        "01_a": "evidence step_a_summary failed path/digest verification"
    }
    assert latest["01_a"]["status"] == "resume_evidence_invalid"
    assert latest["02_b"]["status"] == "ok"


def test_resume_cannot_hide_tampered_summary_by_dropping_its_required_id(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_omitted_summary_authority"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, summary, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="step",
    )
    Path(verified_run_evidence_path(run_dir, summary)).write_text(
        '{"estimate": 999}',
        encoding="utf-8",
    )
    checkpoint["evidence_ids"] = [
        evidence_id
        for evidence_id in checkpoint["evidence_ids"]
        if evidence_id != summary.evidence_id
    ]
    state = {
        # A mutable checkpoint cannot hide the corrupt summary by retaining
        # only the other still-valid authorities in its evidence set.
        "per_step_records": [checkpoint],
        "findings": [],
    }
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

    assert invalidated == {
        "01_model": (
            "successful checkpoint step_summary_evidence_id step_summary "
            "is absent from evidence_ids"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


@pytest.mark.parametrize(
    "authority_field",
    [
        "step_summary_evidence_id",
        "script_evidence_id",
        "interpretation_evidence_id",
    ],
)
def test_resume_requires_each_explicit_step_authority_in_evidence_ids(
    tmp_path: Path,
    authority_field: str,
) -> None:
    run_dir = tmp_path / authority_field
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="required",
    )
    authority_id = checkpoint[authority_field]
    checkpoint["evidence_ids"].remove(authority_id)
    state = {
        "per_step_records": [checkpoint],
        "findings": [],
    }
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

    assert invalidated == {
        "01_model": (
            f"successful checkpoint {authority_field} {authority_id} "
            "is absent from evidence_ids"
        )
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_run_level_evidence_cannot_substitute_for_required_step_authority(
    tmp_path: Path,
) -> None:
    missing_field = "step_summary_evidence_id"
    run_dir = tmp_path / missing_field
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="model",
    )
    checkpoint.pop(missing_field)
    run_receipt_path = run_dir / "run_receipt.json"
    run_receipt_path.write_text('{"status": "ok"}', encoding="utf-8")
    run_receipt = evidence.register_file(
        kind="statistic",
        description="Digest-valid run-level receipt, not a step authority.",
        source_path=run_receipt_path,
        evidence_id="run_receipt",
    )
    checkpoint["evidence_ids"].append(run_receipt.evidence_id)
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_model": f"successful checkpoint is missing required {missing_field}"
    }
    assert updated["per_step_records"][-1]["status"] == "resume_evidence_invalid"


def test_resume_explicit_run_level_receipt_cannot_impersonate_step_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "explicit_run_level_substitution"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, summary, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="model",
    )
    run_receipt_path = run_dir / "run_receipt.json"
    run_receipt_path.write_text('{"status": "ok"}', encoding="utf-8")
    run_receipt = evidence.register_file(
        kind="statistic",
        description="Run-level receipt.",
        source_path=run_receipt_path,
        evidence_id="run_receipt",
    )
    checkpoint["step_summary_evidence_id"] = run_receipt.evidence_id
    checkpoint["evidence_ids"].remove(summary.evidence_id)
    checkpoint["evidence_ids"].append(run_receipt.evidence_id)
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    _, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_model": (
            "successful checkpoint step_summary_evidence_id run_receipt "
            "is not owned by step 01_model"
        )
    }


def test_resume_requires_interpretation_field_when_analyzer_evidence_exists(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "omitted_interpretation_field"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    checkpoint, _, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_model",
        prefix="model",
    )
    checkpoint.pop("interpretation_evidence_id")
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    _, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={"per_step_records": [checkpoint], "findings": []},
        records=records,
    )

    assert invalidated == {
        "01_model": (
            "successful checkpoint is missing required interpretation_evidence_id"
        )
    }


def test_resume_invalidates_downstream_success_when_upstream_evidence_is_bad(
    tmp_path: Path,
):
    run_dir = tmp_path / "run_upstream_evidence_invalidation"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    upstream_checkpoint, _, upstream, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_upstream",
        prefix="upstream",
    )
    downstream_checkpoint, _, downstream, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="02_downstream",
        prefix="downstream",
        summary_inputs=(upstream.evidence_id,),
    )
    state = {
        "per_step_records": [upstream_checkpoint, downstream_checkpoint],
        "findings": [],
    }
    Path(verified_run_evidence_path(run_dir, upstream)).unlink()
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

    assert set(invalidated) == {"01_upstream", "02_downstream"}
    assert (
        "upstream_summary failed path/digest verification"
        in invalidated["02_downstream"]
    )
    latest = {record["step_id"]: record for record in updated["per_step_records"]}
    assert latest["01_upstream"]["status"] == "resume_evidence_invalid"
    assert latest["02_downstream"]["status"] == "resume_evidence_invalid"


def test_resume_transitively_invalidates_intact_downstream_authorities(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_transitive_checkpoint_invalidation"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    upstream_checkpoint, _, upstream_summary, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="01_upstream",
        prefix="upstream",
    )
    middle_checkpoint, _, middle_summary, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="02_middle",
        prefix="middle",
        summary_inputs=(upstream_summary.evidence_id,),
    )
    downstream_checkpoint, _, _, _ = _register_resume_step_authorities(
        run_dir=run_dir,
        evidence=evidence,
        step_id="03_downstream",
        prefix="downstream",
        summary_inputs=(middle_summary.evidence_id,),
    )
    # Only checkpoint metadata is damaged. Every evidence blob and every
    # digest-bound input closure remains readable, so downstream invalidation
    # must follow producer authority rather than another file-integrity error.
    # Remove an authority that cannot be reconstructed from another explicit
    # checkpoint field.  A missing script field alone is a supported legacy
    # migration when the exact summary already binds the listed code evidence.
    upstream_checkpoint.pop("step_summary_evidence_id")
    records = {
        record["evidence_id"]: record
        for record in json.loads(
            (run_dir / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
        )
    }

    updated, invalidated = invalidate_unverified_successful_steps(
        run_dir=run_dir,
        resume_state={
            "per_step_records": [
                upstream_checkpoint,
                middle_checkpoint,
                downstream_checkpoint,
            ],
            "findings": [],
        },
        records=records,
    )

    assert invalidated == {
        "01_upstream": (
            "successful checkpoint is missing required step_summary_evidence_id"
        ),
        "02_middle": (
            "successful checkpoint depends on invalidated step 01_upstream "
            f"via evidence {upstream_summary.evidence_id}"
        ),
        "03_downstream": (
            "successful checkpoint depends on invalidated step 02_middle "
            f"via evidence {middle_summary.evidence_id}"
        ),
    }
    current = {record["step_id"]: record for record in updated["per_step_records"]}
    assert {
        current[step_id]["status"]
        for step_id in ("01_upstream", "02_middle", "03_downstream")
    } == {"resume_evidence_invalid"}


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
    partial["evidence"] = [item.model_dump(mode="json") for item in evidence.records()]
    partial_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
    Path(verified_run_evidence_path(run_dir, record)).unlink()

    observed = {}

    def capture_resume_state(**kwargs):
        observed.update(kwargs["resume_state"])
        raise RuntimeError("captured invalidated resume state")

    monkeypatch.setattr(pipeline, "_run_plan_phase", capture_resume_state)
    with pytest.raises(RuntimeError, match="captured invalidated"):
        pipeline.run(**run_kwargs)

    current = {item["step_id"]: item for item in observed["per_step_records"]}
    assert current["01_summary"]["status"] == "resume_evidence_invalid"
    assert any(
        finding.get("validator") == "resume_evidence_integrity"
        for finding in observed["findings"]
    )


@pytest.mark.parametrize("omit_outcome_roster", [False, True])
def test_legacy_completed_resume_is_adopted_only_from_verified_context_and_cohort(
    ra,
    tmp_path: Path,
    omit_outcome_roster: bool,
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

    capsule_record = evidence.get("run_input_capsule")
    assert capsule_record is not None
    capsule_evidence_path = run_dir / capsule_record.relative_path
    # Emulate a completed run created before RunInputCapsule existed while
    # keeping the modern evidence ledger authoritative. Flat projection edits
    # cannot remove selected evidence once a full-state generation exists.
    with evidence._lock:
        evidence._records = [
            record
            for record in evidence._records
            if record.evidence_id != "run_input_capsule"
        ]
        evidence._aliases = {
            alias: evidence_id
            for alias, evidence_id in evidence._aliases.items()
            if evidence_id != "run_input_capsule"
        }
        evidence._save()
    capsule_evidence_path.unlink()
    (run_dir / RUN_INPUT_CAPSULE_FILENAME).unlink()

    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
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
    if omit_outcome_roster:
        scientific_identity["outcome_columns"] = []
        with pytest.raises(RunInputIdentityError, match="outcome_columns"):
            prepare_existing_resume_input(
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
        assert not (run_dir / RUN_INPUT_CAPSULE_FILENAME).exists()
        return
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
    from easyicu.research_agent.authority.plan_scope import (
        _serializable_plan_scientific_scope_signature,
    )

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
    original_path.write_text(original.model_dump_json(indent=2), encoding="utf-8")
    revision_path = run_dir / "analysis_plan_revision_2.json"
    revision_path.write_text(revision.model_dump_json(indent=2), encoding="utf-8")
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
        "per_step_records": [{"step_id": "00_probe", "status": "ok"}]
        + [
            {
                "step_id": step.step_id,
                "status": "ok",
                "planned_analysis_role": step.planned_analysis_role,
                "analysis_request": {"step": step.model_dump(mode="json")},
                "plan_scientific_signature": (
                    _serializable_plan_scientific_scope_signature(revision)
                ),
            }
            for step in revision.steps
            if step.step_id in {"01_cohort", "02_table"}
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
    from easyicu.research_agent.authority.plan_scope import (
        _serializable_plan_scientific_scope_signature,
    )

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
            {
                "step_id": "01_current_success",
                "status": "ok",
                "planned_analysis_role": plan.steps[0].planned_analysis_role,
                "analysis_request": {"step": plan.steps[0].model_dump(mode="json")},
                "plan_scientific_signature": (
                    _serializable_plan_scientific_scope_signature(plan)
                ),
            },
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


def test_resume_prefers_review_bound_plan_after_rejected_newer_migration(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.plan_scope import (
        _serializable_plan_scientific_scope_signature,
    )
    from easyicu.research_agent.canonical_json import canonical_sha256

    run_dir = tmp_path / "run_review_bound_resume"
    run_dir.mkdir()
    reviewed = AnalysisPlan(
        research_question="Resume the approved plan.",
        steps=[
            AnalysisStep(
                step_id="01_complete",
                intent="Completed scientific work.",
                expected_outputs=["table:result"],
            ),
            AnalysisStep(
                step_id="02_display",
                planned_analysis_role="auxiliary",
                intent="Render the approved result.",
                method="visualization",
                inputs=["table:result"],
                expected_outputs=["figure:result"],
            ),
        ],
    )
    unreviewed = reviewed.model_copy(
        update={
            "revision": 3,
            "steps": [
                reviewed.steps[0],
                reviewed.steps[1].model_copy(
                    update={"inputs": ["table:result", "table:extra"]}
                ),
            ],
        }
    )
    reviewed_path = run_dir / "analysis_plan.json"
    unreviewed_path = run_dir / "analysis_plan_revision_3.json"
    review_path = run_dir / "scientific_plan_review.json"
    reviewed_path.write_text(reviewed.model_dump_json(indent=2), encoding="utf-8")
    unreviewed_path.write_text(unreviewed.model_dump_json(indent=2), encoding="utf-8")
    review_path.write_text(
        json.dumps(
            {"plan_sha256": canonical_sha256(reviewed.model_dump(mode="json"))}
        ),
        encoding="utf-8",
    )
    evidence = EvidenceStore(run_dir)
    reviewed_record = evidence.register_file(
        kind="log",
        description="Reviewed plan.",
        source_path=reviewed_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    evidence.register_file(
        kind="log",
        description="Rejected newer migration.",
        source_path=unreviewed_path,
        evidence_id="analysis_plan_revision_3",
        producer="runtime_supervisor",
        generation_mode="system",
    )
    evidence.register_file(
        kind="log",
        description="Exact scientific review binding.",
        source_path=review_path,
        evidence_id="scientific_plan_review",
        producer="plan_scientific_review",
        generation_mode="deterministic_skill",
    )
    resume_state = {
        "per_step_records": [
            {
                "step_id": "01_complete",
                "status": "ok",
                "planned_analysis_role": reviewed.steps[0].planned_analysis_role,
                "analysis_request": {
                    "step": reviewed.steps[0].model_dump(mode="json")
                },
                "plan_scientific_signature": (
                    _serializable_plan_scientific_scope_signature(reviewed)
                ),
            }
        ]
    }

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state=resume_state,
    )

    assert selected == reviewed
    assert selected_path == verified_run_evidence_path(run_dir, reviewed_record)


def test_review_bound_crash_resume_does_not_mutate_the_plan(tmp_path: Path) -> None:
    from easyicu.research_agent.canonical_json import canonical_sha256

    plan = AnalysisPlan(
        research_question="Resume exactly what the user approved.",
        steps=[
            AnalysisStep(
                step_id="01_display",
                planned_analysis_role="auxiliary",
                intent="Render the reviewed table.",
                method="visualization",
                inputs=["table:result"],
                expected_outputs=["figure:result"],
            )
        ],
    )
    review_path = tmp_path / "scientific_plan_review.json"
    review_path.write_text(
        json.dumps({"plan_sha256": canonical_sha256(plan.model_dump(mode="json"))}),
        encoding="utf-8",
    )
    evidence = EvidenceStore(tmp_path)
    evidence.register_file(
        kind="log",
        description="Exact scientific review binding.",
        source_path=review_path,
        evidence_id="scientific_plan_review",
        producer="plan_scientific_review",
        generation_mode="deterministic_skill",
    )
    findings = []

    resumed, migrated_path, mode = _apply_resume_plan_migrations(
        plan=plan,
        agent_context=None,
        run_dir=tmp_path,
        resume_state={"per_step_records": []},
        resume_from_step_id="01_display",
        role_resolver=lambda _role: None,
        evidence=evidence,
        prompt_version="test",
        llm_signature="mock",
        max_prompt_tokens=None,
        submission_profile_name=None,
        plan_generation_mode="resumed",
        migrated_plan_path=None,
        findings=findings,
        scientific_runtime_authorities=None,
    )

    assert resumed == plan
    assert migrated_path is None
    assert mode == "resumed"
    assert findings[-1].detail["reason"] == "review_bound_resume_plan_preserved"


def test_resume_plan_skips_newest_revision_without_locked_cohort(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        write_locked_cohort_definition,
    )

    run_dir = tmp_path / "run_locked_cohort_revision"
    run_dir.mkdir()
    original = AnalysisPlan(
        research_question="Resume the locked cohort.",
        cohort=CohortDefinition(name="locked_primary_cohort"),
        steps=[
            AnalysisStep(
                step_id="01_summary",
                intent="Summarize the locked cohort.",
                expected_outputs=["table:summary"],
            )
        ],
    )
    drifted_revision = original.model_copy(update={"cohort": None, "revision": 2})
    original_path = run_dir / "analysis_plan.json"
    revision_path = run_dir / "analysis_plan_revision_2.json"
    original_path.write_text(original.model_dump_json(indent=2), encoding="utf-8")
    revision_path.write_text(
        drifted_revision.model_dump_json(indent=2), encoding="utf-8"
    )
    evidence = EvidenceStore(run_dir)
    original_record = evidence.register_file(
        kind="log",
        description="Original plan with locked cohort.",
        source_path=original_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    evidence.register_file(
        kind="log",
        description="Incomplete replan without cohort authority.",
        source_path=revision_path,
        evidence_id="analysis_plan_revision_2",
        producer="replanner",
        generation_mode="llm",
    )
    write_locked_cohort_definition(
        run_dir=run_dir,
        plan=original,
        evidence=evidence,
        prompt_pack_version=None,
        llm_signature="test",
    )

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state={"per_step_records": []},
    )

    assert selected == original
    assert selected_path == verified_run_evidence_path(run_dir, original_record)


def test_resume_plan_accepts_implicit_primary_cohort_lock(tmp_path: Path) -> None:
    from easyicu.research_agent.cohort.schema import write_locked_cohort_definition
    from easyicu.research_agent.authority.plan_scope import (
        _serializable_plan_scientific_scope_signature,
    )

    run_dir = tmp_path / "run_implicit_primary_cohort"
    run_dir.mkdir()
    step = AnalysisStep(
        step_id="01_summary",
        intent="Summarize the canonical implicit primary cohort.",
        planned_analysis_role="auxiliary",
        expected_outputs=["table:summary"],
    )
    plan = AnalysisPlan(
        research_question="Resume the implicit primary cohort.",
        cohort=None,
        steps=[step],
    )
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    plan_record = evidence.register_file(
        kind="log",
        description="Plan using the canonical implicit primary cohort.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    write_locked_cohort_definition(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
        prompt_pack_version=None,
        llm_signature="test",
    )

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state={
            "per_step_records": [
                {
                    "step_id": step.step_id,
                    "status": "ok",
                    "planned_analysis_role": step.planned_analysis_role,
                    "analysis_request": {"step": step.model_dump(mode="json")},
                    "plan_scientific_signature": (
                        _serializable_plan_scientific_scope_signature(plan)
                    ),
                }
            ]
        },
    )

    assert selected == plan
    assert selected_path == verified_run_evidence_path(run_dir, plan_record)


def test_resume_plan_rejects_completed_role_mismatch(tmp_path: Path) -> None:
    from easyicu.research_agent.authority.plan_scope import (
        _serializable_plan_scientific_scope_signature,
    )

    run_dir = tmp_path / "run_role_mismatch"
    run_dir.mkdir()
    saved_step = AnalysisStep(
        step_id="01_model",
        intent="Estimate the result.",
        planned_analysis_role="auxiliary",
        expected_outputs=["table:estimate"],
    )
    saved_plan = AnalysisPlan(research_question="q", steps=[saved_step])
    path = run_dir / "analysis_plan.json"
    path.write_text(saved_plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Saved plan.",
        source_path=path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    executed_step = saved_step.model_copy(update={"planned_analysis_role": "primary"})
    executed_plan = saved_plan.model_copy(update={"steps": [executed_step]})
    selected, selected_path = _load_compatible_resume_plan(
        run_dir=run_dir,
        resume_state={
            "per_step_records": [
                {
                    "step_id": executed_step.step_id,
                    "status": "ok",
                    "planned_analysis_role": "primary",
                    "analysis_request": {"step": executed_step.model_dump(mode="json")},
                    "plan_scientific_signature": (
                        _serializable_plan_scientific_scope_signature(executed_plan)
                    ),
                }
            ]
        },
    )
    assert selected is None
    assert selected_path is None


def test_implicit_resume_offers_only_latest_contract_failed_code_once(
    tmp_path: Path,
) -> None:
    """Normal resume may replay one exact failed-contract script, not history."""

    from easyicu.research_agent.execution.phase import (
        _failed_contract_code_can_be_reused_before_coder,
        _serializable_plan_scientific_scope_signature,
    )
    from easyicu.research_agent.orchestration.resume import ResumeController

    step = AnalysisStep(
        step_id="01_summary",
        intent="Summarize the declared cohort.",
        inputs=["stay_id"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )
    plan = AnalysisPlan(
        research_question="Summarize this ICU cohort.",
        steps=[step],
    )
    run_dir = tmp_path / "run_implicit_contract_reuse"
    run_dir.mkdir()
    source_path = tmp_path / "implicit_contract_candidate.py"
    code = "import pandas as pd\nprint(pd.__version__)\n"
    source_path.write_text(code, encoding="utf-8")
    evidence_record = EvidenceStore(run_dir).register_file(
        kind="code",
        description="Agent-generated candidate for implicit resume.",
        source_path=source_path,
        evidence_id="code_01_summary",
        produced_by_step=step.step_id,
        producer="coder",
        generation_mode="llm",
    )
    evidence_payload = evidence_record.model_dump(mode="json")
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    resolved_inputs_sha256 = "a" * 64
    run_input_capsule_sha256 = "b" * 64
    failed_record = {
        "step_id": step.step_id,
        "status": "contract_failed",
        "returncode": 0,
        "timed_out": False,
        "outputs_safe_to_collect": True,
        "executed_code_sha256": digest,
        "concept_approved_code_sha256": digest,
        "script_evidence_id": evidence_record.evidence_id,
        "resolved_inputs_sha256": resolved_inputs_sha256,
        "run_input_capsule_sha256": run_input_capsule_sha256,
        "plan_scientific_signature": (
            _serializable_plan_scientific_scope_signature(plan)
        ),
        "analysis_request": {"step": step.model_dump(mode="json")},
    }

    def controller(records):
        return ResumeController(
            plan=plan,
            run_dir=run_dir,
            resume_state={
                "per_step_records": records,
                "evidence": [evidence_payload],
            },
        )

    candidate = controller([failed_record]).prior_code_for_step(step.step_id)
    assert candidate is not None
    assert candidate[0] == code
    assert _failed_contract_code_can_be_reused_before_coder(
        prior_step_record=failed_record,
        resumed_code=candidate,
        step=step,
        plan=plan,
        resolved_inputs_sha256=resolved_inputs_sha256,
        run_input_capsule_sha256=run_input_capsule_sha256,
    )

    replayed_record = dict(
        failed_record,
        resumed_failed_contract_code_preflight=True,
    )
    assert not _failed_contract_code_can_be_reused_before_coder(
        prior_step_record=replayed_record,
        resumed_code=candidate,
        step=step,
        plan=plan,
        resolved_inputs_sha256=resolved_inputs_sha256,
        run_input_capsule_sha256=run_input_capsule_sha256,
    )

    assert (
        controller(
            [failed_record, {"step_id": step.step_id, "status": "coder_failed"}]
        ).prior_code_for_step(step.step_id)
        is None
    )
    assert (
        controller([{"step_id": step.step_id, "status": "ok"}]).prior_code_for_step(
            step.step_id
        )
        is None
    )


def test_explicit_resume_window_marks_selected_and_downstream_steps(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.orchestration.resume import ResumeController

    steps = [
        AnalysisStep(
            step_id=step_id,
            intent=f"Execute {step_id}.",
            inputs=["stay_id"],
            expected_outputs=[f"table:{step_id}"],
            method="descriptive_summary",
        )
        for step_id in ("01_first", "02_selected", "03_downstream")
    ]
    controller = ResumeController(
        plan=AnalysisPlan(research_question="Test explicit resume.", steps=steps),
        run_dir=tmp_path,
        resume_state={"per_step_records": []},
        resume_from_step_id="02_selected",
    )

    assert controller.explicitly_reruns_step("01_first") is False
    assert controller.explicitly_reruns_step("02_selected") is True
    assert controller.explicitly_reruns_step("03_downstream") is True


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
    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=PatternScriptedMockLLMClient(
            [
                (
                    "INTERPRET THE RESULTS",
                    [KeyboardInterrupt("simulate interruption after runner outputs")],
                )
            ],
            contextual_default=True,
        ),
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
    partial = json.loads(
        (run_dirs[-1] / "manifest_partial.json").read_text(encoding="utf-8")
    )
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

    partial_after = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
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
            i
            for i, record in enumerate(records)
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


@pytest.mark.parametrize(
    (
        "reuse_step_code",
        "mark_prior_contract_failed",
        "explicit_resume",
    ),
    [
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, True, False),
    ],
)
def test_resume_from_step_prefers_verified_capsule_over_legacy_code_reuse(
    ra,
    tmp_path: Path,
    monkeypatch,
    reuse_step_code: bool,
    mark_prior_contract_failed: bool,
    explicit_resume: bool,
):
    """A verified capsule outranks opt-in legacy code-evidence recovery."""

    _disable_article_contract(monkeypatch)
    plan = {
        "research_question": "Is SOFA associated with ICU mortality?",
        "steps": [
            {
                "step_id": "04_primary_association",
                "planned_analysis_role": "primary",
                "intent": "Estimate SOFA and ICU mortality association.",
                "inputs": ["sofa2", "death"],
                "expected_outputs": ["table:cohort_summary"],
                "method": "descriptive",
                "icu_rule_refs": ["aggregation_rule_for"],
            }
        ],
        "rationale": "single-step resume code reuse test",
    }
    code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
with open(
    os.environ["EASYICU_RESOLVED_INPUTS_JSON"], "r", encoding="utf-8"
) as f:
    resolved = json.load(f)
plausibility_audit = {}
for column, contract in resolved["raw_input_contracts"]["contracts"].items():
    bounds = contract.get("analysis_plausibility_range")
    if bounds is None:
        continue
    numeric = pd.to_numeric(df[column], errors="coerce")
    lower = bounds.get("minimum")
    upper = bounds.get("maximum")
    below_n = int((numeric < lower).sum()) if lower is not None else 0
    above_n = int((numeric > upper).sum()) if upper is not None else 0
    plausibility_audit[column] = {
        "policy": "retain_and_flag",
        "below_minimum_n": below_n,
        "above_maximum_n": above_n,
        "out_of_range_n": below_n + above_n,
    }
summary = {
    "predictor": "sofa2",
    "n": int(len(df)),
    "sofa2_median": float(df["sofa2"].median()),
    "mortality_rate": float(df["death"].mean()),
    "plausibility_audit": plausibility_audit,
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
summary["output_files"] = {
    "table:cohort_summary": "cohort_summary.csv"
}
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""

    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "sofa2": [0, 1, 3, 6],
            "death": [1, 0, 0, 1],
        }
    )
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=_pattern_llm(
            plan=plan,
            code=code,
            interpretation=("The cohort table is available {evidence:cohort_summary}."),
            manuscript=(
                "# Title\n\n## Results\n\n"
                "The table is available {evidence:cohort_summary}."
            ),
        ),
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        max_step_provider_calls=20,
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
    if mark_prior_contract_failed:
        for ledger_name in ("per_step_records", "step_attempt_history"):
            prior_record = next(
                record
                for record in reversed(partial.get(ledger_name, []))
                if record.get("step_id") == "04_primary_association"
            )
            prior_record["status"] = "contract_failed"
    bad_code_path = run_dir / "evidence" / "code_bad__analysis.py"
    bad_code_path.write_text("{}", encoding="utf-8")
    partial["evidence"].append(
        {
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
        }
    )
    partial.setdefault("findings", []).extend(
        [
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
        ]
    )
    partial_path.write_text(
        json.dumps(partial, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.delenv("EASYICU_RESUME_REUSE_STEP_CODE", raising=False)
    if reuse_step_code:
        monkeypatch.setenv("EASYICU_RESUME_REUSE_STEP_CODE", "1")
    second_llm = _pattern_llm(
        plan=plan,
        code=RuntimeError("simulated coder outage"),
    )
    second_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=second_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        max_step_provider_calls=20,
    )
    second = second_pipeline.run(
        question="Is SOFA associated with ICU mortality?",
        cohort=cohort,
        cohort_name="resume_code_reuse_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id=("04_primary_association" if explicit_resume else None),
        stop_after_step_id="04_primary_association",
        stop_after_analysis=True,
    )
    assert second.run_id == first.run_id
    assert _prompt_calls(second_llm, "WRITE THE PYTHON CODE") == []

    partial = json.loads(
        (Path(second.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in partial["per_step_records"]
        if record.get("step_id") == "04_primary_association"
    ]
    assert records[-1]["status"] == "ok"
    if explicit_resume:
        assert records[-1]["step_authority_capsule_reused"] is True
    assert "resumed_code_evidence_id" not in records[-1]
    final_code_records = [
        record
        for record in partial["evidence"]
        if record.get("kind") == "code"
        and record.get("produced_by_step") == "04_primary_association"
        and record.get("generation_mode") == "resumed_code_reuse"
    ]
    if explicit_resume:
        assert len(final_code_records) == 1
        assert records[-1]["script_evidence_id"] == final_code_records[0]["evidence_id"]
    else:
        assert final_code_records == []
    assert records[-1]["script_evidence_id"] != "code_bad"
    if explicit_resume:
        assert not any(
            "stale pre-resume" in finding.get("message", "")
            for finding in partial["findings"]
        )
    assert records[-1].get("resumed_failed_contract_code_preflight") is None


def test_concept_repair_failure_resumes_quarantined_draft_fail_closed(
    ra, tmp_path: Path, monkeypatch
):
    """A rejected draft is repaired on resume, never reused as executable code."""

    _disable_article_contract(monkeypatch)
    from easyicu.research_agent.audits.validators import LLMConceptAuditor
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.runner import CodeRunner
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_evidence_records,
        current_successful_step_ids,
    )

    audit_state = {"emit_error": True, "reject_marker": None}
    persisted_message = "Displayed percentage is not reconciled to its denominator."

    def fake_audit(
        self,
        *,
        context,
        script_text,
        step,
        provider_budget=None,
        study_endpoint=None,
        plan_step_roster=None,
    ):
        # The declaration arguments are named rather than swallowed by `**_`.
        # A double that accepts anything would have kept this test green while
        # the real call site passed a keyword the auditor never read; naming them
        # is what made the mismatch visible when the interface grew.
        del self, context, study_endpoint, plan_step_roster
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
    runner_calls: list[str] = []
    original_run = CodeRunner.run

    def recording_run(self, *, step_id, code, resolved_inputs_path=None):
        runner_calls.append(step_id)
        return original_run(
            self,
            step_id=step_id,
            code=code,
            resolved_inputs_path=resolved_inputs_path,
        )

    monkeypatch.setattr(CodeRunner, "run", recording_run)

    draft_code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {
    "n": int(len(df)),
    "draft_marker": True,
    "output_files": {
        "table:cohort_summary": "cohort_summary.csv"
    },
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "cohort_summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
    repaired_code = draft_code.replace(
        '"draft_marker": True', '"status": "ok", "repaired_marker": True'
    )

    plan = {
        "research_question": "Summarize the ICU cohort.",
        "steps": [
            {
                "step_id": "01_summary",
                "planned_analysis_role": "auxiliary",
                "intent": "Produce a descriptive cohort summary.",
                "inputs": ["stay_id"],
                "expected_outputs": ["table:cohort_summary"],
                "method": "descriptive_summary",
                "icu_rule_refs": [],
            }
        ],
        "rationale": "single-step quarantine resume test",
    }

    def quarantine_llm(
        repair_response: str | BaseException,
    ) -> PatternScriptedMockLLMClient:
        return _pattern_llm(
            plan=plan,
            code=draft_code,
            repairs=[repair_response],
            interpretation=(
                "The cohort summary is available {evidence:cohort_summary}."
            ),
            manuscript="# Title\n\n## Results\n\nSummary {evidence:cohort_summary}.",
        )

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = quarantine_llm(RuntimeError("simulated repair quota exhaustion"))
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=first_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=20,
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
    assert len(_prompt_calls(first_llm, "WRITE THE PYTHON CODE")) == 1
    assert len(_prompt_calls(first_llm, "REPAIR THE PYTHON CODE")) == 1
    assert runner_calls == ["01_summary"]
    assert (run_dir / "steps" / "01_summary" / "analysis.py").exists()
    assert not any((run_dir / "steps" / "01_summary" / "outputs").iterdir())
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()
    assert any(
        record.get("kind") == "code" and record.get("produced_by_step") == "01_summary"
        for record in first_partial["evidence"]
    )
    assert "01_summary" not in current_successful_step_ids(
        first_partial["per_step_records"]
    )
    assert not any(
        record.get("produced_by_step") == "01_summary"
        for record in current_evidence_records(
            first_partial["evidence"],
            first_partial["per_step_records"],
        )
    )

    # Simulate a nondeterministic auditor forgetting its prior error. The saved
    # error must still force REPAIR, and another outage must still not execute.
    audit_state["emit_error"] = False
    second_llm = quarantine_llm(RuntimeError("simulated repair quota exhaustion"))
    second_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=second_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=20,
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
    assert _prompt_calls(second_llm, "WRITE THE PYTHON CODE") == []
    second_repair_prompts = _prompt_calls(
        second_llm,
        "REPAIR THE PYTHON CODE",
        full=True,
    )
    assert len(second_repair_prompts) == 1
    assert persisted_message in second_repair_prompts[-1]
    assert runner_calls == ["01_summary"]
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()

    # Both allowed logical repairs have now been spent across two processes.
    # Constructing another pipeline must not buy a fresh attempt, even if that
    # model claims it could return a materially inert edit.
    noop_llm = quarantine_llm(
        (
            draft_code
            + "\n# claimed repair, no semantic change\npass\n'claimed repair'\n"
        )
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
        max_step_provider_calls=20,
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
    assert _prompt_calls(noop_llm, "WRITE THE PYTHON CODE") == []
    assert _prompt_calls(noop_llm, "REPAIR THE PYTHON CODE") == []
    assert noop_record["status"] == "blocked_by_concept_audit"
    assert noop_record["quarantined_repair_succeeded"] is False
    assert noop_record["step_llm_repair_attempts"] == 2
    assert noop_record["step_llm_repair_budget"] == 2
    assert noop_record["step_llm_repair_budget_exhausted"] is True
    assert runner_calls == ["01_summary"]
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()

    # Once the cross-resume logical repair budget is exhausted, another resume
    # cannot buy a fresh repair attempt by constructing a new pipeline object.
    # The last durable candidate remains quarantined and unexecuted.
    partial_code = draft_code.replace("draft_marker", "partial_marker")
    audit_state["reject_marker"] = "partial_marker"
    partial_llm = quarantine_llm(partial_code)
    partial_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=partial_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=True,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=20,
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
    from easyicu.research_agent.orchestration.resume import (
        load_quarantined_concept_draft,
    )

    latest_draft = load_quarantined_concept_draft(run_dir=run_dir, step_id="01_summary")
    assert latest_draft is not None
    assert _prompt_calls(partial_llm, "REPAIR THE PYTHON CODE") == []
    assert "draft_marker" in latest_draft.code
    assert "partial_marker" not in latest_draft.code
    assert runner_calls == ["01_summary"]


def test_resume_repair_ticket_uses_only_current_deterministic_coordinates(
    ra, tmp_path: Path, monkeypatch
):
    _disable_article_contract(monkeypatch)
    from easyicu.research_agent.audits.validators import ConceptUsageAuditor
    from easyicu.research_agent.contracts.runtime import ValidationFinding

    coordinate = {"call_line": 10}

    def deterministic_finding(
        self,
        *,
        context,
        script_text,
        step,
        provider_budget=None,
        study_endpoint=None,
        plan_step_roster=None,
    ):
        del self, context, script_text
        call_line = coordinate["call_line"]
        return [
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message="A deterministic provenance result needs repair.",
                detail={
                    "reason": "provenance_audit_not_fail_closed",
                    "issues": [
                        {
                            "failure_mode": (
                                "provenance_helper_result_not_immediately_guarded"
                            ),
                            "helper_name": "provenance_audit",
                            "call_line": call_line,
                            "following_guard_line": call_line + 1,
                        }
                    ],
                },
            )
        ]

    monkeypatch.setattr(ConceptUsageAuditor, "audit", deterministic_finding)

    draft_code = """
import json
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
out = os.environ["STEP_OUT_DIR"]
summary = {"n": int(len(df))}
pd.DataFrame([summary]).to_csv(os.path.join(out, "summary.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""

    plan = {
        "research_question": "Summarize the cohort.",
        "steps": [
            {
                "step_id": "01_summary",
                "planned_analysis_role": "auxiliary",
                "intent": "Produce a descriptive cohort summary.",
                "inputs": ["stay_id"],
                "expected_outputs": ["table:summary"],
                "method": "descriptive_summary",
                "icu_rule_refs": [],
            }
        ],
        "rationale": "coordinate refresh regression",
    }

    def coordinate_llm() -> PatternScriptedMockLLMClient:
        return _pattern_llm(
            plan=plan,
            code=draft_code,
            repairs=[RuntimeError("stop after recording repair prompt")],
        )

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = coordinate_llm()
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=first_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=20,
    )
    first = first_pipeline.run(
        question="Summarize the cohort.",
        cohort=cohort,
        cohort_name="current_coordinate_resume",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    first_repair_prompts = _prompt_calls(
        first_llm,
        "REPAIR THE PYTHON CODE",
        full=True,
    )
    assert '"call_line": 10' in first_repair_prompts[-1]

    coordinate["call_line"] = 20
    resumed_llm = coordinate_llm()
    resumed_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=resumed_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        max_step_provider_calls=20,
    )
    resumed_pipeline.run(
        question="Summarize the cohort.",
        cohort=cohort,
        cohort_name="current_coordinate_resume",
        database="synthetic",
        target_outcome="death",
        resume_run_id=first.run_id,
        resume_from_step_id="01_summary",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )

    prompt = _prompt_calls(
        resumed_llm,
        "REPAIR THE PYTHON CODE",
        full=True,
    )[-1]
    authority_payload = prompt.split(
        "HOST-OWNED REPAIR AUTHORITY (typed; verbatim):", 1
    )[1].split("REPAIR THE PYTHON CODE", 1)[0]
    parsed_authority = json.loads(authority_payload)
    current_ticket = json.dumps(parsed_authority["typed_ticket"], sort_keys=True)
    assert '"call_line": 20' in current_ticket
    assert '"call_line": 10' not in current_ticket
    assert "provenance_helper_result_not_immediately_guarded" in current_ticket
    assert parsed_authority["host_guidance"] == {}


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
    from easyicu.research_agent.execution.phase import (
        _python_repair_is_materially_changed,
    )

    before = "import os\nvalue = 1\n"

    assert not _python_repair_is_materially_changed(before, after)
    assert _python_repair_is_materially_changed(before, "import os\nvalue = 2\n")


def test_logical_repair_budget_restore_is_monotonic_across_early_failure() -> None:
    from easyicu.research_agent.execution.phase import (
        _monotonic_step_llm_repair_history,
    )

    attempts, classes, invalid = _monotonic_step_llm_repair_history(
        [
            {
                "step_id": "01_summary",
                "status": "blocked_by_concept_audit",
                "step_llm_repair_attempts": 2,
                "step_llm_repair_classes": ["concept", "concept"],
            },
            {
                "step_id": "01_summary",
                "status": "contract_failed",
                "provider_call_budget_receipt_invalid": True,
            },
        ],
        limit=2,
    )

    assert attempts == 2
    assert classes == ["concept", "concept"]
    assert invalid is False

    attempts, _, invalid = _monotonic_step_llm_repair_history(
        [{"step_id": "01_summary", "step_llm_repair_attempts": "unknown"}],
        limit=2,
    )
    assert attempts == 2
    assert invalid is True


def test_resume_retires_unchanged_draft_after_deterministic_policy_supersession(
    ra, tmp_path: Path, monkeypatch
) -> None:
    """A validator-policy fix may retire its own stale error without code churn."""

    _disable_article_contract(monkeypatch)
    from easyicu.research_agent.audits.validators import LLMConceptAuditor
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.runner import CodeRunner

    audit_state = {"old_policy": True}

    def policy_transition_audit(
        self,
        *,
        context,
        script_text,
        step,
        provider_budget=None,
        study_endpoint=None,
        plan_step_roster=None,
    ):
        del self, context, script_text, study_endpoint, plan_step_roster
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
    "output_files": {
        "table:cohort_summary": "cohort_summary.csv"
    },
}
pd.DataFrame([summary]).to_csv(
    os.path.join(out, "cohort_summary.csv"), index=False
)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""

    plan = {
        "research_question": "Summarize the cohort for in-hospital mortality.",
        "steps": [
            {
                "step_id": "01_summary",
                "planned_analysis_role": "auxiliary",
                "intent": "Produce a descriptive cohort summary.",
                "inputs": ["death"],
                "expected_outputs": ["table:cohort_summary"],
                "method": "descriptive_summary",
                "icu_rule_refs": [],
            }
        ],
        "rationale": "single-step policy transition test",
    }

    def policy_transition_llm() -> PatternScriptedMockLLMClient:
        return _pattern_llm(
            plan=plan,
            code=draft_code,
            repairs=[RuntimeError("simulated old-policy repair outage")],
            interpretation=(
                "The cohort summary is available {evidence:cohort_summary}."
            ),
            manuscript="# Title\n\n## Results\n\nSummary {evidence:cohort_summary}.",
        )

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = policy_transition_llm()
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
    assert len(_prompt_calls(first_llm, "WRITE THE PYTHON CODE")) == 1
    assert len(_prompt_calls(first_llm, "REPAIR THE PYTHON CODE")) == 1
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
    resumed_llm = policy_transition_llm()
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
    assert _prompt_calls(resumed_llm, "WRITE THE PYTHON CODE") == []
    assert _prompt_calls(resumed_llm, "REPAIR THE PYTHON CODE") == []
    assert quarantine_absent_at_runner == [True]
    assert record["status"] == "ok"
    assert record["resumed_quarantined_draft"] is True
    assert record["quarantined_repair_succeeded"] is False
    assert record["quarantine_policy_superseded"] is True
    assert record["quarantine_policy_superseded_findings"][0]["downgraded_reason"]
    assert record["quarantine_retired_by"] == (
        "deterministic_validator_policy_supersession"
    )
    assert record["quarantine_retired"] is True
    assert record["quarantined_requires_repair"] is False
    assert not (run_dir / "steps" / "01_summary" / ".quarantine").exists()


@pytest.mark.parametrize(
    "legacy_stale_checkpoint",
    [False, True],
    ids=["repair-on-resume", "already-repaired-stale-finding"],
)
def test_resume_reaudits_material_deterministic_quarantine_repair(
    ra, tmp_path: Path, monkeypatch, legacy_stale_checkpoint: bool
) -> None:
    """A deterministic replay retires stale findings without a new coder call."""

    import easyicu.research_agent.repairs.coordination as coordination_module

    real_repair = coordination_module.deterministic_concept_audit_repair
    repair_enabled = {"value": False}
    from easyicu.research_agent.agents.core import PlannerAgent

    original_planner_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_planner_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)

    def gated_repair(
        code,
        messages,
        *,
        repair_reasons=(),
        repair_findings=(),
        step=None,
        on_semantic_escalation=None,
    ):
        # `step` is forwarded by `authorized_deterministic_concept_repair` since
        # the all-rows profile-roles repair was registered. Without it here the
        # call raises TypeError, the whole deterministic-repair path dies, and the
        # symptom surfaces as "zero coder repairs" on the FIRST run -- which reads
        # as a deliberate lifecycle change rather than a crashed double.
        if not repair_enabled["value"]:
            return code, []
        return real_repair(
            code,
            messages,
            repair_reasons=repair_reasons,
            repair_findings=repair_findings,
            step=step,
            on_semantic_escalation=on_semantic_escalation,
        )

    monkeypatch.setattr(
        coordination_module,
        "deterministic_concept_audit_repair",
        gated_repair,
    )
    draft_code = """
import json
import os
from pathlib import Path
import pandas as pd

def main():
    frame = pd.read_parquet(os.environ["COHORT_PARQUET"])
    out = Path(os.environ["STEP_OUT_DIR"])
    invalid_pair_n = int(frame["stay_id"].isna().sum())
    discordant_n = int((frame["stay_id"] < 0).sum())
    audit = {
        "role": "audit_only",
        "invalid_pair_n": invalid_pair_n,
        "discordant_n": discordant_n,
    }
    summary = {
        "status": "ok",
        "measurement_provenance_audit": audit,
        "registered_outputs": {
            "table:cohort_summary": "cohort_summary.csv",
        },
        "output_files": {
            "table:cohort_summary": "cohort_summary.csv",
        },
    }
    if invalid_pair_n > 0 or discordant_n > 0:
        summary["status"] = "failed_provenance_audit"
    pd.DataFrame([{"n": int(len(frame))}]).to_csv(
        out / "cohort_summary.csv", index=False
    )
    with open(out / "step_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle)

if __name__ == "__main__":
    main()
"""

    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

    plan_response = json.dumps(
        {
            "research_question": "Summarize the ICU cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Produce a descriptive cohort summary.",
                    "inputs": ["stay_id"],
                    "expected_outputs": ["table:cohort_summary"],
                    "method": "descriptive_summary",
                    "icu_rule_refs": [],
                }
            ],
            "rationale": "single-step deterministic resume test",
        }
    )

    def deterministic_resume_llm() -> PatternScriptedMockLLMClient:
        return PatternScriptedMockLLMClient(
            [
                ("ICU-AWARE RESEARCH PLAN", [plan_response] * 4),
                ("WRITE THE PYTHON CODE", [draft_code]),
                (
                    "REPAIR THE PYTHON CODE",
                    [RuntimeError("simulated first-pass repair outage")],
                ),
                (
                    "INTERPRET THE RESULTS",
                    ["The cohort summary is available {evidence:cohort_summary}."],
                ),
                (
                    "MANUSCRIPT SCAFFOLD",
                    ["# Title\n\n## Results\n\nSummary {evidence:cohort_summary}."],
                ),
            ]
        )

    def call_count(llm: PatternScriptedMockLLMClient, marker: str) -> int:
        count = 0
        for messages, _kwargs in llm.calls:
            prompt = "\n".join(
                message.content.upper()
                for message in messages
                if message.role == "user"
            )
            if marker not in prompt:
                continue
            if marker == "WRITE THE PYTHON CODE" and "REPAIR THE PYTHON CODE" in prompt:
                continue
            count += 1
        return count

    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0]})
    first_llm = deterministic_resume_llm()
    first_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=first_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        max_step_provider_calls=20,
        runner_kind="subprocess",
    )
    first = first_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="deterministic_quarantine_resume",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    run_dir = Path(first.workdir)
    assert call_count(first_llm, "WRITE THE PYTHON CODE") == 1
    assert call_count(first_llm, "REPAIR THE PYTHON CODE") == 1
    assert (run_dir / "steps" / "01_summary" / ".quarantine").is_dir()
    first_partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    first_record = next(
        item
        for item in first_partial["per_step_records"]
        if item.get("step_id") == "01_summary"
    )
    first_provider_categories = list(
        first_record.get("step_provider_call_categories") or []
    )
    # Deterministic revalidation/repair runs before the coder-repair budget
    # branch. Exhaust the NEW durable logical ledger without buying another
    # provider call; resume must recover this receipt-ahead-of-step-snapshot
    # state and still permit deterministic revalidation.
    provider_receipt = provider_call_budget_receipt_path(
        run_dir,
        step_id="01_summary",
    )
    provider_state = load_provider_call_budget_state(
        provider_receipt,
        step_id="01_summary",
        expected_reserved_final_category=None,
    )
    durable_budget = StepProviderCallBudget(
        provider_state.limit,
        step_id="01_summary",
        consumed_categories=provider_state.categories,
        logical_repair_entries=provider_state.logical_repairs,
        receipt_path=provider_receipt,
        reserved_final_category=None,
    )
    assert durable_budget.logical_repair_classes == ("concept",)
    assert durable_budget.reserve_logical_repair("concept", max_repairs=2) == 2

    if legacy_stale_checkpoint:
        from easyicu.research_agent.orchestration.resume import (
            load_quarantined_concept_draft,
            store_quarantined_concept_draft,
        )

        stale_draft = load_quarantined_concept_draft(
            run_dir=run_dir,
            step_id="01_summary",
        )
        assert stale_draft is not None
        stale_messages = [
            value
            for finding in stale_draft.findings
            for value in (
                finding.get("message"),
                (finding.get("detail") or {}).get("reason"),
            )
            if value
        ]
        repaired_checkpoint_code, repair_names = real_repair(
            stale_draft.code,
            stale_messages,
        )
        assert repair_names == ["provenance_fail_closed_guard_v1"]
        assert repaired_checkpoint_code != stale_draft.code
        # Reproduce the legacy checkpoint written by the pre-fix execute loop:
        # the exact digest is already repaired, but its pre-repair deterministic
        # finding was accidentally persisted beside it.
        repaired_checkpoint = store_quarantined_concept_draft(
            run_dir=run_dir,
            step_id="01_summary",
            code=repaired_checkpoint_code,
            findings=list(stale_draft.findings),
        )
        # Reproduce the legacy writer's internally consistent checkpoint: its
        # step record pointed at the repaired draft digest even though the
        # attached deterministic finding was stale. A digest mismatch is a
        # different authority failure and must correctly trigger regeneration.
        for collection_name in ("per_step_records", "step_attempt_history"):
            for record in first_partial.get(collection_name, []):
                if (
                    record.get("step_id") != "01_summary"
                    or record.get("quarantined_requires_repair") is not True
                ):
                    continue
                record["quarantined_draft_sha256"] = repaired_checkpoint.sha256
                record["quarantined_draft_relative_path"] = (
                    repaired_checkpoint.relative_path
                )
        (run_dir / "manifest_partial.json").write_text(
            json.dumps(first_partial, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    repair_enabled["value"] = True
    from easyicu.research_agent.execution.runner import CodeRunner

    quarantine_absent_at_runner = []
    original_run = CodeRunner.run

    def run_after_quarantine_revalidation(
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

    monkeypatch.setattr(CodeRunner, "run", run_after_quarantine_revalidation)
    resumed_llm = deterministic_resume_llm()
    resumed_pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=resumed_llm,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        max_step_provider_calls=20,
        runner_kind="subprocess",
    )
    resumed_pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=cohort,
        cohort_name="deterministic_quarantine_resume",
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
    assert call_count(resumed_llm, "WRITE THE PYTHON CODE") == int(
        legacy_stale_checkpoint
    )
    assert call_count(resumed_llm, "REPAIR THE PYTHON CODE") == 0
    resumed_provider_categories = list(
        record.get("step_provider_call_categories") or []
    )
    assert resumed_provider_categories[: len(first_provider_categories)] == (
        first_provider_categories
    )
    new_provider_categories = resumed_provider_categories[
        len(first_provider_categories) :
    ]
    if legacy_stale_checkpoint:
        # Once evidence authority has been selected, a mutable legacy manifest
        # cannot re-authorize hand-rewritten quarantine code. Fail closed and
        # regenerate exactly once; do not silently promote the stale draft.
        assert new_provider_categories.count("initial_generation") == 1
    else:
        assert not any(
            category == "initial_generation"
            or category.endswith(("_patch", "_full_rewrite"))
            for category in new_provider_categories
        )
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_llm_repair_budget"] == 2
    assert quarantine_absent_at_runner == [not legacy_stale_checkpoint]
    assert record["status"] == "ok"
    if legacy_stale_checkpoint:
        assert record["deterministic_concept_repairs"] == 1
        assert not record.get("quarantined_repair_succeeded")
        assert not record.get("quarantine_deterministic_revalidation_succeeded")
        assert not record.get("quarantine_retired")
    else:
        assert record["deterministic_concept_repairs"] == 1
        assert record["quarantined_repair_materially_changed"] is True
        assert record["quarantined_repair_succeeded"] is True
        assert not record.get("quarantine_deterministic_revalidation_succeeded")
        assert record["quarantine_retired"] is True
    assert not record.get("monotonic_concept_constraints")
    assert not (run_dir / "steps" / "01_summary" / ".quarantine").exists()


def test_quarantine_deterministic_revalidation_is_fail_closed() -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
        _quarantined_deterministic_errors_resolved_by_current_gate,
    )

    script = "import os\nvalue = 1\n"
    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
    mechanical_error = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="A deterministic mechanical error from the prior digest.",
    )

    resolved = _quarantined_deterministic_errors_resolved_by_current_gate(
        prior_errors=[mechanical_error],
        current_findings=[],
        script_text=script,
        quarantined_script_sha256=digest,
    )
    assert resolved is not None
    assert resolved[0]["quarantined_script_sha256"] == digest
    assert resolved[0]["deterministic_gate_fingerprint"]

    assert (
        _quarantined_deterministic_errors_resolved_by_current_gate(
            prior_errors=[mechanical_error],
            current_findings=[],
            script_text=script + "value = 2\n",
            quarantined_script_sha256=digest,
        )
        is None
    )
    assert (
        _quarantined_deterministic_errors_resolved_by_current_gate(
            prior_errors=[mechanical_error],
            current_findings=[mechanical_error],
            script_text=script,
            quarantined_script_sha256=digest,
        )
        is None
    )


@pytest.mark.parametrize(
    "foreign_validator",
    [
        "llm_concept_auditor",
        "provider_call_budget",
        "provider_call_budget_receipt",
    ],
)
def test_quarantine_deterministic_revalidation_never_retires_foreign_or_mixed_errors(
    foreign_validator: str,
) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
        _quarantined_deterministic_errors_resolved_by_current_gate,
    )

    script = "import os\nvalue = 1\n"
    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
    deterministic_error = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="A deterministic mechanical error from the prior digest.",
    )
    foreign_error = ValidationFinding(
        validator=foreign_validator,
        severity="error",
        message="A non-replayable error remains binding.",
    )

    for prior_errors in ([foreign_error], [deterministic_error, foreign_error]):
        assert (
            _quarantined_deterministic_errors_resolved_by_current_gate(
                prior_errors=prior_errors,
                current_findings=[],
                script_text=script,
                quarantined_script_sha256=digest,
            )
            is None
        )


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
    from easyicu.research_agent.contracts.runtime import ValidationFinding

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
    from easyicu.research_agent.execution.phase import (
        _quarantined_errors_superseded_by_current_policy,
    )

    context, script = _policy_supersession_context_and_script(ra)
    result = _quarantined_errors_superseded_by_current_policy(
        prior_errors=[_stored_horizon_error(ra)],
        current_findings=[],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
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


def test_quarantine_policy_supersession_reclassifies_isolated_raw_branch_false_override(
    ra,
) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
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
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
product_contract = exposure_binding['product_contract']
def resolve_exposure(definition, product_contract, frame):
    if not isinstance(definition, pd.DataFrame):
        raise RuntimeError('finalized table required')
    executable = product_contract['executable_column']
    finalized = pd.to_numeric(definition[executable], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    return finalized.astype(int)
def resolve_raw_exposure(definition, frame):
    return reconcile_binary_event_presence(frame)
if isinstance(exposure_definition, pd.DataFrame):
    treatment = resolve_exposure(exposure_definition, product_contract, frame)
else:
    treatment = resolve_raw_exposure(exposure_definition, frame).values
model = sm.Logit(outcome, pd.DataFrame({'treatment': treatment}))
"""
    stored = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Finalized exposure is overwritten.",
        detail={
            "issue_code": "finalized_exposure_overridden",
            "context": (
                "The script replaces treatment with "
                "reconcile_binary_event_presence values."
            ),
        },
    )

    result = _quarantined_errors_superseded_by_current_policy(
        prior_errors=[stored],
        current_findings=[],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
    )

    assert result is not None
    reclassified, provenance = result
    assert reclassified[0].severity == "warning"
    assert provenance[0]["reclassified_severity"] == "warning"


def test_quarantine_policy_does_not_trust_artifact_literal_decoy_flow(ra) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
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
decoy = pd.DataFrame({'treatment': [0, 1]})
exposure_definition = ('artifact:primary_exposure_definition', decoy)[1]
finalized = pd.to_numeric(exposure_definition['treatment'], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid exposure')
def consume():
    return finalized.astype(int)
treatment = consume()
"""
    stored = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Finalized exposure is overwritten.",
        detail={"issue_code": "finalized_exposure_overridden"},
    )

    assert (
        _quarantined_errors_superseded_by_current_policy(
            prior_errors=[stored],
            current_findings=[],
            context=context,
            script_text=script,
            quarantined_script_sha256=hashlib.sha256(
                script.encode("utf-8")
            ).hexdigest(),
        )
        is None
    )


def test_quarantine_policy_does_not_trust_uncalled_return_as_consumption(ra) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
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
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
finalized = pd.to_numeric(exposure_definition['treatment'], errors='coerce')
if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
    raise RuntimeError('invalid exposure')
def never_called():
    return finalized.astype(int)
treatment = helper_result.values
"""
    stored = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Finalized exposure is overwritten.",
        detail={"issue_code": "finalized_exposure_overridden"},
    )

    assert (
        _quarantined_errors_superseded_by_current_policy(
            prior_errors=[stored],
            current_findings=[],
            context=context,
            script_text=script,
            quarantined_script_sha256=hashlib.sha256(
                script.encode("utf-8")
            ).hexdigest(),
        )
        is None
    )


def test_quarantine_policy_does_not_trust_audit_only_authority_flow(ra) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
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
exposure_binding = resolved_inputs['artifact:primary_exposure_definition']
exposure_definition = exposure_binding['value']
if isinstance(exposure_definition, pd.DataFrame):
    finalized = pd.to_numeric(exposure_definition['treatment'], errors='coerce')
    if finalized.isna().any() or not np.isfinite(finalized).all() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid exposure')
    treatment = finalized.astype(int)
else:
    treatment = reconcile_binary_event_presence(frame).values
pd.DataFrame({'audited_treatment': treatment}).to_csv(audit_path, index=False)
raw_wrong = pd.to_numeric(frame['raw_wrong'], errors='coerce')
model = sm.Logit(outcome, pd.DataFrame({'treatment': raw_wrong}))
"""
    stored = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Finalized exposure is overwritten.",
        detail={"issue_code": "finalized_exposure_overridden"},
    )

    assert (
        _quarantined_errors_superseded_by_current_policy(
            prior_errors=[stored],
            current_findings=[],
            context=context,
            script_text=script,
            quarantined_script_sha256=hashlib.sha256(
                script.encode("utf-8")
            ).hexdigest(),
        )
        is None
    )


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
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
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
            ValidationFinding.model_validate(finding) for finding in current_findings
        ],
        context=context,
        script_text=script,
        quarantined_script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
    )

    assert result is None


def test_quarantine_policy_supersession_requires_zero_current_errors(ra) -> None:
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
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
        quarantined_script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
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
    step_ids_before = [
        s["step_id"] for s in json.loads(plan_path.read_text(encoding="utf-8"))["steps"]
    ]
    assert step_ids_before, "first run produced no plan steps"

    class DifferentPlanLLM:
        """Planner here returns a plan with a step_id that must never appear
        if the locked plan is reused."""

        name = "different-plan-llm"

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            upper = user.upper()
            if "ICU-AWARE RESEARCH PLAN" in upper:
                return json.dumps(
                    {
                        "research_question": "Is admission SOFA-2 associated with ICU mortality?",
                        "steps": [
                            {
                                "step_id": "88_resume_should_ignore_this",
                                "planned_analysis_role": "primary",
                                "intent": "This plan must be ignored on resume.",
                                "inputs": ["sofa2", "death"],
                                "expected_outputs": ["table:ignored"],
                                "method": "descriptive",
                                "icu_rule_refs": ["aggregation_rule_for"],
                            }
                        ],
                        "rationale": "resume must not use this plan",
                    }
                )
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

    step_ids_after = [
        s["step_id"] for s in json.loads(plan_path.read_text(encoding="utf-8"))["steps"]
    ]
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

    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

    llm = PatternScriptedMockLLMClient([], contextual_default=True)
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
    planner_calls = [
        message.content
        for messages, _kwargs in llm.calls
        for message in messages
        if message.role == "user"
        and message.content.upper().startswith(
            (
                "PRODUCE AN ICU-AWARE RESEARCH PLAN",
                "REVISE THE ICU-AWARE RESEARCH PLAN",
            )
        )
    ]
    assert not planner_calls
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
    assert partial["plan_path"] == revision_record.relative_path
    assert partial["current_plan_authority"] == {
        "schema_version": "easyicu.current_plan_authority/1",
        "evidence_id": revision_record.evidence_id,
        "relative_path": revision_record.relative_path,
        "sha256": revision_record.sha256,
        "revision": 4,
    }
    assert any(
        finding.get("validator") == "planner_schema_migration"
        and (finding.get("detail") or {}).get("kind") == "legacy_figure_render_edge"
        for finding in partial["findings"]
    )


def test_resume_to_nonexistent_run_id_starts_fresh_directory(
    ra, synthetic_cohort, tmp_path: Path
):
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


def test_final_manifest_keeps_step_records_for_metered_hosted_stub(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The final manifest must keep per-step resume records outside Mock paths.

    The real hosted path is wrapped in ``MeteredClient`` when cost
    tracking is enabled. This stub exercises that routing without a
    network call and pins the final ``manifest.json`` contract that
    paper/provenance tooling reads.
    """

    _disable_article_contract(monkeypatch)
    plan = {
        "research_question": "Is SOFA associated with ICU mortality?",
        "steps": [
            {
                "step_id": "04_primary_association",
                "planned_analysis_role": "primary",
                "intent": "Estimate SOFA and ICU mortality association.",
                "inputs": ["sofa2", "death"],
                "expected_outputs": ["table:primary_association"],
                "method": "descriptive",
                "icu_rule_refs": ["aggregation_rule_for"],
            }
        ],
        "rationale": "single-step metered mock resume test",
    }
    code = """
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
    "output_files": {
        "table:primary_association": "primary_association.csv"
    },
}
pd.DataFrame([summary]).to_csv(os.path.join(out, "primary_association.csv"), index=False)
with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f)
"""
    llm = _pattern_llm(
        plan=plan,
        code=code,
        interpretation=(
            "The primary association table is available {evidence:primary_association}."
        ),
        manuscript=(
            "# Title\n\n## Results\n\n"
            "The table is available {evidence:primary_association}."
        ),
    )

    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "sofa2": [0, 1, 3, 6],
            "death": [1, 0, 0, 1],
        }
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
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
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )

    assert manifest["per_step_records"], "final manifest dropped per-step records"
    assert manifest["per_step_records"] == partial["per_step_records"]
    assert manifest["per_step_records"][0]["status"] == "ok"
    assert manifest["current_plan_authority"] == partial["current_plan_authority"]
    assert manifest["plan_path"] == manifest["current_plan_authority"]["relative_path"]
    assert manifest["step_attempt_history"] == []
    assert manifest["step_attempt_history_ref"]["record_count"] == len(
        partial["step_attempt_history"]
    )
    from easyicu.research_agent.authority.runtime_artifacts import (
        load_run_artifact_authority,
    )

    hydrated = load_run_artifact_authority(run_dir)
    assert hydrated is not None
    assert hydrated["step_attempt_history"] == partial["step_attempt_history"]
    assert manifest["cost_records"], "hosted-stub path should be metered"
