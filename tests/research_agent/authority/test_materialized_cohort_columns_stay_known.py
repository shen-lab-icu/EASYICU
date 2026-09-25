"""A cohort that filters a column its run materialized stays readable.

Such a column (``<concept>_max``) is known to cohort validation only inside
the run's scoped concept registration.  A replan, a resumed run, the Web
readiness projection, report repair and the primary-population step all read
the plan or the cohort lock again after that scope closed.  Each presents the
roster of the run's sealed context -- or, for a definition or plan already
built, its own ids -- and nothing broader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.agents.planner import PlannerAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_authority import normalize_replan_candidate
from easyicu.research_agent.authority.plan_scope import (
    _serializable_plan_scientific_scope_signature,
    verified_plan_scientific_scope_count,
)
from easyicu.research_agent.authority.resume_plan import load_compatible_resume_plan
from easyicu.research_agent.authority.run_input import (
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
)
from easyicu.research_agent.cohort.schema import (
    _load_locked_cohort_definition,
    registered_run_cohort_concept_ids,
    write_locked_cohort_definition,
)
from easyicu.research_agent.execution.runners.primary_population_descriptive import (
    _bound_categorical_plan,
)
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    cohort_concept_id_scope,
    cohort_definition_sha,
    concept_id_exists,
    sealed_cohort_concept_ids,
)
from easyicu.research_agent.planning.runtime_suffix import (
    RuntimePlanSuffixRevision,
    merge_runtime_plan_suffix,
)
from easyicu.research_agent.reporting.registered_report_inputs import (
    ReadOnlyReportEvidence,
    _registered_plan_and_context,
)
from easyicu.research_agent.reporting.writer_only_migration import (
    prepare_writer_only_migration,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver.scientific_readiness_projection import (
    build_scientific_readiness_projection,
)

COLUMN = "fixture_flag_max"
QUESTION = "Describe the flagged stays."


def _cohort_payload(column: str = COLUMN) -> dict:
    return {
        "name": "flagged_stays",
        "inclusion": [
            {
                "concept_id": column,
                "time_window": {
                    "anchor": "icu_admission",
                    "start_offset_hours": 0.0,
                    "end_offset_hours": 24.0,
                },
                "aggregation": "any",
                "op": "==",
                "value": 1.0,
            }
        ],
        "exclusion": [],
        "selection_mode": "predicate_filtered",
    }


def _steps() -> list[AnalysisStep]:
    return [
        AnalysisStep(
            step_id="01_describe",
            planned_analysis_role="auxiliary",
            intent="Describe the flagged stays.",
            inputs=["death"],
            expected_outputs=["table:summary"],
            method="descriptive",
        ),
        AnalysisStep(
            step_id="02_report",
            planned_analysis_role="auxiliary",
            intent="Report the description.",
            inputs=["table:summary"],
            expected_outputs=["table:report"],
            method="descriptive",
        ),
    ]


def _plan() -> AnalysisPlan:
    with cohort_concept_id_scope([COLUMN]):
        return AnalysisPlan(
            research_question=QUESTION,
            revision=1,
            cohort=_cohort_payload(),
            steps=_steps(),
        )


def _context(tmp_path: Path):
    cohort_path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "death": [0, 1, 0, 1],
            COLUMN: [1.0, 1.0, 0.0, 1.0],
        }
    ).to_parquet(cohort_path, index=False)
    return build_research_context(
        research_question=QUESTION,
        cohort=cohort_path,
        cohort_name="flagged",
        database="synthetic",
        target_outcome="death",
    )


def _register(evidence: EvidenceStore, name: str, evidence_id: str) -> None:
    evidence.register_file(
        kind="log",
        description=f"Registered {evidence_id}.",
        source_path=evidence.root / name,
        evidence_id=evidence_id,
        producer="pipeline",
        generation_mode="system",
    )


def _run(tmp_path: Path, *, register_context: bool = True):
    """A run that registered its context, plan and cohort lock."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = _context(tmp_path)
    plan = _plan()
    evidence = EvidenceStore(run_dir)
    if register_context:
        (run_dir / "research_context.json").write_text(
            context.model_dump_json(indent=2), encoding="utf-8"
        )
        _register(evidence, "research_context.json", "research_context")
    (run_dir / "analysis_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    _register(evidence, "analysis_plan.json", "analysis_plan")
    write_locked_cohort_definition(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test-prompts/v1",
        llm_signature="test",
        cohort_concept_ids=sealed_cohort_concept_ids(context),
    )
    return run_dir, context, plan, evidence


def test_the_run_roster_comes_from_its_registered_context(tmp_path: Path) -> None:
    run_dir, context, _plan_, _evidence = _run(tmp_path)

    assert not concept_id_exists(COLUMN)
    assert COLUMN in sealed_cohort_concept_ids(context)
    assert registered_run_cohort_concept_ids(run_dir) == sealed_cohort_concept_ids(
        context
    )
    assert registered_run_cohort_concept_ids(tmp_path / "missing") == ()


def test_a_built_definition_has_one_digest_in_and_out_of_scope() -> None:
    with cohort_concept_id_scope([COLUMN]):
        definition = CohortDefinition.from_dict(_cohort_payload())
        inside = cohort_definition_sha(definition)

    assert cohort_definition_sha(definition) == inside
    assert not concept_id_exists(COLUMN)


def test_resume_reads_the_lock_and_plan_with_the_run_roster(tmp_path: Path) -> None:
    run_dir, context, plan, _evidence = _run(tmp_path)
    step = plan.steps[0]
    resume_state = {
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
    }

    assert _load_locked_cohort_definition(run_dir) == plan.cohort
    roster = sealed_cohort_concept_ids(context)
    stored = [run_dir / "analysis_plan.json"]
    assert verified_plan_scientific_scope_count(stored, cohort_concept_ids=roster) == 1
    assert verified_plan_scientific_scope_count(stored) == 0
    for supplied in (None, context):
        selected, path = load_compatible_resume_plan(
            run_dir=run_dir, resume_state=resume_state, context=supplied
        )
        assert selected == plan
        assert path is not None and path.name.startswith("analysis_plan")
    assert not concept_id_exists(COLUMN)


def test_a_pre_scope_host_checkpoint_counts_the_plans_of_such_a_run(
    tmp_path: Path,
) -> None:
    """A legacy host checkpoint inherits the plan scope only if one plan counts."""

    run_dir, context, plan, _evidence = _run(tmp_path)
    step = plan.steps[0]
    record = {
        "step_id": step.step_id,
        "status": "ok",
        "planned_analysis_role": step.planned_analysis_role,
        "analysis_request": {"step": step.model_dump(mode="json")},
        "step_authority_kind": _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
        "generation_mode": _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
    }

    selected, _path = load_compatible_resume_plan(
        run_dir=run_dir, resume_state={"per_step_records": [record]}, context=context
    )

    assert selected == plan


def test_without_the_run_context_the_column_is_still_unknown(tmp_path: Path) -> None:
    run_dir, _context_, _plan_, _evidence = _run(tmp_path, register_context=False)

    with pytest.raises(CohortSchemaError, match=f"unknown concept_id: {COLUMN}"):
        _load_locked_cohort_definition(run_dir)


def test_readiness_reads_the_cohort_of_such_a_run(tmp_path: Path) -> None:
    run_dir, _context_, plan, _evidence = _run(tmp_path)
    definition = plan.cohort.to_dict()
    (run_dir / "cohort_provenance.json").write_text(
        json.dumps(
            {
                "database": "synthetic",
                "cohort_definition": None,
                "export_authority": {"authority_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cohort_analysis_provenance.json").write_text(
        json.dumps(
            {
                "cohort_definition": definition,
                "cohort_sha256": cohort_definition_sha(plan.cohort),
                "n_universe": 4,
                "n_analysis_cohort": 3,
            }
        ),
        encoding="utf-8",
    )

    projection = build_scientific_readiness_projection(
        run_id="run-flagged",
        run_dir=run_dir,
        axes={
            "analysis_validated": True,
            "manuscript_ready": False,
            "publication_ready": False,
            "paper_authorized": False,
        },
        literature_evidence={},
        study={},
    )

    data = next(domain for domain in projection.domains if domain.domain == "data")
    assert data.status == "passed"
    assert projection.facts["data"]["cohort_definition_explicit"] is True
    assert not concept_id_exists(COLUMN)


def test_report_repair_and_the_population_step_read_the_bound_plan(
    tmp_path: Path,
) -> None:
    run_dir, _context_, plan, _evidence = _run(tmp_path)
    (run_dir / "manifest_partial.json").write_text(
        json.dumps({"plan_path": "analysis_plan.json"}), encoding="utf-8"
    )

    registered, _registered_context = _registered_plan_and_context(
        ReadOnlyReportEvidence(run_dir)
    )

    assert registered == plan
    assert _bound_categorical_plan(run_dir) == plan
    assert not concept_id_exists(COLUMN)


def test_writer_only_repair_validates_the_plan_of_such_a_run(tmp_path: Path) -> None:
    run_dir, _context_, plan, _evidence = _run(tmp_path)
    (run_dir / "preplan_literature_bundle.json").write_text(
        LiteratureBundle(research_question=QUESTION, citations=[]).model_dump_json(),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold.md").write_text(
        "# Flagged stays\n\n## Results\n\nThe flagged stays were described.\n",
        encoding="utf-8",
    )
    (run_dir / "writer_evidence_digest.md").write_text("digest", encoding="utf-8")

    prepared = prepare_writer_only_migration(run_dir)

    # Not demoted to a report-only legacy plan it cannot read.
    assert prepared.plan_validation_status == "validated"
    assert prepared.plan == plan
    assert not concept_id_exists(COLUMN)


def test_a_planner_may_filter_on_a_context_column_and_nothing_else(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    raw = json.dumps(
        {
            "research_question": QUESTION,
            "cohort": _cohort_payload(),
            "steps": [step.model_dump(mode="json") for step in _steps()],
            "rationale": "Describe the flagged stays.",
        }
    )

    parsed = PlannerAgent.__new__(PlannerAgent)._parse(raw, context)

    assert parsed.cohort == _plan().cohort
    assert not concept_id_exists(COLUMN)
    with pytest.raises(ValueError, match="unknown concept_id: fixture_other_max"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            raw.replace(COLUMN, "fixture_other_max"), context
        )


def test_a_replan_restores_the_cohort_scope_and_keeps_its_column(
    tmp_path: Path,
) -> None:
    current = _plan()
    candidate = current.model_copy(
        update={"revision": 2, "research_question": "A different question."}
    )

    result = normalize_replan_candidate(
        current_plan=current,
        candidate_plan=candidate,
        completed_records=[],
        context=_context(tmp_path),
        max_total_steps=0,
        locked_robustness_specs=[],
    )

    assert result.plan.revision == 2
    assert result.plan.research_question == current.research_question
    assert result.plan.cohort == current.cohort
    assert not any(
        finding.detail.get("error_type") for finding in result.findings
    ), result.findings
    assert not concept_id_exists(COLUMN)


def test_a_runtime_suffix_keeps_the_cohort_it_never_touches() -> None:
    plan = _plan()
    replacement = plan.steps[1].model_copy(update={"intent": "Report it again."})

    merged = merge_runtime_plan_suffix(
        current_plan=plan,
        completed_step_ids=["01_describe"],
        revision=RuntimePlanSuffixRevision(
            replace_from_step_id="02_report",
            replacement_step=replacement,
            rationale="The executed description governs the remaining report.",
        ),
    )

    assert merged.cohort == plan.cohort
    assert merged.steps[1].intent == "Report it again."
    assert not concept_id_exists(COLUMN)
