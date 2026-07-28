"""Resume authority for versioned measurement-companion plan closure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_input_closure import (
    close_measurement_companion_inputs,
)
from easyicu.research_agent.authority.plan_scope import (
    _serializable_plan_scientific_scope_signature,
    measurement_companion_input_closure_evidence_id,
    verified_plan_evidence_rank,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    verified_run_evidence_path,
)
from easyicu.research_agent.pipeline import _load_compatible_resume_plan
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _context(tmp_path: Path):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "mortality": [0, 1, 0],
            "sofa_measured": [1, 1, 0],
            "sofa_n": [6, 5, 0],
        }
    ).to_parquet(cohort_path, index=False)
    return build_research_context(
        research_question="Audit measurement availability.",
        cohort=cohort_path,
        cohort_name="resume_closure",
        database="synthetic",
        target_outcome="mortality",
    )


def _plan(*, revision: int, intent: str) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Audit measurement availability.",
        revision=revision,
        steps=[
            AnalysisStep(
                step_id="01_measurement_audit",
                intent=intent,
                inputs=["sofa_measured"],
                expected_outputs=["table:measurement_audit"],
                method="missingness_source_availability_audit",
            )
        ],
    )


def _register_plan(
    evidence: EvidenceStore,
    *,
    plan: AnalysisPlan,
    evidence_id: str,
) -> None:
    path = evidence.root / f"{evidence_id}.json"
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Planner-owned analysis plan.",
        source_path=path,
        evidence_id=evidence_id,
        producer="planner",
        generation_mode="llm",
    )


def _successful_record(plan: AnalysisPlan) -> dict:
    step = plan.steps[0]
    return {
        "step_id": step.step_id,
        "status": "ok",
        "planned_analysis_role": step.planned_analysis_role,
        "analysis_request": {"step": step.model_dump(mode="json")},
        "plan_scientific_signature": (
            _serializable_plan_scientific_scope_signature(plan)
        ),
    }


def test_versioned_input_closure_rank_requires_exact_host_authority() -> None:
    digest = "a" * 64
    evidence_id = measurement_companion_input_closure_evidence_id(
        revision=7,
        sha256=digest,
    )
    record = {
        "evidence_id": evidence_id,
        "sha256": digest,
        "producer": "runtime_supervisor",
        "generation_mode": "system",
        "metadata": {
            "reason": "measurement_companion_input_closure",
            "source_plan_revision": 7,
            "closure_sha256": digest,
        },
    }

    assert verified_plan_evidence_rank(record) == 7
    assert verified_plan_evidence_rank({**record, "producer": "planner"}) is None
    assert (
        verified_plan_evidence_rank(
            {
                **record,
                "metadata": {
                    **record["metadata"],
                    "source_plan_revision": 6,
                },
            }
        )
        is None
    )
    assert (
        verified_plan_evidence_rank(
            {**record, "evidence_id": evidence_id[:-8] + "b" * 8}
        )
        is None
    )


def test_resume_migrates_only_verified_plan_to_exact_closed_step(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    base = _plan(revision=1, intent="Audit the original measurement definition.")
    revised = _plan(revision=2, intent="Audit the revised measurement definition.")
    stale_closed, stale_findings = close_measurement_companion_inputs(
        plan=base,
        context=context,
    )
    revised_closed, revised_findings = close_measurement_companion_inputs(
        plan=revised,
        context=context,
    )
    assert stale_findings and revised_findings

    evidence = EvidenceStore(tmp_path)
    _register_plan(evidence, plan=base, evidence_id="analysis_plan")
    stale_path = tmp_path / "analysis_plan_input_closure.json"
    stale_path.write_text(stale_closed.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Legacy fixed-slot input closure.",
        source_path=stale_path,
        evidence_id="analysis_plan_input_closure",
        producer="runtime_supervisor",
        generation_mode="system",
        metadata={"reason": "measurement_companion_input_closure"},
    )
    _register_plan(evidence, plan=revised, evidence_id="analysis_plan_revision_2")
    resume_state = {"per_step_records": [_successful_record(revised_closed)]}

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state=resume_state,
        context=context,
        evidence=evidence,
        prompt_pack_version="test-prompts/v1",
    )

    assert selected == revised_closed
    versioned = [
        record
        for record in evidence.records()
        if record.evidence_id.startswith("analysis_plan_input_closure_revision_2_")
    ]
    assert len(versioned) == 1
    assert selected_path == verified_run_evidence_path(tmp_path, versioned[0])
    assert versioned[0].metadata["closure_sha256"] == versioned[0].sha256
    assert (
        tmp_path / "analysis_plan_input_closure.json"
    ).read_text(encoding="utf-8") == revised_closed.model_dump_json(indent=2)

    repeated, repeated_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state=resume_state,
        context=context,
        evidence=evidence,
        prompt_pack_version="test-prompts/v1",
    )
    assert repeated == selected
    assert repeated_path == selected_path
    assert len(evidence.records()) == 4


def test_resume_never_promotes_mutable_or_nonmatching_plan_closure(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    immutable = _plan(revision=1, intent="Immutable planner intent.")
    evidence = EvidenceStore(tmp_path)
    _register_plan(evidence, plan=immutable, evidence_id="analysis_plan")

    mutable = _plan(revision=99, intent="Unregistered mutable drift.")
    mutable_closed, findings = close_measurement_companion_inputs(
        plan=mutable,
        context=context,
    )
    assert findings
    (tmp_path / "analysis_plan.json").write_text(
        mutable.model_dump_json(indent=2),
        encoding="utf-8",
    )

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state={"per_step_records": [_successful_record(mutable_closed)]},
        context=context,
        evidence=evidence,
        prompt_pack_version="test-prompts/v1",
    )

    assert selected is None
    assert selected_path is None
    assert evidence.ids() == ["analysis_plan"]
