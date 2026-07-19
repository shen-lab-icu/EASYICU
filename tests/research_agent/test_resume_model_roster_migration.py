"""Fail-closed migration for legacy resume plans without typed model rosters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.pipeline import (
    LegacyResumePlanMigrationError,
    _load_compatible_resume_plan,
    _migrate_legacy_resume_model_requirements,
)
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    PlannedModelRequirement,
)


def _legacy_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Estimate an adjusted ICU association.",
        analysis_type="cohort_definition_sensitivity",
        revision=1,
        rationale="Planner-owned scientific rationale.",
        steps=[
            AnalysisStep(
                step_id="01_completed_description",
                intent="Describe the fixed analytic cohort.",
                inputs=["baseline"],
                expected_outputs=["table:description"],
                method="descriptive_baseline",
                icu_rule_refs=["fixed_rule"],
            ),
            AnalysisStep(
                step_id="02_remaining_closed",
                intent=(
                    "Fit separate planner-selected adjusted models for the "
                    "binary endpoint and continuous length outcome."
                ),
                inputs=["exposure", "endpoint", "length_outcome"],
                expected_outputs=[
                    "table:adjusted_association_estimates",
                    "table:analytic_population",
                ],
                method="adjusted_association_models",
                icu_rule_refs=["fixed_model_rule"],
            ),
            AnalysisStep(
                step_id="03_method_only_not_closed",
                intent="A different product remains agent-owned.",
                expected_outputs=["table:unrelated_product"],
                method="adjusted_association_models",
            ),
            AnalysisStep(
                step_id="04_product_only_not_closed",
                intent="The closed product name alone is insufficient.",
                expected_outputs=["table:adjusted_association_estimates"],
                method="descriptive_baseline",
            ),
        ],
    )


def _planner_requirement() -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="planner_selected_binary_endpoint",
        outcome="agent_selected_endpoint",
        outcome_type="binary",
        method_family="statsmodels_glm_binomial",
        exposure_source="agent_selected_transform",
        analysis_role="primary",
        analysis_set="complete_case",
        required_for_step_success=True,
    )


def _continuous_planner_requirement() -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="planner_selected_continuous_outcome",
        outcome="length_outcome",
        outcome_type="continuous",
        method_family="median_quantile_regression",
        exposure_source="agent_selected_transform",
        analysis_role="secondary",
        analysis_set="complete_case",
        required_for_step_success=True,
    )


def _evidence_with_base_plan(run_dir: Path, plan: AnalysisPlan) -> EvidenceStore:
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Legacy analysis plan.",
        source_path=plan_path,
        evidence_id="analysis_plan",
        producer="planner",
        generation_mode="llm",
    )
    return evidence


def _context(tmp_path: Path, plan: AnalysisPlan):
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure": [0.1, 0.4, 0.8, 1.2],
            "endpoint": [0, 1, 0, 1],
            "length_outcome": [1.5, 2.0, 3.25, 5.0],
            "baseline": [50, 61, 72, 83],
        }
    ).to_parquet(cohort_path, index=False)
    return build_research_context(
        research_question=plan.research_question,
        cohort=cohort_path,
        cohort_name="migration_packet_test",
        database="synthetic",
        target_outcome="endpoint",
        primary_exposure="exposure",
    )


def _valid_packet_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "02_remaining_closed",
                    "model_requirements": [
                        _planner_requirement().model_dump(mode="json"),
                        _continuous_planner_requirement().model_dump(mode="json"),
                    ],
                }
            ]
        }
    )


def test_resume_migration_uses_planner_packet_and_registers_revision(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    context = _context(tmp_path, plan)
    base_bytes = (tmp_path / "analysis_plan.json").read_bytes()
    resume_state = {
        "plan_path": "analysis_plan.json",
        "per_step_records": [
            {"step_id": "01_completed_description", "status": "ok"},
            {"step_id": "02_remaining_closed", "status": "ok"},
        ],
    }
    calls = []

    class PacketLLM:
        name = "packet-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            calls.append((messages, max_tokens, temperature))
            return _valid_packet_json()

    roles = []

    revised, revision_path, target_ids = _migrate_legacy_resume_model_requirements(
        plan=plan,
        context=context,
        run_dir=tmp_path,
        resume_state=resume_state,
        resume_from_step_id="02_remaining_closed",
        role_resolver=lambda role: roles.append(role) or PacketLLM(),
        evidence=evidence,
        prompt_version="test-pack",
        llm_signature="test-planner",
    )

    assert roles == ["planner"]
    assert len(calls) == 1
    prompt = "\n".join(message.content for message in calls[0][0])
    assert "02_remaining_closed" in prompt
    assert "binary endpoint and continuous length outcome" in prompt
    assert '"inputs": [' in prompt
    assert '"expected_outputs": [' in prompt
    assert '"icu_rule_refs": [' in prompt
    assert "requirement_id" in prompt
    assert "outcome_type" in prompt
    assert "analysis_role" in prompt
    assert "analysis_set" in prompt
    assert "primary, secondary, sensitivity" in prompt
    assert "source_aware, complete_case" in prompt
    assert "every adjusted outcome/model pre-specified" in prompt
    assert "target_outcome is not an exhaustive roster" in prompt
    assert "secondary outcome/model to be omitted" in prompt
    assert "exactly one analysis_role=primary" in prompt
    assert "Planner chooses which requirement is primary" in prompt
    assert "READ-ONLY PLAN-LEVEL COMMITMENTS" in prompt
    for field in (
        "research_question",
        "analysis_type",
        "rationale",
        "cohort",
        "robustness_specs",
    ):
        assert f'"{field}"' in prompt
    assert target_ids == ("02_remaining_closed",)
    assert revision_path == tmp_path / "analysis_plan_revision_2.json"
    assert revision_path.exists()
    assert (tmp_path / "analysis_plan.json").read_bytes() == base_bytes
    assert revised.steps[1].model_requirements == [
        _planner_requirement(),
        _continuous_planner_requirement(),
    ]
    assert not revised.steps[2].model_requirements
    assert not revised.steps[3].model_requirements

    revision_evidence = evidence.get("analysis_plan_revision_2")
    assert revision_evidence is not None
    assert revision_evidence.producer == "planner"
    assert revision_evidence.generation_mode == "llm"
    assert revision_evidence.metadata["reason"] == (
        "legacy_missing_model_requirements"
    )
    assert revision_evidence.metadata["target_step_ids"] == [
        "02_remaining_closed"
    ]

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state=resume_state,
    )
    assert selected_path == verified_run_evidence_path(
        tmp_path,
        revision_evidence,
    )
    assert selected == revised


def test_resume_loader_ignores_mutable_live_plan_and_unregistered_revision(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    immutable_record = evidence.get("analysis_plan")
    assert immutable_record is not None

    live_drift = plan.model_copy(
        update={
            "steps": [
                plan.steps[0].model_copy(update={"intent": "Mutable live drift."}),
                *plan.steps[1:],
            ]
        }
    )
    (tmp_path / "analysis_plan.json").write_text(
        live_drift.model_dump_json(indent=2),
        encoding="utf-8",
    )
    unregistered_revision = live_drift.model_copy(update={"revision": 99})
    (tmp_path / "analysis_plan_revision_99.json").write_text(
        unregistered_revision.model_dump_json(indent=2),
        encoding="utf-8",
    )

    selected, selected_path = _load_compatible_resume_plan(
        run_dir=tmp_path,
        resume_state={
            "plan_path": "analysis_plan_revision_99.json",
            "per_step_records": [
                {"step_id": "01_completed_description", "status": "ok"},
            ],
        },
    )

    assert selected == plan
    assert selected_path == verified_run_evidence_path(
        tmp_path,
        immutable_record,
    )


def test_resume_migration_retries_real_incomplete_requirement_shape(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    context = _context(tmp_path, plan)
    llm_calls = []

    class IncompleteThenValidPacketLLM:
        name = "incomplete-then-valid-packet-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            llm_calls.append((messages, max_tokens, temperature))
            if len(llm_calls) == 1:
                # Exact real failure shape: outcome/outcome_type/analysis_set
                # absent and analysis_role is a prose label, not the enum.
                return json.dumps(
                    {
                        "steps": [
                            {
                                "step_id": "02_remaining_closed",
                                "model_requirements": [
                                    {
                                        "requirement_id": "incomplete_secondary",
                                        "method_family": "logistic_regression",
                                        "exposure_source": "exposure",
                                        "analysis_role": (
                                            "secondary_adjusted_association"
                                        ),
                                        "required_for_step_success": True,
                                    }
                                ],
                            }
                        ]
                    }
                )
            return _valid_packet_json()

    revised, revision_path, target_ids = _migrate_legacy_resume_model_requirements(
        plan=plan,
        context=context,
        run_dir=tmp_path,
        resume_state={
            "per_step_records": [
                {"step_id": "01_completed_description", "status": "ok"},
            ]
        },
        resume_from_step_id=None,
        role_resolver=lambda role: (
            IncompleteThenValidPacketLLM() if role == "planner" else None
        ),
        evidence=evidence,
        prompt_version="test-pack",
        llm_signature="test-planner",
    )

    assert len(llm_calls) == 2
    retry_prompt = "\n".join(message.content for message in llm_calls[1][0])
    assert "secondary_adjusted_association" in retry_prompt
    assert "outcome" in retry_prompt
    assert "outcome_type" in retry_prompt
    assert "analysis_set" in retry_prompt
    assert revision_path == tmp_path / "analysis_plan_revision_2.json"
    assert target_ids == ("02_remaining_closed",)
    assert revised.analysis_type == plan.analysis_type
    assert revised.rationale == plan.rationale
    assert revised.cohort == plan.cohort
    assert revised.robustness_specs == plan.robustness_specs
    for original_step, revised_step in zip(plan.steps, revised.steps):
        if original_step.step_id == "02_remaining_closed":
            assert revised_step.model_requirements == [
                _planner_requirement(),
                _continuous_planner_requirement(),
            ]
            revised_step = revised_step.model_copy(update={"model_requirements": []})
        assert revised_step == original_step


@pytest.mark.parametrize("invalid_primary_count", [0, 2])
def test_resume_migration_retries_zero_or_multiple_primary_roles(
    tmp_path: Path,
    invalid_primary_count: int,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    context = _context(tmp_path, plan)
    calls = []

    class InvalidThenValidPrimaryPacketLLM:
        name = "invalid-then-valid-primary-packet-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            calls.append(messages)
            if len(calls) == 1:
                payload = json.loads(_valid_packet_json())
                roles = (
                    ["secondary", "secondary"]
                    if invalid_primary_count == 0
                    else ["primary", "primary"]
                )
                for requirement, role in zip(
                    payload["steps"][0]["model_requirements"],
                    roles,
                ):
                    requirement["analysis_role"] = role
                return json.dumps(payload)
            return _valid_packet_json()

    revised, revision_path, _target_ids = (
        _migrate_legacy_resume_model_requirements(
            plan=plan,
            context=context,
            run_dir=tmp_path,
            resume_state={"per_step_records": []},
            resume_from_step_id=None,
            role_resolver=lambda _role: InvalidThenValidPrimaryPacketLLM(),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )
    )

    assert len(calls) == 2
    retry_prompt = "\n".join(message.content for message in calls[1])
    assert "exactly one analysis_role='primary'" in retry_prompt
    assert revision_path is not None
    assert [
        requirement.analysis_role
        for requirement in revised.steps[1].model_requirements
    ] == ["primary", "secondary"]


def test_resume_migration_does_not_touch_completed_or_nonclosed_steps(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    unchanged, revision_path, target_ids = _migrate_legacy_resume_model_requirements(
        plan=plan,
        context=object(),
        run_dir=tmp_path,
        resume_state={
            "per_step_records": [
                {"step_id": "02_remaining_closed", "status": "ok"},
            ]
        },
        resume_from_step_id=None,
        role_resolver=lambda _role: (_ for _ in ()).throw(
            AssertionError("completed/non-closed steps must not invoke the LLM")
        ),
        evidence=evidence,
        prompt_version="test-pack",
        llm_signature="test-planner",
    )

    assert unchanged is plan
    assert revision_path is None
    assert target_ids == ()
    assert not list(tmp_path.glob("analysis_plan_revision_*.json"))


@pytest.mark.parametrize("extra_location", ["packet", "step"])
def test_resume_migration_packet_rejects_out_of_scope_fields(
    tmp_path: Path,
    extra_location: str,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    context = _context(tmp_path, plan)
    calls = []

    class OutOfScopePacketLLM:
        name = "out-of-scope-packet-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            calls.append(messages)
            payload = json.loads(_valid_packet_json())
            if extra_location == "packet":
                payload["research_question"] = "Attempted plan rewrite."
            else:
                payload["steps"][0]["intent"] = "Attempted step rewrite."
            return json.dumps(payload)

    with pytest.raises(LegacyResumePlanMigrationError):
        _migrate_legacy_resume_model_requirements(
            plan=plan,
            context=context,
            run_dir=tmp_path,
            resume_state={
                "per_step_records": [
                    {"step_id": "01_completed_description", "status": "ok"},
                ]
            },
            resume_from_step_id=None,
            role_resolver=lambda _role: OutOfScopePacketLLM(),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )

    assert len(calls) == 3
    assert not list(tmp_path.glob("analysis_plan_revision_*.json"))
    assert evidence.get("analysis_plan_revision_2") is None


def test_resume_migration_llm_failure_has_no_default_model_fallback(
    tmp_path: Path,
) -> None:
    plan = _legacy_plan()
    evidence = _evidence_with_base_plan(tmp_path, plan)
    context = _context(tmp_path, plan)
    calls = []

    class FailingPacketLLM:
        name = "failing-packet-llm"

        def complete(self, messages, *, max_tokens=4096, temperature=0.1):
            calls.append(messages)
            raise RuntimeError("planner unavailable")

    with pytest.raises(
        LegacyResumePlanMigrationError,
        match="stopped without a default model",
    ):
        _migrate_legacy_resume_model_requirements(
            plan=plan,
            context=context,
            run_dir=tmp_path,
            resume_state={"per_step_records": []},
            resume_from_step_id=None,
            role_resolver=lambda _role: FailingPacketLLM(),
            evidence=evidence,
            prompt_version="test-pack",
            llm_signature="test-planner",
        )

    assert len(calls) == 1
    assert all(not step.model_requirements for step in plan.steps)
    assert not list(tmp_path.glob("analysis_plan_revision_*.json"))
    assert evidence.get("analysis_plan_revision_2") is None
