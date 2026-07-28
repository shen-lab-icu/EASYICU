"""A step that no deterministic owner claims must say so in the record.

Both deterministic-ownership defects found in the E1 canary were invisible
until someone rebuilt the step by hand and called the ownership predicates one
at a time.  These tests pin the report that makes the decline readable, and
pin that it observes the selector rather than re-deciding for it.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest

from easyicu.research_agent.execution import phase as execution_phase
from easyicu.research_agent.execution.runners import selection as selection_module
from easyicu.research_agent.execution.runners.selection_report import (
    STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION,
    standard_executor_candidate_report,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe the cohort.",
        robustness_specs=[],
        steps=[],
    )


def _report(step: AnalysisStep, **kwargs: Any) -> Dict[str, Any]:
    return standard_executor_candidate_report(step, plan=_plan(), **kwargs)


def _kinds(report: Dict[str, Any]) -> Dict[str, bool]:
    return {entry["analysis_kind"]: entry["owns"] for entry in report["candidates"]}


def test_report_names_every_owner_the_selector_consults() -> None:
    """The report must not quietly omit a candidate the selector can pick."""

    # If a new executor is wired into the selector without a report entry, its
    # decline becomes invisible again — which is the whole defect being fixed.
    selector_source = inspect.getsource(selection_module.select_standard_executor)
    reported = set(_kinds(_report(AnalysisStep(
        step_id="00_unclaimed",
        intent="Do something no executor owns.",
        method="unknown_method",
        inputs=[],
        expected_outputs=["table:unknown_product"],
    ))))
    selected_kinds = {
        line.split('analysis_kind="', 1)[1].split('"', 1)[0]
        for line in selector_source.splitlines()
        if 'analysis_kind="' in line
    }

    assert selected_kinds
    missing = {
        kind
        for kind in selected_kinds
        # The missingness family is reported through one base entry plus its
        # three closed contracts; the selector spells its three variants out.
        if not kind.startswith("missingness_")
    } - reported
    assert missing == set()


def test_a_step_no_owner_claims_reports_every_candidate_declining() -> None:
    step = AnalysisStep(
        step_id="09_bespoke_analysis",
        intent="Something outside every closed contract.",
        method="bespoke_method",
        inputs=["age"],
        expected_outputs=["table:bespoke_product"],
    )

    report = _report(step)

    assert report["schema_version"] == STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION
    assert report["claimed_by"] is None
    assert report["declared_method"] == "bespoke_method"
    assert report["declared_outputs"] == ["table:bespoke_product"]
    assert report["declared_raw_input_count"] == 1
    assert report["declared_typed_inputs"] == []
    assert report["owning_candidates"] == []
    assert not any(
        entry["owns"] for entry in report["candidates"] if entry["kind"] == "owner"
    )


def _missingness_step(**overrides: Any) -> AnalysisStep:
    payload: Dict[str, Any] = {
        "step_id": "04_missingness_and_event_timing_audit",
        "intent": "Audit per-concept measurement availability.",
        "method": "missingness_measurement_audit",
        "planned_analysis_role": "auxiliary",
        "inputs": [
            "artifact:analysis_cohort",
            "sep3_sofa2_max",
            "sep3_sofa2_measured",
        ],
        "expected_outputs": ["table:missingness_measurement_audit"],
    }
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def test_the_report_names_the_exact_contract_that_claimed_the_step() -> None:
    """The E1 Step 04 shape: which of the closed contracts matched, by name."""

    compact = _kinds(_report(_missingness_step()))
    enriched = _kinds(
        _report(
            _missingness_step(
                method="measurement_bias_audit",
                expected_outputs=[
                    "table:missingness_measurement_audit",
                    "table:measurement_process_audit",
                    "table:exposure_component_completeness_audit",
                ],
            )
        )
    )

    assert compact["missingness_audit"] is True
    assert compact["missingness_audit:compact_contract"] is True
    assert compact["missingness_audit:measurement_bias_contract"] is False
    assert enriched["missingness_audit"] is True
    assert enriched["missingness_audit:measurement_bias_contract"] is True
    assert enriched["missingness_audit:compact_contract"] is False


def test_an_unrecognised_enrichment_shows_scope_kept_but_contract_lost() -> None:
    """Inputs still legal, every closed contract gone — the readable decline."""

    # A product set no contract covers is the case the old record could not
    # distinguish from "this analysis is not deterministic at all".
    unrecognised = _kinds(
        _report(
            _missingness_step(
                method="measurement_bias_audit",
                expected_outputs=[
                    "table:missingness_measurement_audit",
                    "table:measurement_process_audit",
                    "table:exposure_component_completeness_audit",
                    "table:informative_visit_process_model",
                ],
            )
        )
    )

    assert unrecognised["missingness_audit"] is False
    assert unrecognised["missingness_audit:input_scope"] is True
    assert not any(
        unrecognised[key]
        for key in (
            "missingness_audit:availability_contract",
            "missingness_audit:complete_case_contract",
            "missingness_audit:compact_contract",
            "missingness_audit:measurement_bias_contract",
        )
    )


def test_a_raising_predicate_is_recorded_not_propagated() -> None:
    """Observability must never be able to fail a step."""

    step = AnalysisStep(
        step_id="09_bespoke_analysis",
        intent="Something outside every closed contract.",
        method="bespoke_method",
        inputs=[],
        expected_outputs=["table:bespoke_product"],
    )

    class Exploding:
        @property
        def display_labels(self) -> Dict[str, str]:
            raise RuntimeError("plan display_labels exploded")

    report = standard_executor_candidate_report(step, plan=Exploding())

    assert report["claimed_by"] is None
    assert report["owning_candidates"] == []
    errored = [entry for entry in report["candidates"] if entry.get("error")]
    assert errored, "an exploding plan surface must be recorded on some candidate"
    assert all(entry["owns"] is False for entry in errored)
    assert all("RuntimeError" in entry["error"] for entry in errored)


def test_execute_phase_records_the_report_for_claimed_and_unclaimed_steps() -> None:
    source = inspect.getsource(execution_phase.run_execute_phase)

    assert 'step_record["standard_executor_candidates"] = (' in source
    # Written outside the `if standard_executor is not None:` body, so an
    # unclaimed step — the case that needs explaining — is recorded too.
    claimed_branch = source.index("if standard_executor is not None:")
    write_site = source.index('step_record["standard_executor_candidates"] = (')
    preflight_assignment = source.index(
        "preflight_standard_code = standard_executor.code", claimed_branch
    )
    assert write_site > preflight_assignment


@pytest.mark.parametrize("claimed", ["grouped_table_one", None])
def test_claimed_by_is_recorded_verbatim(claimed: str | None) -> None:
    step = AnalysisStep(
        step_id="03_table_one",
        intent="Compare baseline characteristics.",
        method="descriptive_table_one",
        inputs=[],
        expected_outputs=["table:table_one"],
    )

    report = _report(step, claimed_by=claimed)

    assert report["claimed_by"] == claimed
