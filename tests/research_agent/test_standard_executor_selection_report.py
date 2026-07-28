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


def _report(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan | None = None,
    plausibility_scope: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Report what the real selector concluded for this exact step.

    The report is a renderer over the selector's trace, so a test that built the
    record any other way would be testing a path production never takes.
    """

    plan = _plan() if plan is None else plan
    trace: list[selection_module.StandardExecutorCandidate] = []
    selection = selection_module.select_standard_executor(
        step,
        plan=plan,
        plausibility_scope=plausibility_scope,
        trace=trace,
    )
    kwargs.setdefault(
        "claimed_by", None if selection is None else selection.analysis_kind
    )
    return standard_executor_candidate_report(
        step,
        plan=plan,
        trace=trace,
        **kwargs,
    )


def _kinds(report: Dict[str, Any]) -> Dict[str, bool]:
    return {
        entry["analysis_kind"]: bool(
            entry["contract_matches"] if entry["kind"] == "owner" else entry["matches"]
        )
        for entry in report["candidates"]
    }


def test_report_names_every_owner_the_selector_consults() -> None:
    """The report must not quietly omit a candidate the selector can pick."""

    # If a new executor is wired into the selector without a report entry, its
    # decline becomes invisible again — which is the whole defect being fixed.
    selector_source = inspect.getsource(selection_module.select_standard_executor)
    reported = set(
        _kinds(
            _report(
                AnalysisStep(
                    step_id="00_unclaimed",
                    intent="Do something no executor owns.",
                    method="unknown_method",
                    inputs=[],
                    expected_outputs=["table:unknown_product"],
                )
            )
        )
    )
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
    assert report["trace_available"] is True
    assert not any(
        entry["contract_matches"]
        for entry in report["candidates"]
        if entry["kind"] == "owner"
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


def test_a_raising_detail_classifier_is_recorded_not_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observability must never be able to fail a step."""

    from easyicu.research_agent.execution.runners import deterministic_missingness

    def _explode(*_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("contract classifier exploded")

    monkeypatch.setattr(
        deterministic_missingness,
        "missingness_audit_input_scope_supported",
        _explode,
    )

    step = AnalysisStep(
        step_id="09_bespoke_analysis",
        intent="Something outside every closed contract.",
        method="bespoke_method",
        inputs=[],
        expected_outputs=["table:bespoke_product"],
    )

    report = standard_executor_candidate_report(step, plan=_plan(), trace=[])

    assert report["claimed_by"] is None
    assert report["owning_candidates"] == []
    errored = [entry for entry in report["candidates"] if entry.get("error")]
    assert errored, "a raising classifier must be recorded on its candidate"
    assert all(entry["matches"] is False for entry in errored)
    assert all("RuntimeError" in entry["error"] for entry in errored)


def test_the_report_cannot_claim_an_owner_the_selector_declined() -> None:
    """The second-registry defect, pinned.

    ``prevalence_mortality_figure``'s contract matches this step, but the
    selector declines it because the step also owes a host-verified
    plausibility receipt its deterministic code cannot emit -- it reads two
    parent tables, not the ranged raw columns.  A report that re-ran the
    ownership predicate reported that owner as available, which is a
    diagnostic that lies exactly where someone is trying to find out why the
    Coder ran.  Ownership must be readable as "matched but declined", not as a
    claim.

    ``descriptive_cohort_summary`` used to be the instance here.  It now
    renders the receipt itself and is selected, so the case moved to an owner
    that still genuinely cannot emit one.
    """

    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )

    step = AnalysisStep.model_validate(
        {
            "step_id": "04_prevalence_mortality_figure",
            "intent": "Render the sealed prevalence and mortality tables.",
            "method": "visualization",
            "planned_analysis_role": "auxiliary",
            "inputs": ["table:cohort_summary", "table:outcome_incidence"],
            "expected_outputs": ["figure:prevalence_mortality"],
            "input_consumption_contracts": [
                {"input_key": "table:cohort_summary", "mode": "all_rows"},
                {"input_key": "table:outcome_incidence", "mode": "all_rows"},
            ],
        }
    )
    scope = FlagOnlyPlausibilityScope(
        step_id="04_prevalence_mortality_figure",
        expected_columns=("sofa2_liver_max",),
        source_contracts_sha256="0" * 64,
        authority_kind="raw_universe",
    )

    report = _report(step, plausibility_scope=scope)

    assert report["claimed_by"] is None
    assert report["owning_candidates"] == ["prevalence_mortality_figure"]
    assert report["declined_after_match"] == ["prevalence_mortality_figure"]
    outcome = {
        entry["analysis_kind"]: entry["outcome"]
        for entry in report["candidates"]
        if entry["kind"] == "owner"
    }
    assert outcome["prevalence_mortality_figure"] == "declined_receipt_required"


def test_a_report_without_a_trace_says_so_instead_of_guessing() -> None:
    """An absent diagnostic is recoverable; a confident wrong one is not."""

    step = AnalysisStep(
        step_id="09_bespoke_analysis",
        intent="Something outside every closed contract.",
        method="bespoke_method",
        inputs=[],
        expected_outputs=["table:bespoke_product"],
    )

    report = standard_executor_candidate_report(step, plan=_plan())

    assert report["trace_available"] is False
    assert report["owning_candidates"] == []
    assert not any(entry["kind"] == "owner" for entry in report["candidates"])


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
