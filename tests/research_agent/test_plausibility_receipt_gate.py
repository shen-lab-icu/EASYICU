"""The run-artifact half of the `retain_and_flag` obligation.

The static gate next door proves a script is *shaped* to record its
out-of-range counts. A shape is not a result: the script can be shaped
correctly and still leave nothing behind, and no amount of reading the source
can tell the difference. These tests are about what the step actually produced.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.gates.plausibility_receipt import (
    RECEIPT_CONTRACT_CLAUSE,
    RECEIPT_CONTRACT_SENTENCE,
    plausibility_audit_receipt_findings,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)

# Enough of a script to put the step under the obligation: it reads the sealed
# range, so it owes a receipt.
READS_THE_RANGE = 'bounds = contract.get("analysis_plausibility_range")\n'

COMPLIANT = {
    "policy": "retain_and_flag",
    "below_minimum_n": 0,
    "above_maximum_n": 0,
    "out_of_range_n": 0,
}


def _context(*, ranged: bool = True) -> ResearchContext:
    return ResearchContext(
        research_question="Assess a continuous ICU marker.",
        cohort=CohortDescriptor(
            cohort_name="c", database="synthetic", n_stays=3, n_patients=3
        ),
        variables=[
            ConceptDescriptor(
                name="marker",
                dtype="float64",
                valid_range=[0.0, 10.0] if ranged else None,
            )
        ],
    )


def _step() -> AnalysisStep:
    return AnalysisStep(step_id="06_primary", intent="associate", method="logistic")


def _reasons(summary, *, ranged: bool = True, script: str = READS_THE_RANGE):
    findings = plausibility_audit_receipt_findings(
        step_summary=summary,
        context=_context(ranged=ranged),
        step=_step(),
        script_text=script,
    )
    assert all(finding.severity == "error" for finding in findings)
    # Every block quotes the contract, so a repair is told the exact shape it
    # owes rather than being asked to guess one.
    assert all(RECEIPT_CONTRACT_SENTENCE in finding.message for finding in findings)
    return {str((finding.detail or {}).get("reason")) for finding in findings}


def test_code_that_claims_to_write_the_summary_but_leaves_no_receipt_is_blocked():
    """The gap this half exists to close.

    The script reads the range and writes a summary, so the static gate is
    satisfied -- and the artifact carries nothing. Only reading the run output
    can see that.
    """

    assert _reasons({"status": "completed", "n_full": 94456}) == {
        "plausibility_audit_receipt_absent"
    }


def test_a_receipt_of_all_zeros_is_a_result_and_passes():
    """Zero is the answer, not the absence of one."""

    assert _reasons({"plausibility_audit": {"marker": COMPLIANT}}) == set()


def test_a_receipt_reporting_real_counts_passes():
    assert (
        _reasons(
            {
                "plausibility_audit": {
                    "marker": {
                        **COMPLIANT,
                        "below_minimum_n": 2,
                        "above_maximum_n": 1,
                        "out_of_range_n": 3,
                    }
                }
            }
        )
        == set()
    )


def test_a_list_of_records_naming_its_own_variable_is_accepted():
    """The fields are the contract; the container is not worth a repair."""

    assert (
        _reasons({"plausibility_audit": [{"variable": "marker", **COMPLIANT}]}) == set()
    )


def test_a_receipt_echoing_the_hosts_own_policy_object_is_accepted():
    """Taken verbatim from a real run's step summary.

    The script wrote back the whole `plausibility_policy` object the host had
    handed it in the contract rather than the action string. Reading only the
    string would have blocked a receipt that was, if anything, more faithful --
    and it would have blocked it on the first real step this gate ever saw.
    """

    assert (
        _reasons(
            {
                "plausibility_audit": {
                    "marker": {
                        **COMPLIANT,
                        "minimum": 0.0,
                        "maximum": 10.0,
                        "policy": {
                            "out_of_range_action": "retain_and_flag",
                            "range_policy": "flag_only",
                        },
                    }
                }
            }
        )
        == set()
    )


def test_a_policy_object_naming_a_different_action_is_still_blocked():
    """Reading the object must not become accepting any object."""

    assert _reasons(
        {
            "plausibility_audit": {
                "marker": {
                    **COMPLIANT,
                    "policy": {"out_of_range_action": "exclude"},
                }
            }
        }
    ) == {"plausibility_audit_policy_mismatch"}


@pytest.mark.parametrize(
    ("receipt", "reason"),
    [
        pytest.param({}, "plausibility_audit_receipt_empty", id="empty"),
        pytest.param(
            {"marker": 0}, "plausibility_audit_record_untyped", id="bare_number"
        ),
        pytest.param(
            {"other_column": COMPLIANT},
            "plausibility_audit_variable_not_declared",
            id="undeclared_variable",
        ),
        pytest.param(
            {"marker": {**COMPLIANT, "policy": "exclude"}},
            "plausibility_audit_policy_mismatch",
            id="wrong_policy",
        ),
        pytest.param(
            {"marker": {"policy": "retain_and_flag", "below_minimum_n": 0}},
            "plausibility_audit_count_missing",
            id="count_omitted",
        ),
        pytest.param(
            {"marker": {**COMPLIANT, "above_maximum_n": True}},
            "plausibility_audit_count_missing",
            id="count_is_a_flag",
        ),
        pytest.param(
            {"marker": {**COMPLIANT, "below_minimum_n": 2}},
            "plausibility_audit_count_inconsistent",
            id="counts_do_not_add_up",
        ),
    ],
)
def test_a_receipt_that_does_not_carry_the_counts_is_blocked(receipt, reason):
    """`above_maximum_n: True` is the one worth naming: `bool` is an `int` in
    Python, so a receipt can record that something happened while recording
    nothing about how much."""

    assert _reasons({"plausibility_audit": receipt}) == {reason}


def test_a_run_that_declares_no_range_owes_no_receipt():
    assert _reasons({}, ranged=False) == set()


def test_a_step_that_never_reads_the_range_owes_no_receipt():
    """Same trigger as the static gate, so the two halves cannot disagree about
    which steps are under the obligation."""

    assert _reasons({}, script="rows = [1, 2, 3]\n") == set()


def test_the_repair_receipt_alone_puts_a_step_under_the_obligation():
    """After the deterministic repair the script may no longer name the range,
    and that is exactly when the LLM auditor goes quiet. The marker keeps the
    obligation alive."""

    repaired = "pass  # _easyicu_flag_only_plausibility_range_retained_v1\n"
    assert _reasons({}, script=repaired) == {"plausibility_audit_receipt_absent"}


def test_the_published_contract_and_the_gate_share_their_field_names():
    """A gate demanding a shape nobody was told to write blocks everything, so
    the Coder's clause is rendered from the same constants the gate reads."""

    for field in ("plausibility_audit", "below_minimum_n", "above_maximum_n"):
        assert field in RECEIPT_CONTRACT_CLAUSE
        assert field in RECEIPT_CONTRACT_SENTENCE


def test_the_receipt_gate_runs_inside_the_final_deterministic_gates():
    """It has to reach the pipeline, not only its own unit test."""

    from easyicu.research_agent.execution import phase

    from .test_gate_evaluator_contract import gate_call_order

    order = gate_call_order(
        phase._evaluate_final_deterministic_gates,
        [
            "_demote_step_contract_for_primary_runner",
            "_demote_result_figure_shape_for_family_renderer",
            "plausibility_audit_receipt_findings",
        ],
    )
    assert "plausibility_audit_receipt_findings" in order
    # After the demotion passes: a fail-closed obligation is not something a
    # narrow family demotion should be able to soften.
    assert (
        order["_demote_result_figure_shape_for_family_renderer"]
        < order["plausibility_audit_receipt_findings"]
    )


def test_a_missing_receipt_enters_the_repair_loop_instead_of_sealing_the_step():
    """Where the check runs decides whether the step can recover.

    Evaluated only in the final gate, a missing or malformed receipt arrives
    after evidence registration and becomes a terminal `contract_failed`
    record -- the step dies holding a finding that says exactly what to write.
    It has to be raised in the pre-registration gate, before the repair
    dispatch, so the Coder gets its one attempt at it.
    """

    from easyicu.research_agent.execution import phase

    from .test_gate_evaluator_contract import gate_call_order

    order = gate_call_order(
        phase.run_execute_phase,
        [
            "_step_deterministic_contract_findings",
            "_demote_step_contract_for_primary_runner",
            "plausibility_audit_receipt_findings",
            "_deterministic_summary_repair",
            "deterministic_contract_repair",
            "_step_contract_repair_guidance",
        ],
    )
    assert "plausibility_audit_receipt_findings" in order, (
        "the receipt check never runs in the pre-registration gate, so a "
        "missing receipt can only ever be terminal"
    )
    # Raised with the other early contract findings...
    assert (
        order["_step_deterministic_contract_findings"]
        < order["plausibility_audit_receipt_findings"]
    )
    assert (
        order["_demote_step_contract_for_primary_runner"]
        < order["plausibility_audit_receipt_findings"]
    )
    # ...and before every repair the loop can spend on them.
    for repair in (
        "_deterministic_summary_repair",
        "deterministic_contract_repair",
        "_step_contract_repair_guidance",
    ):
        assert order["plausibility_audit_receipt_findings"] < order[repair], repair


def test_a_receipt_finding_is_not_one_of_the_unrepairable_terminal_classes():
    """Being early is not enough if the finding is classified as terminal.

    `_locked_measurement_data_quality_issues` short-circuits the repair loop
    for locked-cohort facts generated code cannot fix. A missing receipt is the
    opposite of that: the script simply has to write one.
    """

    from easyicu.research_agent.execution.phase import (
        _locked_measurement_data_quality_issues,
    )

    findings = plausibility_audit_receipt_findings(
        step_summary={"rows": 1},
        context=_context(),
        step=_step(),
        script_text=READS_THE_RANGE,
    )
    assert findings
    assert _locked_measurement_data_quality_issues(findings) == []
