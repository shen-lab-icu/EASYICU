"""The prespecified stability rule's outcome is a host claim, whichever way it falls.

A selected class solution is frozen only when every planned subsample refit
reproduces it and the refits' mean agreement reaches the planner's minimum.
When the rule rejected the solution, the owner reported an execution failure
and the study had no manuscript; when the rule accepted it, no claim stated
the result the class-count rule had handed on.  The stability owner now
states the rule's outcome in a closed, versioned envelope, its disposition
must follow from its own numbers, and every number in its sentences binds to
the owner.  Synthetic values only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.prespecified_rule_outcomes import (
    RULE_OUTCOME_SCHEMA_VERSION,
    derive_rule_outcome_claim_payloads,
    validate_rule_outcome,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

STEP = "02_class_stability"


def _stability(**updates) -> dict:
    payload = {
        "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
        "rule": "class_solution_stability",
        "metric": "adjusted_rand_index",
        "selected_class_count": 3,
        "planned_resamples": 50,
        "successful_resamples": 50,
        "minimum_successful_resamples": 50,
        "mean_stability": 0.412,
        "minimum_mean_stability": 0.7,
        "disposition": "stability_below_threshold",
    }
    payload.update(updates)
    return payload


_MET = {"mean_stability": 0.861, "disposition": "stability_threshold_met"}
_SHORT = {
    "successful_resamples": 47,
    "mean_stability": None,
    "disposition": "too_few_successful_refits",
}


@pytest.mark.parametrize(
    "updates, result, conclusion",
    [
        (
            {},
            "the mean adjusted Rand index across 50 subsample refits was 0.41, "
            "below the prespecified minimum of 0.70; under the prespecified rule no "
            "class solution was frozen.",
            "did not meet the prespecified resampling stability rule",
        ),
        (
            _MET,
            "the mean adjusted Rand index across 50 subsample refits was 0.86, at or "
            "above the prespecified minimum of 0.70, and the candidate solution was "
            "frozen.",
            "met the prespecified resampling stability rule",
        ),
        (
            _SHORT,
            "47 of 50 subsample refits converged and realized every class, fewer "
            "than the 50 the rule requires; stability was not established",
            "did not meet the prespecified resampling stability rule",
        ),
    ],
)
def test_each_disposition_states_its_own_rule(updates, result, conclusion) -> None:
    outcome = validate_rule_outcome(_stability(**updates))

    assert outcome.result_sentence().startswith(
        "In the prespecified resampling stability assessment of the 3-class "
        "candidate solution, "
    )
    assert result in outcome.result_sentence()
    assert conclusion in outcome.conclusion_sentence()
    assert outcome.report_section == "primary"


def test_a_report_only_design_can_still_fail_its_refit_minimum() -> None:
    outcome = validate_rule_outcome(_stability(**_SHORT, minimum_mean_stability=None))

    assert outcome.disposition == "too_few_successful_refits"
    assert "minimum of" not in outcome.result_sentence()


def test_a_mean_at_the_minimum_meets_the_rule() -> None:
    outcome = validate_rule_outcome(_stability(**{**_MET, "mean_stability": 0.7}))

    assert "was 0.70, at or above the prespecified minimum of 0.70," in (
        outcome.result_sentence()
    )


def test_a_mean_just_below_the_minimum_does_not_read_as_equal() -> None:
    outcome = validate_rule_outcome(_stability(mean_stability=0.696))

    assert "was 0.696, below the prespecified minimum of 0.700;" in (
        outcome.result_sentence()
    )


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param({"mean_stability": 0.82}, id="met_but_called_below"),
        pytest.param({**_MET, "mean_stability": 0.41}, id="below_but_called_met"),
        pytest.param({**_SHORT, "mean_stability": 0.5}, id="unestablished_with_a_mean"),
        pytest.param({"mean_stability": None}, id="established_without_a_mean"),
        pytest.param({"successful_resamples": 51}, id="more_refits_than_planned"),
        pytest.param({"minimum_mean_stability": None}, id="decision_without_a_minimum"),
        pytest.param({**_SHORT, "successful_resamples": 50}, id="enough_refits_called_short"),
    ],
)
def test_an_envelope_that_contradicts_itself_fails_closed(updates) -> None:
    with pytest.raises(ValidationError):
        validate_rule_outcome(_stability(**updates))


def test_the_rule_outcome_is_a_primary_claim_named_for_the_rule() -> None:
    [payload] = derive_rule_outcome_claim_payloads(
        {"status": "ok", "reportable_rule_outcomes": [_stability()]}
    )

    assert payload["claim_id"] == "class_stability_rule"
    assert payload["analysis_role"] == "primary"
    assert payload["direction"] == "descriptive_only"
    assert payload["estimand"] == "prespecified rule outcome: stability below threshold"


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param({}, id="below"),
        pytest.param(_MET, id="met"),
        pytest.param(_SHORT, id="short"),
        pytest.param({"mean_stability": 0.696}, id="just_below"),
    ],
)
def test_every_number_in_a_stability_sentence_binds_to_its_owner(
    tmp_path: Path, updates
) -> None:
    summary = {"status": "ok", "reportable_rule_outcomes": [_stability(**updates)]}
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic", description="Owner summary", source_path=source,
        evidence_id="owner_summary", produced_by_step=STEP, producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id="owner_summary", summary=summary,
    )
    ledger = [{"step_id": STEP, "status": "ok", "evidence_ids": ["owner_summary"]}]
    [claim] = store.authoritative_scientific_claims(ledger)

    for sentence in (claim.render_reader_text(),
                     claim.render_reader_text(include_estimate=False)):
        bound = store.bind_manuscript(
            f"## Results\n\n{sentence} {{evidence:{claim.evidence_id}}}\n",
            per_step_records=ledger,
        )
        _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
        assert not untraced, sentence
