"""A prespecified rule's formal outcome is a host claim, even when it selects nothing.

A signed owner can run to completion while its rule rejects the result: a
class-count criterion lowest at the grid's upper boundary, a class description
with no frozen solution, a planned analysis the inputs cannot support.  No
claim producer covered these outcomes, so the strict Results grammar had
nothing to state and a run whose owners all succeeded left its required
Results empty.  Owners now emit closed, versioned rule-outcome envelopes, and
the host derives a claim from each.  Synthetic values only.
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
from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    ScientificClaimDraft,
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
    scientific_claim_compilation_requested,
)

STEP = "01_class_candidates"


def _class_count(**updates) -> dict:
    payload = {
        "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
        "rule": "information_criterion_class_count",
        "criterion": "bic",
        "candidate_class_counts": [2, 3, 4, 5],
        "criterion_minimum_class_count": 5,
        "n_records": 1240,
        "smallest_class_fraction": 0.081,
        "minimum_class_fraction": 0.05,
        "disposition": "minimum_at_upper_boundary",
    }
    payload.update(updates)
    return payload


def _eligibility(**updates) -> dict:
    payload = {
        "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
        "rule": "minimum_observed_windows",
        "anchor": "icu_admission",
        "window_start_hours": 0,
        "window_end_hours": 48,
        "window_width_hours": 8,
        "n_windows": 6,
        "minimum_observed_windows": 3,
        "input_n": 1402,
        "included_n": 1240,
        "excluded_n": 162,
    }
    payload.update(updates)
    return payload


def _claims(summary: dict, step_id: str = STEP) -> list[ScientificClaim]:
    drafts = derive_scientific_claim_drafts(summary)
    return bind_scientific_claim_drafts(
        [draft.model_dump(mode="json") for draft in drafts],
        step_id=step_id,
        evidence_id=f"{step_id}_summary",
    )


def test_an_upper_boundary_minimum_is_reported_as_no_class_solution() -> None:
    [claim] = _claims({"status": "ok", "reportable_rule_outcomes": [_class_count()]})

    assert claim.claim_ref == f"{STEP}.class_count_rule"
    assert claim.schema_version == "easyicu.scientific_claim/4"
    assert claim.render_reader_text() == (
        "Among 1,240 records in the class model, the Bayesian information criterion "
        "across the prespecified candidate range of 2 to 5 classes was lowest at the "
        "upper boundary of 5 classes; under the prespecified rule this is not an "
        "interior solution, and no class solution was selected."
    )
    assert claim.render_reader_text(include_estimate=False) == (
        "The prespecified class-count rule selected no class solution within the "
        "candidate range, so no classes are described or interpreted."
    )
    # The machine form the Writer sees is the same result, with its role.
    assert claim.render_text().endswith(
        "no class solution was selected (prespecified rule outcome; analysis role: primary)."
    )


@pytest.mark.parametrize(
    ("updates", "result", "conclusion"),
    [
        pytest.param(
            {"criterion_minimum_class_count": 3, "smallest_class_fraction": 0.031,
             "disposition": "smallest_class_below_minimum"},
            "was lowest at 3 classes, but the smallest class held a proportion of "
            "0.031 of records, below the prespecified minimum of 0.050; under the "
            "prespecified rule no class solution was selected.",
            "minimum class-size rules selected no class solution",
            id="smallest_class_below_minimum",
        ),
        pytest.param(
            {"criterion_minimum_class_count": 3, "disposition": "minimum_selected"},
            "was lowest at 3 classes, and the smallest class held a proportion of "
            "0.081 of records; this candidate solution proceeded to the prespecified "
            "stability assessment.",
            "selected a 3-class candidate solution for the prespecified stability",
            id="minimum_selected",
        ),
    ],
)
def test_each_disposition_states_its_own_rule(updates, result, conclusion) -> None:
    [claim] = _claims(
        {"status": "ok", "reportable_rule_outcomes": [_class_count(**updates)]}
    )

    assert claim.render_reader_text().endswith(result)
    assert conclusion in claim.render_reader_text(include_estimate=False)


def test_the_eligibility_rule_states_its_window_and_counts() -> None:
    [claim] = _claims(
        {"status": "ok", "reportable_rule_outcomes": [_eligibility()]},
        step_id="00_panel",
    )

    assert claim.claim_ref == "00_panel.observed_window_rule"
    assert claim.analysis_role == "auxiliary"
    assert claim.rule_outcome.report_section == "cohort"
    assert claim.render_reader_text() == (
        "Of 1,402 records in the longitudinal panel, 1,240 had at least 3 of the 6 "
        "prespecified 8-hour windows from 0 to 48 hours after ICU admission observed "
        "and entered the class model; 162 were excluded."
    )


def test_wording_follows_the_grid_and_anchor_it_is_given() -> None:
    [sparse] = _claims(
        {"status": "ok", "reportable_rule_outcomes": [
            _class_count(candidate_class_counts=[2, 4, 6],
                         criterion_minimum_class_count=6)
        ]}
    )
    [before] = _claims(
        {"status": "ok", "reportable_rule_outcomes": [
            _eligibility(anchor="hospital_admission", window_start_hours=-12,
                         window_end_hours=36)
        ]},
        step_id="00_panel",
    )

    assert "across the prespecified candidates of 2, 4, and 6 classes" in (
        sparse.render_reader_text()
    )
    assert "from -12 to 36 hours relative to hospital admission" in (
        before.render_reader_text()
    )


def test_class_description_and_feasibility_outcomes_are_claims() -> None:
    [description] = _claims(
        {"status": "ok", "reportable_rule_outcomes": [{
            "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
            "rule": "frozen_class_description",
            "disposition": "no_frozen_solution",
        }]},
        step_id="05_class_description",
    )
    [feasibility] = _claims(
        {"status": "ok", "reportable_rule_outcomes": [{
            "schema_version": RULE_OUTCOME_SCHEMA_VERSION,
            "rule": "planned_analysis_feasibility",
            "disposition": "not_executable_from_sealed_inputs",
            "planned_analysis_role": "sensitivity",
        }]},
        step_id="08_repeat_measure_protocol",
    )

    assert description.render_reader_text().startswith("No class solution was frozen")
    assert feasibility.analysis_role == "sensitivity"
    assert feasibility.render_reader_text() == (
        "A prespecified sensitivity analysis was not executable from the study "
        "inputs, and no estimate was produced for it."
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_class_count(disposition="minimum_selected"),
                     id="boundary_minimum_called_selected"),
        pytest.param(_class_count(criterion_minimum_class_count=3,
                                  smallest_class_fraction=0.02,
                                  disposition="minimum_selected"),
                     id="small_class_called_selected"),
        pytest.param(_class_count(criterion_minimum_class_count=3,
                                  disposition="smallest_class_below_minimum"),
                     id="adequate_class_called_small"),
        pytest.param(_class_count(criterion_minimum_class_count=7),
                     id="minimum_outside_the_grid"),
        pytest.param(_class_count(candidate_class_counts=[2, 4, 3, 5]),
                     id="grid_not_increasing"),
        pytest.param(_class_count(candidate_class_counts=[1, 2, 5]),
                     id="grid_below_two"),
        pytest.param(_class_count(n_records=True), id="boolean_count"),
        pytest.param(_class_count(note="free text"), id="extra_key"),
        pytest.param(_eligibility(included_n=1241), id="counts_do_not_add_up"),
        pytest.param(_eligibility(n_windows=5), id="grid_does_not_tile_window"),
        pytest.param(_eligibility(minimum_observed_windows=7), id="minimum_exceeds_grid"),
        pytest.param(_eligibility(anchor="ICU admission"), id="anchor_not_a_key"),
        pytest.param({"schema_version": RULE_OUTCOME_SCHEMA_VERSION,
                      "rule": "planned_analysis_feasibility",
                      "disposition": "not_executable_from_sealed_inputs",
                      "planned_analysis_role": "auxiliary"},
                     id="auxiliary_feasibility"),
        pytest.param({**_class_count(), "rule": "significance_threshold"},
                     id="unknown_rule"),
        pytest.param({**_class_count(), "schema_version": "easyicu.prespecified_rule_outcome/2"},
                     id="unknown_version"),
    ],
)
def test_an_envelope_that_contradicts_itself_fails_closed(payload) -> None:
    with pytest.raises(ValidationError):
        validate_rule_outcome(payload)
    with pytest.raises(ValueError):
        derive_scientific_claim_drafts(
            {"status": "ok", "reportable_rule_outcomes": [payload]}
        )


def test_the_envelope_list_itself_is_closed() -> None:
    for summary in (
        {"status": "ok", "reportable_rule_outcomes": []},
        {"status": "ok", "reportable_rule_outcomes": _class_count()},
        {"status": "ok", "reportable_rule_outcomes": [_class_count(), _class_count()]},
        {"status": "failed", "reportable_rule_outcomes": [_class_count()]},
    ):
        with pytest.raises(ValueError):
            derive_rule_outcome_claim_payloads(summary)


def test_a_rule_outcome_claim_carries_no_estimate_and_no_other_claim_carries_one() -> None:
    [payload] = derive_rule_outcome_claim_payloads(
        {"status": "ok", "reportable_rule_outcomes": [_class_count()]}
    )
    for forged in (
        {**payload, "point_estimate": 1.0, "interval_lower": 0.5, "interval_upper": 2.0},
        {**payload, "adjusted_for": ["age"]},
        {**payload, "direction": "positive"},
        {**payload, "schema_version": "easyicu.scientific_claim/2"},
        {key: value for key, value in payload.items() if key != "rule_outcome"},
    ):
        with pytest.raises(ValidationError):
            ScientificClaimDraft.model_validate(forged)
    association = {
        "claim_id": "adjusted_association", "claim_type": "association",
        "exposure": "lactate max", "outcome": "icu readmission",
        "direction": "no_clear_association", "estimand": "adjusted odds ratio",
        "population": "the complete case analysis set", "analysis_role": "primary",
        "status": "supported",
    }
    with pytest.raises(ValidationError):
        ScientificClaimDraft.model_validate(
            {**association, "rule_outcome": payload["rule_outcome"]}
        )


def test_older_claim_payloads_keep_their_exact_bytes() -> None:
    summary = {
        "interpretation_class": "adjusted_association",
        "exposure": "lactate_max",
        "outcome": "icu_readmission",
        "effect_scale": "odds_ratio",
        "primary_estimate_interval": [0.82, 1.31],
        "analysis_set": "complete_case",
        "analysis_role": "primary",
        "adjustment_covariates": ["age", "sex"],
    }
    [draft] = derive_scientific_claim_drafts(summary)

    assert "rule_outcome" not in draft.model_dump(mode="json")
    assert "rule_outcome" not in json.dumps(
        bind_scientific_claim_drafts(
            [draft.model_dump(mode="json")], step_id="04_model", evidence_id="e",
        )[0].model_dump(mode="json")
    )


def test_a_rule_outcome_sits_beside_another_envelope_in_one_summary() -> None:
    summary = {
        "status": "ok",
        "interpretation_class": "adjusted_association",
        "exposure": "lactate_max",
        "outcome": "icu_readmission",
        "effect_scale": "odds_ratio",
        "primary_estimate_interval": [0.82, 1.31],
        "analysis_set": "complete_case",
        "analysis_role": "primary",
        "adjustment_covariates": ["age"],
        "reportable_rule_outcomes": [_eligibility()],
    }

    assert scientific_claim_compilation_requested(summary)
    assert [draft.claim_id for draft in derive_scientific_claim_drafts(summary)] == [
        "observed_window_rule", "adjusted_association",
    ]


def _register(tmp_path: Path, summary: dict, *, generation_mode: str) -> EvidenceStore:
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir)
    store.register_file(
        kind="statistic",
        description="Candidate selection summary.",
        source_path=source,
        evidence_id="candidate_summary",
        produced_by_step=STEP,
        producer="runner",
        generation_mode=generation_mode,
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id="candidate_summary", summary=summary,
    )
    return store


def test_only_a_deterministic_owner_obtains_the_claim(tmp_path: Path) -> None:
    summary = {"status": "ok", "reportable_rule_outcomes": [_class_count()]}

    store = _register(tmp_path / "owner", summary, generation_mode="deterministic_standard")
    [claim] = store.scientific_claims()
    assert claim.claim_ref == f"{STEP}.class_count_rule"
    assert claim.evidence_id == "candidate_summary"
    # The claim reloads by re-deriving it from the sealed summary bytes.
    assert EvidenceStore(tmp_path / "owner" / "run").scientific_claims() == [claim]

    # A generated script cannot issue itself the same claim.
    scripted = _register(tmp_path / "script", summary, generation_mode="llm")
    assert scripted.scientific_claims() == []


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(_class_count(), id="upper_boundary"),
        # The grid holds a 5 and the minimum is 0.05: the two must not collide.
        pytest.param(_class_count(criterion_minimum_class_count=3,
                                  smallest_class_fraction=0.031,
                                  disposition="smallest_class_below_minimum"),
                     id="smallest_class_below_minimum"),
        pytest.param(_class_count(criterion_minimum_class_count=4,
                                  disposition="minimum_selected"),
                     id="minimum_selected"),
        pytest.param(_eligibility(), id="observed_windows"),
    ],
)
def test_every_number_in_a_rule_sentence_binds_to_its_owner(tmp_path: Path, outcome) -> None:
    summary = {"status": "ok", "reportable_rule_outcomes": [outcome]}
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
