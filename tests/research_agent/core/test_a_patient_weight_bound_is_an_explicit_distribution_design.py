"""The same declared weights and assumptions reach computation and drawing."""

import copy
import hashlib
import json

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    _distribution_rows,
    _proportion_interval,
    _verify_product,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import _validate


from .patient_weighted_distribution_fixtures import ASSUMPTIONS, _spec


@pytest.mark.parametrize("event", [0, 1])
def test_declared_bound_handles_both_boundaries_and_is_drawable(event):
    frame = pd.DataFrame({"patient": ["a"] * 99 + ["b", "c", "d"],
                          "group": [0] * 100 + [1] * 2, "event": [event] * 102})
    spec = _spec()
    rows = _distribution_rows(frame, spec=spec)
    _verify_product(rows, spec=spec)
    _, _, design, _ = _validate(pd.DataFrame(rows))
    assert design["interval_method"] == "patient_weighted_hoeffding"
    assert json.loads(design["patient_weighted_interval_assumptions"]) == ASSUMPTIONS
    assert rows[0]["outcome_interval_cluster_size_square_sum"] == 9802
    assert rows[0]["outcome_interval_cluster_count"] == 2
    assert rows[0]["outcome_standard_error_pct"] is None
    assert (rows[0]["ci_low_pct"], rows[0]["ci_high_pct"]) == (0.0, 100.0)


def test_weights_are_recomputed_on_the_actual_complete_case_denominator():
    frame = pd.DataFrame({"patient": ["a"] * 99 + ["b"], "event": [0] * 98 + [None, 0]})
    denominator = frame.event.notna()
    receipt = _proportion_interval(
        frame, spec=_spec(denominator_policy="observed_outcome_rows",
                          missing_outcome_policy="exclude_from_denominator"),
        denominator_mask=denominator, numerator_mask=frame.event.eq(1),
    )
    assert receipt["cluster_size_square_sum"] == 98**2 + 1
    assert receipt["cluster_count"] == 2


@pytest.mark.parametrize("updates", [
    {"patient_weighted_interval_assumptions": None},
    {"repeated_unit_interval_method": "patient_cluster_robust_wald"},
    {"patient_weighted_interval_assumptions": {**ASSUMPTIONS, "weights": "equal_patients"}},
])
def test_a_method_or_assumption_change_cannot_be_silently_accepted(updates):
    with pytest.raises(ValidationError):
        _spec(**updates)


@pytest.mark.parametrize("column,wrong", [
    ("outcome_interval_cluster_size_square_sum", 5000),
    ("ci_high_pct", 65.761977),
    ("outcome_interval_covariance", "cluster_robust"),
    ("outcome_standard_error_pct", 0.0),
    ("patient_weighted_interval_assumptions", None),
])
def test_bound_weights_uncertainty_and_assumptions_cannot_be_tampered(column, wrong):
    frame = pd.DataFrame({"patient": ["a"] * 99 + ["b", "c", "d"],
                          "group": [0] * 100 + [1] * 2, "event": [0] * 102})
    table = pd.DataFrame(_distribution_rows(frame, spec=_spec()))
    table.loc[0, column] = wrong
    with pytest.raises(ValueError):
        _validate(table)


def test_no_grouping_authority_can_be_invented_to_execute_the_bound():
    frame = pd.DataFrame({"event": [0, 0, 0]})
    with pytest.raises(RuntimeError, match="host-bound patient groups"):
        _proportion_interval(frame, spec=_spec(dependence=None),
                             denominator_mask=frame.event.notna(),
                             numerator_mask=frame.event.eq(1))


def _run_product(tmp_path, monkeypatch):
    from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
        run_exposure_outcome_distribution_from_env,
    )

    frame = pd.DataFrame({"patient": ["a"] * 99 + ["b", "c", "d"],
                          "group": [0] * 100 + [1] * 2, "event": [0] * 102})
    outputs = tmp_path / "steps/distribution/outputs"
    outputs.mkdir(parents=True)
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    resolved = tmp_path / "resolved_inputs.json"
    resolved.write_text(json.dumps({"step_id": "distribution", "inputs": {
        "artifact:analysis_cohort": {
            "relative_path": "cohort.parquet", "declared_kind": "artifact",
            "product": "analysis_cohort", "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
            "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        },
    }}))
    monkeypatch.setenv("STEP_OUT_DIR", str(outputs))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved))
    summary = run_exposure_outcome_distribution_from_env(
        spec_payload=_spec().model_dump(mode="json"),
        typed_cohort_input="artifact:analysis_cohort", analysis_role="primary",
    )
    return summary, outputs / "exposure_outcome_distribution.csv"


def test_public_producer_drawer_and_claim_authority_share_one_declared_bound(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.scientific_claims import ScientificClaim, derive_scientific_claim_drafts
    from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
        run_exposure_outcome_distribution_figure,
    )

    summary, table = _run_product(tmp_path, monkeypatch)
    risks = summary["descriptive_estimates"]["outcome_absolute_risks"]
    assert risks[0]["cluster_size_square_sum"] == 9802
    drafts = derive_scientific_claim_drafts(summary)
    assert len(drafts) == 2
    assert {draft.interval_method for draft in drafts} == {"patient_weighted_hoeffding"}
    assert all("conditional on fixed observed patient weights" in draft.estimand for draft in drafts)
    assert all("covariance:" not in draft.estimand for draft in drafts)
    claims = [ScientificClaim(**draft.model_dump(), step_id="distribution", evidence_id="synthetic-summary") for draft in drafts]
    assert all("conditional on the observed patient weights" in claim.render_reader_text() for claim in claims)
    assert all("conditional on the observed patient weights" in claim.render_reader_text(include_estimate=False) for claim in claims)
    assert all(draft.patient_weighted_interval_assumptions.model_dump() == ASSUMPTIONS for draft in drafts)
    manifest = {"step_id": "figure", "inputs": {"table:exposure_outcome_distribution": {
        "relative_path": str(table.relative_to(tmp_path)), "declared_kind": "table",
        "evidence_kind": "table", "product": "exposure_outcome_distribution",
        "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "product_contract": {"columns": list(pd.read_csv(table).columns), "row_count": 3},
        "consumption_contract": {
            "input_key": "table:exposure_outcome_distribution", "mode": "all_rows",
            "artifact_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        },
    }}}
    figure_dir = tmp_path / "figure"
    run_exposure_outcome_distribution_figure(
        out_dir=figure_dir, run_dir=tmp_path, resolved_inputs=manifest,
        step_id="figure", figure_product="weighted_proportion",
    )
    contract = json.loads((figure_dir / "weighted_proportion.figure_contract.json").read_text())
    assert "Hoeffding" in contract["reader_caption"]
    assert "conditional" in contract["reader_caption"]
    assert "0–100%" in contract["reader_caption"]


@pytest.mark.parametrize("change", ["weights", "assumptions", "covariance", "standard_error", "interval"])
def test_scientific_authority_refuses_bound_receipt_tampering(tmp_path, monkeypatch, change):
    from easyicu.research_agent.authority.scientific_claims import derive_scientific_claim_drafts

    summary, _ = _run_product(tmp_path, monkeypatch)
    altered = copy.deepcopy(summary)
    risk = altered["descriptive_estimates"]["outcome_absolute_risks"][0]
    if change == "weights":
        risk["cluster_size_square_sum"] = 5000
    elif change == "assumptions":
        altered["descriptive_estimates"].pop("patient_weighted_interval_assumptions")
    elif change == "covariance":
        risk["covariance"] = "cluster_robust"
    elif change == "standard_error":
        risk["standard_error_pct"] = 0.0
    else:
        risk["ci_high_pct"] = 65.761977
    with pytest.raises(ValueError):
        derive_scientific_claim_drafts(altered)
