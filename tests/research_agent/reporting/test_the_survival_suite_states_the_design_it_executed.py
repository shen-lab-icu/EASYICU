"""A signed survival suite states the design it executed as a Methods fact.

The strict Methods grammar admits a numeric design detail only as an exact
host fact.  The signed landmark survival suite recorded no executed design,
so every Writer sentence with its landmark, horizon, alpha or interval
cutpoints was deleted, and Methods could not say how the survival model was
run.  The suite now records its design from the sealed contract it executed;
the Methods owner renders it as one exact fact whose numbers bind to the
summary, also when the summary is larger than the per-step numeric cap.  The
Writer is told that Methods numbers are the host's, so it describes the same
choices in words instead of writing sentences the gate deletes.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    LandmarkSurvivalDesign,
    validate_executed_method_design,
)
from easyicu.research_agent.authority.manuscript_claim_policy import filter_evidence_bound_scaffold
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_sections import MANUSCRIPT_SECTION_SPECS
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"


def _summary(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    summary = run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")
    return authority, json.loads(json.dumps(summary))


def _store(tmp_path, summary, *, max_leaves=None):
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic", description="Signed survival suite summary", source_path=source,
        evidence_id=EVIDENCE, produced_by_step=STEP, producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id=EVIDENCE, summary=summary, max_leaves=max_leaves,
    )
    return store, [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]


def _bound_without_loss(store, ledger, fact):
    scaffold = f"## Methods\n\n### Variables\n\n{fact.scaffold}\n"
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold, per_step_records=ledger)
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced


def test_the_summary_records_the_sealed_design(tmp_path):
    authority, summary = _summary(tmp_path)

    design = validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY])

    assert isinstance(design, LandmarkSurvivalDesign)
    assert design.landmark_hours == authority.landmark_hours
    assert design.endpoint_horizon_days == authority.endpoint_horizon_days
    assert design.exposure_window_end_hours == authority.exposure_window_hours[1]
    assert design.prevalent_exposure_cutoff_hours == authority.prevalent_exposure_cutoff_hours
    assert design.n_adjustment_covariates == len(authority.adjustment_columns)
    assert design.proportional_hazards_alpha == authority.proportional_hazards_alpha
    assert tuple(design.time_varying_cutpoints_days) == tuple(authority.time_varying_interval_cutpoints_days)
    assert (design.rmst_horizon_days is None) == (authority.rmst_product is None)


def test_the_design_is_one_exact_bound_methods_fact(tmp_path):
    authority, summary = _summary(tmp_path)
    store, ledger = _store(tmp_path, summary)

    (fact,) = [fact for fact in store.manuscript_method_facts(ledger) if fact.source_field.endswith(EXECUTED_METHOD_DESIGN_KEY)]

    assert fact.text.startswith("Executed survival design: ")
    assert f"a landmark {authority.landmark_hours:g} hours after" in fact.text
    assert f"follow-up ending at day {authority.endpoint_horizon_days:g}" in fact.text
    assert f"alpha of {authority.proportional_hazards_alpha:g}" in fact.text
    assert fact.evidence_id == EVIDENCE
    _bound_without_loss(store, ledger, fact)


def test_the_design_binds_when_the_summary_exceeds_the_numeric_cap(tmp_path):
    _authority, summary = _summary(tmp_path)
    store, ledger = _store(tmp_path, summary, max_leaves=12)

    (fact,) = [fact for fact in store.manuscript_method_facts(ledger) if fact.source_field.endswith(EXECUTED_METHOD_DESIGN_KEY)]

    _bound_without_loss(store, ledger, fact)


def test_a_writer_cannot_forge_the_design_statement(tmp_path):
    _authority, summary = _summary(tmp_path)
    store, ledger = _store(tmp_path, summary)
    # No number, so only the reserved form can stop it.
    forged = (
        "Executed survival design: the risk set comprised records alive at the "
        f"landmark, without any exclusion {{evidence:{EVIDENCE}}}."
    )

    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(
            f"## Methods\n\n### Variables\n\n{forged}\n", per_step_records=ledger,
        )


BASE = {
    "schema_version": "easyicu.executed_method_design/1",
    "design_kind": "landmark_survival",
    "time_origin": "ICU admission",
    "landmark_hours": 24.0,
    "endpoint_horizon_days": 90.0,
    "prevalent_exposure_cutoff_hours": 6.0,
    "exposure_window_end_hours": 24.0,
    "n_adjustment_covariates": 3,
    "effect_model": "cox_proportional_hazards_efron_ties",
    "interval_method": "wald_95_ci",
    "proportional_hazards_test": "schoenfeld_residuals",
    "proportional_hazards_alpha": 0.05,
    "time_varying_cutpoints_days": [7.0, 30.0],
    "rmst_horizon_days": 89.0,
}


@pytest.mark.parametrize(
    "change",
    [
        {"landmark_hours": 2400.0, "time_varying_cutpoints_days": [], "rmst_horizon_days": None},
        {"exposure_window_end_hours": 36.0},
        {"prevalent_exposure_cutoff_hours": 24.0},
        {"time_varying_cutpoints_days": [30.0, 7.0]},
        {"time_varying_cutpoints_days": [7.0, 89.0]},
        {"rmst_horizon_days": 90.0},
    ],
    ids=["landmark_after_horizon", "window_after_landmark", "cutoff_at_window_end",
         "cutpoints_unordered", "cutpoint_at_followup_end", "rmst_not_followup_end"],
)
def test_a_disordered_design_is_refused(change):
    validate_executed_method_design(BASE)
    with pytest.raises(ValueError):
        validate_executed_method_design({**BASE, **change})


def test_the_writer_is_told_methods_numbers_are_host_facts():
    (methods,) = [spec for spec in MANUSCRIPT_SECTION_SPECS if spec.key == "methods"]

    assert "deletes any other Methods sentence that states a number" in methods.instruction
    assert "without restating their numbers" in methods.instruction


def test_the_gate_keeps_the_worded_choice_and_deletes_the_numbered_one():
    numbered = f"Follow-up was censored at 90 days after the landmark {{evidence:{EVIDENCE}}}."
    worded = f"Follow-up was censored at the horizon of the executed design {{evidence:{EVIDENCE}}}."

    result = filter_evidence_bound_scaffold(
        f"## Methods\n\n### Statistical analysis\n\n{numbered}\n\n{worded}\n",
        resolve_claim=lambda ref: None, resolve_evidence=lambda ref: ref == EVIDENCE,
    )

    assert result.removed_result_sentences == (numbered,)
    assert worded in result.scaffold
