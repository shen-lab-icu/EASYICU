"""The signed survival suite reports an interval model the data cannot estimate.

The binary landmark survival suite fits its prespecified piecewise Cox model
whenever its plan seals interval cut points.  A data condition that leaves
that model without an estimate (an exposure group without an event in one
interval, a fit that does not converge, a contrast without a valid variance,
a non-finite estimate, or follow-up ending by the last cut point) ended the
whole run as an executor failure, also while the PH test let the constant
hazard ratio stand as the result.

Now, while the test holds, the interval model is reported as not estimable,
with its reason, and the constant estimate is the result.  When the test
rejects the constant estimate the interval estimates are the result, so the
suite stops with its typed stop: the stop names the method's reason and
leaves a record the host reads.  An input the method does not accept still
fails as before, and a sensitivity re-fit names the same reasons.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality): every death falls in the first four weeks after the
landmark, so no exposure group has an event from day 28.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.authority.survival_scientific_claims import (
    CONSTANT_HAZARD_RATIO_CLAIM_ID,
    SurvivalReporting,
)
from easyicu.research_agent.contracts import executor_stop
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.manuscript_result_structure import (
    PRIMARY_RESULT_HEADINGS_BY_FAMILY,
)
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    build_survival_manuscript_projection,
)
from easyicu.research_agent.methods import time_varying_cox
from easyicu.research_agent.methods.time_varying_cox import TimeVaryingCoxError
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from tests.support.survival_sealed import (
    render_signed_survival_figure,
    run_signed_suite,
    sealed_survival,
    synthetic_crossing_hazard_rows,
    synthetic_survival_rows,
)

pytest.importorskip("lifelines")

SUITE = "signed_landmark_survival_suite"
STOP = "landmark_survival_interval_result_not_estimable"
STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"
TIME_VARYING = "landmark_time_varying_cox_summary.csv"


def _deaths_by(day: float, *, crossing: bool = False) -> pd.DataFrame:
    """The sealed study's rows with every death by ``day`` after ICU admission.

    Survivors stay followed to the 90-day horizon, as the fixed-horizon
    endpoint requires.  Deaths come from proportional exponential hazards
    after the 24-hour landmark, and none after ``day``.  ``crossing`` has
    exposed stays die in the first week after the landmark and comparator
    stays in its third and fourth, so the PH test rejects.
    """

    rng = np.random.default_rng(20261007)
    rows = synthetic_survival_rows()
    n = len(rows)
    exposed = rows["rrt"].to_numpy() == 1
    if crossing:
        time = np.where(exposed, rng.uniform(1.5, 8.0, n), rng.uniform(15.0, day, n))
        died = rows["mort_90d"].to_numpy() == 1
    else:
        time = 1.0 + rng.exponential(1.0 / (0.012 * np.exp(0.6 * exposed)), n)
        died = time < day
    rows["mort_90d"] = died.astype("int64")
    rows["followup_days_90d"] = np.where(died, time, 90.0)
    return rows


def _run(tmp_path, rows, name: str = "suite"):
    out = tmp_path / name / "out"
    out.mkdir(parents=True)  # the runner creates the step's output directory first
    _context, authority = sealed_survival(tmp_path / name)
    summary = json.loads(json.dumps(run_signed_suite(authority, rows, out)))
    return summary, authority, out


def _strict_store(directory, summary):
    """The summary registered as the host runner registers a signed owner's."""

    source = directory / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(directory, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic",
        description="Signed survival suite summary",
        source_path=source,
        evidence_id=EVIDENCE,
        produced_by_step=STEP,
        producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=STEP, evidence_id=EVIDENCE, summary=summary
    )
    return store, [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]


def _untraced(directory, summary) -> list:
    """Every claim of the run in its Results section, bound under the strict gate."""

    store, ledger = _strict_store(directory, summary)
    claims = store.authoritative_scientific_claims(ledger)
    heading = PRIMARY_RESULT_HEADINGS_BY_FAMILY["survival"]
    results = "\n\n".join(claim.placeholder for claim in claims)
    bound = store.bind_manuscript(
        f"## Results\n\n### {heading}\n\n{results}\n", per_step_records=ledger
    )
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    return untraced


def _methods_fact(directory, summary) -> str:
    """The executed design as its Methods fact, bound without an untraced number."""

    store, ledger = _strict_store(directory, summary)
    (fact,) = [
        fact
        for fact in store.manuscript_method_facts(ledger)
        if fact.source_field.endswith(EXECUTED_METHOD_DESIGN_KEY)
    ]
    scaffold = f"## Methods\n\n### Variables\n\n{fact.scaffold}\n"
    safe, removed = store.enforce_evidence_bound_scaffold(
        scaffold, per_step_records=ledger
    )
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=ledger)
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert not untraced
    return fact.text


def test_an_interval_without_an_event_is_reported_not_estimable_while_ph_holds(
    tmp_path,
):
    summary, authority, out = _run(tmp_path, _deaths_by(27.0))

    envelope = summary["reportable_survival_results"]
    assert (
        envelope["proportional_hazards_test"]["disposition"]
        == "assumption_not_rejected"
    )
    assert envelope["constant_hazard_ratio_authorized"] is True
    assert envelope["time_varying_adjusted_association"] == {
        "status": "not_estimable",
        "reason": "interval_without_event",
        "method": "piecewise_time_varying_cox",
        "adjustment_columns": list(authority.adjustment_columns),
    }
    SurvivalReporting.model_validate(envelope)
    # The constant estimate is the result; no interval estimate is claimed.
    claims = derive_scientific_claim_drafts(summary)
    assert claims[0].claim_id == CONSTANT_HAZARD_RATIO_CLAIM_ID
    assert claims[0].analysis_role == "primary"
    assert not [claim for claim in claims if claim.claim_id.startswith("interval_")]
    abstract = [
        claim["scientific_claim_id"]
        for claim in envelope["manuscript_projection"]["claims"]
        if claim["claim_id"].startswith("abstract_")
    ]
    assert abstract == [CONSTANT_HAZARD_RATIO_CLAIM_ID]
    # The table, the receipt and the executed design say why.
    table = pd.read_csv(out / TIME_VARYING)
    assert table[["term", "model_status", "not_estimable_reason"]].to_dict(
        "records"
    ) == [
        {
            "term": authority.derived_exposure_column,
            "model_status": "not_estimable",
            "not_estimable_reason": "interval_without_event",
        }
    ]
    assert (
        summary["scientific_runtime_receipt"]["time_varying_status"]
        == "interval_without_event"
    )
    design = validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY])
    assert design.interval_model_not_estimable_reason == "interval_without_event"
    assert (
        "a piecewise Cox model split at days 7, 14 and 28 after the landmark was prespecified "
        "for interval-specific contrasts and was not estimable, because a follow-up interval "
        "had no event"
    ) in _methods_fact(tmp_path / "methods", summary)
    # Every number the claims print binds to the run.
    assert _untraced(tmp_path / "results", summary) == []
    # The figure draws the constant estimate, as it does whenever the test holds.
    render_signed_survival_figure(authority, out, tmp_path / "figure")
    receipt = json.loads(
        (
            tmp_path / "figure" / "landmark_survival_figure_runtime_receipt.json"
        ).read_text()
    )
    assert receipt["promoted_effect_measure"] == authority.effect_measure


def test_an_estimated_interval_model_keeps_the_envelope_it_always_had(tmp_path):
    summary, _authority, out = _run(tmp_path, synthetic_survival_rows())

    association = summary["reportable_survival_results"][
        "time_varying_adjusted_association"
    ]
    assert set(association) == {"method", "adjustment_columns", "intervals"}
    assert len(association["intervals"]) == 4
    assert summary["scientific_runtime_receipt"]["time_varying_status"] == "estimated"
    assert set(pd.read_csv(out / TIME_VARYING)["model_status"]) == {"estimated"}
    design = summary[EXECUTED_METHOD_DESIGN_KEY]
    assert "interval_model_not_estimable_reason" not in design
    assert (
        "interval-specific contrasts came from a piecewise Cox model split at days 7, 14 and 28"
        in (_methods_fact(tmp_path / "methods", summary))
    )


def test_a_rejected_ph_test_without_interval_estimates_stops_with_its_reason(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(executor_stop.ExecutorStop) as caught:
        run_signed_suite(authority, _deaths_by(27.0, crossing=True), out)

    stop = caught.value
    assert (stop.owner, stop.reason_code, stop.cause_code) == (
        SUITE,
        STOP,
        "interval_without_event",
    )
    assert str(stop).startswith(f"{STOP}: ")
    assert isinstance(stop.__cause__, TimeVaryingCoxError)
    assert stop.__cause__.reason == "interval_without_event"
    # No table was written: the stop's record is the only file.
    record = out / executor_stop.EXECUTOR_STOP_RECORD_NAME
    assert [path.name for path in out.iterdir()] == [record.name]
    recorded = executor_stop.parse_executor_stop_record(
        record.read_bytes(), expected_owner=SUITE
    )
    assert (recorded.reason_code, recorded.cause_code) == (
        STOP,
        "interval_without_event",
    )
    step_record = {
        "deterministic_standard_analysis": SUITE,
        "executor_stop_reason_code": STOP,
        "executor_stop_cause_code": "interval_without_event",
    }
    assert executor_stop.executor_stop_codes(step_record) == {
        "reason_code": STOP,
        "cause_code": "interval_without_event",
    }
    assert executor_stop.EXECUTOR_STOP_REASONS[STOP].repeats_on_unchanged_retry is True
    # The stop is this suite's, and names only the method's data conditions.
    assert executor_stop.registered_executor_stop(STOP, "invalid_input") is None
    assert (
        executor_stop.registered_executor_stop(
            STOP, "did_not_converge", owner="signed_landmark_continuous_survival_suite"
        )
        is None
    )


def test_a_rejected_ph_test_with_interval_estimates_reports_them(tmp_path):
    summary, _authority, _out = _run(tmp_path, synthetic_crossing_hazard_rows())

    envelope = summary["reportable_survival_results"]
    assert envelope["proportional_hazards_test"]["disposition"] == "assumption_rejected"
    assert len(envelope["time_varying_adjusted_association"]["intervals"]) == 4
    assert summary["scientific_runtime_receipt"]["time_varying_status"] == "estimated"


def test_an_input_the_interval_model_refuses_still_fails_as_before(
    tmp_path, monkeypatch
):
    def refuse(*_args, **_kwargs):
        raise TimeVaryingCoxError("time-varying Cox covariates must be unique")

    monkeypatch.setattr(time_varying_cox, "fit_piecewise_time_varying_cox", refuse)

    with pytest.raises(TimeVaryingCoxError, match="must be unique") as caught:
        _run(tmp_path, synthetic_survival_rows())
    assert not isinstance(caught.value, executor_stop.ExecutorStop)


def test_a_sensitivity_refit_names_why_its_interval_model_had_no_estimate(
    tmp_path, monkeypatch
):
    fit = time_varying_cox.fit_piecewise_time_varying_cox
    calls = []

    def primary_only(*args, **kwargs):
        # The primary interval model is estimated; each re-fit's is not.
        calls.append(1)
        if len(calls) == 1:
            return fit(*args, **kwargs)
        raise TimeVaryingCoxError("no event", reason="interval_without_event")

    monkeypatch.setattr(
        time_varying_cox, "fit_piecewise_time_varying_cox", primary_only
    )

    summary, _authority, out = _run(tmp_path, synthetic_crossing_hazard_rows())

    assert len(calls) > 1
    table = pd.read_csv(out / "landmark_prevalence_sensitivity.csv")
    refits = table.loc[table["analysis"] == "sensitivity"]
    assert set(refits["not_reported_reason"]) == {
        "the interval model is not estimable: interval_without_event"
    }
    fits = summary["reportable_survival_results"]["prevalence_definition_sensitivity"][
        "fits"
    ]
    assert all("interval_hazard_ratios" not in item for item in fits)


def test_the_envelope_refuses_an_interval_model_that_contradicts_its_test(tmp_path):
    held, _authority, _out = _run(tmp_path, _deaths_by(27.0), name="held")
    rejected, _authority, _out = _run(
        tmp_path, synthetic_crossing_hazard_rows(), name="rejected"
    )
    not_estimable = held["reportable_survival_results"][
        "time_varying_adjusted_association"
    ]

    # A rejected test makes the interval estimates the result.
    forged = copy.deepcopy(rejected["reportable_survival_results"])
    forged["time_varying_adjusted_association"] = copy.deepcopy(not_estimable)
    forged.pop("prevalence_definition_sensitivity", None)
    with pytest.raises(ValueError, match="makes the interval estimates the result"):
        SurvivalReporting.model_validate(forged)
    # A model the data left without an estimate carries no interval and a known reason.
    for change in (
        {
            "intervals": rejected["reportable_survival_results"][
                "time_varying_adjusted_association"
            ]["intervals"]
        },
        {"reason": "invalid_input"},
        {"adjustment_columns": []},
    ):
        envelope = copy.deepcopy(held["reportable_survival_results"])
        envelope["time_varying_adjusted_association"].update(change)
        with pytest.raises(ValueError):
            SurvivalReporting.model_validate(envelope)


def test_the_design_states_a_reason_only_for_a_prespecified_interval_model(tmp_path):
    summary, _authority, _out = _run(tmp_path, _deaths_by(27.0))
    design = dict(summary[EXECUTED_METHOD_DESIGN_KEY])

    without_intervals = {**design, "time_varying_cutpoints_days": []}
    with pytest.raises(ValueError, match="only a prespecified interval model"):
        validate_executed_method_design(without_intervals)
    with pytest.raises(ValueError):
        validate_executed_method_design(
            {**design, "interval_model_not_estimable_reason": "invalid_input"}
        )


def test_the_projection_asks_for_interval_claims_only_when_they_are_the_result():
    held = build_survival_manuscript_projection(
        interval_count=0, proportional_hazards_rejected=False
    )
    assert [
        claim["scientific_claim_id"]
        for claim in held["claims"]
        if "scientific_claim_id" in claim
    ] == [CONSTANT_HAZARD_RATIO_CLAIM_ID]
    with pytest.raises(ValueError, match="requires intervals"):
        build_survival_manuscript_projection(
            interval_count=0, proportional_hazards_rejected=True
        )
